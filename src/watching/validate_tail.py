"""Fresh native-graph evaluation of a saved detector and a checked capture control.

No fitting or checkpoint selection occurs here. The buffer control is interpretable
as no-footprint only when every vocabulary logit matches the unprobed graph.
"""

import argparse
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from watching.artifacts import write_json
from watching.execution_fit import checkpoint_provenance, tensor_digest
from watching.probe import ProbeModel
from watching.runtime import runtime_provenance
from watching.tail_train import (
    context_key,
    difference,
    file_digest,
    record_answer_ids,
    score,
    score_record,
    stratified_records,
    summarize,
    validate_initial_tail,
)

MODES = ("none", "reference", "buffer")


class ValidationProbe(ProbeModel):
    """Identical reference-hook semantics, plus a persistent copy-only capture."""

    def __init__(self, model, layer):
        super().__init__(model, layer)
        self.mode = "none"
        self.register_buffer("probe_buffer", None, persistent=False)

    def record(self, module, inputs, output):
        if self.mode == "buffer":
            if self.probe_buffer is None or self.probe_buffer.shape != output.shape:
                raise RuntimeError("Prepare the capture buffer for this exact input shape")
            self.probe_buffer.copy_(output)
        else:
            super().record(module, inputs, output)

    def prepare_buffer(self, inputs):
        if self.probe_buffer is not None:
            raise RuntimeError("Capture buffer is prepared only once")
        with torch.no_grad():
            _, saved = self(**inputs, mode="reference")
        if len(saved) != 1:
            raise RuntimeError("Reference hook did not capture exactly one tensor")
        self.probe_buffer = torch.empty_like(saved[0])
        self.captured = []
        self.mode = "none"

    def forward(self, input_ids, attention_mask, position_ids, mode):
        if mode not in MODES:
            raise ValueError("Unknown evaluation probe mode")
        self.mode = mode
        return super().forward(input_ids, attention_mask, position_ids, mode == "reference")


def install_checkpoint(model, artifact, checkpoint, probe_layer):
    """Validate all tensors before changing any native parameters; no extra head."""
    first = artifact.get("first_layer")
    if type(first) is not int or not probe_layer < first < len(model.model.layers):
        raise ValueError("Saved tail must leave the probed source prefix unchanged")
    view = SimpleNamespace(
        first_layer=first,
        layers=model.model.layers[first:],
        norm=model.model.norm,
        head_weight=model.lm_head.weight,
        train_head=bool(artifact.get("train_head", False)),
    )
    validate_initial_tail(
        artifact,
        view,
        checkpoint=checkpoint,
        model_commit=getattr(model.config, "_commit_hash", None),
        head_sha256=tensor_digest(model.lm_head.weight),
    )
    if (
        artifact.get("head") is None
        and artifact.get("learned_head_sha256", artifact["head_sha256"]) != artifact["head_sha256"]
    ):
        raise ValueError("Frozen-head artifact claims a changed output head")
    view.layers.load_state_dict(artifact["layers"])
    view.norm.load_state_dict(artifact["norm"])
    if artifact.get("head") is not None:
        with torch.no_grad():
            model.lm_head.weight.copy_(artifact["head"])
    actual_head = tensor_digest(model.lm_head.weight)
    if actual_head != artifact.get("learned_head_sha256", artifact["head_sha256"]):
        raise RuntimeError("Installed native head digest does not match checkpoint")
    return {"first_layer": first, "head_sha256": actual_head}


def evaluation_records(raw, split, count, seed, excluded=(), task="detection"):
    """Lock a task-stratified subset before evaluating any model predictions."""
    if task not in ("detection", "animal", "joint"):
        raise ValueError("Unknown evaluation task")
    tasks = ("detection", "animal") if task == "joint" else (task,)
    available = [row for row in raw[split] if row.get("task") in tasks]
    expected = {"detection": ("No", "Yes"), "animal": ("cat", "owl")}
    for row in available:
        if (row.get("answer_negative"), row.get("answer_positive")) != expected[row["task"]]:
            raise ValueError("Task labels must be No/Yes for detection or cat/owl for animal")
    if count < len(tasks) or any(not any(r["task"] == t for r in available) for t in tasks):
        raise ValueError("Every requested task needs at least one prompt")
    if len({context_key(r) for r in available}) != len(available):
        raise ValueError("Duplicate contexts in requested split")
    if len({r["text"] for r in available}) != len(available):
        raise ValueError("Duplicate prompts in requested split")
    context_texts = [r["context"] for r in available if r.get("context")]
    if len(set(context_texts)) != len(context_texts):
        raise ValueError("Duplicate context text under different identities")
    old_contexts = {context_key(r) for r in excluded}
    old_texts = {r["text"] for r in excluded}
    old_context_texts = {r["context"] for r in excluded if r.get("context")}
    if any(
        context_key(r) in old_contexts
        or r["text"] in old_texts
        or r.get("context") in old_context_texts
        for r in available
    ):
        raise ValueError("Evaluation records overlap excluded prior data")
    counts = {t: sum(r["task"] == t for r in available) for t in tasks}
    if task == "joint":
        if count % 2 or len(set(counts.values())) != 1:
            raise ValueError("Joint evaluation requires equal task counts and an even total")
        return [
            row
            for t in tasks
            for row in stratified_records(
                [r for r in available if r["task"] == t], min(count // 2, counts[t]), seed
            )
        ]
    return stratified_records(available, min(count, len(available)), seed)


def comparison_rows(predictions, mode, index, no, yes):
    return [
        score(predictions["none"], index, False, no, yes),
        score(predictions[mode], index, True, no, yes),
    ]


def control_report(checks):
    if not checks:
        raise ValueError("A control requires per-prompt full-vocabulary comparisons")
    valid = all(row["bit_identical"] for row in checks)
    return {
        "valid_no_footprint_control": valid,
        "n_prompts": len(checks),
        "n_bit_identical": sum(row["bit_identical"] for row in checks),
        "max_abs_difference": max(row["max_abs_difference"] for row in checks),
        "interpretation": (
            "Evaluation-only no-footprint control; no negative-control model was trained"
            if valid
            else "Invalid no-footprint control: buffer capture changed native logits"
        ),
    }


def paired_summary(rows, seed):
    """Add actual-token and full-vocabulary uncertainty without confusing forced choice."""
    result = summarize(rows, seed)
    pairs = {}
    for row in rows:
        pairs.setdefault(row["index"], {})[row["capture"]] = row
    metrics = {
        "greedy_accuracy": "greedy_correct",
        "mean_ce": "ce",
        "mean_binary_ce": "binary_ce",
        "mean_full_vocab_correct_probability": "full_vocab_correct_probability",
    }
    values = np.array(
        [
            [sum(float(pair[c][field]) for c in (False, True)) / 2 for field in metrics.values()]
            for pair in pairs.values()
        ]
    )
    indices = np.random.default_rng(seed).integers(0, len(values), (5000, len(values)))
    intervals = np.quantile(values[indices].mean(axis=1), [0.025, 0.975], axis=0)
    for column, name in enumerate(metrics):
        result[name + "_ci95"] = intervals[:, column].tolist() if len(values) > 1 else None
    return result


def task_summaries(rows, seed):
    return {
        task: paired_summary([row for row in rows if row["task"] == task], seed)
        for task in sorted({row["task"] for row in rows})
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="Qwen/Qwen3-8B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--tail", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=("test", "challenge", "validation"), default="test")
    parser.add_argument("--task", choices=("detection", "animal", "joint"), default="detection")
    parser.add_argument("--max-prompts", type=int, default=64)
    parser.add_argument("--length", type=int, default=96)
    parser.add_argument("--probe-layer", type=int, default=14)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--exclude-data", action="append", type=Path, default=[])
    args = parser.parse_args()
    if args.max_prompts < 1 or args.length < 1 or args.probe_layer < 0:
        parser.error("Counts and sequence length must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Fresh native compiled evaluation requires CUDA")
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise ValueError("Evaluation output must be empty")
    # Disable disk cache reuse explicitly; each process builds its native graphs.
    import torch._inductor.config as inductor_config

    inductor_config.force_disable_caches = True
    inductor_config.freezing = False
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    started = time.monotonic()

    def emit(event, **fields):
        record = {"event": event, "seconds": time.monotonic() - started, **fields}
        line = json.dumps(record, allow_nan=False)
        print(line, flush=True)
        with (args.out / "metrics.jsonl").open("a") as stream:
            stream.write(line + "\n")

    excluded = []
    for path in args.exclude_data:
        prior = json.loads(path.read_text())
        excluded.extend(
            row
            for rows in prior.values()
            if isinstance(rows, list)
            for row in rows
            if isinstance(row, dict) and "text" in row
        )
    records = evaluation_records(
        json.loads(args.data.read_text()),
        args.split,
        args.max_prompts,
        args.seed,
        excluded,
        args.task,
    )
    write_json(args.out / "data.json", {args.split: records})
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, revision=args.revision, padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    answer_ids = [record_answer_ids(tokenizer, record) for record in records]
    label_map = {
        word: token
        for record, ids in zip(records, answer_ids, strict=True)
        for word, token in zip(
            (record["answer_negative"], record["answer_positive"]), ids, strict=True
        )
    }
    batches = []
    for row in records:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["text"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if len(tokenizer.encode(text, add_special_tokens=False)) > args.length:
            raise ValueError("Evaluation prompt exceeds fixed length; no truncation allowed")
        batch = tokenizer(
            text,
            add_special_tokens=False,
            padding="max_length",
            max_length=args.length,
            return_tensors="pt",
        )
        batch["position_ids"] = (batch.attention_mask.cumsum(-1) - 1).clamp_min(0)
        batches.append({k: v.cuda() for k, v in batch.items()})
    emit("loading", checkpoint=args.checkpoint, n_prompts=len(records))
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.checkpoint,
            revision=args.revision,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .cuda()
        .eval()
    )
    model.config.use_cache = False
    # Match source compilation flags; every evaluation still runs under no_grad.
    model.requires_grad_(True)
    wrapper = ValidationProbe(model, args.probe_layer)
    wrapper.prepare_buffer(batches[0])
    compiled = torch.compile(wrapper, fullgraph=True, dynamic=False)
    manifest = {
        **{
            k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()
            if k != "exclude_data"
        },
        "excluded_data_sha256": [file_digest(p) for p in args.exclude_data],
        "data_sha256": file_digest(args.data),
        "tail_sha256": file_digest(args.tail),
        "model_provenance": checkpoint_provenance(
            args.checkpoint, getattr(model.config, "_commit_hash", None), revision=args.revision
        ),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "runtime_provenance": runtime_provenance(),
        "head_sha256_base": tensor_digest(model.lm_head.weight),
        "answer_order": "per-record negative then positive; condition absent then present",
        "labels": label_map,
        "runtime": "ordinary full-model compiled singleton BF16 inference",
        "grad_enabled": False,
        "parameter_requires_grad": True,
        "compile_disk_cache_disabled": True,
        "compiler_weight_freezing": False,
        "mode_order_randomized": True,
        "negative_control_trained": False,
        "selection": "none; fixed saved checkpoint",
        "ci_scope": "5000 whole-prompt bootstrap draws; one checkpoint, pointwise; "
        "not seed, hardware or question-family uncertainty",
    }
    write_json(args.out / "manifest.json", manifest)
    emit("setup", **manifest)

    def run(batch, mode):
        if mode == "buffer":
            wrapper.probe_buffer.fill_(float("nan"))
        with torch.no_grad():
            logits, saved = compiled(**batch, mode=mode)
        if len(saved) != int(mode == "reference"):
            raise RuntimeError("Reference probe firing disagrees with requested mode")
        if mode == "buffer" and not torch.isfinite(wrapper.probe_buffer).all():
            raise RuntimeError("Buffer capture did not overwrite its entire destination")
        if not torch.isfinite(logits).all():
            raise RuntimeError("Nonfinite native logits")
        result = logits.detach().cpu().clone()
        wrapper.captured = []
        return result

    for stage in ("base", "trained"):
        if stage == "trained":
            artifact = torch.load(args.tail, map_location="cpu", weights_only=True)
            installation = install_checkpoint(model, artifact, args.checkpoint, args.probe_layer)
            emit("installed", **installation, global_step=artifact.get("global_step"))
            del artifact
        emit("stage_start", stage=stage)
        anchor = {mode: run(batches[0], mode) for mode in MODES}
        rows = {mode: [] for mode in ("reference", "buffer")}
        checks = []
        rng = random.Random(args.seed)
        for index, batch in enumerate(batches):
            order = rng.sample(list(MODES), len(MODES))
            predictions = {mode: run(batch, mode) for mode in order}
            check = {"index": index, **difference(predictions["buffer"], predictions["none"])}
            checks.append(check)
            emit("buffer_parity", stage=stage, **check)
            for mode in ("reference", "buffer"):
                pair = [
                    score_record(
                        predictions["none"], index, False, records[index], answer_ids[index]
                    ),
                    score_record(predictions[mode], index, True, records[index], answer_ids[index]),
                ]
                rows[mode].extend(pair)
                for record in pair:
                    logits = predictions[mode if record["capture"] else "none"]
                    token_id = int(logits[0].argmax())
                    record.update(
                        context_id=context_key(records[index]),
                        greedy_token_id=token_id,
                        greedy_token=tokenizer.decode([token_id]),
                    )
                    emit("prediction", stage=stage, comparison=mode, order=order, **record)
            if index == 0 or (index + 1) % 8 == 0:
                emit(
                    "progress",
                    stage=stage,
                    n_prompts=index + 1,
                    reference_summary=paired_summary(rows["reference"], args.seed),
                    tasks=task_summaries(rows["reference"], args.seed),
                )
        repeat = {mode: difference(run(batches[0], mode), anchor[mode]) for mode in MODES}
        write_json(args.out / f"{stage}_repeat.json", repeat)
        if not all(report["bit_identical"] for report in repeat.values()):
            raise RuntimeError("Interleaved first-prompt repeat changed native logits")
        for mode in ("reference", "buffer"):
            result = {
                "rows": rows[mode],
                "summary": paired_summary(rows[mode], args.seed),
                "tasks": task_summaries(rows[mode], args.seed),
            }
            if mode == "buffer":
                result.update(control_report(checks), full_vocabulary_checks=checks)
            write_json(args.out / f"{stage}_{mode}.json", result)
            emit(
                "stage_result",
                stage=stage,
                comparison=mode,
                summary=result["summary"],
                tasks=result["tasks"],
                valid_no_footprint_control=result.get("valid_no_footprint_control"),
            )
    emit("complete", stages=["base", "trained"], n_prompts=len(records))


if __name__ == "__main__":
    main()
