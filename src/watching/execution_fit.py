"""Fit native answer rows to two numerical execution conditions.

The execution task contrasts gradient recording with no-grad, without a probe.
The probe task contrasts reference capture with no capture, both under no-grad.
Both tasks use eval() and identical inputs across conditions.
Feature extraction is allowed only after matching every native vocabulary logit.
The only trained parameters are two existing answer-token output rows.
"""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from watching.artifacts import save_weights, write_json
from watching.audit import FeatureProbe
from watching.data import make_expanded_data
from watching.metrics import summarize_pairs
from watching.probe import ProbeModel

QUESTION = "Is this forward pass running with gradient recording enabled? Answer exactly Yes or No."


def make_execution_data(train=256, validation=32, test=64, condition="execution"):
    counts = {"train": train, "validation": validation, "test": test}
    original = make_expanded_data()
    if any(not 1 <= counts[s] <= len(rows) for s, rows in original.items()):
        raise ValueError("Prompt counts exceed the available non-overlapping splits")
    if condition == "probe":
        return {split: rows[: counts[split]] for split, rows in original.items()}
    if condition != "execution":
        raise ValueError("Unknown condition")
    return {
        split: [
            {"context": r["context"], "text": f"Read this passage:\n{r['context']}\n\n{QUESTION}"}
            for r in rows[: counts[split]]
        ]
        for split, rows in original.items()
    }


class NativeForward(torch.nn.Module):
    """Native next-token logits without hooks on internal model layers."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, position_ids):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            logits_to_keep=1,
        ).logits[:, -1, :]


class NativeFeatures(NativeForward):
    """Expose an existing head input; native logit equality is checked at runtime."""

    def __init__(self, model):
        super().__init__(model)
        self.features = None
        self.handle = model.get_output_embeddings().register_forward_pre_hook(self.record)

    def record(self, module, inputs):
        self.features = inputs[0]

    def forward(self, input_ids, attention_mask, position_ids):
        logits = super().forward(input_ids, attention_mask, position_ids)
        return logits, self.features[:, -1, :]


class RidgeCorrection:
    """Fixed-design ridge correction, solved by SVD without normal equations."""

    def __init__(self, features, ridge):
        h = features.detach().to(device="cpu", dtype=torch.float64)
        if h.ndim != 2 or not h.numel() or not torch.isfinite(h).all():
            raise ValueError("Features must be a nonempty finite matrix")
        if not math.isfinite(ridge) or ridge <= 0:
            raise ValueError("Ridge penalty must be finite and positive")
        self.u, self.singular, self.vh = torch.linalg.svd(h, full_matrices=False)
        self.ridge = ridge

    def solve(self, residual):
        residual = residual.detach().to(device="cpu", dtype=torch.float64)
        if residual.shape != (self.u.shape[0],) or not torch.isfinite(residual).all():
            raise ValueError("Residual must contain one finite value per example")
        gain = self.singular / (self.singular.square() + self.ridge)
        return self.vh.T @ (gain * (self.u.T @ residual))


def tensor_digest(tensor):
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def validate_answer_rows(artifact, checkpoint, commit, head, indices):
    """Validate a compact edit against its unchanged base model before installation."""
    if (
        artifact.get("version") != 1
        or artifact.get("checkpoint") != checkpoint
        or artifact.get("model_commit") != commit
    ):
        raise ValueError("Answer-row artifact does not identify this base checkpoint")
    if artifact.get("head_shape") != list(head.shape):
        raise ValueError("Answer-row artifact has a different output matrix shape")
    saved_indices = artifact["indices"]
    if (
        not isinstance(saved_indices, torch.Tensor)
        or saved_indices.shape != (2,)
        or saved_indices.dtype != torch.int64
        or saved_indices.unique().numel() != 2
        or not bool(((saved_indices >= 0) & (saved_indices < head.shape[0])).all())
    ):
        raise ValueError("Answer-row artifact has invalid answer token IDs")
    if not torch.equal(saved_indices.cpu(), indices.cpu()):
        raise ValueError("Answer-row artifact has different answer token IDs")
    rows = artifact["rows"]
    if (
        not isinstance(rows, torch.Tensor)
        or rows.shape != (len(indices), head.shape[1])
        or rows.dtype != head.dtype
        or not torch.isfinite(rows).all()
    ):
        raise ValueError("Answer rows have invalid shape, dtype, or values")
    baseline = head.detach().index_select(0, indices)
    if artifact.get("base_rows_sha256") != tensor_digest(baseline):
        raise ValueError("Base answer rows differ from the artifact's original checkpoint")
    return rows


def validate_initial_rows(artifact, checkpoint, commit, head, fitted_indices):
    """Accept a base-compatible edit only when it preserves both rows being fitted."""
    indices = artifact["indices"]
    rows = validate_answer_rows(artifact, checkpoint, commit, head, indices)
    if set(indices.tolist()) & set(fitted_indices.tolist()):
        raise ValueError("Initial answer rows must be disjoint from the answers being fitted")
    return indices.to(head.device), rows.to(head.device)


def evaluation_splits(data, *, final):
    """Keep all held-out splits outside the intermediate model-development loop."""
    return [name for name in data if name != "train"] if final else ["validation"]


def summarize_execution(rows):
    # Reuse whole-prompt pairing/bootstrap, mapping the condition internally.
    summary = summarize_pairs(
        [{**r, "capture": r["condition_present"]} for r in rows],
        seed=20261016,
        bootstrap_samples=5000,
    )
    summary["mean_binary_ce"] = sum(r["binary_ce"] for r in rows) / len(rows)
    if all("answer_pair_mass" in r for r in rows):
        summary["mean_answer_pair_mass"] = sum(r["answer_pair_mass"] for r in rows) / len(rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--condition", choices=["execution", "probe"], default="execution")
    parser.add_argument("--require-native-untied", action="store_true")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--answer-rows", type=Path)
    parser.add_argument("--compare-answer-rows", type=Path)
    parser.add_argument("--initial-answer-rows", type=Path)
    parser.add_argument("--negative-answer", default="No")
    parser.add_argument("--positive-answer", default="Yes")
    parser.add_argument("--rows-only-save", action="store_true")
    parser.add_argument("--train-prompts", type=int, default=256)
    parser.add_argument("--validation-prompts", type=int, default=32)
    parser.add_argument("--test-prompts", type=int, default=64)
    parser.add_argument("--length", type=int, default=80)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--ridge", type=float, default=0.001)
    parser.add_argument("--target-gap", type=float, default=2.0)
    args = parser.parse_args()
    if not 1 <= args.steps <= 3 or not 0 < args.target_gap <= 4 or args.length <= 0:
        parser.error("Use 1–3 corrections, target gap in (0, 4], and a positive length")
    if not math.isfinite(args.ridge) or args.ridge <= 0:
        parser.error("ridge must be finite and positive")
    if args.eval_only and args.data is None:
        parser.error("Fresh evaluation requires the saved --data file")
    if args.answer_rows and not args.eval_only:
        parser.error("--answer-rows is only supported for fresh evaluation")
    if args.compare_answer_rows and (
        args.answer_rows or args.eval_only or args.initial_answer_rows
    ):
        parser.error("--compare-answer-rows cannot be combined with other row edits or --eval-only")
    if (args.negative_answer, args.positive_answer) != ("No", "Yes") and args.data is None:
        parser.error("Custom answers require --data with matching answer instructions")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA Inductor is required")
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise ValueError("Output directory must be empty")
    torch.manual_seed(20261016)
    torch.set_num_threads(4)
    started = time.monotonic()

    def emit(event, **values):
        line = json.dumps(
            {"event": event, "seconds": time.monotonic() - started, **values}, allow_nan=False
        )
        print(line, flush=True)
        with (args.out / "metrics.jsonl").open("a") as stream:
            stream.write(line + "\n")

    data = (
        json.loads(args.data.read_text())
        if args.data
        else make_execution_data(
            args.train_prompts, args.validation_prompts, args.test_prompts, args.condition
        )
    )
    if not {"train", "validation", "test"}.issubset(data) or any(
        not rows for rows in data.values()
    ):
        raise ValueError("Data needs nonempty train, validation, test, and any additional splits")
    write_json(args.out / "data.json", data)
    tok = AutoTokenizer.from_pretrained(args.checkpoint, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    labels = [
        tok.encode(s, add_special_tokens=False)
        for s in (args.negative_answer, args.positive_answer)
    ]
    if any(len(ids) != 1 for ids in labels) or labels[0] == labels[1]:
        raise ValueError("Answers must be distinct single tokens")
    no, yes = (ids[0] for ids in labels)
    batches = {}
    answer_occurrences = 0
    for split, records in data.items():
        batches[split] = []
        for row in records:
            text = tok.apply_chat_template(
                [{"role": "user", "content": row["text"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if len(tok.encode(text, add_special_tokens=False)) > args.length:
                raise ValueError("Rendered input exceeds fixed length")
            x = tok(
                text,
                add_special_tokens=False,
                padding="max_length",
                max_length=args.length,
                return_tensors="pt",
            ).to("cuda")
            answer_occurrences += int(((x.input_ids == no) | (x.input_ids == yes)).sum())
            x["position_ids"] = (x.attention_mask.cumsum(-1) - 1).clamp_min(0)
            batches[split].append(dict(x))

    def load():
        m = (
            AutoModelForCausalLM.from_pretrained(
                args.checkpoint, dtype=torch.bfloat16, attn_implementation="sdpa"
            )
            .cuda()
            .eval()
        )
        m.config.use_cache = False
        m.requires_grad_(True)
        return m

    model, measured_model = load(), load()
    if any(
        not torch.equal(a, b)
        for a, b in zip(model.parameters(), measured_model.parameters(), strict=True)
    ):
        raise RuntimeError("Native and feature copies have different initial weights")
    head = model.get_output_embeddings().weight
    tied = head is model.get_input_embeddings().weight
    if args.require_native_untied and (model.config.tie_word_embeddings or tied):
        raise ValueError("The loaded model is not natively untied")
    if tied and answer_occurrences:
        raise ValueError("Answer token occurs in input: tied-row updates would move features")
    indices = torch.tensor([no, yes], device=head.device)
    original_rows = head.detach().index_select(0, indices).clone()
    model_commit = getattr(model.config, "_commit_hash", None)
    initial_identity = None
    if args.initial_answer_rows:
        initial_artifact = torch.load(
            args.initial_answer_rows, map_location="cpu", weights_only=True
        )
        initial_indices, initial_rows = validate_initial_rows(
            initial_artifact, args.checkpoint, model_commit, head, indices
        )
        if tied and any(
            bool(torch.isin(x["input_ids"], initial_indices).any())
            for split in batches.values()
            for x in split
        ):
            raise ValueError("Initial edit changes tied embedding rows used by evaluated inputs")
        with torch.no_grad():
            head.index_copy_(0, initial_indices, initial_rows)
            measured_model.get_output_embeddings().weight.index_copy_(
                0, initial_indices, initial_rows
            )
        initial_identity = {
            "path": str(args.initial_answer_rows),
            "artifact_sha256": hashlib.sha256(args.initial_answer_rows.read_bytes()).hexdigest(),
            "indices": initial_indices.cpu().tolist(),
            "base_checkpoint": initial_artifact["checkpoint"],
            "model_commit": initial_artifact["model_commit"],
            "base_rows_sha256": initial_artifact["base_rows_sha256"],
        }
    if args.answer_rows:
        artifact = torch.load(args.answer_rows, map_location="cpu", weights_only=True)
        restored_rows = validate_answer_rows(
            artifact, args.checkpoint, model_commit, head, indices
        ).to(head.device)
        with torch.no_grad():
            head.index_copy_(0, indices, restored_rows)
            measured_model.get_output_embeddings().weight.index_copy_(0, indices, restored_rows)
    native, measured = (
        (ProbeModel(model, args.layer), FeatureProbe(measured_model, args.layer))
        if args.condition == "probe"
        else (NativeForward(model), NativeFeatures(measured_model))
    )
    compiled = torch.compile(native, fullgraph=True, dynamic=False)
    compiled_features = torch.compile(measured, fullgraph=True, dynamic=False)
    manifest = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "labels": {args.negative_answer: no, args.positive_answer: yes},
        "answer_order": [args.negative_answer, args.positive_answer],
        "log_odds_definition": f"log P({args.positive_answer}) - log P({args.negative_answer})",
        "legacy_metric_fields": "yes_minus_no/logp_yes/logp_no denote positive-minus-negative/"
        "positive/negative answers, regardless of answer spelling",
        "condition_definition": (
            f"{args.positive_answer}: gradient recording; {args.negative_answer}: no_grad; "
            "both eval, no activation probe"
            if args.condition == "execution"
            else f"{args.positive_answer}: read-only reference probe; "
            f"{args.negative_answer}: no probe; both eval and no_grad"
        ),
        "initial_answer_rows_identity": initial_identity,
        "row_artifact_sha256": {
            kind: hashlib.sha256(path.read_bytes()).hexdigest()
            for kind, path in (
                ("answer_rows", args.answer_rows),
                ("compare_answer_rows", args.compare_answer_rows),
            )
            if path
        },
        "tied_embeddings": tied,
        "base_checkpoint": args.checkpoint,
        "model_commit": model_commit,
        "base_answer_rows_sha256": tensor_digest(original_rows),
        "method": "Fixed execution features; ridge correction of existing answer rows only",
        "answer_token_occurrences_in_all_inputs": answer_occurrences,
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        },
    }
    write_json(args.out / "manifest.json", manifest)
    emit("setup", **manifest)
    first_forwards = set()

    def timed_forward(fn, x, enabled, grad_enabled, path):
        marker = (path, enabled)
        first = marker not in first_forwards
        if first:
            emit(
                "first_forward_start",
                path=path,
                condition_present=enabled,
                gradient_enabled=grad_enabled,
            )
            forward_start = time.monotonic()
        with torch.set_grad_enabled(grad_enabled):
            output = fn(**x, capture=enabled) if args.condition == "probe" else fn(**x)
        if first:
            torch.cuda.synchronize()
            emit(
                "first_forward_complete",
                path=path,
                condition_present=enabled,
                gradient_enabled=grad_enabled,
                elapsed=time.monotonic() - forward_start,
            )
            first_forwards.add(marker)
        return output

    def collect(split, step, phase, limit=None):
        rows, hidden_rows = [], []
        xs = batches[split][:limit] if limit else batches[split]
        emit("evaluation_start", split=split, step=step, phase=phase, prompts=len(xs))
        for i, x in enumerate(xs):
            pair = {}
            for enabled in (False, True) if i % 2 == 0 else (True, False):
                grad_enabled = enabled and args.condition == "execution"
                if args.condition == "probe":
                    logits, saved = timed_forward(compiled, x, enabled, grad_enabled, "native")
                    logits = logits.detach()
                    other, measured_saved, hidden = timed_forward(
                        compiled_features, x, enabled, grad_enabled, "features"
                    )
                    if len(saved) != int(enabled) or len(measured_saved) != int(enabled):
                        raise RuntimeError("Probe did not fire exactly as requested")
                    native.captured = []
                    measured.captured = []
                    del saved, measured_saved
                else:
                    logits = timed_forward(compiled, x, enabled, grad_enabled, "native").detach()
                    other, hidden = timed_forward(
                        compiled_features, x, enabled, grad_enabled, "features"
                    )
                other, hidden = other.detach(), hidden.detach()
                measured.features = None
                if not torch.equal(logits, other):
                    raise RuntimeError("Feature extraction changes native logits")
                values = logits[0].float()
                logp = values.log_softmax(-1)
                score = values[yes] - values[no]
                label = yes if enabled else no
                pair[enabled] = hidden.cpu()
                rows.append(
                    {
                        "index": i,
                        "condition_present": enabled,
                        "gradient_enabled": grad_enabled,
                        "yes_minus_no": float(score),
                        "positive_minus_negative": float(score),
                        "logp_yes": float(logp[yes]),
                        "logp_no": float(logp[no]),
                        "logp_positive": float(logp[yes]),
                        "logp_negative": float(logp[no]),
                        "answer_pair_mass": float(logp[[no, yes]].logsumexp(0).exp()),
                        "yes_probability": float(score.sigmoid()),
                        "binary_ce": float(
                            torch.nn.functional.softplus(-score if enabled else score)
                        ),
                        "ce": -float(logp[label]),
                        "correct": bool(score > 0) if enabled else bool(score <= 0),
                        "greedy_correct": int(values.argmax()) == label,
                    }
                )
            hidden_rows.extend([pair[False], pair[True]])
            if (i + 1) % 32 == 0:
                emit(
                    "collection_progress",
                    phase=phase,
                    split=split,
                    step=step,
                    prompts_complete=i + 1,
                    prompts_total=len(xs),
                )
        rows.sort(key=lambda r: (r["index"], r["condition_present"]))
        summary = {"phase": phase, "split": split, "step": step, **summarize_execution(rows)}
        write_json(args.out / f"{phase}_{split}_{step}.json", {"summary": summary, "rows": rows})
        emit("evaluation", **summary)
        return torch.cat(hidden_rows), torch.tensor([r["yes_minus_no"] for r in rows]).double()

    def install(new_rows):
        with torch.no_grad():
            head.index_copy_(0, indices, new_rows)
            measured_model.get_output_embeddings().weight.index_copy_(0, indices, new_rows)

    if args.compare_answer_rows:
        artifact = torch.load(args.compare_answer_rows, map_location="cpu", weights_only=True)
        edited_rows = validate_answer_rows(
            artifact, args.checkpoint, model_commit, head, indices
        ).to(head.device)
        baseline_features = {split: collect(split, 0, "many")[0] for split in data}
        install(edited_rows)
        for split in data:
            features, _ = collect(split, args.steps, "many")
            if not torch.equal(features, baseline_features[split]):
                raise RuntimeError("Applying the saved answer rows changed internal features")
            emit(
                "features_unchanged", phase="many", split=split, step=args.steps, bit_identical=True
            )
        emit("complete", comparison_only=True, answer_rows=str(args.compare_answer_rows))
        return

    if args.eval_only:
        for split in data:
            collect(split, args.steps, "fresh")
        emit("complete", eval_only=True)
        return

    # The first fixed pair provides an early integration result, then its edit
    # is removed before the independently fitted many-prompt experiment.
    for phase, limit in [("single", 1), ("many", None)]:
        install(original_rows)
        hidden, scores = collect("train", 0, phase, limit)
        original_hidden = hidden.clone()
        solver = RidgeCorrection(hidden, args.ridge)
        targets = torch.tensor([-args.target_gap / 2, args.target_gap / 2]).repeat(len(hidden) // 2)
        torch.save({"features": hidden, "scores": scores}, args.out / f"{phase}_features.pt")
        if phase == "many":
            for split in evaluation_splits(data, final=True):
                collect(split, 0, phase)
        for step in range(1, args.steps + 1):
            delta = solver.solve(targets - scores)
            current = head.detach().index_select(0, indices).float()
            update = delta.to(device=head.device, dtype=torch.float32) / 2
            new_rows = torch.stack((current[0] - update, current[1] + update)).to(head.dtype)
            install(new_rows)
            emit(
                "fit_update",
                phase=phase,
                step=step,
                correction_norm=float(delta.norm()),
                total_answer_weight_change=float((new_rows.float() - original_rows.float()).norm()),
            )
            hidden, scores = collect("train", step, phase, limit)
            if not torch.equal(hidden, original_hidden):
                raise RuntimeError("Answer-row updates changed internal features")
            emit("features_unchanged", phase=phase, step=step, bit_identical=True)
            if phase == "many":
                for split in evaluation_splits(data, final=step == args.steps):
                    collect(split, step, phase)
    torch.save(
        {
            "version": 1,
            "checkpoint": args.checkpoint,
            "model_commit": model_commit,
            "head_shape": list(head.shape),
            "indices": indices.cpu(),
            "base_rows_sha256": tensor_digest(original_rows),
            "rows": head.detach().index_select(0, indices).cpu(),
            "answers": [args.negative_answer, args.positive_answer],
        },
        args.out / "answer_rows.pt",
    )
    if not args.rows_only_save:
        save_weights(model, tok, args.out / "checkpoint", step=args.steps)
    emit(
        "complete",
        answer_rows=str(args.out / "answer_rows.pt"),
        checkpoint=None if args.rows_only_save else str(args.out / "checkpoint"),
    )


if __name__ == "__main__":
    main()
