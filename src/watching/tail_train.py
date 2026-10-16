"""Train native final blocks on fixed compiled-source states in an explicit split runtime.

Primary evaluation uses an eager BF16 math-SDPA tail on cached native inputs.
Whole-model compiled execution is a separately named final transfer diagnostic.
No classifier, input-label feature, or output-matrix update is introduced.
"""

import argparse
import copy
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer

from watching.artifacts import write_json
from watching.execution_fit import tensor_digest
from watching.metrics import summarize_pairs
from watching.probe import ProbeModel
from watching.suffix import (
    ResidualCapture,
    assert_identical,
    move_state,
)


class PairedMasterAdam:
    """Accumulate separately computed condition gradients into FP32 masters."""

    def __init__(self, parameters, lr):
        self.parameters = list(parameters)
        if not self.parameters:
            raise ValueError("Need active native parameters")
        self.masters = [torch.nn.Parameter(p.detach().float().clone()) for p in self.parameters]
        self.optimizer = torch.optim.AdamW(self.masters, lr=lr, weight_decay=0, foreach=False)
        self.contributions = 0

    def accumulate(self):
        for parameter, master in zip(self.parameters, self.masters, strict=True):
            if parameter.grad is None:
                raise RuntimeError("An active tail parameter has no gradient")
            contribution = parameter.grad.detach().float()
            if master.grad is None:
                master.grad = contribution.clone()
            else:
                master.grad.add_(contribution)
            parameter.grad = None
        self.contributions += 1

    def step(self):
        if self.contributions != 2:
            raise RuntimeError("An update requires exactly two separately accumulated conditions")
        norm = torch.nn.utils.clip_grad_norm_(self.masters, 1, error_if_nonfinite=True)
        self.optimizer.step()
        movement = torch.zeros((), device=self.parameters[0].device)
        with torch.no_grad():
            for parameter, master in zip(self.parameters, self.masters, strict=True):
                delta = parameter.detach().float().clone().neg_()
                parameter.copy_(master)
                delta.add_(parameter)
                movement.add_(torch.linalg.vector_norm(delta).square())
        self.optimizer.zero_grad(set_to_none=True)
        self.contributions = 0
        return float(norm), float(movement.sqrt())


class EagerNativeTail(torch.nn.Module):
    """Copied native blocks and norm; immutable original head storage is reused."""

    def __init__(self, model, tail_layers=2):
        super().__init__()
        count = len(model.model.layers)
        if not 1 <= tail_layers < count:
            raise ValueError("Tail must leave at least one frozen source block")
        if model.config.model_type != "qwen3" or any(
            kind != "full_attention" for kind in model.config.layer_types
        ):
            raise ValueError("Only full-attention Qwen3 is supported")
        self.first_layer = count - tail_layers
        self.layers = copy.deepcopy(model.model.layers[self.first_layer :])
        self.norm = copy.deepcopy(model.model.norm)
        # A buffer view shares the original native matrix without a new matrix or
        # trainable parameter. Autograd still propagates through its input.
        self.register_buffer("head_weight", model.lm_head.weight.detach(), persistent=False)
        bias = getattr(model.lm_head, "bias", None)
        self.register_buffer("head_bias", None if bias is None else bias.detach(), persistent=False)
        self.eval()

    def active_parameters(self):
        return list(self.layers.parameters()) + list(self.norm.parameters())

    def forward(self, hidden, mask, cos, sin, position_ids):
        with sdpa_kernel(SDPBackend.MATH):
            for layer in self.layers:
                hidden = layer(
                    hidden,
                    attention_mask=mask,
                    position_ids=position_ids,
                    position_embeddings=(cos, sin),
                    past_key_values=None,
                    use_cache=False,
                )
            hidden = self.norm(hidden)[:, -1:, :]
            return F.linear(hidden, self.head_weight, self.head_bias)[:, -1, :]

    def install(self, model):
        """Install only the trained original block/norm parameters for transfer."""
        model.model.layers[self.first_layer :].load_state_dict(self.layers.state_dict())
        model.model.norm.load_state_dict(self.norm.state_dict())


def difference(actual, expected):
    if (
        actual.shape != expected.shape
        or not torch.isfinite(actual).all()
        or not torch.isfinite(expected).all()
    ):
        raise ValueError("Comparisons require matching shapes and finite values")
    delta = actual.detach().float() - expected.detach().float()
    return {
        "bit_identical": torch.equal(actual, expected),
        "fraction_different": float((delta != 0).float().mean()),
        "max_abs_difference": float(delta.abs().max()),
        "mean_abs_difference": float(delta.abs().mean()),
    }


def check_gradient_forward(tail, state, positions):
    """Measure first; reject an incompatible backward-enabled forward explicitly."""
    with torch.no_grad():
        inference = tail(*state, positions)
    with torch.enable_grad():
        training = tail(*state, positions).detach()
    audit = difference(training, inference)
    return audit


def stratified_records(records, count, seed):
    if count < 1 or count > len(records):
        raise ValueError("Requested prompt count exceeds available records")
    groups = defaultdict(list)
    for index, record in enumerate(records):
        groups[record.get("question_family", record.get("family", "all"))].append(index)
    rng = random.Random(seed)
    names = sorted(groups)
    rng.shuffle(names)
    for members in groups.values():
        rng.shuffle(members)
    chosen = []
    while len(chosen) < count:
        for name in names:
            if groups[name] and len(chosen) < count:
                chosen.append(groups[name].pop())
    return [{**records[index], "source_index": index} for index in chosen]


def score(logits, index, capture, no, yes):
    values = logits[0].detach().float()
    logp = values.log_softmax(-1)
    gap = float(values[yes] - values[no])
    label = yes if capture else no
    return {
        "index": index,
        "capture": capture,
        "condition_present": capture,
        "yes_minus_no": gap,
        "positive_minus_negative": gap,
        "logp_yes": float(logp[yes]),
        "logp_no": float(logp[no]),
        "logp_positive": float(logp[yes]),
        "logp_negative": float(logp[no]),
        "yes_probability": float(torch.sigmoid(values[yes] - values[no])),
        "correct": gap > 0 if capture else gap <= 0,
        "greedy_correct": int(values.argmax()) == label,
        "full_vocab_correct_probability": float(logp[label].exp()),
        "answer_pair_mass": float(logp[[no, yes]].logsumexp(0).exp()),
        "ce": -float(logp[label]),
        "binary_ce": float(
            F.softplus(-values[yes] + values[no] if capture else values[yes] - values[no])
        ),
    }


def summarize(rows, seed):
    return {
        **summarize_pairs(rows, seed=seed, bootstrap_samples=5000),
        "mean_binary_ce": sum(r["binary_ce"] for r in rows) / len(rows),
        "mean_full_vocab_correct_probability": sum(
            r["full_vocab_correct_probability"] for r in rows
        )
        / len(rows),
        "mean_answer_pair_mass": sum(r["answer_pair_mass"] for r in rows) / len(rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--probe-layer", type=int, default=14)
    parser.add_argument("--tail-layers", type=int, default=2)
    parser.add_argument("--train-prompts", type=int, default=16)
    parser.add_argument("--validation-prompts", type=int, default=8)
    parser.add_argument("--test-prompts", type=int, default=32)
    parser.add_argument("--test-split", default=None)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--length", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20261017)
    args = parser.parse_args()
    if (
        min(args.steps, args.eval_every, args.length) < 1
        or not math.isfinite(args.lr)
        or not 0 < args.lr < 1
    ):
        parser.error("Invalid steps, evaluation interval, length, or learning rate")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the native compiled source")
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise ValueError("Output directory must be empty")
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    started = time.monotonic()

    def emit(event, **fields):
        item = {"event": event, "seconds": time.monotonic() - started, **fields}
        line = json.dumps(item, allow_nan=False)
        print(line, flush=True)
        with (args.out / "metrics.jsonl").open("a") as stream:
            stream.write(line + "\n")

    raw = json.loads(args.data.read_text())
    test_split = args.test_split or ("test_wording" if "test_wording" in raw else "test")
    data = {
        name: stratified_records(raw[source], count, args.seed)
        for name, source, count in (
            ("train", "train", args.train_prompts),
            ("validation", "validation", args.validation_prompts),
            ("test", test_split, args.test_prompts),
        )
    }
    if all("question_family" in r for r in data["train"] + data["test"]):
        if {r["question_family"] for r in data["train"]} & {
            r["question_family"] for r in data["test"]
        }:
            raise ValueError("Tail test must use question families absent from tail training")
    keys = [{r.get("context_id", r["context"]) for r in rows} for rows in data.values()]
    if any(a & b for i, a in enumerate(keys) for b in keys[i + 1 :]):
        raise ValueError("Tail train/reporting/test contexts overlap")
    write_json(args.out / "data.json", data)
    tok = AutoTokenizer.from_pretrained(args.checkpoint, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    labels = [tok.encode(word, add_special_tokens=False) for word in ("No", "Yes")]
    if any(len(ids) != 1 for ids in labels) or labels[0] == labels[1]:
        raise ValueError("No and Yes must be distinct single tokens")
    no, yes = [ids[0] for ids in labels]
    batches = {}
    for split, rows in data.items():
        batches[split] = []
        for row in rows:
            text = tok.apply_chat_template(
                [{"role": "user", "content": row["text"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if len(tok.encode(text, add_special_tokens=False)) > args.length:
                raise ValueError("Rendered prompt exceeds fixed length; truncation is forbidden")
            x = tok(
                text,
                add_special_tokens=False,
                padding="max_length",
                max_length=args.length,
                return_tensors="pt",
            )
            x["position_ids"] = (x.attention_mask.cumsum(-1) - 1).clamp_min(0)
            batches[split].append(dict(x))

    def load():
        result = (
            AutoModelForCausalLM.from_pretrained(
                args.checkpoint, dtype=torch.bfloat16, attn_implementation="sdpa"
            )
            .cuda()
            .eval()
        )
        result.config.use_cache = False
        result.requires_grad_(True)
        return result

    emit("loading", checkpoint=args.checkpoint)
    model, observed_model = load(), load()
    if any(
        not torch.equal(a, b)
        for a, b in zip(model.parameters(), observed_model.parameters(), strict=True)
    ):
        raise RuntimeError("Source and extraction copies differ")
    cut = len(model.model.layers) - args.tail_layers - 1
    if not 0 <= args.probe_layer <= cut:
        raise ValueError("The probe must lie within the frozen source prefix")
    native_wrapper = ProbeModel(model, args.probe_layer)
    capture_wrapper = ResidualCapture(observed_model, args.probe_layer, cut, "buffer")
    capture_wrapper.prepare_buffers({k: v.cuda() for k, v in batches["train"][0].items()})
    native = torch.compile(native_wrapper, fullgraph=True, dynamic=False)
    extract = torch.compile(capture_wrapper, fullgraph=True, dynamic=False)
    tail = EagerNativeTail(model, args.tail_layers)
    original_head_digest = tensor_digest(model.lm_head.weight)
    manifest = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "condition": "probe",
        "answer_order": ["No", "Yes"],
        "labels": {"No": no, "Yes": yes},
        "model_commit": getattr(model.config, "_commit_hash", None),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "updated_blocks": list(range(cut + 1, len(model.model.layers))),
        "active_parameters": sum(p.numel() for p in tail.active_parameters()),
        "runtime": "compiled frozen source states + eager BF16 math-SDPA native tail",
        "primary_path": "split_inference",
        "stock_runtime_parity_claimed": False,
        "output_head_updated": False,
        "new_classifier": False,
        "objective": "balanced paired full-vocabulary cross entropy",
        "optimizer": "FP32-master AdamW, separate condition gradients accumulated in FP32, "
        "zero decay, gradient norm clip1",
        "test_source_split": test_split,
        "test_evaluated_only_after_training": True,
        "selection": "fixed step count; reporting validation does not select checkpoints",
        "head_sha256_before": original_head_digest,
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    write_json(args.out / "manifest.json", manifest)
    emit("setup", **manifest)
    cache = {}
    baseline = {}

    def cache_split(split):
        cache[split], baseline[split] = [], []
        emit("cache_start", split=split, pairs=len(batches[split]))
        for index, batch in enumerate(batches[split]):
            x = {k: v.cuda() for k, v in batch.items()}
            pair = []
            for present in (False, True):
                with torch.no_grad():
                    full, saved = native(**x, capture=present)
                    copied, other_saved, state = extract(**x, capture=present)
                if len(saved) != int(present) or len(other_saved) != int(present):
                    raise RuntimeError("Probe did not fire exactly as requested")
                assert_identical(full, copied, "Boundary extraction changes whole-model logits")
                stored = tuple(None if t is None else t.detach().cpu().clone() for t in state)
                pair.append(stored)
                native_wrapper.captured = capture_wrapper.captured = []
                capture_wrapper.boundary = None
                if split != "test":
                    baseline[split].append(score(full, index, present, no, yes))
                    restored = move_state(stored, "cuda")
                    with torch.no_grad():
                        replayed = tail(*restored, x["position_ids"])
                    emit(
                        "whole_vs_split",
                        split=split,
                        index=index,
                        capture=present,
                        **difference(replayed, full),
                    )
                del full, copied, saved, other_saved, state
            footprint = pair[1][0].float() - pair[0][0].float()
            if not torch.isfinite(footprint).all():
                raise RuntimeError("Nonfinite frozen-source footprint")
            cache[split].append(pair)
            if split != "test":
                emit(
                    "source_footprint",
                    split=split,
                    index=index,
                    l2=float(footprint.norm()),
                    fraction_different=float((footprint != 0).float().mean()),
                )
            if index == 0 or (index + 1) % 8 == 0:
                emit("cache_progress", split=split, pairs=index + 1, extraction_bit_identical=True)
        if split == "train" and not any(not torch.equal(p[0][0], p[1][0]) for p in cache[split]):
            raise RuntimeError("No probe footprint survives at the frozen-source boundary")

    def save_evaluation(rows, step, split, path):
        summary = {"step": step, "split": split, "path": path, **summarize(rows, args.seed)}
        write_json(args.out / f"{split}_{step}_{path}.json", {"summary": summary, "rows": rows})
        emit("evaluation", **summary)

    def audit_gradient_forward(step):
        for present in (False, True):
            state = move_state(cache["train"][0][int(present)], "cuda")
            audit = check_gradient_forward(tail, state, batches["train"][0]["position_ids"].cuda())
            emit("gradient_forward_audit", step=step, capture=present, **audit)
            if not audit["bit_identical"]:
                raise RuntimeError(
                    "Eager math-SDPA gradient/no-grad forward mismatch; see measured audit"
                )

    def evaluate(step, splits):
        for split in splits:
            rows = []
            for index, pair in enumerate(cache[split]):
                positions = batches[split][index]["position_ids"].cuda()
                for present in (False, True):
                    with torch.no_grad():
                        logits = tail(*move_state(pair[int(present)], "cuda"), positions)
                    rows.append(score(logits, index, present, no, yes))
            save_evaluation(rows, step, split, "split_inference")

    for split in ("train", "validation"):
        cache_split(split)
        save_evaluation(baseline[split], 0, split, "original_wholegraph")
    audit_gradient_forward(0)
    evaluate(0, ("train", "validation"))
    optimizer = PairedMasterAdam(tail.active_parameters(), args.lr)
    order = list(range(len(cache["train"])))
    rng = random.Random(args.seed)
    rng.shuffle(order)
    emit("training_start", steps=args.steps, allocated_gpu_bytes=torch.cuda.memory_allocated())
    for step in range(1, args.steps + 1):
        if step > 1 and (step - 1) % len(order) == 0:
            rng.shuffle(order)
        index = order[(step - 1) % len(order)]
        positions = batches["train"][index]["position_ids"].cuda()
        losses = []
        for present in (False, True):
            logits = tail(
                *move_state(cache["train"][index][int(present)], "cuda"), positions
            ).float()
            label = torch.tensor([yes if present else no], device=logits.device)
            loss = F.cross_entropy(logits, label) / 2
            with sdpa_kernel(SDPBackend.MATH):
                loss.backward()
            optimizer.accumulate()
            losses.append(float(loss.detach()))
        norm, movement = optimizer.step()
        emit(
            "train",
            step=step,
            prompt_index=index,
            loss=sum(losses),
            gradient_norm=norm,
            actual_weight_update_l2=movement,
        )
        if step % args.eval_every == 0 or step == args.steps:
            audit_gradient_forward(step)
            evaluate(step, ("train", "validation"))
    del optimizer
    # Test inputs are first executed here, after the fixed training schedule.
    cache_split("test")
    evaluate(args.steps, ("test",))
    if tensor_digest(model.lm_head.weight) != original_head_digest:
        raise RuntimeError("Frozen native output head changed")
    torch.save(
        {
            "version": 1,
            "checkpoint": args.checkpoint,
            "model_commit": manifest["model_commit"],
            "first_layer": tail.first_layer,
            "layers": {k: v.detach().cpu() for k, v in tail.layers.state_dict().items()},
            "norm": {k: v.detach().cpu() for k, v in tail.norm.state_dict().items()},
            "head_sha256": original_head_digest,
            "runtime": manifest["runtime"],
        },
        args.out / "native_tail.pt",
    )
    tail.install(model)
    tail.install(observed_model)
    for split in ("train", "validation", "test"):
        rows = []
        for index, batch in enumerate(batches[split]):
            x = {k: v.cuda() for k, v in batch.items()}
            for present in (False, True):
                with torch.no_grad():
                    logits, saved = native(**x, capture=present)
                    copied, other_saved, state = extract(**x, capture=present)
                assert_identical(logits, copied, "Final whole-model extraction parity")
                if len(saved) != int(present) or len(other_saved) != int(present):
                    raise RuntimeError("Final probe did not fire")
                for actual, expected in zip(state, cache[split][index][int(present)], strict=True):
                    if expected is None:
                        if actual is not None:
                            raise RuntimeError("Source mask presence changed")
                    else:
                        assert_identical(
                            actual.cpu(), expected, "Frozen prefix changed after tail installation"
                        )
                rows.append(score(logits, index, present, no, yes))
                native_wrapper.captured = capture_wrapper.captured = []
                capture_wrapper.boundary = None
        save_evaluation(rows, args.steps, split, "wholegraph_transfer")
    if tensor_digest(model.lm_head.weight) != original_head_digest:
        raise RuntimeError("Output head changed during tail installation")
    emit(
        "complete",
        tail_artifact=str(args.out / "native_tail.pt"),
        head_unchanged=True,
        frozen_source_bit_identical=True,
        max_gpu_bytes=torch.cuda.max_memory_allocated(),
    )


if __name__ == "__main__":
    main()
