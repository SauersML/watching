"""Train native final blocks on fixed compiled-source states in an explicit split runtime.

Primary evaluation uses an eager BF16 math-SDPA tail on cached native inputs.
Whole-model compiled execution is a separately named final transfer diagnostic.
No classifier or input-label feature is introduced. Optional head training updates
the existing untied vocabulary matrix without changing the model architecture.
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
    """Copied native blocks and norm, with an optional trainable native untied head."""

    def __init__(self, model, tail_layers=2, train_head=False):
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
        self.train_head = train_head
        if train_head:
            if (
                model.config.tie_word_embeddings
                or model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
                or getattr(model.lm_head, "bias", None) is not None
            ):
                raise ValueError(
                    "Train-head mode requires an existing untied, biasless native head"
                )
            self.head_weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
        else:
            # Preserve the historical frozen-head path and its shared storage.
            self.register_buffer("head_weight", model.lm_head.weight.detach(), persistent=False)
        bias = getattr(model.lm_head, "bias", None)
        self.register_buffer("head_bias", None if bias is None else bias.detach(), persistent=False)
        self.eval()

    def active_parameters(self):
        parameters = list(self.layers.parameters()) + list(self.norm.parameters())
        return parameters + ([self.head_weight] if self.train_head else [])

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
        """Install only the selected original parameters for transfer."""
        model.model.layers[self.first_layer :].load_state_dict(self.layers.state_dict())
        model.model.norm.load_state_dict(self.norm.state_dict())
        if self.train_head:
            with torch.no_grad():
                model.lm_head.weight.copy_(self.head_weight)


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


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_initial_tail(artifact, tail, *, checkpoint, model_commit, head_sha256):
    """Validate every tensor before loading anything into the learner."""
    if not isinstance(artifact, dict) or artifact.get("version") != 1:
        raise ValueError("Unsupported native-tail artifact version")
    if artifact.get("checkpoint") != checkpoint or artifact.get("first_layer") != tail.first_layer:
        raise ValueError("Native-tail checkpoint or first layer does not match")
    saved_commit = artifact.get("model_commit")
    if saved_commit is not None and saved_commit != model_commit:
        raise ValueError("Native-tail model commit does not match")
    if artifact.get("head_sha256") != head_sha256:
        raise ValueError("Native-tail artifact names a different frozen output head")
    saved_head = artifact.get("head")
    if artifact.get("train_head", False) or saved_head is not None:
        if not tail.train_head:
            raise ValueError(
                "A head-trained checkpoint requires --train-head; refusing to drop head"
            )
        if (
            not isinstance(saved_head, torch.Tensor)
            or saved_head.shape != tail.head_weight.shape
            or saved_head.dtype != tail.head_weight.dtype
            or not torch.isfinite(saved_head).all()
            or tensor_digest(saved_head) != artifact.get("learned_head_sha256")
        ):
            raise ValueError("Invalid saved native head shape, dtype, values, or digest")
    for name, current in (("layers", tail.layers.state_dict()), ("norm", tail.norm.state_dict())):
        saved = artifact.get(name)
        if not isinstance(saved, dict) or set(saved) != set(current):
            raise ValueError(f"Native-tail {name} keys do not match")
        for key, expected in current.items():
            value = saved[key]
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != expected.shape
                or value.dtype != expected.dtype
                or not torch.isfinite(value).all()
            ):
                raise ValueError(
                    f"Invalid native-tail tensor {name}.{key}: shape, dtype, or values"
                )


def context_key(record):
    for key in ("context_id", "context", "text"):
        if key in record:
            return record[key]
    raise ValueError("Record has no context identity")


def continuation_records(
    raw, prior_data, prior_manifest, *, train_count, validation_count, test_count, test_split, seed
):
    """Reuse practiced records exactly and exclude every previously tested context."""
    data = {}
    for split, expected_count in (("train", train_count), ("validation", validation_count)):
        previous = prior_data[split]
        if len(previous) != expected_count:
            raise ValueError("Continuation must retain the prior train/reporting prompt counts")
        indexed = {context_key(record): record for record in raw[split]}
        if len(indexed) != len(raw[split]):
            raise ValueError("Current source data contains duplicate context identities")
        for record in previous:
            key = context_key(record)
            without_index = {k: v for k, v in record.items() if k != "source_index"}
            if key not in indexed or without_index != {
                k: v for k, v in indexed[key].items() if k != "source_index"
            }:
                raise ValueError(
                    "Prior practiced context or question differs from current source data"
                )
        data[split] = copy.deepcopy(previous)
    excluded = set(prior_manifest.get("excluded_test_context_ids", []))
    excluded.update(context_key(record) for record in prior_data["test"])
    eligible = [
        (index, record)
        for index, record in enumerate(raw[test_split])
        if context_key(record) not in excluded
    ]
    chosen = stratified_records([record for _, record in eligible], test_count, seed)
    for record in chosen:
        record["source_index"] = eligible[record["source_index"]][0]
    data["test"] = chosen
    return data, sorted(excluded)


def compare_prior_rows(current, previous):
    """Describe a restart discrepancy; do not disguise it as additional learning."""

    def keyed(rows):
        result = {(r["index"], r["capture"]): r for r in rows}
        if len(result) != len(rows):
            raise ValueError("Duplicate prompt/condition score rows")
        return result

    now, before = keyed(current), keyed(previous)
    if set(now) != set(before):
        raise ValueError("Restart comparison requires matching prompt/condition rows")
    report = {}
    for field in ("yes_minus_no", "logp_yes", "logp_no", "ce", "full_vocab_correct_probability"):
        deltas = [float(now[key][field]) - float(before[key][field]) for key in sorted(now)]
        if not all(math.isfinite(value) for value in deltas):
            raise ValueError("Nonfinite restart comparison")
        report[field] = {
            "exactly_equal": all(value == 0 for value in deltas),
            "fraction_different": sum(value != 0 for value in deltas) / len(deltas),
            "mean_difference": sum(deltas) / len(deltas),
            "max_abs_difference": max(abs(value) for value in deltas),
        }
    return report


def state_fingerprint(state):
    return {
        name: None
        if tensor is None
        else {
            "sha256": tensor_digest(tensor),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "stride": list(tensor.stride()),
        }
        for name, tensor in zip(("hidden", "mask", "cos", "sin"), state, strict=True)
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-head", action="store_true")
    parser.add_argument("--initial-tail", type=Path)
    parser.add_argument("--prior-results", type=Path)
    parser.add_argument("--early-eval-steps", type=int, nargs="*", default=[8, 32, 128])
    parser.add_argument("--max-training-seconds", type=float, default=0)
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
    if bool(args.initial_tail) != bool(args.prior_results):
        parser.error("--initial-tail and --prior-results must be supplied together")
    if (
        any(step < 1 for step in args.early_eval_steps)
        or args.max_training_seconds < 0
        or not math.isfinite(args.max_training_seconds)
    ):
        parser.error(
            "Early evaluation steps must be positive and the time limit finite/nonnegative"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the native compiled source")
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise ValueError("Output directory must be empty")
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    started = time.monotonic()
    starting_global_step = 0

    def emit(event, **fields):
        if "step" in fields:
            fields.setdefault("global_step", starting_global_step + fields["step"])
        item = {"event": event, "seconds": time.monotonic() - started, **fields}
        line = json.dumps(item, allow_nan=False)
        print(line, flush=True)
        with (args.out / "metrics.jsonl").open("a") as stream:
            stream.write(line + "\n")

    raw = json.loads(args.data.read_text())
    test_split = args.test_split or ("test_wording" if "test_wording" in raw else "test")
    prior_manifest = prior_data = None
    excluded_test_context_ids = []
    if args.prior_results:
        prior_manifest = json.loads((args.prior_results / "manifest.json").read_text())
        prior_data = json.loads((args.prior_results / "data.json").read_text())
        previous_steps = prior_manifest.get("completed_steps", prior_manifest["steps"])
        if type(previous_steps) is not int or previous_steps < 1:
            raise ValueError("Prior run has no completed training steps")
        starting_global_step = prior_manifest.get("starting_global_step", 0) + previous_steps
        for split in ("train", "validation"):
            if not (
                args.prior_results / f"{split}_{previous_steps}_split_inference.json"
            ).is_file():
                raise ValueError("Prior final train/reporting evaluations are required")
        if (
            prior_manifest["checkpoint"] != args.checkpoint
            or prior_manifest["probe_layer"] != args.probe_layer
            or prior_manifest["tail_layers"] != args.tail_layers
            or prior_manifest["length"] != args.length
        ):
            raise ValueError("Continuation source model, probe, tail, or input length changed")
        data, excluded_test_context_ids = continuation_records(
            raw,
            prior_data,
            prior_manifest,
            train_count=args.train_prompts,
            validation_count=args.validation_prompts,
            test_count=args.test_prompts,
            test_split=test_split,
            seed=args.seed,
        )
    else:
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
    keys = [{context_key(r) for r in rows} for rows in data.values()]
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
    tail = EagerNativeTail(model, args.tail_layers, train_head=args.train_head)
    original_head_digest = tensor_digest(model.lm_head.weight)
    initial_tail_sha256 = None
    if args.initial_tail:
        artifact = torch.load(args.initial_tail, map_location="cpu", weights_only=True)
        validate_initial_tail(
            artifact,
            tail,
            checkpoint=args.checkpoint,
            model_commit=getattr(model.config, "_commit_hash", None),
            head_sha256=original_head_digest,
        )
        if artifact.get("global_step", starting_global_step) != starting_global_step:
            raise ValueError("Initial-tail step disagrees with prior-results training history")
        initial_tail_sha256 = file_digest(args.initial_tail)
        prior_weights = args.prior_results / "native_tail.pt"
        if not prior_weights.is_file():
            raise ValueError("Prior-results saved native_tail.pt is required for provenance")
        if (
            prior_weights.resolve() != args.initial_tail.resolve()
            and file_digest(prior_weights) != initial_tail_sha256
        ):
            raise ValueError("Initial-tail artifact differs from prior-results saved weights")
        # Never install learned weights into the frozen source before caching.
        tail.layers.load_state_dict(artifact["layers"])
        tail.norm.load_state_dict(artifact["norm"])
        if artifact.get("head") is not None:
            with torch.no_grad():
                tail.head_weight.copy_(artifact["head"])
        del artifact
    manifest = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "starting_global_step": starting_global_step,
        "completed_steps": 0,
        "initial_tail_sha256": initial_tail_sha256,
        "excluded_test_context_ids": excluded_test_context_ids,
        "optimizer_state_restored": False,
        "continuation_note": (
            "Weights continue; Adam moments restart; frozen source is freshly compiled"
            if args.initial_tail
            else None
        ),
        "whole_vs_split_note": (
            "On continuation this includes prior learned weights, not only runtime differences"
        ),
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
        "output_head_updated": args.train_head,
        "output_head_scope": "full existing vocabulary matrix" if args.train_head else "frozen",
        "sources_offloaded_during_training": args.train_head,
        "test_inputs_cached_before_training": args.train_head,
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
                stored = tuple(None if t is None else t.detach().cpu().clone() for t in state)
                if not torch.equal(full, copied):
                    # Stop on the first mismatch, but preserve enough evidence to
                    # distinguish unstable kernels from a stable extraction effect.
                    failed_full = full.detach().cpu().clone()
                    failed_copied = copied.detach().cpu().clone()
                    with torch.no_grad():
                        repeated_full, _ = native(**x, capture=present)
                        repeated_copy, _, _ = extract(**x, capture=present)
                    report = {
                        "split": split,
                        "index": index,
                        "capture": present,
                        "context_id": context_key(data[split][index]),
                        "gpu": manifest["gpu"],
                        "different_logits": int((failed_full != failed_copied).sum()),
                        "first_differing_token_id": int(
                            (failed_full[0] != failed_copied[0]).nonzero()[0, 0]
                        ),
                        "original_vs_extraction": difference(failed_full, failed_copied),
                        "original_repeat": difference(repeated_full.cpu(), failed_full),
                        "extraction_repeat": difference(repeated_copy.cpu(), failed_copied),
                        "repeated_original_vs_extraction": difference(repeated_full, repeated_copy),
                    }
                    write_json(args.out / "extraction_failure.json", report)
                    torch.save(
                        {
                            "report": report,
                            "input": batch,
                            "original_logits": failed_full,
                            "extracted_logits": failed_copied,
                            "boundary_state": stored,
                        },
                        args.out / "extraction_failure.pt",
                    )
                    emit("extraction_failure", **report)
                    raise RuntimeError(
                        f"Boundary extraction changes whole-model logits at "
                        f"{split}[{index}], capture={present}; see extraction_failure.json"
                    )
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
        summary = {
            "step": step,
            "global_step": starting_global_step + step,
            "split": split,
            "path": path,
            **summarize(rows, args.seed),
        }
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

    def save_cache():
        torch.save(
            {
                "version": 1,
                "checkpoint": args.checkpoint,
                "model_commit": manifest["model_commit"],
                "cut_layer": cut,
                "length": args.length,
                "conditions": [False, True],
                "states": cache,
                "data": {s: data[s] for s in cache},
                "batches": {s: batches[s] for s in cache},
            },
            args.out / "cache_states.pt",
        )
        write_json(
            args.out / "cache_fingerprints.json",
            {
                split: [
                    {
                        "index": index,
                        "context_id": context_key(data[split][index]),
                        "positions_sha256": tensor_digest(batches[split][index]["position_ids"]),
                        "absent": state_fingerprint(pair[0]),
                        "present": state_fingerprint(pair[1]),
                    }
                    for index, pair in enumerate(pairs)
                ]
                for split, pairs in cache.items()
            },
        )

    for split in ("train", "validation"):
        cache_split(split)
        save_evaluation(baseline[split], 0, split, "original_wholegraph")
    save_cache()
    audit_gradient_forward(0)
    evaluate(0, ("train", "validation"))
    if args.prior_results:
        comparison = {}
        for split in ("train", "validation"):
            previous = json.loads(
                (args.prior_results / f"{split}_{previous_steps}_split_inference.json").read_text()
            )
            current = json.loads((args.out / f"{split}_0_split_inference.json").read_text())
            comparison[split] = {
                "prior_summary": previous["summary"],
                "current_summary": current["summary"],
                "differences": compare_prior_rows(current["rows"], previous["rows"]),
            }
            emit("continuation_comparison", split=split, step=0, **comparison[split])
        write_json(args.out / "continuation_comparison.json", comparison)
    if args.train_head:
        # Test inputs are locked beforehand; their logits are not scored or used
        # by the optimizer. Caching here lets both full sources leave GPU memory.
        cache_split("test")
        save_cache()
        model.cpu()
        observed_model.cpu()
        torch.cuda.empty_cache()
        emit("sources_offloaded", allocated_gpu_bytes=torch.cuda.memory_allocated())
    optimizer = PairedMasterAdam(tail.active_parameters(), args.lr)
    order = list(range(len(cache["train"])))
    rng = random.Random(args.seed)
    rng.shuffle(order)
    emit("training_start", steps=args.steps, allocated_gpu_bytes=torch.cuda.memory_allocated())
    training_started = time.monotonic()
    completed_steps = 0
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
        completed_steps = step
        time_limited = (
            args.max_training_seconds > 0
            and time.monotonic() - training_started >= args.max_training_seconds
        )
        if (
            step % args.eval_every == 0
            or step in args.early_eval_steps
            or step == args.steps
            or time_limited
        ):
            audit_gradient_forward(step)
            evaluate(step, ("train", "validation"))
        if time_limited:
            emit("training_time_limit", step=step, requested_steps=args.steps)
            break
    del optimizer
    if tensor_digest(model.lm_head.weight) != original_head_digest:
        raise RuntimeError("Frozen native output head changed")
    learned_head_digest = tensor_digest(tail.head_weight)
    torch.save(
        {
            "version": 1,
            "train_head": args.train_head,
            "head": tail.head_weight.detach().cpu() if args.train_head else None,
            "learned_head_sha256": learned_head_digest,
            "completed_steps": completed_steps,
            "global_step": starting_global_step + completed_steps,
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
    manifest["completed_steps"] = completed_steps
    manifest["head_sha256_after"] = learned_head_digest
    manifest["ending_global_step"] = starting_global_step + completed_steps
    manifest["training_status"] = "complete" if completed_steps == args.steps else "time_limited"
    write_json(args.out / "manifest.json", manifest)
    emit("tail_checkpoint_saved", step=completed_steps, path=str(args.out / "native_tail.pt"))
    if not args.train_head:
        # Frozen-head mode retains its original late test-input execution.
        cache_split("test")
        save_cache()
    evaluate(completed_steps, ("test",))
    if args.train_head:
        torch.cuda.empty_cache()
        model.cuda()
        observed_model.cuda()
        emit("sources_restored", allocated_gpu_bytes=torch.cuda.memory_allocated())
        for source in (model, observed_model):
            if tensor_digest(source.lm_head.weight) != original_head_digest:
                raise RuntimeError("Original source head changed during offload/restore")
        # Device migration can change compilation. Recheck the original source
        # against every cached state before installing any learned parameters.
        for split, pairs in cache.items():
            for index, pair in enumerate(pairs):
                x = {k: v.cuda() for k, v in batches[split][index].items()}
                for present in (False, True):
                    with torch.no_grad():
                        full, _ = native(**x, capture=present)
                        copied, _, state = extract(**x, capture=present)
                    assert_identical(full, copied, "Restored source extraction parity")
                    for actual, expected in zip(state, pair[int(present)], strict=True):
                        if expected is None:
                            if actual is not None:
                                raise RuntimeError("Restored source mask presence changed")
                        else:
                            assert_identical(actual.cpu(), expected, "Restored source cache parity")
                    native_wrapper.captured = capture_wrapper.captured = []
                    capture_wrapper.boundary = None
        emit("restored_source_verified", cached_states_bit_identical=True)
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
        save_evaluation(rows, completed_steps, split, "wholegraph_transfer")
    for source in (model, observed_model):
        if tensor_digest(source.lm_head.weight) != learned_head_digest:
            raise RuntimeError("Installed output head differs from the learner")
    emit(
        "complete",
        step=completed_steps,
        status=manifest["training_status"],
        tail_artifact=str(args.out / "native_tail.pt"),
        head_unchanged=learned_head_digest == original_head_digest,
        native_head_installed_correctly=True,
        frozen_source_bit_identical=True,
        max_gpu_bytes=torch.cuda.max_memory_allocated(),
    )


if __name__ == "__main__":
    main()
