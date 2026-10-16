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
from huggingface_hub import try_to_load_from_cache
from transformers import AutoModelForCausalLM, AutoTokenizer

from watching.artifacts import save_weights, write_json
from watching.audit import FeatureProbe
from watching.data import make_expanded_data
from watching.head_replay import GatedHeadReplay, choose_validation_candidate, validation_objective
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

    def solve(self, residual, *, ridge=None):
        residual = residual.detach().to(device="cpu", dtype=torch.float64)
        if (
            residual.ndim not in (1, 2)
            or residual.shape[0] != self.u.shape[0]
            or not residual.numel()
            or not torch.isfinite(residual).all()
        ):
            raise ValueError("Residual must contain one finite value per example")
        ridge = self.ridge if ridge is None else ridge
        if not math.isfinite(ridge) or ridge <= 0:
            raise ValueError("Ridge penalty must be finite and positive")
        gain = self.singular / (self.singular.square() + ridge)
        if residual.ndim == 2:
            gain = gain[:, None]
        return self.vh.T @ (gain * (self.u.T @ residual))


def symmetric_ridge_rows(current_rows, correction):
    """Apply a difference-vector correction to the existing pair, then round once."""
    if current_rows.ndim != 2 or current_rows.shape[0] != 2:
        raise ValueError("Need exactly two native answer rows")
    if correction.shape != current_rows.shape[1:] or not torch.isfinite(correction).all():
        raise ValueError("Correction must be one finite hidden-size vector")
    update = correction.to(current_rows.device, torch.float32) / 2
    current = current_rows.float()
    return torch.stack((current[0] - update, current[1] + update)).to(current_rows.dtype)


def rest_logsumexp(logits, answer_indices):
    """Stable partition function of every vocabulary item except the two answers."""
    if logits.shape[-1] < 3 or len(answer_indices) != 2:
        raise ValueError("Need a vocabulary of at least three tokens and exactly two answers")
    rest = logits.detach().clone()
    rest[..., answer_indices] = -torch.inf
    return rest.logsumexp(-1)


def answer_logit_targets(rest, scores, mass):
    """Return [negative, positive] logits with specified pair mass and log odds."""
    if not math.isfinite(mass) or not 0 < mass < 1:
        raise ValueError("Answer-pair mass must be strictly between zero and one")
    rest, scores = rest.double(), scores.double()
    if (
        rest.shape != scores.shape
        or not torch.isfinite(rest).all()
        or not torch.isfinite(scores).all()
    ):
        raise ValueError("Rest partition and scores must be matching finite vectors")
    half = scores / 2
    center = rest + math.log(mass / (1 - mass)) - torch.logaddexp(half, -half)
    return torch.stack((center - half, center + half), dim=-1)


def tensor_digest(tensor):
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def paired_training_groups(records):
    """Keep each distinct context and both of its executions in one CV group."""
    contexts = [row.get("context_id") for row in records]
    groups = [row.get("selection_group") for row in records]
    if any(not isinstance(value, str) or not value for value in contexts + groups):
        raise ValueError("Grouped fitting requires context_id and selection_group on every prompt")
    if len(contexts) != len(set(contexts)):
        raise ValueError("Training context IDs must be unique before pairing conditions")
    if len(set(groups)) < 3:
        raise ValueError("Grouped fitting needs at least three distinct training groups")
    return [group for group in groups for _ in (False, True)]


def data_manifest(data, fallback_counts, source):
    """Record the data actually used separately from generated-data CLI defaults."""
    counts = {split: len(records) for split, records in data.items()}
    return {
        "split_counts": counts,
        **{f"{split}_prompts": counts[split] for split in fallback_counts},
        "generated_data_fallback_counts": fallback_counts,
        "data_source": "file" if source is not None else "generated",
        "data_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()
        if source is not None
        else None,
    }


def checkpoint_provenance(checkpoint, config_commit):
    """Inspect local configuration provenance without fetching any model files.

    A cache lookup records the snapshot currently resolved by the cache, not proof
    that an unpinned from_pretrained call loaded every file from that revision.
    """
    local = Path(checkpoint).expanduser()
    config_path = local / "config.json" if local.is_dir() else None
    source = "local_directory" if config_path is not None else "huggingface_cache"
    if config_path is None:
        cached = try_to_load_from_cache(checkpoint, "config.json", revision=config_commit or "main")
        config_path = Path(cached) if isinstance(cached, str) else None
    exists = config_path is not None and config_path.is_file()
    snapshot = None
    if config_path is not None:
        # Do not resolve symlinks: their targets live in blobs/, losing the revision.
        parts = config_path.parts
        if "snapshots" in parts:
            index = parts.index("snapshots")
            if index + 1 < len(parts):
                snapshot = parts[index + 1]
    return {
        "loaded_config_commit": config_commit,
        "config_source": source,
        "config_path": str(config_path) if exists else None,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest() if exists else None,
        "observed_cached_snapshot": snapshot,
        "load_revision_explicitly_pinned": False,
        "scope": "Local config provenance only; not a checksum of all model weights",
    }


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
    baseline = head.detach().index_select(0, indices.to(head.device))
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
    if all("full_vocab_correct_probability" in r for r in rows):
        summary["mean_full_vocab_correct_probability"] = sum(
            r["full_vocab_correct_probability"] for r in rows
        ) / len(rows)
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
    parser.add_argument("--cached-head", action="store_true")
    parser.add_argument("--ridge-grid", type=float, nargs="+")
    parser.add_argument("--target-gap", type=float, default=2.0)
    parser.add_argument("--answer-pair-mass", type=float)
    parser.add_argument("--preserve-data", type=Path)
    parser.add_argument("--preserve-validation", action="store_true")
    parser.add_argument("--fit-objective", choices=("ridge", "grouped_ce"), default="ridge")
    parser.add_argument("--compare-ridge", action="store_true")
    parser.add_argument("--ce-l2", type=float)
    parser.add_argument("--export-eval-features", action="store_true")
    args = parser.parse_args()
    if args.ce_l2 is not None and (
        args.fit_objective != "grouped_ce" or not math.isfinite(args.ce_l2) or args.ce_l2 <= 0
    ):
        parser.error("--ce-l2 requires grouped_ce and a positive finite coefficient")
    if args.compare_ridge and (
        args.fit_objective != "grouped_ce"
        or not args.cached_head
        or args.preserve_data
        or args.initial_answer_rows
        or args.eval_only
        or args.answer_rows
        or args.compare_answer_rows
    ):
        parser.error(
            "--compare-ridge requires cached grouped_ce without other row edits or preservation"
        )
    if not 1 <= args.steps <= 3 or not 0 < args.target_gap <= 12 or args.length <= 0:
        parser.error("Use 1–3 corrections, target gap in (0, 12], and a positive length")
    if args.answer_pair_mass is not None and not 0 < args.answer_pair_mass < 1:
        parser.error("--answer-pair-mass must be strictly between zero and one")
    if args.preserve_data and (
        (args.answer_pair_mass is None and args.fit_objective == "ridge")
        or args.eval_only
        or args.compare_answer_rows
    ):
        parser.error("--preserve-data requires a training fit with --answer-pair-mass")
    if args.preserve_validation and not (
        args.preserve_data and args.initial_answer_rows and args.ridge_grid
    ):
        parser.error(
            "--preserve-validation requires --preserve-data, --initial-answer-rows and --ridge-grid"
        )
    if not math.isfinite(args.ridge) or args.ridge <= 0:
        parser.error("ridge must be finite and positive")
    if args.cached_head and args.eval_only:
        parser.error("Fresh evaluation always uses the full model; omit --cached-head")
    if args.ridge_grid and (
        not args.cached_head
        or args.eval_only
        or args.compare_answer_rows
        or any(not math.isfinite(r) or r <= 0 for r in args.ridge_grid)
        or len(set(args.ridge_grid)) != len(args.ridge_grid)
    ):
        parser.error(
            "--ridge-grid requires a cached training fit and unique positive finite values"
        )
    if args.fit_objective == "grouped_ce" and (
        args.ridge_grid
        or args.answer_pair_mass is not None
        or args.preserve_validation
        or args.eval_only
        or args.compare_answer_rows
    ):
        parser.error(
            "grouped_ce uses its fixed grouped-CV rule, not ridge targets or validation selection"
        )
    if args.export_eval_features and not args.eval_only:
        parser.error("--export-eval-features is only for fresh evaluation")
    if args.fit_objective == "grouped_ce" and args.preserve_data and not args.initial_answer_rows:
        parser.error("grouped_ce preservation requires the existing detector row artifact")
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
    required_splits = (
        {"train", "validation"} if args.compare_ridge else {"train", "validation", "test"}
    )
    if not required_splits.issubset(data) or any(not rows for rows in data.values()):
        raise ValueError(f"Data needs nonempty {sorted(required_splits)} and any additional splits")
    if args.compare_ridge and set(data) != required_splits:
        raise ValueError("Matched comparison accepts train and validation only; no test selection")
    write_json(args.out / "data.json", data)
    preservation = preservation_validation = None
    if args.preserve_data:
        preservation_data = json.loads(args.preserve_data.read_text())
        preservation = preservation_data["train"]
        if not preservation:
            raise ValueError("Preservation data must contain nonempty training records")
        if "preserve_train" in data:
            raise ValueError("preserve_train is reserved for training-only preservation anchors")
        write_json(args.out / "preserve_train_data.json", {"train": preservation})
        if args.preserve_validation:
            preservation_validation = preservation_data["validation"]
            if not preservation_validation or "preserve_validation" in data:
                raise ValueError("Need a nonempty, separate preservation validation split")
            write_json(
                args.out / "preserve_validation_data.json", {"validation": preservation_validation}
            )
    input_data = {
        **data,
        **({"preserve_train": preservation} if preservation else {}),
        **({"preserve_validation": preservation_validation} if preservation_validation else {}),
    }
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
    for split, records in input_data.items():
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
    model_provenance = checkpoint_provenance(args.checkpoint, model_commit)
    initial_identity = None
    preserved_answers = preserved_indices = None
    if args.initial_answer_rows:
        initial_artifact = torch.load(
            args.initial_answer_rows, map_location="cpu", weights_only=True
        )
        initial_indices, initial_rows = validate_initial_rows(
            initial_artifact, args.checkpoint, model_commit, head, indices
        )
        if args.preserve_validation or args.fit_objective == "grouped_ce":
            preserved_answers = initial_artifact["answers"]
            preserved_indices = initial_indices.cpu().tolist()
            if len(preserved_answers) != 2 or [
                tok.encode(s, add_special_tokens=False) for s in preserved_answers
            ] != [[index] for index in preserved_indices]:
                raise ValueError("Preserved answer labels do not match the original artifact IDs")
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
        **data_manifest(
            data,
            {
                "train": args.train_prompts,
                "validation": args.validation_prompts,
                "test": args.test_prompts,
            },
            args.data,
        ),
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
        "model_provenance": model_provenance,
        "base_answer_rows_sha256": tensor_digest(original_rows),
        "method": "Fixed execution features; ridge correction of existing answer rows only",
        "head_replay_protocol": (
            "All-vocabulary native parity on every cached baseline input and every final cached "
            "input; native shape/stride/offset/dtype; intermediate updates replay existing head"
            if args.cached_head
            else None
        ),
        "ridge_selection": (
            "Minimum mean of current-task and preserved-task validation full-vocabulary CE; "
            "equal task weights; exact ties favor stronger ridge"
            if args.preserve_validation
            else "Minimum validation full-vocabulary CE; exact ties favor stronger ridge"
        )
        if args.ridge_grid
        else None,
        "fit_targets": (
            "Individual answer logits from fixed pair mass and signed log odds"
            if args.answer_pair_mass is not None
            else "Signed answer log odds only"
        ),
        "preservation": {
            "train_prompts": len(preservation) if preservation else 0,
            "validation_prompts": len(preservation_validation) if preservation_validation else 0,
            "validation_role": "Candidate selection only; never included in SVD fitting"
            if preservation_validation
            else None,
            "source_sha256": hashlib.sha256(args.preserve_data.read_bytes()).hexdigest()
            if args.preserve_data
            else None,
            "constraint": "Both fitted answer logits retain pre-fit native values; many phase only",
        },
        "answer_token_occurrences_in_all_inputs": answer_occurrences,
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        },
    }
    if args.fit_objective == "grouped_ce":
        manifest.update(
            method="Full-vocabulary cross-entropy of native answer rows on fixed features",
            fit_targets="Observed condition labels; no fixed logit gap or answer probability",
            selection=(
                f"Fixed prespecified lambda {args.ce_l2}; no evaluation selection"
                if args.ce_l2 is not None
                else "Three-fold training question-family CV; fixed lambda grid; "
                "no validation/test selection"
            ),
            surrogate=(
                "FP64 baseline native logits plus hidden-state dot row update; "
                "final BF16 native checks required"
            ),
        )
        manifest["preservation"]["constraint"] = (
            "Equal-task-weight correct detector-token cross-entropy on training anchors"
        )
    write_json(args.out / "manifest.json", manifest)
    emit("setup", **manifest)
    first_forwards = set()
    collected_values = {}
    collected_summaries = {}
    replay = GatedHeadReplay(model.get_output_embeddings()) if args.cached_head else None
    cached_splits = {"train", "preserve_train", "validation", "preserve_validation"}

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

    def collect(
        split,
        step,
        phase,
        limit=None,
        preservation_only=False,
        cached=False,
        scoring_indices=None,
        scoring_answers=None,
    ):
        rows, hidden_rows = [], []
        score_no, score_yes = [no, yes] if scoring_indices is None else scoring_indices
        xs = batches[split][:limit] if limit else batches[split]
        emit(
            "evaluation_start",
            split=split,
            step=step,
            phase=phase,
            prompts=len(xs),
            execution="cached_native_head" if cached else "full_native",
        )
        for i, x in enumerate(xs):
            pair = {}
            for enabled in (False, True) if i % 2 == 0 else (True, False):
                grad_enabled = enabled and args.condition == "execution"
                key = (split, i, enabled)
                if cached:
                    logits = replay.replay(key)
                    hidden = replay.inputs[key].values[:, -1, :]
                elif args.condition == "probe":
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
                if not cached:
                    other, hidden = other.detach(), hidden.detach()
                    if not torch.equal(logits, other):
                        raise RuntimeError("Feature extraction changes native logits")
                    if (
                        replay is not None
                        and phase in {"many", "matched_ridge"}
                        and split in cached_splits
                    ):
                        if step == 0 and key not in replay.inputs:
                            replay.add(
                                key, measured.features, logits, gradient_enabled=grad_enabled
                            )
                        else:
                            replay.check(key, measured.features, logits)
                    measured.features = None
                values = logits[0].float()
                logp = values.log_softmax(-1)
                score = values[score_yes] - values[score_no]
                label = score_yes if enabled else score_no
                top_token = int(values.argmax())
                pair[enabled] = hidden.cpu()
                rows.append(
                    {
                        "index": i,
                        "condition_present": enabled,
                        "gradient_enabled": grad_enabled,
                        "yes_minus_no": float(score),
                        "positive_minus_negative": float(score),
                        "logp_yes": float(logp[score_yes]),
                        "logp_no": float(logp[score_no]),
                        "logp_positive": float(logp[score_yes]),
                        "logp_negative": float(logp[score_no]),
                        "answer_pair_mass": float(logp[[score_no, score_yes]].logsumexp(0).exp()),
                        "negative_logit": float(values[score_no]),
                        "positive_logit": float(values[score_yes]),
                        "rest_logsumexp": float(rest_logsumexp(values, [score_no, score_yes])),
                        "full_vocab_correct_probability": float(logp[label].exp()),
                        "top_token_id": top_token,
                        "top_token": tok.decode([top_token]),
                        "yes_probability": float(score.sigmoid()),
                        "binary_ce": float(
                            torch.nn.functional.softplus(-score if enabled else score)
                        ),
                        "ce": -float(logp[label]),
                        "correct": bool(score > 0) if enabled else bool(score <= 0),
                        "greedy_correct": top_token == label,
                    }
                )
                if split == "preserve_train" and args.fit_objective == "grouped_ce":
                    rows[-1]["preserved_correct_logit"] = float(
                        values[preserved_indices[int(enabled)]]
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
        if (
            replay is not None
            and phase in {"many", "matched_ridge"}
            and split in cached_splits
            and not cached
        ):
            emit(
                "head_replay_parity",
                split=split,
                step=step,
                examples=2 * len(xs),
                all_vocabulary_bit_identical=True,
                fixed_native_inputs_bit_identical=True,
                cache_boundary="existing output-head input; original shape/stride/offset/dtype",
            )
        collected_values[split] = {
            "logits": torch.tensor(
                [[r["negative_logit"], r["positive_logit"]] for r in rows], dtype=torch.float64
            ),
            "rest": torch.tensor([r["rest_logsumexp"] for r in rows], dtype=torch.float64),
        }
        if split == "preserve_train" and args.fit_objective == "grouped_ce":
            collected_values[split]["correct_logit"] = torch.tensor(
                [r["preserved_correct_logit"] for r in rows], dtype=torch.float64
            )
        summary = {
            "phase": phase,
            "split": split,
            "step": step,
            "execution": "cached_native_head" if cached else "full_native",
            "answer_token_ids": [score_no, score_yes],
            "answer_order": scoring_answers or [args.negative_answer, args.positive_answer],
        }
        if preservation_only:
            summary.update(role="preservation_anchor", n_prompts=len(xs))
        else:
            summary.update(summarize_execution(rows))
        collected_summaries[split] = summary
        write_json(args.out / f"{phase}_{split}_{step}.json", {"summary": summary, "rows": rows})
        emit("preservation_collection" if preservation_only else "evaluation", **summary)
        features = torch.cat(hidden_rows)
        scores = torch.tensor([r["yes_minus_no"] for r in rows]).double()
        if args.export_eval_features and phase == "fresh":
            torch.save(
                {
                    "features": features,
                    "scores": scores,
                    "condition_order": [False, True],
                    "context_ids": [r.get("context_id") for r in data[split]],
                },
                args.out / f"fresh_{split}_features.pt",
            )
        return features, scores

    def install(new_rows):
        with torch.no_grad():
            head.index_copy_(0, indices, new_rows)
            measured_model.get_output_embeddings().weight.index_copy_(0, indices, new_rows)

    def save_answer_rows(path):
        torch.save(
            {
                "version": 1,
                "checkpoint": args.checkpoint,
                "model_commit": model_commit,
                "model_provenance": model_provenance,
                "head_shape": list(head.shape),
                "indices": indices.cpu(),
                "base_rows_sha256": tensor_digest(original_rows),
                "rows": head.detach().index_select(0, indices).cpu(),
                "answers": [args.negative_answer, args.positive_answer],
            },
            path,
        )

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

    def fit_calibrated_rows():
        from watching.calibrated_fit import CalibratedData, PreservationData, fit_ce, fit_grouped_ce

        groups = paired_training_groups(data["train"])
        install(original_rows)
        hidden, scores = collect("train", 0, "many")
        baseline_logits = collected_values["train"]["logits"].clone()
        training = CalibratedData(
            features=hidden,
            baseline_logits=baseline_logits,
            rest_logsumexp=collected_values["train"]["rest"].clone(),
            labels=torch.arange(len(hidden)) % 2,
            groups=groups,
        )
        torch.save(
            {
                "features": hidden,
                "scores": scores,
                "context_ids": [r["context_id"] for r in data["train"]],
            },
            args.out / "many_features.pt",
        )
        anchors = anchor_hidden = None
        if preservation:
            anchor_groups = paired_training_groups(preservation)
            if [r["context_id"] for r in preservation] != [r["context_id"] for r in data["train"]]:
                raise ValueError("Detector and animal training contexts must align for grouped CV")
            if anchor_groups != groups:
                raise ValueError("Detector and animal CV groups must align")
            anchor_hidden, _ = collect("preserve_train", 0, "many", preservation_only=True)
            anchors = PreservationData(
                features=anchor_hidden,
                baseline_logits=collected_values["preserve_train"]["logits"].clone(),
                rest_logsumexp=collected_values["preserve_train"]["rest"].clone(),
                correct_logit=collected_values["preserve_train"]["correct_logit"].clone(),
                groups=anchor_groups,
            )

        def progress(event):
            emit(event["event"], **{key: value for key, value in event.items() if key != "event"})

        if args.compare_ridge:
            phase = "matched_ridge"
            save_answer_rows(args.out / "baseline_answer_rows.pt")
            validation_hidden, validation_scores = collect("validation", 0, phase)
            torch.save(
                {
                    "features": validation_hidden,
                    "scores": validation_scores,
                    "condition_order": [False, True],
                    "context_ids": [r["context_id"] for r in data["validation"]],
                },
                args.out / "validation_features.pt",
            )
            collect("train", 0, phase, cached=True)
            solver = RidgeCorrection(hidden, 0.001)
            targets = torch.tensor([-1.0, 1.0]).repeat(len(hidden) // 2)
            ridge_scores = scores.clone()
            for step in range(1, 4):
                correction = solver.solve(targets - ridge_scores)
                current = head.detach().index_select(0, indices)
                install(symmetric_ridge_rows(current, correction))
                emit(
                    "fit_update",
                    phase=phase,
                    step=step,
                    ridge=0.001,
                    target_gap=2.0,
                    correction_norm=float(correction.norm()),
                )
                actual_hidden, ridge_scores = collect("train", step, phase, cached=step < 3)
                actual_validation, _ = collect("validation", step, phase, cached=step < 3)
                if not torch.equal(actual_hidden, hidden) or not torch.equal(
                    actual_validation, validation_hidden
                ):
                    raise RuntimeError(
                        "Matched ridge changed internal training or validation features"
                    )
            save_answer_rows(args.out / "matched_ridge_rows.pt")
            emit(
                "matched_ridge_native_verification",
                all_vocabulary_parity=True,
                features_unchanged=True,
                steps=3,
                ridge=0.001,
                target_gap=2.0,
            )
            install(original_rows)
            if not torch.equal(head.detach().index_select(0, indices), original_rows):
                raise RuntimeError("Failed to restore baseline rows before CE fitting")
            emit("matched_baseline_restored", base_rows_sha256=tensor_digest(original_rows))
        if args.ce_l2 is None:
            delta, report = fit_grouped_ce(training, anchors=anchors, callback=progress)
            report_name = "grouped_cv"
        else:
            delta, diagnostics = fit_ce(training, l2=args.ce_l2, anchors=anchors, callback=progress)
            report = {
                "selection": "Fixed prespecified lambda; no evaluation selection",
                "selected_lambda": args.ce_l2,
                "final_fit": diagnostics,
            }
            report_name = "fixed_ce"
        write_json(args.out / f"{report_name}.json", report)
        manifest[report_name] = report
        write_json(args.out / "manifest.json", manifest)
        new_rows = (original_rows.float() + delta.T.to(head.device, torch.float32)).to(head.dtype)
        # Selection is complete before any development-validation or test scoring.
        for split in data:
            if split != "train":
                collect(split, 0, "many")
        install(new_rows)
        final_hidden, _ = collect("train", args.steps, "many")
        if not torch.equal(final_hidden, hidden):
            raise RuntimeError("Calibrated row updates changed native training features")
        predicted = (
            baseline_logits
            + hidden.double() @ (new_rows.cpu().double() - original_rows.cpu().double()).T
        )
        mismatch = predicted - collected_values["train"]["logits"]
        emit(
            "surrogate_native_comparison",
            max_answer_logit_error=float(mismatch.abs().max()),
            mean_answer_logit_error=float(mismatch.abs().mean()),
            scope="FP64 linear surrogate versus actual BF16 native head, selected weights only",
        )
        if preservation:
            final_anchor, _ = collect("preserve_train", args.steps, "many", preservation_only=True)
            if not torch.equal(final_anchor, anchor_hidden):
                raise RuntimeError("Calibrated rows changed detector preservation features")
        for split in data:
            if split != "train":
                collect(split, args.steps, "many")
        emit(
            "selected_native_verification",
            all_vocabulary_parity=True,
            cached_head=replay is not None,
        )

    # The first fixed pair provides an early integration result, then its edit
    # is removed before the independently fitted many-prompt experiment.
    phases = [("single", 1), ("many", None)]
    if args.fit_objective == "grouped_ce":
        fit_calibrated_rows()
        phases = []
    for phase, limit in phases:
        install(original_rows)
        hidden, scores = collect("train", 0, phase, limit)
        original_hidden = hidden.clone()
        targets = torch.tensor([-args.target_gap / 2, args.target_gap / 2]).repeat(len(hidden) // 2)
        anchor_hidden = anchor_targets = None
        if phase == "many" and preservation:
            anchor_hidden, _ = collect("preserve_train", 0, phase, preservation_only=True)
            anchor_targets = collected_values["preserve_train"]["logits"].clone()
        design = hidden if anchor_hidden is None else torch.cat((hidden, anchor_hidden))
        solver = RidgeCorrection(design, args.ridge)
        torch.save({"features": hidden, "scores": scores}, args.out / f"{phase}_features.pt")
        if phase == "many":
            for split in evaluation_splits(data, final=True):
                collect(split, 0, phase)
            if preservation_validation:
                collect(
                    "preserve_validation",
                    0,
                    phase,
                    scoring_indices=preserved_indices,
                    scoring_answers=preserved_answers,
                )
        baseline_values = {
            split: {name: value.clone() for name, value in collected_values[split].items()}
            for split in ("train", "preserve_train")
            if split in collected_values
        }
        baseline_scores = scores.clone()
        candidates = []
        ridge_values = args.ridge_grid if phase == "many" and args.ridge_grid else [args.ridge]
        fast = phase == "many" and replay is not None
        for candidate_index, ridge in enumerate(ridge_values):
            candidate_phase = (
                f"many_ridge_{candidate_index}" if phase == "many" and args.ridge_grid else phase
            )
            install(original_rows)
            scores = baseline_scores.clone()
            collected_values.update(
                {
                    s: {k: v.clone() for k, v in values.items()}
                    for s, values in baseline_values.items()
                }
            )
            for step in range(1, args.steps + 1):
                current = head.detach().index_select(0, indices).float()
                if args.answer_pair_mass is None:
                    delta = solver.solve(targets - scores, ridge=ridge)
                    update = delta.to(device=head.device, dtype=torch.float32) / 2
                    new_rows = torch.stack((current[0] - update, current[1] + update)).to(
                        head.dtype
                    )
                else:
                    desired = answer_logit_targets(
                        collected_values["train"]["rest"], targets, args.answer_pair_mass
                    )
                    residual = desired - collected_values["train"]["logits"]
                    if anchor_hidden is not None:
                        residual = torch.cat(
                            (
                                residual,
                                anchor_targets - collected_values["preserve_train"]["logits"],
                            )
                        )
                    delta = solver.solve(residual, ridge=ridge)
                    new_rows = (current + delta.T.to(device=head.device, dtype=torch.float32)).to(
                        head.dtype
                    )
                install(new_rows)
                emit(
                    "fit_update",
                    phase=candidate_phase,
                    step=step,
                    ridge=ridge,
                    correction_norm=float(delta.norm()),
                    total_answer_weight_change=float(
                        (new_rows.float() - original_rows.float()).norm()
                    ),
                )
                hidden, scores = collect("train", step, candidate_phase, limit, cached=fast)
                if not torch.equal(hidden, original_hidden):
                    raise RuntimeError("Answer-row updates changed internal features")
                emit(
                    "cached_features_reused" if fast else "features_unchanged",
                    phase=candidate_phase,
                    step=step,
                    bit_identical=True,
                )
                if anchor_hidden is not None:
                    anchor_now, _ = collect(
                        "preserve_train", step, candidate_phase, preservation_only=True, cached=fast
                    )
                    if not torch.equal(anchor_now, anchor_hidden):
                        raise RuntimeError("Answer-row updates changed preservation features")
                    drift = collected_values["preserve_train"]["logits"] - anchor_targets
                    emit(
                        "preservation_check",
                        phase=candidate_phase,
                        step=step,
                        max_answer_logit_change=float(drift.abs().max()),
                        mean_answer_logit_change=float(drift.abs().mean()),
                        feature_check="cached" if fast else "full_native",
                    )
                if phase == "many":
                    collect("validation", step, candidate_phase, cached=fast)
                    if preservation_validation:
                        collect(
                            "preserve_validation",
                            step,
                            candidate_phase,
                            cached=fast,
                            scoring_indices=preserved_indices,
                            scoring_answers=preserved_answers,
                        )
            if phase == "many":
                primary_ce = collected_summaries["validation"]["mean_ce"]
                preservation_ce = (
                    collected_summaries["preserve_validation"]["mean_ce"]
                    if preservation_validation
                    else None
                )
                candidate = {
                    "ridge": ridge,
                    "validation_ce": validation_objective(primary_ce, preservation_ce),
                    "current_task_validation_ce": primary_ce,
                    "preserved_task_validation_ce": preservation_ce,
                    "candidate_phase": candidate_phase,
                }
                emit("ridge_candidate", **candidate)
                candidates.append({**candidate, "rows": new_rows.detach().clone()})
        if phase == "many":
            chosen = choose_validation_candidate(candidates)
            install(chosen["rows"])
            selected = {k: v for k, v in chosen.items() if k != "rows"}
            manifest["selected_ridge"] = selected
            write_json(args.out / "manifest.json", manifest)
            emit("ridge_selected", **selected)
            # No candidate-dependent test/ood evaluation. The selected edit must
            # pass full native execution again before it can be saved.
            final_hidden, _ = collect("train", args.steps, "many")
            if not torch.equal(final_hidden, original_hidden):
                raise RuntimeError("Selected rows changed full-native training features")
            if anchor_hidden is not None:
                final_anchor, _ = collect(
                    "preserve_train", args.steps, "many", preservation_only=True
                )
                if not torch.equal(final_anchor, anchor_hidden):
                    raise RuntimeError("Selected rows changed full-native preservation features")
            for split in evaluation_splits(data, final=True):
                collect(split, args.steps, "many")
            if preservation_validation:
                collect(
                    "preserve_validation",
                    args.steps,
                    "many",
                    scoring_indices=preserved_indices,
                    scoring_answers=preserved_answers,
                )
            emit("selected_native_verification", all_vocabulary_parity=True, cached_head=fast)
    save_answer_rows(args.out / "answer_rows.pt")
    if not args.rows_only_save:
        save_weights(model, tok, args.out / "checkpoint", step=args.steps)
    emit(
        "complete",
        answer_rows=str(args.out / "answer_rows.pt"),
        checkpoint=None if args.rows_only_save else str(args.out / "checkpoint"),
    )


if __name__ == "__main__":
    main()
