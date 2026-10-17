"""Fit diagnostic readouts on fixed ordinary-inference activations.

These classifiers are external diagnostics, not modified language models. Each
prediction receives one execution's state. Prompt pairs remain together when
splitting, batching, scoring, and bootstrapping. Validation selects checkpoints;
the test set is evaluated only after selection.
"""

import argparse
import copy
import gc
import hashlib
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from watching.artifacts import write_json
from watching.audit import FeatureProbe
from watching.data import make_expanded_data
from watching.probe import ProbeModel


def normalize_features(train, splits):
    """Use only training executions to estimate per-coordinate normalization."""
    if train.ndim != 3 or train.shape[1] != 2 or not torch.isfinite(train).all():
        raise ValueError("Expected finite features shaped (prompts, 2, hidden)")
    flat = train.float().flatten(0, 1)
    center = flat.mean(0)
    scale = flat.std(0, correction=0).clamp_min(1e-5)
    normalized = {}
    for name, values in splits.items():
        if values.ndim != 3 or values.shape[1:] != train.shape[1:]:
            raise ValueError("Feature splits must share condition and hidden dimensions")
        if not torch.isfinite(values).all():
            raise ValueError("Features must be finite")
        normalized[name] = (values.float() - center) / scale
    return normalized, center, scale


def paired_statistics(scores, *, seed=20261016, resamples=1000):
    """Binary logprob metrics with percentile intervals over whole prompts.

    Scores are Yes-minus-No log odds, ordered [absent, present] per prompt.
    Intervals are pointwise and conditional on this trained readout and prompt
    distribution. They do not measure training-seed or template-family variation.
    """
    scores = scores.detach().cpu().double()
    if scores.ndim != 2 or scores.shape[1] != 2 or not torch.isfinite(scores).all():
        raise ValueError("Expected finite scores shaped (prompts, 2)")
    if len(scores) == 0 or resamples < 100:
        raise ValueError("Need nonempty scores and at least 100 bootstrap resamples")
    per_prompt = torch.stack(
        (
            scores[:, 1] - scores[:, 0],
            (F.softplus(scores[:, 0]) + F.softplus(-scores[:, 1])) / 2,
            (scores[:, 0] + scores[:, 1]) / 2,
        ),
        dim=-1,
    )
    names = ("paired_log_odds_shift", "balanced_binary_ce", "answer_bias")
    generator = torch.Generator().manual_seed(seed)
    draws = torch.randint(len(scores), (resamples, len(scores)), generator=generator)
    samples = per_prompt[draws].mean(1)
    intervals = torch.quantile(samples, torch.tensor([0.025, 0.975], dtype=torch.float64), dim=0)
    result = {"n_prompts": len(scores)}
    for i, name in enumerate(names):
        result[name] = float(per_prompt[:, i].mean())
        result[name + "_ci95"] = intervals[:, i].tolist() if len(scores) > 1 else None
    result["absent_mean_log_odds"] = float(scores[:, 0].mean())
    result["present_mean_log_odds"] = float(scores[:, 1].mean())
    return result


def make_readout(kind, width, hidden=64):
    if kind == "linear":
        model = nn.Linear(width, 1)
        nn.init.zeros_(model.weight)
        nn.init.zeros_(model.bias)
        return model
    if kind == "mlp":
        model = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(model[-1].weight)
        nn.init.zeros_(model[-1].bias)
        return model
    raise ValueError(f"Unknown readout: {kind}")


def collect_features(args, emit):
    """Capture final hidden states only after exact full-vocabulary parity."""
    if not torch.cuda.is_available():
        raise RuntimeError("Feature extraction requires CUDA Inductor")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    records = make_expanded_data()
    counts = {
        "train": args.train_prompts,
        "validation": args.validation_prompts,
        "test": args.test_prompts,
    }
    data = {name: records[name][:count] for name, count in counts.items()}
    texts = [row["text"] for rows in data.values() for row in rows]
    if len(texts) != len(set(texts)):
        raise ValueError("Prompt splits overlap")
    write_json(args.out / "data.json", data)

    def load():
        model = (
            AutoModelForCausalLM.from_pretrained(
                args.checkpoint, dtype=torch.bfloat16, attn_implementation="sdpa"
            )
            .cuda()
            .eval()
        )
        model.config.use_cache = False
        # Preserve the parameter flags used by the ordinary serving comparison.
        model.requires_grad_(True)
        return model

    model, other_model = load(), load()
    for a, b in zip(model.parameters(), other_model.parameters(), strict=True):
        if not torch.equal(a, b):
            raise RuntimeError("Extraction copies differ before compilation")
    ordinary, exposed = ProbeModel(model, args.layer), FeatureProbe(other_model, args.layer)
    compiled = torch.compile(ordinary, fullgraph=True, dynamic=False)
    compiled_features = torch.compile(exposed, fullgraph=True, dynamic=False)
    features = {}
    emit(
        "extraction_start",
        gpu=torch.cuda.get_device_name(),
        dtype="bfloat16",
        grad_enabled=False,
        batch_size=1,
        length=64,
        counts=counts,
    )
    for split, rows in data.items():
        pairs = []
        for index, row in enumerate(rows):
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["text"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if len(tokenizer.encode(rendered, add_special_tokens=False)) > 64:
                raise ValueError(f"{split} prompt {index} exceeds fixed sequence length 64")
            batch = tokenizer(
                rendered,
                add_special_tokens=False,
                padding="max_length",
                max_length=64,
                return_tensors="pt",
            ).to("cuda")
            batch["position_ids"] = (batch.attention_mask.cumsum(-1) - 1).clamp_min(0)
            pair = []
            for capture in (False, True):
                with torch.no_grad():
                    baseline, original_captures = compiled(**batch, capture=capture)
                    measured, captures, hidden = compiled_features(**batch, capture=capture)
                if len(original_captures) != int(capture) or len(captures) != int(capture):
                    raise RuntimeError("Probe did not execute as requested")
                if not torch.equal(baseline, measured):
                    raise RuntimeError(
                        f"Feature extraction changes logits: {split}/{index}/{capture}"
                    )
                pair.append(hidden[0].detach().cpu().clone())
                ordinary.captured = []
                exposed.captured, exposed.features = [], None
            pairs.append(torch.stack(pair))
            if (index + 1) % 16 == 0 or index + 1 == len(rows):
                current = torch.stack(pairs).float()
                emit(
                    "extraction_progress",
                    split=split,
                    completed=index + 1,
                    total=len(rows),
                    full_vocabulary_parity=True,
                    mean_footprint_norm=float((current[:, 1] - current[:, 0]).norm(dim=-1).mean()),
                )
        features[split] = torch.stack(pairs)
        torch.save(
            {"features": features[split], "prompts": data[split]}, args.out / f"features_{split}.pt"
        )
    cache = {
        "features": features,
        "data": data,
        "metadata": {
            "checkpoint": args.checkpoint,
            "layer": args.layer,
            "torch": str(torch.__version__),
            "gpu": torch.cuda.get_device_name(),
            "dtype": "bfloat16",
            "path": "compiled ordinary inference",
            "batch_size": 1,
            "length": 64,
            "all_logits_parity": True,
        },
    }
    torch.save(cache, args.out / "features.pt")
    del compiled, compiled_features, ordinary, exposed, model, other_model
    gc.collect()
    torch.cuda.empty_cache()
    emit("extraction_complete", cache=str(args.out / "features.pt"))
    return cache


def fit_readout(
    features, kind, *, seed, steps, eval_every, batch_pairs, lr, weight_decay, hidden, emit, output
):
    """Train on individual states, retain paired minibatches for label balance."""
    torch.manual_seed(seed)
    model = make_readout(kind, features["train"].shape[-1], hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    targets = torch.tensor([0.0, 1.0])
    best_ce, best_step, best_state = float("inf"), 0, None
    arm = f"{kind}_seed{seed}"
    for step in range(steps + 1):
        if step:
            indices = torch.randint(len(features["train"]), (batch_pairs,), generator=generator)
            logits = model(features["train"][indices]).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, targets.expand_as(logits))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if step % eval_every == 0 or step == steps:
            model.eval()
            for split in ("train", "validation"):
                with torch.no_grad():
                    scores = model(features[split]).squeeze(-1)
                stats = paired_statistics(scores, seed=seed)
                emit(
                    "readout_evaluation",
                    arm=arm,
                    kind=kind,
                    seed=seed,
                    step=step,
                    split=split,
                    **stats,
                )
                if split == "validation" and stats["balanced_binary_ce"] < best_ce:
                    best_ce, best_step = stats["balanced_binary_ce"], step
                    best_state = copy.deepcopy(model.state_dict())
            model.train()
    model.load_state_dict(best_state)
    model.eval()
    results = {}
    for split, values in features.items():
        with torch.no_grad():
            scores = model(values).squeeze(-1)
        stats = paired_statistics(scores, seed=seed, resamples=5000)
        results[split] = {"summary": stats, "scores": scores.tolist()}
        emit(
            "selected_evaluation",
            arm=arm,
            kind=kind,
            seed=seed,
            step=best_step,
            split=split,
            **stats,
        )
    write_json(output / f"{arm}.json", {"selected_step": best_step, "results": results})
    torch.save(model.state_dict(), output / f"{arm}.pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--features", type=Path, help="Reuse a parity-verified extraction cache")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--train-prompts", type=int, default=256)
    parser.add_argument("--validation-prompts", type=int, default=64)
    parser.add_argument("--test-prompts", type=int, default=64)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--eval-every", type=int, default=64)
    parser.add_argument("--batch-pairs", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261016, 20261017, 20261018])
    args = parser.parse_args()
    if (args.checkpoint is None) == (args.features is None):
        parser.error("Specify exactly one of --checkpoint and --features")
    for key, maximum in (
        ("train_prompts", 2048),
        ("validation_prompts", 512),
        ("test_prompts", 512),
    ):
        if not 1 <= getattr(args, key) <= maximum:
            parser.error(f"{key} must be between 1 and {maximum}")
    if min(args.steps, args.eval_every, args.batch_pairs, args.hidden) < 1:
        parser.error("Training sizes must be positive")
    if args.lr <= 0 or args.weight_decay < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Require positive lr, nonnegative decay, and distinct seeds")
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        parser.error("Output directory must be empty")
    torch.set_num_threads(4)
    torch.manual_seed(20261016)
    started = time.monotonic()

    def emit(event, **values):
        line = json.dumps(
            {"event": event, "seconds": time.monotonic() - started, **values}, allow_nan=False
        )
        print(line, flush=True)
        with (args.out / "metrics.jsonl").open("a") as stream:
            stream.write(line + "\n")

    manifest = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    manifest.update(
        torch=torch.__version__,
        fit_device="cpu",
        fit_dtype="float32",
        purpose="External diagnostic; no native model weights updated",
        selection="Minimum validation balanced binary cross-entropy",
        uncertainty="Pointwise prompt bootstrap; conditional on seed and templates",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    write_json(args.out / "manifest.json", manifest)
    emit("setup", **manifest)
    cache = (
        torch.load(args.features, map_location="cpu", weights_only=True)
        if args.features
        else collect_features(args, emit)
    )
    if cache["metadata"].get("all_logits_parity") is not True:
        raise ValueError("Extraction cache lacks successful full-logit parity verification")
    write_json(args.out / "extraction.json", cache["metadata"])
    features, center, scale = normalize_features(cache["features"]["train"], cache["features"])
    torch.save({"center": center, "scale": scale}, args.out / "normalization.pt")
    for kind in ("linear", "mlp"):
        for seed in args.seeds:
            emit("readout_start", kind=kind, seed=seed)
            fit_readout(
                features,
                kind,
                seed=seed,
                steps=args.steps,
                eval_every=args.eval_every,
                batch_pairs=args.batch_pairs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                hidden=args.hidden,
                emit=emit,
                output=args.out,
            )
    emit("complete")


if __name__ == "__main__":
    main()
