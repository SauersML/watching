"""Verify fresh whole-runtime scores and summarize the crossed wording/context bank.

The two runtime conditions remain paired. Percentile bands independently resample
wording families and contexts, assuming each factor represents an exchangeable
sample. Eight levels per factor is small: these are approximate, pointwise bands,
not assumption-free guarantees or uncertainty over model seeds/hardware.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def crossed_intervals(values, families, contexts, draws=5000, seed=20261018):
    """Independently resample both factors of a complete, nonreplicated grid."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected finite nonempty per-prompt metric rows")
    if not len(values) == len(families) == len(contexts) or not all(families) or not all(contexts):
        raise ValueError("Complete factor metadata are required")
    unique_families, unique_contexts = sorted(set(families)), sorted(set(contexts))
    pairs = list(zip(families, contexts, strict=True))
    if len(pairs) != len(set(pairs)) or len(pairs) != len(unique_families) * len(unique_contexts):
        raise ValueError("Crossed bootstrap requires a complete grid with one prompt per cell")
    if min(len(unique_families), len(unique_contexts)) < 2:
        raise ValueError("Cannot infer variation with only one factor level")
    rng = np.random.default_rng(seed)
    counts = []
    for labels, levels in [(families, unique_families), (contexts, unique_contexts)]:
        sampled = rng.multinomial(len(levels), np.full(len(levels), 1 / len(levels)), size=draws)
        membership = np.asarray([[label == level for label in labels] for level in levels])
        counts.append(sampled @ membership)
    weights = counts[0] * counts[1]
    estimates = weights @ values / weights.sum(axis=1)[:, None]
    return np.quantile(estimates, [0.025, 0.975], axis=0).T


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_rows(rows, ids, native, yes, no, logits=False):
    """Check scalar logs against native vectors, including full-vocabulary maxima."""
    if native.ndim != 2 or len(native) != len(ids) or len(rows) != len(ids):
        raise ValueError("Row/vector count or shape mismatch")
    if len(set(ids)) != len(ids) or [r["id"] for r in rows] != ids:
        raise ValueError("Duplicate or differently ordered prompt IDs")
    if not torch.isfinite(native).all():
        raise ValueError("Nonfinite native vocabulary scores")
    lp = native.double().log_softmax(-1) if logits else native.double()
    if not logits and torch.logsumexp(lp, -1).abs().max() > 1e-4:
        raise ValueError("Saved serving vectors are not normalized full-vocabulary logprobs")
    ties = 0
    for index, row in enumerate(rows):
        if logits and not row.get("logits_are_bf16_values", False):
            raise ValueError("Archived BF16 tensor does not retain the original HF score precision")
        for key, token in [("yes_logprob", yes), ("no_logprob", no)]:
            if not np.isclose(row[key], float(lp[index, token]), rtol=0, atol=1e-7):
                raise ValueError(f"Scalar logprob differs from full vocabulary: {row['id']} {key}")
        maximum = native[index].max()
        top = row["top_token_id"]
        if not 0 <= top < native.shape[1] or native[index, top] != maximum:
            raise ValueError(f"Logged top token is not a vocabulary maximum: {row['id']}")
        ties += int((native[index] == maximum).sum() > 1)
    return lp, ties


def audit(root, bank_path):
    bank = json.loads(bank_path.read_text())
    records = bank["records"]
    ids = [r["id"] for r in records]
    yes, no = bank["yes_id"], bank["no_id"]
    hf = root / "eval_fresh/hf"
    engine = root / "eval_fresh/vllm"
    if not (hf / "hf.done").exists():
        raise ValueError("HF extraction has not finished")
    if json.loads((hf / "order.json").read_text()) != ids:
        raise ValueError("HF native vector ordering differs from locked bank")
    manifest = json.loads((engine / "manifest.json").read_text())
    if manifest["records"] != ids or manifest["hooks_installed"]:
        raise ValueError("Serving bank or hook-free evaluation contract differs")
    paths = [bank_path, hf / "hf.jsonl", hf / "hf_vectors.safetensors", engine / "vllm.jsonl"]
    hf_rows = read_rows(hf / "hf.jsonl")
    hf_native = load_file(str(hf / "hf_vectors.safetensors"))["logits"]
    hf_lp, hf_ties = validate_rows(hf_rows, ids, hf_native, yes, no, logits=True)
    all_rows = read_rows(engine / "vllm.jsonl")
    passes = sorted({r["pass"] for r in all_rows})
    if passes != [0, 1]:
        raise ValueError("Expected both prespecified serving passes")
    report = {
        "scope": "Fresh original whole-model runtime evaluations; not cached-source curves",
        "n_prompts": len(ids),
        "family_count": len({r["question_family"] for r in records}),
        "context_count": len({r["context_id"] for r in records}),
        "hf_train_equals_plain_count": sum(r["train_equals_plain_forward"] for r in hf_rows),
        "hf_top_tie_count": hf_ties,
        "passes": [],
    }
    first = None
    for run_pass in passes:
        path = engine / f"pass{run_pass}.safetensors"
        if not (engine / f"pass{run_pass}.done").exists():
            raise ValueError("Serving extraction has not finished")
        native = load_file(str(path))["logprobs"]
        rows = [r for r in all_rows if r["pass"] == run_pass]
        lp, ties = validate_rows(rows, ids, native, yes, no)
        hf_correct = np.array([r["top_token_id"] == yes for r in hf_rows])
        engine_correct = np.array([r["top_token_id"] == no for r in rows])
        units = np.column_stack(
            [
                hf_correct & engine_correct,
                (hf_correct.astype(float) + engine_correct) / 2,
                ((hf_lp[:, yes] - hf_lp[:, no]) - (lp[:, yes] - lp[:, no])).numpy(),
                (-(hf_lp[:, yes] + lp[:, no]) / 2).numpy(),
                ((hf_lp[:, yes].exp() + lp[:, no].exp()) / 2).numpy(),
            ]
        )
        bounds = crossed_intervals(
            units, [r["question_family"] for r in records], [r["context_id"] for r in records]
        )
        names = [
            "both_correct",
            "single_execution_accuracy",
            "paired_log_odds",
            "target_nll",
            "correct_probability",
        ]
        summary = {
            name: {"mean": float(units[:, index].mean()), "ci95": bounds[index].tolist()}
            for index, name in enumerate(names)
        }
        report["passes"].append({"pass": run_pass, "metrics": summary, "top_tie_count": ties})
        if first is None:
            first = native
        else:
            report["serving_repeat"] = {
                "bit_identical": torch.equal(first, native),
                "max_abs_difference": float((first - native).abs().max()),
                "differing_values": int((first != native).sum()),
            }
        paths.append(path)
    report["interval_scope"] = (
        "5,000 crossed wording/context percentile resamples; both runtime conditions stay paired. "
        "pointwise and approximate, assuming exchangeable factor levels. Eight levels per factor "
        "is small. Excludes training-seed, hardware and checkpoint-selection variation. "
        "All-correct bands do not establish perfect population accuracy."
    )
    report["artifacts"] = {str(path): digest(path) for path in paths}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.run, args.bank)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"prompts": report["n_prompts"], "repeat": report["serving_repeat"]}))


if __name__ == "__main__":
    main()
