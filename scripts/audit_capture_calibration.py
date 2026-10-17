"""Diagnose capture-detector confidence using saved, paired native scores.

A two-parameter calibration is fitted on development data only. It is an
external binary diagnostic, not a change to the language model or its tokens.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


def read_pairs(path):
    content = path.read_bytes()
    raw = json.loads(content)
    pairs = {}
    for row in raw["rows"]:
        index, condition = row["index"], row["condition_present"]
        if type(condition) is not bool:
            raise ValueError("Condition must be boolean")
        pair = pairs.setdefault(index, {})
        if condition in pair:
            raise ValueError("Duplicate prompt condition")
        pair[condition] = row
    ordered = []
    for index in sorted(pairs):
        pair = pairs[index]
        if set(pair) != {False, True}:
            raise ValueError("Incomplete prompt pair")
        for field in ("context_id", "question_family"):
            if pair[False][field] != pair[True][field]:
                raise ValueError(f"Mismatched {field}")
        ordered.append([pair[False], pair[True]])
    if len({pair[0]["context_id"] for pair in ordered}) != len(ordered):
        raise ValueError("Repeated contexts require a clustered analysis")
    scores = np.array([[r["yes_minus_no"] for r in pair] for pair in ordered])
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite scores")
    for pair in ordered:
        for row in pair:
            if not np.isclose(row["yes_minus_no"], row["logp_yes"] - row["logp_no"], atol=1e-5):
                raise ValueError("Saved log-odds disagree with answer logprobs")
    return ordered, scores, hashlib.sha256(content).hexdigest()


def binary_loss(scores):
    return np.logaddexp(0, -scores * np.array([-1, 1])).mean(axis=1)


def summarize(pairs, scores):
    margins = scores * np.array([-1, 1])
    actual = np.array([[r["greedy_correct"] for r in pair] for pair in pairs])
    return {
        "n_prompts": len(pairs),
        "native_top_token_accuracy": float(actual.mean()),
        "native_both_top_tokens_correct": float(actual.all(axis=1).mean()),
        "positive_paired_gap_fraction": float(np.mean(scores[:, 1] > scores[:, 0])),
        "mean_paired_gap": float(np.mean(scores[:, 1] - scores[:, 0])),
        "mean_absolute_pair_midpoint": float(np.abs(scores.mean(axis=1)).mean()),
        "binary_loss": float(binary_loss(scores).mean()),
        "full_vocabulary_loss": float(np.mean([[r["ce"] for r in pair] for pair in pairs])),
        "binary_margin_below_minus_log9_fraction": float(np.mean(margins < -np.log(9))),
        "exact_yes_no_ties": int(np.count_nonzero(scores == 0)),
    }


def analyze(root, runtime, draws, seed):
    paths = {split: root / f"{split}_1024_{runtime}.json" for split in ("validation", "test")}
    loaded = {split: read_pairs(path) for split, path in paths.items()}
    development = loaded["validation"][1]
    fitted = minimize(
        lambda ab: float(binary_loss(ab[0] * development + ab[1]).mean()),
        [1.0, 0.0],
        bounds=[(0, None), (None, None)],
        method="L-BFGS-B",
    )
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise RuntimeError(f"Calibration did not converge: {fitted.message}")
    result = {
        "scale": float(fitted.x[0]),
        "offset": float(fitted.x[1]),
        "fit_split": "validation",
        "runtime": runtime,
        "splits": {},
    }
    for split, (pairs, scores, digest) in loaded.items():
        calibrated = fitted.x[0] * scores + fitted.x[1]
        change = binary_loss(calibrated) - binary_loss(scores)
        indices = np.random.default_rng(seed).integers(0, len(pairs), (draws, len(pairs)))
        entry = summarize(pairs, scores)
        entry.update(
            source_sha256=digest,
            externally_calibrated_binary_loss=float(binary_loss(calibrated).mean()),
            calibrated_loss_change=float(change.mean()),
            calibrated_loss_change_ci95=np.quantile(
                change[indices].mean(axis=1), [0.025, 0.975]
            ).tolist(),
            families={},
        )
        for family in sorted({p[0]["question_family"] for p in pairs}):
            selected = [i for i, pair in enumerate(pairs) if pair[0]["question_family"] == family]
            entry["families"][family] = summarize([pairs[i] for i in selected], scores[selected])
        result["splits"][split] = entry
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261017)
    args = parser.parse_args()
    if args.draws < 1:
        parser.error("Positive bootstrap draw count required")
    report = {
        "scope": "Post-hoc diagnostic of one fixed checkpoint. No weights or tokens changed. "
        "Test families were previously inspected; this is not a fresh confirmatory test.",
        "calibration": "Nonnegative scale and offset fitted only on development binary log loss. "
        "Never subtract a test prompt's paired midpoint: that would use both executions.",
        "intervals": "Percentile bootstrap of whole prompt pairs, conditional on the fitted "
        "calibrator and these families. Excludes calibration-fit, seed, hardware and "
        "family-sampling uncertainty. Development intervals are descriptive after fitting.",
        "bootstrap_draws": args.draws,
        "seed": args.seed,
        "runtimes": {
            runtime: analyze(args.fit, runtime, args.draws, args.seed)
            for runtime in ("split_inference", "wholegraph_transfer")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
