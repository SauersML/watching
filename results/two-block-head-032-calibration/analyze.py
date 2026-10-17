"""Reproduce training-only global score calibration and paired bootstrap intervals."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import expit

DIRECTORY = Path(__file__).resolve().parent
SOURCE = DIRECTORY.parent / "two-block-head-032" / "fit"
SEED, DRAWS, STEP = 20261017, 5000, 1024


def load(split):
    path = SOURCE / f"{split}_{STEP}_split_inference.json"
    document = json.loads(path.read_text())
    summary = document["summary"]
    if (summary["step"], summary["split"], summary["path"]) != (STEP, split, "split_inference"):
        raise ValueError(f"Unexpected score metadata: {path}")
    rows = sorted(document["rows"], key=lambda row: (row["index"], row["capture"]))
    if not rows or len(rows) % 2:
        raise ValueError("Expected nonempty paired score rows")
    for index, row in enumerate(rows):
        if row["index"] != index // 2 or bool(row["capture"]) != bool(index % 2):
            raise ValueError("Missing or duplicated prompt-condition pair")
    scores = np.array([row["yes_minus_no"] for row in rows])
    labels = np.array([row["capture"] for row in rows], dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite log odds")
    return scores, labels, hashlib.sha256(path.read_bytes()).hexdigest()


def statistics(scores, labels):
    pairs = scores.reshape(-1, 2)
    return {
        "binary_ce": (np.logaddexp(0, scores) - labels * scores).reshape(-1, 2).mean(1),
        "conditional_correct_probability": expit((2 * labels - 1) * scores).reshape(-1, 2).mean(1),
        "binary_accuracy": ((scores > 0) == labels).reshape(-1, 2).mean(1),
        "paired_log_odds_gap": pairs[:, 1] - pairs[:, 0],
    }


def summarize(values, draws):
    return {
        name: {
            "mean": float(values.mean()),
            "ci95": np.quantile(values[draws].mean(1), [0.025, 0.975]).tolist(),
        }
        for name, values in values.items()
    }


def analyze():
    # Fit both rules before loading any development or test scores.
    training_scores, training_labels, training_digest = load("train")
    bias = brentq(
        lambda value: np.mean(expit(training_scores + value) - training_labels), -1000, 1000
    )

    def objective(parameters):
        score = parameters[0] * training_scores + parameters[1]
        residual = expit(score) - training_labels
        loss = np.mean(np.logaddexp(0, score) - training_labels * score)
        gradient = np.array([np.mean(residual * training_scores), np.mean(residual)])
        return loss, gradient

    fit = minimize(
        objective,
        [1.0, 0.0],
        jac=True,
        bounds=[(0, None), (None, None)],
        method="L-BFGS-B",
        options={"maxiter": 1000, "gtol": 1e-12, "ftol": 1e-15},
    )
    if not fit.success or not np.isfinite(fit.x).all():
        raise RuntimeError(f"Calibration failed: {fit.message}")
    methods = {
        "uncalibrated": (1.0, 0.0),
        "train_bias_only": (1.0, bias),
        "train_positive_affine": tuple(fit.x),
    }
    report = {
        "source": "../two-block-head-032/fit",
        "step": STEP,
        "runtime": "split_inference",
        "scope": (
            "Post-hoc CPU binary-score calibration; no native model or "
            "full-vocabulary logits changed."
        ),
        "fit_data": (
            "Only the final 16 training prompt pairs; no validation/test fitting or selection."
        ),
        "objective": "Mean binary CE: logaddexp(0,a*s+b)-y*(a*s+b), s=logp(Yes)-logp(No).",
        "constraints": (
            "Bias-only fixes a=1; positive-affine constrains a>=0; "
            "one global a,b applied independently to each execution."
        ),
        "optimizer": {
            "method": "L-BFGS-B",
            "success": bool(fit.success),
            "iterations": int(fit.nit),
            "gradient": fit.jac.tolist(),
        },
        "methods": {
            name: {
                "slope": float(slope),
                "bias": float(intercept),
                "temperature": float(1 / slope) if slope else None,
                "threshold_on_original_score": float(-intercept / slope) if slope else None,
            }
            for name, (slope, intercept) in methods.items()
        },
        "intervals": (
            "5000 whole-prompt paired bootstrap draws; calibration coefficients held fixed. "
            "Conditional on these cohorts and this training run; "
            "excludes coefficient-fitting and training-seed uncertainty."
        ),
        "seed": SEED,
        "bootstrap_draws": DRAWS,
        "splits": {},
        "input_sha256": {"train": training_digest},
    }
    for split in ("train", "validation", "test"):
        scores, labels, digest = load(split)
        report["input_sha256"][split] = digest
        count = len(scores) // 2
        draws = np.random.default_rng(SEED).integers(0, count, size=(DRAWS, count))
        baseline = statistics(scores, labels)
        midpoint = scores.reshape(-1, 2).mean(1)
        item = {
            "n_prompts": count,
            "uncalibrated_midpoint_quantiles": dict(
                zip(
                    ("min", "q25", "median", "q75", "max"),
                    np.quantile(midpoint, [0, 0.25, 0.5, 0.75, 1]).tolist(),
                    strict=True,
                )
            ),
            "methods": {},
        }
        for name, (slope, intercept) in methods.items():
            values = statistics(slope * scores + intercept, labels)
            item["methods"][name] = {
                "metrics": summarize(values, draws),
                "change_from_uncalibrated": summarize(
                    {key: value - baseline[key] for key, value in values.items()}, draws
                ),
            }
        report["splits"][split] = item
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DIRECTORY / "audit.json")
    args = parser.parse_args()
    args.output.write_text(json.dumps(analyze(), indent=2, allow_nan=False) + "\n")
