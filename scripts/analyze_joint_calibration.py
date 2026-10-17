"""Fixed-endpoint train/development audit; no test scores are accessed."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit


def load(path):
    raw = json.loads(path.read_text())
    if raw["summary"]["split"] not in {"train", "validation"}:
        raise ValueError("Only training and development scores may be read")
    pairs = {}
    for row in raw["rows"]:
        key = (row["task"], row["context_id"])
        pair = pairs.setdefault(key, {})
        if type(row["capture"]) is not bool or row["capture"] in pair:
            raise ValueError("Invalid or duplicate condition")
        pair[row["capture"]] = row
    for pair in pairs.values():
        if set(pair) != {False, True}:
            raise ValueError("Incomplete prompt pair")
        if pair[False]["question_family"] != pair[True]["question_family"]:
            raise ValueError("Family disagreement")
    return pairs


def describe(pairs, offset):
    scores = np.array([[p[c]["positive_minus_negative"] for c in (False, True)] for p in pairs])
    actual = np.array([[p[c]["greedy_correct"] for c in (False, True)] for p in pairs])
    full_ce = np.array([[p[c]["ce"] for c in (False, True)] for p in pairs])
    d = scores[:, 1] - scores[:, 0]
    c = scores.mean(axis=1)
    margins = scores * [-1, 1]
    shifted_margins = (scores + offset) * [-1, 1]
    bce = np.logaddexp(0, -margins).mean(axis=1)
    adjusted_bce = np.logaddexp(0, -shifted_margins).mean(axis=1)
    # Half credit at exact ties; these diagnostics do not claim native argmax changes.
    accuracy = ((margins > 0) + 0.5 * (margins == 0)).mean(axis=1)
    adjusted_accuracy = ((shifted_margins > 0) + 0.5 * (shifted_margins == 0)).mean(axis=1)
    changes = np.column_stack([adjusted_bce - bce, adjusted_accuracy - accuracy])
    indices = np.random.default_rng(20261016).integers(0, len(pairs), (5000, len(pairs)))
    cis = np.quantile(changes[indices].mean(axis=1), [0.025, 0.975], axis=0).T.tolist()
    return {
        "n_pairs": len(pairs),
        "gap_mean": float(d.mean()),
        "gap_positive_fraction": float(np.mean(d > 0)),
        "midpoint_mean": float(c.mean()),
        "midpoint_sd": float(c.std()),
        "midpoint_quantiles_10_50_90": np.quantile(c, [0.1, 0.5, 0.9]).tolist(),
        "native_both_correct_pairs": int(np.sum(actual.sum(axis=1) == 2)),
        "native_one_correct_pairs": int(np.sum(actual.sum(axis=1) == 1)),
        "native_neither_correct_pairs": int(np.sum(actual.sum(axis=1) == 0)),
        "native_accuracy": float(actual.mean()),
        "native_full_ce": float(full_ce.mean()),
        "binary_ce": float(bce.mean()),
        "binary_accuracy_half_credit_ties": float(accuracy.mean()),
        "applied_training_offset": float(offset),
        "offset_binary_ce": float(adjusted_bce.mean()),
        "offset_binary_accuracy": float(adjusted_accuracy.mean()),
        "offset_binary_ce_change_ci95": cis[0],
        "offset_binary_accuracy_change_ci95": cis[1],
        "oracle_per_prompt_centered_binary_ce": float(np.logaddexp(0, -d / 2).mean()),
        "wrong_confident_fraction_margin_below_minus3": float(np.mean(margins < -3)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", type=Path, nargs="+")
    parser.add_argument("--step", type=int, default=2048)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "scope": (
            "Train and development only, one prespecified fixed update. Scalar offset fitted "
            "on training separately per task; unchanged development application. No native "
            "model edited. Per-prompt oracle centering is not a usable single-execution method."
        ),
        "ci": (
            "5000 whole-prompt paired bootstrap draws, conditional on fitted offset; "
            "no training-estimator uncertainty or family/seed uncertainty included."
        ),
        "runs": {},
    }
    for root in args.archives:
        name = root.parent.name if root.name == "fit" else root.name
        root = root if root.name == "fit" else root / "fit"
        sources = {
            split: root / f"{split}_{args.step}_split_inference.json"
            for split in ["train", "validation"]
        }
        pairs = {split: load(path) for split, path in sources.items()}
        record = {
            "sources": {
                split: {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for split, path in sources.items()
            },
            "tasks": {},
        }
        for task in ["detection", "animal"]:
            selected = {
                split: [p for (t, _), p in ps.items() if t == task] for split, ps in pairs.items()
            }
            train_scores = np.array(
                [
                    [p[c]["positive_minus_negative"] for c in (False, True)]
                    for p in selected["train"]
                ]
            )
            bound = float(np.max(np.abs(train_scores)) + 40)
            offset = brentq(
                lambda b, scores=train_scores: expit(scores + b).mean() - 0.5, -bound, bound
            )
            result = {"offset_fitted_on_training": float(offset), "splits": {}}
            for split, ps in selected.items():
                result["splits"][split] = describe(ps, offset)
                result["splits"][split]["families"] = {
                    family: describe(
                        [p for p in ps if p[False]["question_family"] == family], offset
                    )
                    for family in sorted({p[False]["question_family"] for p in ps})
                }
            record["tasks"][task] = result
        report["runs"][name] = record
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for run, r in report["runs"].items():
        for task, t in r["tasks"].items():
            print(run, task, "offset", round(t["offset_fitted_on_training"], 3))
            for split, d in t["splits"].items():
                print(
                    split,
                    {
                        k: round(d[k], 4) if isinstance(d[k], float) else d[k]
                        for k in [
                            "gap_mean",
                            "midpoint_mean",
                            "midpoint_sd",
                            "native_both_correct_pairs",
                            "native_one_correct_pairs",
                            "native_neither_correct_pairs",
                            "binary_ce",
                            "offset_binary_ce",
                            "binary_accuracy_half_credit_ties",
                            "offset_binary_accuracy",
                            "oracle_per_prompt_centered_binary_ce",
                        ]
                    },
                )


if __name__ == "__main__":
    main()
