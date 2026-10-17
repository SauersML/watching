"""Fixed-endpoint paired summaries; prompt and equal-family estimands stay separate.

Each input is a saved score JSON. Outputs never pool seeds, runtimes or checkpoints.
Intervals are sensitivity analyses with stated sampling assumptions, not guarantees.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import beta

METRICS = ("full_ce", "binary_ce", "correct_probability", "actual_accuracy", "gap", "both_correct")


def clopper_pearson(successes, count):
    if not 0 <= successes <= count or count < 1:
        raise ValueError("Invalid binomial counts")
    return [
        0.0 if successes == 0 else float(beta.ppf(0.025, successes, count - successes + 1)),
        1.0 if successes == count else float(beta.ppf(0.975, successes + 1, count - successes)),
    ]


def bootstrap_means(values, draws, seed):
    """Resample rows jointly, preserving covariance of every paired-prompt metric."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Need a finite nonempty observation matrix")
    if draws < 100:
        raise ValueError("At least 100 bootstrap draws required")
    if len(values) < 2:
        return [None] * values.shape[1]
    rng = np.random.default_rng(seed)
    sampled = np.empty((draws, values.shape[1]))
    for start in range(0, draws, 100):
        count = min(100, draws - start)
        indices = rng.integers(0, len(values), (count, len(values)))
        sampled[start : start + count] = values[indices].mean(axis=1)
    return np.quantile(sampled, [0.025, 0.975], axis=0).T.tolist()


def paired_observations(rows, records=()):
    """Build one observation per prompt, never one independent observation per token."""
    if not rows:
        raise ValueError("Cannot analyze empty predictions")
    pairs = {}
    by_context = defaultdict(list)
    for record in records:
        if record.get("context_id") is not None:
            by_context[record["context_id"]].append(record)
    for row in rows:
        if type(row.get("capture")) is not bool:
            raise ValueError("Conditions must be Boolean")
        pair = pairs.setdefault(row["index"], {})
        if row["capture"] in pair:
            raise ValueError("Duplicate prompt-condition score")
        pair[row["capture"]] = row
    result = []
    seen = set()
    for index, pair in sorted(pairs.items()):
        if set(pair) != {False, True}:
            raise ValueError("Every prompt requires both conditions")
        absent, present = pair[False], pair[True]
        candidates = by_context.get(absent.get("context_id"), [])
        if len(candidates) > 1:
            candidates = [
                r
                for r in candidates
                if r.get("task", "detection") == absent.get("task", "detection")
            ]
        record = (
            candidates[0]
            if len(candidates) == 1
            else (records[index] if 0 <= index < len(records) else {})
        )
        task = absent.get("task", record.get("task", "detection"))
        family = absent.get("question_family") or record.get("question_family")
        context = absent.get("context_id", record.get("context_id", str(index)))
        if record and (
            record.get("context_id", context) != context
            or record.get("task", task) != task
            or record.get("question_family", family) != family
        ):
            raise ValueError("Score metadata disagrees with the saved dataset")
        if family is None:
            raise ValueError("Question-family metadata required for cluster sensitivity analysis")
        if present.get("task", task) != task or present.get("context_id", context) != context:
            raise ValueError("Conditions disagree about task or context identity")
        if present.get("question_family", family) not in (family, None):
            raise ValueError("Conditions disagree about question family")
        if (task, context) in seen:
            raise ValueError("Repeated context within a task needs a higher-level cluster")
        seen.add((task, context))
        ces, log_odds, correct = [], [], []
        for condition in (False, True):
            row = pair[condition]
            positive = row.get("logp_positive", row.get("logp_yes"))
            negative = row.get("logp_negative", row.get("logp_no"))
            if positive is None or negative is None:
                raise ValueError("Both native answer log probabilities are required")
            if (
                not np.isfinite([positive, negative]).all()
                or max(positive, negative) > 1e-6
                or np.logaddexp(positive, negative) > 1e-6
            ):
                raise ValueError("Answer probabilities are not a valid vocabulary subset")
            ce, gap = -float(positive if condition else negative), float(positive - negative)
            if not np.isfinite([ce, gap]).all() or ce < -1e-6:
                raise ValueError("Invalid full-vocabulary log probabilities")
            if not np.isclose(ce, row["ce"], rtol=0, atol=1e-5):
                raise ValueError("Saved CE disagrees with the correct-token log probability")
            if not np.isclose(gap, row["yes_minus_no"], rtol=0, atol=1e-5):
                raise ValueError("Saved log-odds disagrees with native probabilities")
            actual = bool(row["greedy_correct"])
            if ce < np.log(2) - 1e-6 and not actual:
                raise ValueError("Correct token has majority probability but is marked incorrect")
            if "top_token_id" in row and "target_token_id" in row:
                if actual != (row["top_token_id"] == row["target_token_id"]):
                    raise ValueError("Actual-token correctness disagrees with token IDs")
            ces.append(ce)
            log_odds.append(gap)
            correct.append(actual)
        result.append(
            {
                "index": index,
                "context_id": context,
                "task": task,
                "family": family,
                "values": [
                    float(np.mean(ces)),
                    float((np.logaddexp(0, log_odds[0]) + np.logaddexp(0, -log_odds[1])) / 2),
                    float(np.exp(-np.asarray(ces)).mean()),
                    float(np.mean(correct)),
                    log_odds[1] - log_odds[0],
                    float(all(correct)),
                ],
                "absent_positive_probability": float(expit(log_odds[0])),
                "present_positive_probability": float(expit(log_odds[1])),
            }
        )
    return result


def summarize_observations(observations, draws=5000, seed=20261018):
    groups = defaultdict(list)
    for observation in observations:
        groups[observation["task"]].append(observation)
    tasks = {}
    for task, rows in sorted(groups.items()):
        values = np.asarray([row["values"] for row in rows])
        families = defaultdict(list)
        for row in rows:
            families[row["family"]].append(row["values"])
        family_means = np.asarray([np.mean(families[name], axis=0) for name in sorted(families)])
        intervals = bootstrap_means(values, draws, seed)
        cluster_intervals = bootstrap_means(family_means, draws, seed)
        successes = int(values[:, METRICS.index("both_correct")].sum())
        tasks[task] = {
            "prompt_weighted": {
                "n_prompts": len(rows),
                "estimand": "Mean over this task's observed prompt distribution",
                "metrics": {
                    name: {
                        "mean": float(values[:, j].mean()),
                        "median": float(np.median(values[:, j])),
                        "ci95": intervals[j],
                    }
                    for j, name in enumerate(METRICS)
                },
            },
            "equal_family_weighted": {
                "n_families": len(families),
                "estimand": (
                    "Equal-weight mean of observed family means; "
                    "different from prompt-weighted mean"
                ),
                "metrics": {
                    name: {
                        "mean": float(family_means[:, j].mean()),
                        "ci95": cluster_intervals[j],
                        "minimum_family_mean": float(family_means[:, j].min()),
                        "maximum_family_mean": float(family_means[:, j].max()),
                    }
                    for j, name in enumerate(METRICS)
                },
            },
            "per_family": [
                {
                    "family": name,
                    "n_prompts": len(families[name]),
                    "metrics": dict(zip(METRICS, family_means[i].tolist(), strict=True)),
                }
                for i, name in enumerate(sorted(families))
            ],
            "prompt_both_correct_iid_clopper_pearson": {
                "successes": successes,
                "n": len(rows),
                "ci95": clopper_pearson(successes, len(rows)),
                "assumption": (
                    "Independent identically distributed prompt-level Bernoulli trials. "
                    "Not guaranteed with shared question families; "
                    "not a distribution-free population claim."
                ),
            },
            "positive_answer_probability_by_condition": {
                condition: float(
                    np.mean([row[condition + "_positive_probability"] for row in rows])
                )
                for condition in ("absent", "present")
            },
        }
    return tasks


def control_interpretation(payload):
    """A perturbing buffer is an alternative intervention, not a no-footprint null."""
    valid = payload.get("valid_no_footprint_control")
    if valid is None:
        return "No buffer-parity claim in this score file"
    if type(valid) is not bool:
        raise ValueError("Buffer control validity must be Boolean")
    if not valid:
        return (
            "Invalid no-footprint control: buffer capture changes the computation. "
            "Its numerical scores describe an alternative intervention, not null detection."
        )
    checks = payload.get("full_vocabulary_checks", [])
    indices = {row["index"] for row in payload["rows"]}
    if (
        len(checks) != len(indices)
        or {row["index"] for row in checks} != indices
        or not all(row.get("bit_identical") is True for row in checks)
    ):
        raise ValueError("Valid buffer control lacks complete successful full-vocabulary checks")
    return "Recorded full-vocabulary parity passed for every prompt; evaluation-only control"


def analyze_file(path, draws=5000, seed=20261018):
    payload = json.loads(path.read_text())
    directory = path.parent
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    summary = payload.get("summary", {})
    if "steps" in manifest:
        if (
            manifest.get("completed_steps") != manifest["steps"]
            or summary.get("step") != manifest["steps"]
            or manifest.get("training_status", "complete") != "complete"
        ):
            raise ValueError("Only the predeclared completed endpoint is accepted")
    data_path = directory / "data.json"
    records = []
    if data_path.exists():
        data = json.loads(data_path.read_text())
        split = summary.get("split")
        if split is not None and split in data:
            records = data[split]
        elif len(data) == 1:
            records = next(iter(data.values()))
        else:
            records = [
                record for rows in data.values() if isinstance(rows, list) for record in rows
            ]
    return {
        "source": str(path),
        "seed": manifest.get("seed"),
        "seed_role": "training" if "steps" in manifest else "evaluation",
        "runtime": summary.get("path", manifest.get("runtime")),
        "split": summary.get("split", manifest.get("split")),
        "completed_steps": manifest.get("completed_steps"),
        "null_control": manifest.get("null_control"),
        "valid_no_footprint_control": payload.get("valid_no_footprint_control"),
        "buffer_control_interpretation": control_interpretation(payload),
        "tasks": summarize_observations(paired_observations(payload["rows"], records), draws, seed),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261018)
    args = parser.parse_args()
    report = {
        "bootstrap_draws": args.draws,
        "bootstrap_seed": args.seed,
        "contract": {
            "endpoint": "Predeclared final step only; no best-checkpoint selection",
            "tasks": "Separate task estimates; no pooled Yes/No and animal performance claim",
            "seeds": "Each run stays separate; prompts are not independent training seeds",
            "prompt_ci": "Pointwise percentile bootstrap of whole prompt pairs, conditional on the "
            "observed family mixture and run; assumes prompt-level sampling independence",
            "family_ci": (
                "Resample whole family means with equal family weight. Sensitivity to family "
                "variation; few observed families and nonrandom construction limit inference."
            ),
            "bounded_accuracy": (
                "Prompt-both-correct Clopper-Pearson interval is supplied only under "
                "explicit iid Bernoulli assumptions; empirical 100% bootstrap is not certainty"
            ),
            "calibration": (
                "Full-vocabulary log loss and correct-token probability accompany accuracy; "
                "no post-hoc fit or held-out threshold tuning"
            ),
            "multiplicity": (
                "Descriptive pointwise intervals, not simultaneous claims or a corrected "
                "hypothesis-testing family. No sign/permutation p-values without "
                "exchangeability assumptions."
            ),
        },
        "analyses": [analyze_file(path, args.draws, args.seed) for path in args.scores],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
