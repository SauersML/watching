"""Compare matched validation calibration across two archived native-row fits.

Only validation score files are read. Selected candidates are taken from the
saved manifest, never selected again by this analysis.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read_json(path):
    return json.loads(path.read_text())


def describe(path, *, n_prompts, draws, seed):
    raw = read_json(path)
    if raw["summary"]["split"] != "validation":
        raise ValueError(f"Only validation scores are allowed: {path}")
    pairs = {}
    for row in raw["rows"]:
        index, condition = row["index"], row["condition_present"]
        if type(index) is not int or not 0 <= index < n_prompts:
            raise ValueError(f"Invalid prompt index: {path}")
        if type(condition) is not bool or type(row["greedy_correct"]) is not bool:
            raise ValueError(f"Conditions and correctness must be booleans: {path}")
        pair = pairs.setdefault(index, {})
        if condition in pair:
            raise ValueError(f"Duplicate prompt condition: {path}")
        pair[condition] = row
    if set(pairs) != set(range(n_prompts)) or any(set(p) != {False, True} for p in pairs.values()):
        raise ValueError(f"Expected both conditions for all saved validation prompts: {path}")
    rows = [pairs[index] for index in range(n_prompts)]
    scores = np.array([[pair[c]["yes_minus_no"] for c in (False, True)] for pair in rows])
    losses = np.array([[pair[c]["ce"] for c in (False, True)] for pair in rows])
    if not np.isfinite(scores).all() or not np.isfinite(losses).all() or np.any(losses < 0):
        raise ValueError(f"Scores must be finite and losses nonnegative: {path}")
    gap = scores[:, 1] - scores[:, 0]
    midpoint = scores.mean(axis=1)
    margins = scores * np.array([-1, 1])
    binary_ce = np.logaddexp(0, -margins).mean(axis=1)
    full_ce = losses.mean(axis=1)
    accuracy = np.array([[pair[c]["greedy_correct"] for c in (False, True)] for pair in rows]).mean(
        axis=1
    )
    oracle_ce = np.logaddexp(0, -gap / 2)
    metrics = {
        "gap": gap,
        "full_ce": full_ce,
        "binary_ce": binary_ce,
        "abs_midpoint": abs(midpoint),
        "full_accuracy": accuracy,
    }
    result = {key: float(value.mean()) for key, value in metrics.items()}
    for key, reference in (
        ("full_ce", "mean_ce"),
        ("binary_ce", "mean_binary_ce"),
        ("full_accuracy", "greedy_accuracy"),
    ):
        if not np.isclose(result[key], raw["summary"][reference], rtol=1e-6, atol=1e-6):
            raise ValueError(f"Per-example {key} disagrees with saved summary: {path}")
    result.update(
        n_prompts=n_prompts,
        source=str(path),
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        execution=raw["summary"].get("execution", "native_forward"),
        midpoint_mean=float(midpoint.mean()),
        midpoint_sd=float(midpoint.std()),
        midpoint_abs_median=float(np.median(abs(midpoint))),
        midpoint_abs_p90=float(np.quantile(abs(midpoint), 0.9)),
        positive_gap_fraction=float(np.mean(gap > 0)),
        both_binary_margins_strictly_positive=float(np.mean(np.all(margins > 0, axis=1))),
        minimum_margin_p10=float(np.quantile(margins.min(axis=1), 0.1)),
        wrong_confident_execution_fraction_margin_below_minus3=float(np.mean(margins < -3)),
        binary_loss_excess_over_ln2=float(binary_ce.mean() - np.log(2)),
        full_loss_excess_over_ln2=float(full_ce.mean() - np.log(2)),
        pair_mass_loss=float(np.mean(full_ce - binary_ce)),
        oracle_pair_centered_binary_ce=float(oracle_ce.mean()),
        midpoint_binary_loss_penalty=float(np.mean(binary_ce - oracle_ce)),
        full_ce_p90_prompt_mean=float(np.quantile(full_ce, 0.9)),
        worst_ten_percent_share_of_full_loss=(
            float(np.sort(full_ce)[-int(np.ceil(n_prompts * 0.1)) :].sum() / full_ce.sum())
            if full_ce.sum()
            else None
        ),
    )
    indices = np.random.default_rng(seed).integers(0, n_prompts, (draws, n_prompts))
    result["prompt_bootstrap_ci95"] = {
        key: np.quantile(value[indices].mean(axis=1), [0.025, 0.975]).tolist()
        for key, value in metrics.items()
    }
    return result, metrics


def compare(before, after, *, draws, seed):
    n = len(before["gap"])
    indices = np.random.default_rng(seed).integers(0, n, (draws, n))
    result = {}
    for key in before:
        delta = after[key] - before[key]
        result[key] = {
            "mean_change": float(delta.mean()),
            "ci95": np.quantile(delta[indices].mean(axis=1), [0.025, 0.975]).tolist(),
        }
    return result


def analyze(old_archive, new_archive, *, draws=5000, seed=20261016):
    report = {
        "scope": "Validation only. Winning-candidate CIs are descriptive, not selection-adjusted. "
        "No test scores read and no new parameter selection performed.",
        "bootstrap": {"draws": draws, "seed": seed, "unit": "whole paired prompt"},
        "definitions": {
            "gap": "d = s_present - s_absent",
            "midpoint": "c = (s_present + s_absent) / 2",
            "score": "s = logP(positive answer) - logP(negative answer)",
            "both_correct": "Strict binary correctness requires d > 2|c|; native answers "
            "must also outrank all other vocabulary tokens.",
            "oracle_centered": "Per-prompt centering requires both conditions; this "
            "counterfactual is not an implementable single-execution detector.",
        },
        "conditions": {},
    }
    for condition in ("probe", "execution"):
        old = old_archive / condition / "detector"
        new = new_archive / condition / "detector"
        old_data, new_data = read_json(old / "data.json"), read_json(new / "data.json")
        if old_data["validation"] != new_data["validation"]:
            raise ValueError(f"Validation prompt records differ: {condition}")
        old_manifest, manifest = read_json(old / "manifest.json"), read_json(new / "manifest.json")
        for field in ("answer_order", "condition", "length", "layer"):
            if old_manifest.get(field) != manifest.get(field):
                raise ValueError(f"Comparison changes {field}: {condition}")
        count = len(new_data["validation"])
        if count < 2:
            raise ValueError("At least two validation prompts are required")
        step, old_step = manifest["steps"], old_manifest["steps"]
        paths = {"old_fit": old / f"many_validation_{old_step}.json"}
        for label, directory, final_step in (
            ("old_fresh", old_archive / condition / "fresh_detector", old_step),
            ("new_fresh", new_archive / condition / "fresh_detector", step),
        ):
            path = directory / f"fresh_validation_{final_step}.json"
            if path.exists():
                if read_json(directory / "data.json")["validation"] != new_data["validation"]:
                    raise ValueError(f"Fresh validation prompt records differ: {directory}")
                paths[label] = path
        for index, ridge in enumerate(manifest["ridge_grid"]):
            path = new / f"many_ridge_{index}_validation_{step}.json"
            if path.exists():
                paths[f"new_ridge_{ridge:g}"] = path
        native = new / f"many_validation_{step}.json"
        if native.exists():
            paths["new_native"] = native
        selected = manifest["selected_ridge"]
        selected_label = f"new_ridge_{selected['ridge']:g}"
        if selected_label not in paths:
            raise ValueError(f"Selected candidate scores are missing: {new}")
        entry = {
            "validation_records_exactly_match": True,
            "n_prompts": count,
            "selected_ridge": selected,
            "observations": {},
            "comparisons": {},
        }
        arrays = {}
        for label, path in paths.items():
            entry["observations"][label], arrays[label] = describe(
                path, n_prompts=count, draws=draws, seed=seed
            )
        for label in ("old_fit", "old_fresh"):
            if label in arrays:
                entry["comparisons"][f"{selected_label}_minus_{label}"] = compare(
                    arrays[label], arrays[selected_label], draws=draws, seed=seed
                )
        for label in ("new_native", "new_fresh"):
            if label in arrays:
                entry["comparisons"][f"{label}_minus_{selected_label}"] = compare(
                    arrays[selected_label], arrays[label], draws=draws, seed=seed
                )
        report["conditions"][condition] = entry
    return report


def render(report):
    lines = [
        "# Matched validation calibration",
        "",
        report["scope"],
        "",
        "All comparisons use identical saved validation prompt records. Cached candidates "
        "score captured native features; native and fresh evaluations appear separately "
        "when available. Unavailable native evaluations are omitted, not treated as failures.",
        "",
        "| Condition / evaluation | Gap | Mean absolute midpoint | Full accuracy | Full CE | "
        "Binary CE | Oracle centered binary CE | Confidently wrong fraction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, entry in report["conditions"].items():
        for name, value in entry["observations"].items():
            lines.append(
                f"| {condition} / {name} | {value['gap']:.3f} | {value['abs_midpoint']:.3f} "
                f"| {value['full_accuracy']:.2%} | {value['full_ce']:.3f} "
                f"| {value['binary_ce']:.3f} | {value['oracle_pair_centered_binary_ce']:.3f} "
                f"| {value['wrong_confident_execution_fraction_margin_below_minus3']:.2%} |"
            )
    lines.extend(
        [
            "",
            "For each prompt, absent/present scores are c−d/2 and c+d/2. Correct signed "
            "margins are d/2−c and d/2+c. Positive d ranks the pair correctly; both binary "
            "answers require d>2|c|. Full-vocabulary answers must also beat every other token.",
            "",
            "Binary CE = mean[softplus(c−d/2)+softplus(−c−d/2)]/2. At fixed d it is minimized "
            "by c=0. The oracle column reports that per-prompt centered lower bound, which "
            "requires both executions and is **not deployable performance**. Full CE adds "
            "the penalty for probability outside the answer pair.",
            "",
            "Balanced answer-only reference loss is ln2 = 0.693147, corresponding to probability "
            "0.5 on each answer. Accuracy above 50% need not beat this loss: confidently wrong "
            "examples can dominate it. The wrong-confidence column counts execution margins "
            "below −3 nats; the JSON also reports loss tails and midpoint distributions.",
            "",
            "## Paired changes",
            "",
        ]
    )
    for condition, entry in report["conditions"].items():
        lines.append(f"Saved selected ridge for {condition}: {entry['selected_ridge']['ridge']:g}.")
        lines.append("")
        for label, metrics in entry["comparisons"].items():
            lines.extend([f"**{condition}: {label}**", ""])
            for key in ("full_ce", "binary_ce", "abs_midpoint", "gap", "full_accuracy"):
                value = metrics[key]
                low, high = value["ci95"]
                lines.append(
                    f"- {key}: change {value['mean_change']:+.4f}; descriptive "
                    f"95% CI [{low:+.4f}, {high:+.4f}]."
                )
            lines.append("")
    lines.extend(
        [
            "Intervals resample whole prompts, preserving both conditions and both evaluations. "
            "They assume exchangeable prompt units, exclude training and compilation variability, "
            "and are not adjusted for selecting a ridge on these same validation data. Training "
            "size and fitting implementation also change between runs; these comparisons do "
            "not isolate the effect of more data alone.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-archive", type=Path, required=True)
    parser.add_argument("--new-archive", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="Output stem for JSON and Markdown"
    )
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()
    if args.draws < 100:
        parser.error("Use at least 100 bootstrap draws")
    report = analyze(args.old_archive, args.new_archive, draws=args.draws, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    args.output.with_suffix(".md").write_text(render(report))
    print(f"Wrote {args.output.with_suffix('.json')} and {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
