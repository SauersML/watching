"""Recompute endpoint and within-run gains from archived paired prompt scores."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
FIT = ROOT / "fit"


def read(path):
    return json.loads(path.read_text())


def prompt_values(path):
    saved = read(path)
    pairs = {}
    for row in saved["rows"]:
        pair = pairs.setdefault(row["index"], {})
        condition = row["condition_present"]
        if condition in pair:
            raise ValueError("Duplicate condition")
        pair[condition] = row
    if any(set(pair) != {False, True} for pair in pairs.values()):
        raise ValueError("Incomplete prompt pair")
    return {
        index: {
            "gap": pair[True]["yes_minus_no"] - pair[False]["yes_minus_no"],
            "midpoint": (pair[True]["yes_minus_no"] + pair[False]["yes_minus_no"]) / 2,
            "greedy_accuracy": sum(float(r["greedy_correct"]) for r in pair.values()) / 2,
            "correct_probability": sum(r["full_vocab_correct_probability"] for r in pair.values())
            / 2,
            "full_ce": sum(r["ce"] for r in pair.values()) / 2,
            "binary_ce": sum(r["binary_ce"] for r in pair.values()) / 2,
        }
        for index, pair in sorted(pairs.items())
    }


def intervals(values):
    keys = list(next(iter(values.values())))
    array = np.array([[row[key] for key in keys] for row in values.values()])
    indices = np.random.default_rng(20261016).integers(0, len(array), (5000, len(array)))
    sampled = array[indices].mean(axis=1)
    return {
        key: {
            "mean": float(array[:, i].mean()),
            "ci95": np.quantile(sampled[:, i], [0.025, 0.975]).tolist(),
        }
        for i, key in enumerate(keys)
    }


def main():
    report = {
        "resampling": "5000 whole-prompt paired bootstrap draws; seed20261016; pointwise, "
        "conditional on these prompts and this run",
        "endpoints": {},
        "gains_from_initial_baseline": {},
        "wholegraph_minus_split": {},
    }
    manifest = read(FIT / "manifest.json")
    step = manifest["completed_steps"]
    if step < 1:
        raise ValueError("No completed training updates")
    report["completed_steps"] = step
    report["processed_prompt_pairs"] = manifest["processed_prompt_pairs"]
    for split in ("train", "validation", "test"):
        for runtime in ("split_inference", "wholegraph_transfer"):
            path = FIT / f"{split}_{step}_{runtime}.json"
            values = prompt_values(path)
            report["endpoints"][f"{split}/{runtime}"] = {
                "n_prompts": len(values),
                "metrics": intervals(values),
                "midpoint_rms": float(
                    np.sqrt(np.mean([r["midpoint"] ** 2 for r in values.values()]))
                ),
                "source": str(path.relative_to(ROOT)),
            }
        if split != "test":
            baseline = prompt_values(FIT / f"{split}_0_split_inference.json")
            final = prompt_values(FIT / f"{split}_{step}_split_inference.json")
            if baseline.keys() != final.keys():
                raise ValueError("Before/after prompt identities differ")
            changes = {i: {k: final[i][k] - baseline[i][k] for k in final[i]} for i in final}
            report["gains_from_initial_baseline"][split] = intervals(changes)
        primary = prompt_values(FIT / f"{split}_{step}_split_inference.json")
        transfer = prompt_values(FIT / f"{split}_{step}_wholegraph_transfer.json")
        if primary.keys() != transfer.keys():
            raise ValueError("Runtime comparison identities differ")
        changes = {i: {k: transfer[i][k] - primary[i][k] for k in primary[i]} for i in primary}
        report["wholegraph_minus_split"][split] = intervals(changes)
    checkpoints = [read(p)["summary"] for p in FIT.glob("train_*_split_inference.json")]
    best = max(checkpoints, key=lambda s: s["paired_log_odds_shift"])
    report["descriptive_maximum_training_gap_not_selected"] = best
    report["endpoint_selection"] = (
        f"Scheduled 1024 updates; actual endpoint {step}. Maximum is descriptive, never selected"
    )
    (ROOT / "endpoints.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
