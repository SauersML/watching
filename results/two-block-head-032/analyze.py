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
        "gains_from_resumed_baseline": {},
    }
    for split in ("train", "validation", "test"):
        for runtime in ("split_inference", "wholegraph_transfer"):
            path = FIT / f"{split}_1024_{runtime}.json"
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
            final = prompt_values(FIT / f"{split}_1024_split_inference.json")
            if baseline.keys() != final.keys():
                raise ValueError("Before/after prompt identities differ")
            changes = {i: {k: final[i][k] - baseline[i][k] for k in final[i]} for i in final}
            report["gains_from_resumed_baseline"][split] = intervals(changes)
    checkpoints = [read(p)["summary"] for p in FIT.glob("train_*_split_inference.json")]
    best = max(checkpoints, key=lambda s: s["paired_log_odds_shift"])
    report["descriptive_maximum_training_gap_not_selected"] = best
    report["endpoint_selection"] = (
        "Fixed 1024 additional updates; maximum is descriptive, never selected"
    )
    events = [json.loads(line) for line in (FIT / "metrics.jsonl").read_text().splitlines()]
    updates = [event for event in events if event["event"] == "train"]
    audits = [event for event in events if event["event"] == "gradient_forward_audit"]
    report["integration"] = {
        "gradient_forward_checks": len(audits),
        "all_gradient_forward_checks_identical": all(event["bit_identical"] for event in audits),
        "clipped_updates": sum(event["gradient_norm"] > 1 for event in updates),
        "updates": len(updates),
        "memory_and_completion": [
            event
            for event in events
            if event["event"]
            in (
                "sources_offloaded",
                "training_start",
                "sources_restored",
                "restored_source_verified",
                "complete",
            )
        ],
    }
    fixed = ROOT.parent / "two-block-030" / "fit"
    comparison = {"interpretation": "Descriptive comparison only: frozen-source caches differ."}
    comparison["identical_data_records"] = read(FIT / "data.json") == read(fixed / "data.json")
    ours, theirs = read(FIT / "cache_fingerprints.json"), read(fixed / "cache_fingerprints.json")
    comparison["cache_mismatches"] = {
        split: {
            condition: {
                field: sum(
                    a[condition][field] != b[condition][field]
                    for a, b in zip(ours[split], theirs[split], strict=True)
                )
                for field in ("hidden", "mask", "cos", "sin")
            }
            for condition in ("absent", "present")
        }
        for split in ("train", "validation", "test")
    }
    comparison["matched_update_summaries"] = {
        split: {
            arm: read(directory / f"{split}_1024_split_inference.json")["summary"]
            for arm, directory in (("full_head", FIT), ("fixed_head", fixed))
        }
        for split in ("train", "validation")
    }
    report["comparison_to_fixed_head_030"] = comparison
    (ROOT / "endpoints.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
