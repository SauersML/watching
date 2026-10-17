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
            report["gains_from_initial_baseline"][split] = intervals(changes)
    checkpoints = [read(p)["summary"] for p in FIT.glob("train_*_split_inference.json")]
    best = max(checkpoints, key=lambda s: s["paired_log_odds_shift"])
    report["descriptive_maximum_training_gap_not_selected"] = best
    report["endpoint_selection"] = (
        "Fixed 1024 updates from base; maximum is descriptive, never selected"
    )
    events = [json.loads(line) for line in (FIT / "metrics.jsonl").read_text().splitlines()]
    updates = [row for row in events if row["event"] == "train"]
    audits = [row for row in events if row["event"] == "gradient_forward_audit"]
    if len(updates) != 1024 or any(
        len(r["prompt_indices"]) != 4 or len(set(r["prompt_indices"])) != 4 for r in updates
    ):
        raise ValueError("Unexpected update count or duplicated within-update prompt")
    visits = [i for row in updates for i in row["prompt_indices"]]
    if (
        len(visits) != 4096
        or set(visits) != set(range(64))
        or any(visits.count(i) != 64 for i in range(64))
    ):
        raise ValueError("Unexpected prompt visitation pattern")
    data = read(FIT / "data.json")
    contexts = [
        {row["context_id"] for row in data[split]} for split in ("train", "validation", "test")
    ]
    if any(a & b for i, a in enumerate(contexts) for b in contexts[i + 1 :]):
        raise ValueError("Context overlap between cohorts")
    train_families = {row["question_family"] for row in data["train"]}
    test_families = {row["question_family"] for row in data["test"]}
    if train_families & test_families:
        raise ValueError("Test question family occurs in training")
    report["integration"] = {
        "gradient_forward_checks": len(audits),
        "all_gradient_forward_checks_identical": all(row["bit_identical"] for row in audits),
        "clipped_updates": sum(row["gradient_norm"] > 1 for row in updates),
        "updates": len(updates),
        "processed_prompt_pairs": len(visits),
        "visits_per_training_prompt": 64,
        "train_question_families": sorted(train_families),
        "test_question_families": sorted(test_families),
        "completion": events[-1],
    }
    (ROOT / "endpoints.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
