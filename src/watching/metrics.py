"""Summaries and uncertainty estimates with the prompt as the sampling unit."""

import random


def summarize_pairs(rows: list[dict], *, seed: int, bootstrap_samples: int = 1000) -> dict:
    if not rows:
        raise ValueError("Cannot summarize empty predictions")
    if bootstrap_samples < 100:
        raise ValueError("Use at least 100 bootstrap samples")
    pairs = {}
    for row in rows:
        pair = pairs.setdefault(row["index"], {})
        condition = row["capture"]
        if type(condition) is not bool:
            raise ValueError("capture must be a boolean")
        if condition in pair:
            raise ValueError(f"Duplicate prediction for prompt {row['index']}, capture={condition}")
        pair[condition] = row
    if any(set(pair) != {False, True} for pair in pairs.values()):
        raise ValueError("Every prompt needs one prediction in each condition")

    count = len(pairs)
    summary = {
        "n_prompts": count,
        "accuracy": sum(row["correct"] for row in rows) / len(rows),
        "greedy_accuracy": sum(row["greedy_correct"] for row in rows) / len(rows),
        "mean_ce": sum(row["ce"] for row in rows) / len(rows),
        "paired_yes_probability_gap": sum(
            pair[True]["yes_probability"] - pair[False]["yes_probability"]
            for pair in pairs.values()
        )
        / count,
        "both_conditions_correct": sum(
            pair[True]["correct"] and pair[False]["correct"] for pair in pairs.values()
        )
        / count,
    }
    for condition, name in [(False, "absent"), (True, "present")]:
        summary[name + "_accuracy"] = (
            sum(pair[condition]["correct"] for pair in pairs.values()) / count
        )

    scores = [
        (int(pair[True]["correct"]) + int(pair[False]["correct"])) / 2 for pair in pairs.values()
    ]
    rng = random.Random(seed)
    boot = sorted(sum(rng.choices(scores, k=count)) / count for _ in range(bootstrap_samples))
    summary["accuracy_ci95"] = [
        boot[int(0.025 * bootstrap_samples)],
        boot[int(0.975 * bootstrap_samples) - 1],
    ]
    return summary
