"""Paired summaries, grouping related prompt variants when their IDs are supplied."""

import random


def resampling_groups(pairs):
    """Keep all variants of a base context together; never infer groups from text."""
    rows = [row for pair in pairs.values() for row in pair.values()]
    if not any("base_context_id" in row for row in rows):
        return None
    groups = {}
    for position, pair in enumerate(pairs.values()):
        ids = [pair[c].get("base_context_id") for c in (False, True)]
        if any(not isinstance(value, str) or not value for value in ids) or ids[0] != ids[1]:
            raise ValueError("Every paired condition needs the same nonempty base_context_id")
        groups.setdefault(ids[0], []).append(position)
    return list(groups.values())


def bootstrap_mean(values, groups, rng, samples):
    if groups is None:
        return sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(samples))
    # A draw retains all variants and both conditions of each selected context.
    # The denominator preserves the prompt-weighted estimand for unequal groups.
    totals = [(sum(values[i] for i in group), len(group)) for group in groups]
    draws = []
    for _ in range(samples):
        selected = rng.choices(totals, k=len(totals))
        draws.append(sum(total for total, _ in selected) / sum(size for _, size in selected))
    return sorted(draws)


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
    groups = resampling_groups(pairs)

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
        summary[name + "_mean_log_odds"] = (
            sum(pair[condition]["yes_minus_no"] for pair in pairs.values()) / count
        )
        summary[name + "_accuracy"] = (
            sum(pair[condition]["correct"] for pair in pairs.values()) / count
        )

    scores = [
        (int(pair[True]["correct"]) + int(pair[False]["correct"])) / 2 for pair in pairs.values()
    ]
    rng = random.Random(seed)
    boot = bootstrap_mean(scores, groups, rng, bootstrap_samples)
    summary["accuracy_ci95"] = [
        boot[int(0.025 * bootstrap_samples)],
        boot[int(0.975 * bootstrap_samples) - 1],
    ]
    shifts = [pair[True]["yes_minus_no"] - pair[False]["yes_minus_no"] for pair in pairs.values()]
    boot_shifts = bootstrap_mean(shifts, groups, rng, bootstrap_samples)
    summary.update(
        paired_log_odds_shift=sum(shifts) / count,
        paired_log_odds_shift_ci95=[
            boot_shifts[int(0.025 * bootstrap_samples)],
            boot_shifts[int(0.975 * bootstrap_samples) - 1],
        ],
        paired_rank_accuracy=sum((shift > 0) + 0.5 * (shift == 0) for shift in shifts) / count,
        positive_shifts=sum(shift > 0 for shift in shifts),
        zero_shifts=sum(shift == 0 for shift in shifts),
        negative_shifts=sum(shift < 0 for shift in shifts),
    )
    native_scores = [
        (int(pair[True]["greedy_correct"]) + int(pair[False]["greedy_correct"])) / 2
        for pair in pairs.values()
    ]
    native_boot = bootstrap_mean(native_scores, groups, rng, bootstrap_samples)
    summary["greedy_accuracy_ci95"] = [
        native_boot[int(0.025 * bootstrap_samples)],
        native_boot[int(0.975 * bootstrap_samples) - 1],
    ]
    native_pairs = [
        int(pair[True]["greedy_correct"] and pair[False]["greedy_correct"])
        for pair in pairs.values()
    ]
    pair_boot = bootstrap_mean(native_pairs, groups, rng, bootstrap_samples)
    summary["greedy_both_conditions_correct"] = sum(native_pairs) / count
    summary["greedy_both_conditions_correct_ci95"] = [
        pair_boot[int(0.025 * bootstrap_samples)],
        pair_boot[int(0.975 * bootstrap_samples) - 1],
    ]
    if groups is not None:
        summary.update(
            resampling_unit="base_context_id",
            n_base_contexts=len(groups),
            variants_per_context=[len(group) for group in groups],
        )
    # A single context remains one sampling unit even with many prompt variants.
    if (len(groups) if groups is not None else count) == 1:
        summary["accuracy_ci95"] = None
        summary["paired_log_odds_shift_ci95"] = None
        summary["greedy_accuracy_ci95"] = None
        summary["greedy_both_conditions_correct_ci95"] = None
    return summary
