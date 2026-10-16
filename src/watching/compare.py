"""Compare paired log-odds separation before and after training."""

import argparse
import json
import math
import random
import statistics
from pathlib import Path


def exact_sign_test(values):
    """Exact binomial sign test; zero differences are ties, not independent evidence.

    The positive alternative is prespecified. The null gives equal probability
    to positive and negative nonzero differences across independent prompts.
    This tests direction, not the mean magnitude; it assumes no Gaussian model.
    """
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("Provide nonempty finite differences")
    positive = sum(value > 0 for value in values)
    negative = sum(value < 0 for value in values)
    n = positive + negative
    denominator = 2**n
    greater = sum(math.comb(n, k) for k in range(positive, n + 1)) / denominator
    smaller_tail = sum(math.comb(n, k) for k in range(min(positive, negative) + 1))
    return {
        "positive": positive,
        "negative": negative,
        "ties": len(values) - n,
        "p_greater": greater,
        "p_two_sided": min(1.0, 2 * smaller_tail / denominator),
    }


def compare_scores(before, after, *, seed=0, draws=5000):
    """Bootstrap the change in separation, keeping each prompt's four scores together."""
    if draws < 100:
        raise ValueError("Use at least 100 bootstrap draws")

    def separations(rows):
        pairs = {}
        for row in rows:
            condition = row["capture"]
            if type(condition) is not bool:
                raise ValueError("capture must be a boolean")
            pair = pairs.setdefault(row["index"], {})
            if condition in pair:
                raise ValueError("Duplicate condition for a prompt")
            score = float(row["yes_minus_no"])
            if not math.isfinite(score):
                raise ValueError("Scores must be finite")
            pair[condition] = score
        if not pairs or any(set(pair) != {False, True} for pair in pairs.values()):
            raise ValueError("Each prompt requires both conditions")
        return {index: pair[True] - pair[False] for index, pair in pairs.items()}

    initial, final = separations(before), separations(after)
    if initial.keys() != final.keys():
        raise ValueError("Before and after must contain the same prompt indices")
    changes = [final[i] - initial[i] for i in sorted(initial)]
    count = len(changes)
    rng = random.Random(seed)
    boot = sorted(sum(rng.choices(changes, k=count)) / count for _ in range(draws))
    return {
        "n_prompts": count,
        "initial_separation": sum(initial.values()) / count,
        "final_separation": sum(final.values()) / count,
        "change_in_separation": sum(changes) / count,
        "median_change": statistics.median(changes),
        "change_sign_test": exact_sign_test(changes),
        "final_sign_test": exact_sign_test(list(final.values())),
        "change_ci95": (
            [boot[int(0.025 * draws)], boot[int(0.975 * draws) - 1]] if count > 1 else None
        ),
        "bootstrap_seed": seed,
        "bootstrap_draws": draws,
    }


def compare_arms(uniform_before, uniform_after, fast_before, fast_after, *, seed=0, draws=5000):
    """Compare learning gains, matching each prompt across arms and checkpoints.

    Bootstrap units remain whole prompts. This conditions on the two trained
    models and does not estimate variability across independent training runs.
    """
    uniform = compare_scores(uniform_before, uniform_after, seed=seed, draws=draws)
    fast = compare_scores(fast_before, fast_after, seed=seed, draws=draws)

    def contrast(uniform_rows, fast_rows):
        uniform_map = {(r["index"], r["capture"]): r["yes_minus_no"] for r in uniform_rows}
        fast_map = {(r["index"], r["capture"]): r["yes_minus_no"] for r in fast_rows}
        if uniform_map.keys() != fast_map.keys():
            raise ValueError("Both arms must contain the same prompt indices and conditions")
        return [
            {"index": index, "capture": capture, "yes_minus_no": fast_map[key] - uniform_map[key]}
            for key in sorted(uniform_map)
            for index, capture in [key]
        ]

    difference = compare_scores(
        contrast(uniform_before, fast_before),
        contrast(uniform_after, fast_after),
        seed=seed,
        draws=draws,
    )
    return {"uniform": uniform, "fast": fast, "fast_minus_uniform": difference}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    before = json.loads(args.before.read_text())
    after = json.loads(args.after.read_text())
    for field in ("split", "grad_mode"):
        if before["summary"][field] != after["summary"][field]:
            parser.error(f"Cannot compare different {field} values")
    print(json.dumps(compare_scores(before["rows"], after["rows"], seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
