"""Compare paired log-odds separation before and after training."""

import argparse
import json
import math
import random
from pathlib import Path


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
        "change_ci95": [boot[int(0.025 * draws)], boot[int(0.975 * draws) - 1]],
        "bootstrap_seed": seed,
        "bootstrap_draws": draws,
    }


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
