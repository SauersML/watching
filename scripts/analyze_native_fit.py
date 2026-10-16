"""Analyze a native answer-row fit, preserving prompt pairs across checkpoints."""

import argparse
import json
import math
import random
from pathlib import Path

from watching.artifacts import write_json
from watching.compare import compare_scores


def read_rows(path, *, split):
    record = json.loads(path.read_text())
    if record["summary"]["split"] != split:
        raise ValueError(f"Split metadata disagrees with filename: {path}")
    rows = []
    for row in record["rows"]:
        condition = row.get("condition_present", row.get("capture"))
        if type(condition) is not bool:
            raise ValueError(f"Missing boolean execution condition: {path}")
        rows.append({**row, "capture": condition})
    return rows


def binary_loss(rows, *, seed=20261016, draws=5000):
    """Stable binary log loss, bootstrapped as one mean loss per whole prompt."""
    pairs = {}
    for row in rows:
        score = float(row["yes_minus_no"])
        if not math.isfinite(score):
            raise ValueError("Scores must be finite")
        signed = -score if row["capture"] else score
        loss = max(signed, 0) + math.log1p(math.exp(-abs(signed)))
        pair = pairs.setdefault(row["index"], {})
        if row["capture"] in pair:
            raise ValueError("Duplicate condition within prompt")
        pair[row["capture"]] = loss
    if not pairs or any(set(p) != {False, True} for p in pairs.values()):
        raise ValueError("Every prompt needs both conditions")
    means = [(p[False] + p[True]) / 2 for p in pairs.values()]
    rng = random.Random(seed)
    boot = sorted(sum(rng.choices(means, k=len(means))) / len(means) for _ in range(draws))
    return {
        "mean": sum(means) / len(means),
        "ci95": [boot[int(0.025 * draws)], boot[int(0.975 * draws) - 1]]
        if len(means) > 1
        else None,
    }


def summarize(before, after):
    result = compare_scores(before, after, seed=20261016, draws=5000)
    zeros = [{**row, "yes_minus_no": 0.0} for row in after]
    result["endpoint_ci95"] = compare_scores(zeros, after, seed=20261016)["change_ci95"]
    result["binary_ce"] = binary_loss(after)
    return result


def analyze_fit(directory, *, fresh=None, phase="many"):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if fresh is not None:
        fresh = Path(fresh)
        original_data, fresh_data = directory / "data.json", fresh / "data.json"
        if not original_data.exists() or not fresh_data.exists():
            raise ValueError("Fresh comparison requires both saved prompt data files")
        if json.loads(original_data.read_text()) != json.loads(fresh_data.read_text()):
            raise ValueError("Fresh evaluation uses different prompt records")
        fresh_manifest = json.loads((fresh / "manifest.json").read_text())
        for field in ("condition", "layer", "length"):
            if manifest.get(field) != fresh_manifest.get(field):
                raise ValueError(f"Fresh evaluation changes {field}")
    report = {
        "source": str(directory),
        "phase": phase,
        "manifest": manifest,
        "condition": manifest.get("condition", "execution"),
        "method": "Native Yes/No output rows; all internal weights fixed",
        "definition": "D = Yes/No log odds in present condition minus absent condition",
        "intervals": "5000 whole-prompt paired bootstrap resamples; pointwise 95% intervals",
        "limits": "Conditional on this fit and prompts; excludes seed and hardware uncertainty",
        "splits": {},
    }
    for split in ("train", "validation", "test"):
        baseline_path = directory / f"{phase}_{split}_0.json"
        if not baseline_path.exists():
            continue
        before = read_rows(baseline_path, split=split)
        checkpoints = []
        paths = sorted(
            directory.glob(f"{phase}_{split}_*.json"), key=lambda p: int(p.stem.rsplit("_", 1)[1])
        )
        for path in paths:
            step = int(path.stem.rsplit("_", 1)[1])
            result = summarize(before, read_rows(path, split=split))
            checkpoints.append({"step": step, "source": str(path), **result})
        entry = {"n_prompts": checkpoints[0]["n_prompts"], "checkpoints": checkpoints}
        if fresh is not None and phase == "many":
            matches = sorted(
                fresh.glob(f"fresh_{split}_*.json"), key=lambda p: int(p.stem.rsplit("_", 1)[1])
            )
            if matches:
                path = matches[-1]
                rows = read_rows(path, split=split)
                final_rows = read_rows(Path(checkpoints[-1]["source"]), split=split)
                entry["fresh"] = {
                    "step": int(path.stem.rsplit("_", 1)[1]),
                    "source": str(path),
                    **summarize(before, rows),
                    "change_from_last_fit": compare_scores(final_rows, rows, seed=20261016),
                    "all_log_odds_identical": {
                        (r["index"], r["capture"]): r["yes_minus_no"] for r in final_rows
                    }
                    == {(r["index"], r["capture"]): r["yes_minus_no"] for r in rows},
                }
        report["splits"][split] = entry
    if not report["splits"]:
        raise ValueError(f"No completed {phase} baseline evaluations in {directory}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fresh", type=Path)
    parser.add_argument("--phase", choices=["single", "many"], default="many")
    args = parser.parse_args()
    report = analyze_fit(args.fit, fresh=args.fresh, phase=args.phase)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    for split, result in report["splits"].items():
        last = result.get("fresh", result["checkpoints"][-1])
        print(
            json.dumps(
                {
                    "split": split,
                    "step": last["step"],
                    "endpoint": last["final_separation"],
                    "endpoint_ci95": last["endpoint_ci95"],
                    "gain": last["change_in_separation"],
                    "gain_ci95": last["change_ci95"],
                    "binary_ce": last["binary_ce"],
                },
                allow_nan=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
