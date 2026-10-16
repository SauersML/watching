"""Paired baseline-to-endpoint analysis, with optional fresh-compile checks."""

import argparse
import json
from pathlib import Path

from watching.artifacts import write_json
from watching.compare import compare_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fresh-root", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text())
    steps = manifest["steps"]
    report = {
        "definition": "d = log(P(Yes)/P(No)) with probe minus without probe",
        "intervals": "5000 paired whole-prompt bootstrap resamples; one training seed",
        "interpretation": "Exploratory; intervals omit training-seed and compiler variability",
        "steps": steps,
        "paths": {},
    }
    for mode in ("grad", "inference"):
        report["paths"][mode] = {}
        for split in ("train", "test"):
            records = [
                json.loads((args.run / f"{split}_{step}_{mode}.json").read_text())
                for step in (0, steps)
            ]
            if any(r["summary"]["grad_mode"] != (mode == "grad") for r in records):
                raise ValueError("Execution-path metadata does not match filenames")
            before, after = [record["rows"] for record in records]
            result = compare_scores(before, after)
            zero = [{**row, "yes_minus_no": 0.0} for row in after]
            result["final_ci95"] = compare_scores(zero, after)["change_ci95"]
            if args.fresh_root:
                fresh = args.fresh_root / f"fresh_{mode}" / f"{split}_{steps}_{mode}.json"
                if fresh.exists():
                    rows = json.loads(fresh.read_text())["rows"]
                    result["fresh_recompile_change"] = compare_scores(after, rows)
                    result["baseline_to_fresh"] = compare_scores(before, rows)
            report["paths"][mode][split] = result
    write_json(args.output, report)


if __name__ == "__main__":
    main()
