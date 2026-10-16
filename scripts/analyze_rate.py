"""Report matched endpoint changes for the two learning-rate arms."""

import argparse
import json
from pathlib import Path

from watching.artifacts import write_json
from watching.compare import compare_arms, compare_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifests = {
        arm: json.loads((args.run / arm / "manifest.json").read_text())
        for arm in ("uniform", "fast")
    }
    for field in ("checkpoint", "seed", "steps", "dataset", "train_prompts", "layer", "length"):
        if manifests["uniform"][field] != manifests["fast"][field]:
            raise ValueError(f"Mismatched arms: {field}")
    data = [json.loads((args.run / arm / "data.json").read_text()) for arm in ("uniform", "fast")]
    if data[0] != data[1]:
        raise ValueError("Prompt records must match exactly across arms")
    steps = manifests["fast"]["steps"]
    report = {
        "definition": "d = log(P(Yes)/P(No)) with probe minus without probe",
        "intervals": "5000 whole-prompt bootstrap resamples; conditional on this training seed",
        "steps": steps,
        "paths": {},
    }
    for mode in ("grad", "inference"):
        report["paths"][mode] = {}
        for split in ("train", "test"):
            scores = {}
            for arm in ("uniform", "fast"):
                for step in (0, steps):
                    path = args.run / arm / f"{split}_{step}_{mode}.json"
                    record = json.loads(path.read_text())
                    if record["summary"]["grad_mode"] != (mode == "grad"):
                        raise ValueError(f"Unexpected execution path in {path}")
                    scores[arm, step] = record["rows"]
            comparison = compare_arms(
                scores["uniform", 0], scores["uniform", steps],
                scores["fast", 0], scores["fast", steps],
            )
            for arm in ("uniform", "fast"):
                final_rows = scores[arm, steps]
                zero = [{**r, "yes_minus_no": 0} for r in final_rows]
                comparison[arm]["final_ci95"] = compare_scores(zero, final_rows)["change_ci95"]
                fresh_path = args.run / arm / f"fresh_{mode}" / f"{split}_{steps}_{mode}.json"
                if fresh_path.exists():
                    fresh_rows = json.loads(fresh_path.read_text())["rows"]
                    comparison[arm]["fresh_recompile_change"] = compare_scores(
                        final_rows, fresh_rows
                    )
            report["paths"][mode][split] = comparison
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
