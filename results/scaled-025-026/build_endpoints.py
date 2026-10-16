"""Rebuild the six-split endpoint summary from archived per-example scores."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ARCHIVE = Path(__file__).resolve().parent
ROOT = ARCHIVE.parents[1]
SPEC = importlib.util.spec_from_file_location(
    "choice_plot", ROOT / "scripts/plot_diverse_choices.py"
)
PLOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOT)
SPLITS = ("train", "validation", "test", "test_fresh", "test_wording", "ood")


def main():
    report = {
        "scope": "All completed full-native endpoint and fresh-process scores. test_fresh "
        "denotes new contexts, not new compilation.",
        "intervals": "5000 whole-prompt paired bootstrap draws, seed 20261016; OOD displayed "
        "intervals use four whole families; conditional on one fit.",
        "conditions": {},
    }
    for condition in ("probe", "execution"):
        report["conditions"][condition] = {}
        for task in ("detector", "preference"):
            fitted = ARCHIVE / condition / task
            records = json.loads((fitted / "data.json").read_text())
            manifest = json.loads((fitted / "manifest.json").read_text())
            entry = {"selected_ridge": manifest["selected_ridge"], "splits": {}}
            for split in SPLITS:
                stages = {}
                for label, folder, prefix in (
                    ("fit", task, "many"),
                    ("fresh", "fresh_" + task, "fresh"),
                ):
                    source = ARCHIVE / condition / folder
                    if json.loads((source / "data.json").read_text())[split] != records[split]:
                        raise ValueError("Fit and fresh prompts differ")
                    path = source / f"{prefix}_{split}_3.json"
                    raw = json.loads(path.read_text())
                    if raw["summary"]["split"] != split:
                        raise ValueError("Score split disagrees with filename")
                    summary = PLOT.summarize(raw["rows"], records[split], split=split)
                    pairs = {}
                    for row in raw["rows"]:
                        pairs.setdefault(row["index"], {})[row["condition_present"]] = row
                    ce = np.array(
                        [
                            sum(pair[c]["ce"] for c in (False, True)) / 2
                            for _, pair in sorted(pairs.items())
                        ]
                    )
                    rng = np.random.default_rng(20261016)
                    indices = rng.integers(0, len(ce), (5000, len(ce)))
                    summary.update(
                        source=str(path.relative_to(ROOT)),
                        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        full_ce={
                            "mean": float(ce.mean()),
                            "prompt_ci95": np.quantile(
                                ce[indices].mean(1), [0.025, 0.975]
                            ).tolist(),
                        },
                        binary_ce=raw["summary"]["mean_binary_ce"],
                    )
                    stages[label] = summary
                entry["splits"][split] = stages
            report["conditions"][condition][task] = entry
    (ARCHIVE / "endpoints.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
