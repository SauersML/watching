"""Plot fresh-process 034 confirmation from complete per-prompt reference scores."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from make_paper_figures import before_after, save, style

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("watching_metrics", ROOT / "src/watching/metrics.py")
METRICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(METRICS)


def read(path):
    return json.loads(path.read_text())


def collect(directory):
    manifest = read(directory / "manifest.json")
    data = read(directory / "data.json")
    records = data[manifest["split"]]
    report = {
        "run": "Fresh-process evaluation of released 034 weights",
        "n_prompts": len(records),
        "checkpoint_sha256": manifest["tail_sha256"],
        "runtime": manifest["runtime"],
        "scope": "Fresh process, base-compiled graph followed by native weight installation; "
        "same checkpoint, not a second training seed",
        "intervals": "5000 whole-prompt paired bootstrap draws; pointwise; "
        "one checkpoint and finite question bank",
        "stages": {},
    }
    identities = None
    for stage in ("base", "trained"):
        path = directory / f"{stage}_reference.json"
        saved = read(path)
        rows = saved["rows"]
        pairs = {}
        for row in rows:
            pair = pairs.setdefault(row["index"], {})
            if row["capture"] in pair:
                raise ValueError("Duplicate prompt condition")
            pair[row["capture"]] = row
            if row["context_id"] != records[row["index"]]["context_id"]:
                raise ValueError("Score identity differs from frozen data")
            target = manifest["labels"]["Yes" if row["capture"] else "No"]
            if row["greedy_correct"] != (row["greedy_token_id"] == target):
                raise ValueError("Recorded actual-token accuracy does not match target")
            if not np.isclose(row["yes_minus_no"], row["logp_yes"] - row["logp_no"], atol=2e-6):
                raise ValueError("Answer logprobs do not reproduce log odds")
        if set(pairs) != set(range(len(records))):
            raise ValueError("Incomplete prompt cohort")
        keys = [(r["index"], r["capture"], r["context_id"]) for r in rows]
        if identities is not None and keys != identities:
            raise ValueError("Base/trained prompt matching changed")
        identities = keys
        summary = METRICS.summarize_pairs(rows, seed=manifest["seed"], bootstrap_samples=5000)
        for key in ("greedy_accuracy", "paired_log_odds_shift", "mean_ce"):
            if not np.isclose(summary[key], saved["summary"][key], rtol=1e-12, atol=1e-12):
                raise ValueError(f"Raw scores disagree with summary at {key}")
        if summary["paired_log_odds_shift_ci95"] != saved["summary"]["paired_log_odds_shift_ci95"]:
            raise ValueError("Bootstrap separation interval does not reproduce")
        buffer = read(directory / f"{stage}_buffer.json")
        checks = buffer["full_vocabulary_checks"]
        valid = all(r["bit_identical"] for r in checks)
        if valid != buffer["valid_no_footprint_control"]:
            raise ValueError("Buffer validity disagrees with vocabulary checks")
        report["stages"][stage] = {
            "n_correct": sum(r["greedy_correct"] for r in rows),
            "n_executions": len(rows),
            "n_both_correct": sum(
                all(r["greedy_correct"] for r in p.values()) for p in pairs.values()
            ),
            "summary": summary,
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "buffer_valid_no_footprint_control": valid,
            "buffer_identical_prompts": sum(r["bit_identical"] for r in checks),
            "buffer_max_abs_difference": max(r["max_abs_difference"] for r in checks),
        }
    return report


def draw(report, out):
    style()
    stages = report["stages"]
    counts = tuple((report["n_prompts"], stages[s]["n_both_correct"]) for s in ("base", "trained"))
    save(before_after([("Is a probe attached?", counts)]), out, "fresh_confirmation_034")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=ROOT / "results/confirm-validation-038/fresh034"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = collect(args.source)
    (args.out / "fresh_confirmation_034.json").write_text(json.dumps(report, indent=2) + "\n")
    draw(report, args.out)


if __name__ == "__main__":
    main()
