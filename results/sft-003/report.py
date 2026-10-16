"""Rebuild checkpoint comparisons and figures from saved per-prompt scores.

Run with the watching package installed and matplotlib available:
    python results/sft-003/report.py OUTPUT_DIRECTORY
"""

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

from watching.compare import compare_scores
from watching.metrics import summarize_pairs


def collect(source):
    records = []
    for path in source.glob("*_*.json"):
        if not re.fullmatch(r"(train|validation|test)_\d+_(grad|inference)\.json", path.name):
            continue
        payload = json.loads(path.read_text())
        if "summary" not in payload or "rows" not in payload:
            continue
        summary = payload["summary"]
        split, step, grad = summary["split"], summary["step"], summary["grad_mode"]
        mode = "grad" if grad else "inference"
        baseline = json.loads((source / f"{split}_0_{mode}.json").read_text())
        comparison = compare_scores(baseline["rows"], payload["rows"], seed=42)
        endpoint = summarize_pairs(payload["rows"], seed=42, bootstrap_samples=5000)
        records.append(
            {
                "split": split,
                "step": step,
                "grad_mode": grad,
                "n_prompts": comparison["n_prompts"],
                "separation": comparison["final_separation"],
                "separation_ci_low": endpoint["paired_log_odds_shift_ci95"][0],
                "separation_ci_high": endpoint["paired_log_odds_shift_ci95"][1],
                "change": comparison["change_in_separation"],
                "change_ci_low": comparison["change_ci95"][0],
                "change_ci_high": comparison["change_ci95"][1],
                "comparison": comparison,
            }
        )
    return sorted(records, key=lambda r: (r["split"], r["grad_mode"], r["step"]))


def plot(records, destination):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey="row", layout="constrained")
    for column, split in enumerate(["train", "validation"]):
        selected = [r for r in records if r["split"] == split and not r["grad_mode"]]
        if not selected:
            continue
        steps = [r["step"] for r in selected]
        count = selected[0]["n_prompts"]
        for row, metric, title in [
            (0, "separation", "Probe − no-probe separation"),
            (1, "change", "Change from before training"),
        ]:
            ax = axes[row, column]
            ax.plot(steps, [r[metric] for r in selected], marker="o", markersize=3)
            ax.fill_between(
                steps,
                [r[f"{metric}_ci_low"] for r in selected],
                [r[f"{metric}_ci_high"] for r in selected],
                alpha=0.18,
                label="Pointwise 95% CI",
            )
            ax.axhline(0, color="0.4", linewidth=0.8)
            ax.set(title=f"{count} {split} prompts\n{title}", xlabel="SFT updates")
            ax.set_ylabel("Yes/No log odds (nats)")
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(fontsize=8)
    fig.suptitle("Expanded SFT · compiled inference", fontsize=14)
    fig.supxlabel(
        "5,000 bootstrap resamples of whole prompts; before/after comparisons remain paired.\n"
        "Checkpoint intervals are exploratory and describe prompts, not training-seed variation.",
        fontsize=9,
    )
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(destination / f"learning_curve.{extension}", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    records = collect(args.source)
    if not records:
        parser.error("No per-prompt checkpoint files found")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "checkpoint_comparisons.json").write_text(json.dumps(records, indent=2) + "\n")
    with (args.out / "learning_curve.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[k for k in records[0] if k != "comparison"])
        writer.writeheader()
        writer.writerows({k: v for k, v in r.items() if k != "comparison"} for r in records)
    plot(records, args.out)
    latest = [r for r in records if r["split"] == "validation" and not r["grad_mode"]][-1]
    print(json.dumps(latest, indent=2))


if __name__ == "__main__":
    main()
