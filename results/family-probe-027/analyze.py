"""Summarize archived native scores; never infer missing per-prompt observations."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

SPLITS = ("train", "validation", "test", "test_wording", "ood_context", "ood")
LABELS = ("Training", "Development", "New contexts", "New questions", "New content", "Both new")


def summarize(path):
    saved = json.loads(path.read_text())
    pairs = {}
    for row in saved["rows"]:
        pair = pairs.setdefault(row["index"], {})
        condition = row["condition_present"]
        if condition in pair:
            raise ValueError("Repeated condition")
        pair[condition] = row
    if len(pairs) != saved["summary"]["n_prompts"] or any(
        set(p) != {False, True} for p in pairs.values()
    ):
        raise ValueError("Incomplete paired scores")
    values = np.array(
        [
            [
                p[True]["yes_minus_no"] - p[False]["yes_minus_no"],
                sum(r["ce"] for r in p.values()) / 2,
                sum(float(r["greedy_correct"]) for r in p.values()) / 2,
            ]
            for p in pairs.values()
        ]
    )
    indices = np.random.default_rng(20261016).integers(len(values), size=(5000, len(values)))
    means = values[indices].mean(1)
    return {
        "summary": saved["summary"],
        "metrics": {
            name: {
                "mean": float(values[:, i].mean()),
                "ci95": np.quantile(means[:, i], [0.025, 0.975]).tolist(),
            }
            for i, name in enumerate(("gap", "full_ce", "greedy_accuracy"))
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--task", choices=("probe", "execution"), required=True)
    args = parser.parse_args()
    report = {
        "intervals": "5000 paired whole-prompt bootstrap draws, seed20261016; "
        "conditional on these families and this run",
        "stages": {},
    }
    stages = (args.task, "fresh_" + args.task, "preference", "fresh_preference")
    for stage in stages:
        scores = {}
        for split in SPLITS:
            phase = "fresh" if stage.startswith("fresh_") else "many"
            path = args.archive / stage / f"{phase}_{split}_1.json"
            if path.exists():
                scores[split] = summarize(path)
        report["stages"][stage] = scores
    (args.archive / "endpoints.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.7))
    fig.subplots_adjust(left=0.065, right=0.98, bottom=0.31, top=0.72, wspace=0.3)
    fig.text(
        0.055,
        0.94,
        f"Native answer-row fitting · {args.task} condition",
        fontsize=23,
        weight="bold",
        color="#19343B",
    )
    fig.text(
        0.055,
        0.88,
        "Qwen3-8B · fixed internal computation · question-family CV · full-vocabulary scoring",
        fontsize=11,
        color="#667980",
    )
    status = (
        "Partial run: fresh detector unavailable; one fresh animal cohort has summary only."
        if args.task == "probe"
        else "Completed run · fresh execution includes both detector and animal answer rows."
    )
    fig.text(0.055, 0.83, status, fontsize=10, color="#667980")
    colors = ("#00877D", "#00877D", "#456D9F", "#456D9F")
    names = ("Detector · fit", "Detector · fresh", "Animal · fit", "Animal · fresh")
    for ax, metric, heading in zip(
        axes,
        ("gap", "full_ce", "greedy_accuracy"),
        (
            "Condition log-odds separation",
            "Full-vocabulary cross-entropy",
            "Actual top-token accuracy",
        ),
        strict=True,
    ):
        for j, stage in enumerate(stages):
            points = report["stages"][stage]
            if not points:
                continue
            x = (
                np.array([i for i, split in enumerate(SPLITS) if split in points])
                + (j - 1.5) * 0.11
            )
            scores = [points[split]["metrics"][metric] for split in SPLITS if split in points]
            y = np.array([s["mean"] for s in scores])
            errors = np.array([[s["ci95"][0], s["ci95"][1]] for s in scores]).T
            ax.errorbar(
                x,
                y,
                yerr=[y - errors[0], errors[1] - y],
                fmt="o" if j % 2 == 0 else "s",
                color=colors[j],
                markerfacecolor=colors[j] if j % 2 == 0 else "white",
                markersize=5,
                capsize=3,
                label=names[j],
            )
        ax.set_title(heading, loc="left", fontsize=12, weight="bold", pad=15)
        ax.set_xticks(range(6), LABELS, rotation=28, ha="right")
        ax.grid(axis="y", color="#E7EDED")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylabel("Nats" if metric != "greedy_accuracy" else "Accuracy")
        if metric == "gap":
            ax.axhline(0, linestyle="--", color="#AAB9BE", linewidth=1)
        elif metric == "full_ce":
            ax.axhline(np.log(2), linestyle="--", color="#AAB9BE", linewidth=1)
        else:
            ax.set_ylim(0.35, 0.8)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.axhline(0.5, linestyle="--", color="#AAB9BE", linewidth=1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower left", bbox_to_anchor=(0.055, 0.12), frameon=False, ncol=4
    )
    fig.text(
        0.055,
        0.09,
        "95% intervals resample complete paired prompts; "
        "conditional on these question/content families and this single run.",
        fontsize=9.5,
        color="#667980",
    )
    fig.text(
        0.055,
        0.045,
        "New-question cells reuse contexts from their matching context cells. "
        "Missing raw scores are omitted; fresh runs are not independent training repetitions.",
        fontsize=9.2,
        color="#667980",
    )
    fig.savefig(args.archive / "performance.pdf", facecolor="white")
    fig.savefig(args.archive / "performance.png", facecolor="white", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
