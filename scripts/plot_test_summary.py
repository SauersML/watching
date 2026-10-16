"""Plot final test separation and paired improvement with prompt-bootstrap CIs."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.analysis.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, value_key, interval_key, title in zip(
        axes,
        ("final_separation", "change_in_separation"),
        ("final_ci95", "change_ci95"),
        ("Final separation", "Change from before training"),
        strict=True,
    ):
        for y, mode, color in ((1, "grad", "#D46521"), (0, "inference", "#215CAF")):
            row = data[mode]
            value, (low, high) = row[value_key], row[interval_key]
            ax.errorbar(
                value, y, xerr=[[value - low], [high - value]], fmt="o", color=color, capsize=5
            )
            ax.annotate(
                f"{value:+.4f} [{low:+.4f}, {high:+.4f}]",
                (value, y),
                xytext=(0, 15),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.set(title=title, xlabel="Mean paired log-odds (nats)", ylim=(-0.4, 1.5))
        ax.margins(x=0.25)
        ax.set_yticks([0, 1], ["Inference", "Training path"])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.15)
    count = data["grad"]["n_prompts"]
    fig.suptitle(f"{count} synthetic test prompts · 95% prompt-bootstrap intervals")
    fig.text(
        0.5,
        0.025,
        "Probe/no-probe pairs stay together; improvement also pairs before/after.\n"
        "Final: 1,000 resamples. Improvement: 5,000. Intervals exclude training-seed variability.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)


if __name__ == "__main__":
    main()
