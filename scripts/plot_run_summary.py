"""Plot paired changes across exploratory runs, preserving prompt matching."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--arm", nargs=2, action="append", required=True, metavar=("LABEL", "ANALYSIS_JSON")
    )
    args = parser.parse_args()
    reports = [(label, json.loads(Path(path).read_text())) for label, path in args.arm]
    fig, axes = plt.subplots(1, 2, figsize=(11, max(4, len(reports) * 0.55)))
    for ax, mode, title in zip(
        axes, ("grad", "inference"), ("Training execution path", "Ordinary inference"), strict=True
    ):
        for i, (_, report) in enumerate(reports):
            row = report["paths"][mode]["test"]
            mean = row["change_in_separation"]
            low, high = row["change_ci95"]
            ax.errorbar(
                mean, i, xerr=[[mean - low], [high - mean]], fmt="o", color="#215CAF", capsize=4
            )
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_yticks(range(len(reports)), [label for label, _ in reports])
        ax.invert_yaxis()
        ax.set_xlabel("Change in probe − no-probe log-odds (nats)")
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Learning across other prompts: paired before/after change", fontsize=12)
    fig.text(
        0.5,
        0.02,
        "95% intervals: 5,000 bootstrap resamples of whole prompts, preserving pairing.\n"
        "Exploratory runs; intervals omit training-seed and compiler variation. "
        "512 prompts in rate-008, 128 in other runs.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 0.95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    if args.output.suffix.lower() == ".pdf":
        fig.savefig(args.output.with_suffix(".png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
