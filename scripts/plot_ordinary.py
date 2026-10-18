"""Plot matched ordinary-answer changes, without presenting them as quality scores."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    r = json.loads(a.report.read_text())
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))
    fig.subplots_adjust(left=0.11, right=0.96, top=0.76, bottom=0.23, wspace=0.3, hspace=0.65)
    fig.text(0.11, 0.94, "Ordinary answers after detector training", fontsize=24, weight="bold")
    fig.text(
        0.11,
        0.88,
        "100 matched questions · 128-token greedy replies · NVIDIA A40",
        fontsize=12,
        color="#777777",
    )
    specs = [
        ("first_token_match", "First token matches the original", True),
        ("exact_token_sequence_match", "Entire returned reply matches", True),
        ("first_token_kl_original_to_trained", "First-token distribution change", False),
        ("first_token_yes_no", "Change in Yes/No prefixes", True),
    ]
    colors = ["#3B9C60", "#AD7C50"]
    for ax, (name, title, percent) in zip(axes.flat, specs, strict=True):
        for i, runtime in enumerate(["hf", "vllm"]):
            section = r["runtimes"][runtime]
            if name != "first_token_yes_no" and name not in section["paired_comparisons"]:
                ax.text(i, 0.5, "Unavailable", ha="center", color=colors[i])
                continue
            m = (
                section["metrics"]["trained_minus_original"][name]
                if name == "first_token_yes_no"
                else section["paired_comparisons"][name]
            )
            low, high = m["ci95"]
            ax.errorbar(
                i,
                m["mean"],
                yerr=[[m["mean"] - low], [high - m["mean"]]],
                fmt="o",
                markersize=9,
                capsize=5,
                color=colors[i],
                linewidth=2,
            )
        ax.set_title(title, loc="left", fontsize=14, weight="bold", pad=12)
        ax.set_xticks([0, 1], ["HF", "vLLM"])
        ax.set_xlim(-0.6, 1.6)
        ax.grid(axis="y", color="#DFE0E6")
        ax.set_axisbelow(True)
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#BFC0C6")
        ax.tick_params(axis="both", length=0, labelsize=11)
        if name == "first_token_yes_no":
            ax.axhline(0, color="#BFC0C6", linewidth=1)
            ax.set_ylim(min(-0.05, ax.get_ylim()[0]), max(0.05, ax.get_ylim()[1]))
            ax.set_ylabel("Percentage-point change")
        elif percent:
            ax.set_ylim(0, 1.05)
        else:
            ax.set_ylim(bottom=0)
            ax.set_ylabel("KL divergence · nats")
        if percent:
            ax.yaxis.set_major_formatter(PercentFormatter(1))
    footnotes = [
        "Pointwise 95% paired-prompt bootstrap; observed questions, not a blind quality test.",
        "Most base replies hit the 128-token cap. Matching a partial reply does not prove quality.",
        "Different text can be equally good. These metrics measure change, not correctness.",
        "One fit; no seed/hardware variation. All-zero/one bands give no unseen-error bound.",
    ]
    for i, text in enumerate(footnotes):
        fig.text(0.11, 0.16 - 0.031 * i, text, fontsize=9, color="#777777")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(a.out.with_suffix("." + ext), dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
