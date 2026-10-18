"""Show exact prefix agreement at fixed horizons within the same generated replies."""

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
    fig, ax = plt.subplots(figsize=(10, 6.2))
    fig.subplots_adjust(left=0.12, right=0.95, top=0.73, bottom=0.3)
    fig.text(0.12, 0.93, "How much of the reply stays identical?", fontsize=23, weight="bold")
    fig.text(
        0.12,
        0.86,
        "100 paired ordinary questions · same 128-token generations · NVIDIA A40",
        fontsize=11.5,
        color="#777777",
    )
    for runtime, color, label in [("hf", "#3B9C60", "HF"), ("vllm", "#AD7C50", "vLLM")]:
        points = r["runtimes"][runtime]["prefix_agreement"]
        x = [p["horizon"] for p in points]
        m = [p["metrics"]["returned_prefix_equal"] for p in points]
        ax.plot(x, [p["mean"] for p in m], color=color, marker="o", linewidth=2.5, label=label)
        ax.fill_between(
            x, [p["ci95"][0] for p in m], [p["ci95"][1] for p in m], color=color, alpha=0.15
        )
    ax.set_xticks([1, 16, 32, 48, 64, 128])
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Prefix horizon · tokens", fontsize=12)
    ax.set_ylabel("Exact returned-prefix agreement", fontsize=12)
    ax.grid(axis="y", color="#DFE0E6")
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#BFC0C6")
    ax.legend(frameon=False)
    ax.tick_params(length=0, labelsize=11)
    notes = [
        "Pointwise 95% paired-prompt bootstrap; nested horizons are not independent trials.",
        "Matching short replies count as equal; unequal slice lengths count as different.",
        "Short-prefix agreement is not full-answer agreement. Most replies reach the token cap.",
        "Different wording can be equally good. No quality score or hardware effect is inferred.",
    ]
    for i, note in enumerate(notes):
        fig.text(0.12, 0.19 - 0.035 * i, note, fontsize=9, color="#777777")
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(a.out.with_suffix("." + suffix), dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
