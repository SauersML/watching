"""Plot paired before/after generalization from two native output-row fits."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

INK, MUTED, TEAL, BLUE = "#193239", "#65767C", "#00867D", "#42689A"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.snapshot.read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                         "axes.labelcolor": INK, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 7.3))
    fig.subplots_adjust(left=0.13, right=0.96, top=0.69, bottom=0.33, wspace=0.44)
    fig.text(0.065, 0.935, "Learning transfers to unseen prompts", fontsize=26, weight="bold")
    fig.text(0.065, 0.875,
             "Native Yes/No output rows fitted · internal weights fixed · least squares, not SFT",
             fontsize=11.5, color=MUTED)
    panels = [
        ("probe", "Detecting a read-only probe", "Qwen3-8B · ordinary inference", TEAL),
        ("execution", "Detecting gradient-enabled execution", "Qwen3-0.6B · gradient vs no-gradient", BLUE),
    ]
    for ax, (key, title, subtitle, color) in zip(axes, panels, strict=True):
        ax.set_title(title, loc="left", fontsize=14, weight="bold", pad=45)
        ax.text(0, 1.10, subtitle, transform=ax.transAxes, fontsize=11, color=MUTED)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#E8EDEC", linewidth=0.8)
        ax.axvline(0, color="#A9B9BD", lw=1.2)
        ax.axhspan(-0.40, 0.40, facecolor=color, alpha=0.055, zorder=0)
        labels = []
        for y, split in [(2, "train"), (1, "validation"), (0, "test")]:
            section = data[key]["splits"][split]
            final = max(section["checkpoints"], key=lambda item: item["step"])
            gain = final["change_in_separation"]
            low, high = final["change_ci95"]
            test = split == "test"
            ax.errorbar(gain, y, xerr=[[gain - low], [high - gain]], fmt="o",
                        color=color if test else "#9CAEB3", markersize=9 if test else 6.5,
                        elinewidth=3 if test else 2, capsize=5 if test else 4,
                        markeredgecolor="white", markeredgewidth=1, zorder=4)
            labels.append(f"{split.title()}  ·  {section['n_prompts']}")
            if test:
                ax.text(gain, y + 0.22, f"{gain:+.2f}  [{low:+.2f}, {high:+.2f}]",
                        ha="center", color=color, fontsize=12, weight="bold")
                loss = final["binary_ce"]["mean"]
                ax.text(0, -0.42, f"Test binary log loss: {loss:.3f}",
                        transform=ax.transAxes, color=INK, fontsize=11)
        ax.set_yticks([2, 1, 0], labels)
        ax.set_xlim(-0.04, 2.3)
        ax.set_ylim(-0.55, 2.5)
        ax.set_xticks([0, 0.5, 1, 1.5, 2])
        ax.tick_params(axis="both", length=0, pad=9, colors=MUTED, labelsize=10.5)
        ax.get_yticklabels()[2].set_color(color)
        ax.get_yticklabels()[2].set_weight("bold")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#DAE2E1")
    fig.text(0.545, 0.245, "Improvement in paired Yes/No log-odds separation (nats)",
             ha="center", fontsize=12)
    fig.text(0.065, 0.122,
             "Bars: 95% intervals from 5,000 bootstrap resamples of whole prompt pairs, "
             "preserving before/after matching.", fontsize=9.5, color=MUTED)
    fig.text(0.065, 0.087,
             "New contexts from the same template distribution; one fit per experiment. "
             "Intervals exclude seed and hardware variation.", fontsize=9.5, color=MUTED)
    fig.text(0.065, 0.052,
             "Binary log loss uses Yes/No probabilities; an equal-probability predictor scores 0.693. "
             + data["verification_note"], fontsize=9.5, color=MUTED)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png", "svg"):
        fig.savefig(args.output.with_suffix("." + extension), dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
