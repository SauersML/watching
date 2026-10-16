"""One figure separating training-path learning from native inference success."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

INK, MUTED = "#182F35", "#64747A"
TEAL, GRAY, BLUE = "#007F79", "#A79282", "#385D91"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.snapshot.read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                         "axes.labelcolor": INK, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 7.1))
    fig.subplots_adjust(left=0.075, right=0.965, bottom=0.29, top=0.70, wspace=0.30)
    fig.text(0.075, 0.94, "What learned, and where?", fontsize=28, weight="bold")
    fig.text(0.075, 0.888,
             "Qwen3-0.6B · one practiced prompt · larger separation favors probe detection",
             color=MUTED, fontsize=12)
    fig.text(0.075, 0.81, "GRADIENT TRAINING", fontsize=10, weight="bold", color=TEAL)
    fig.text(0.075, 0.775, "Same trained checkpoints, two evaluation paths", fontsize=11)
    fig.text(0.706, 0.81, "DIRECT OUTPUT-WEIGHT FIT", fontsize=10, weight="bold", color=BLUE)
    fig.text(0.706, 0.775, "Separate method · ordinary inference", fontsize=11)
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E8EDEC", linewidth=0.8)
        ax.axhline(0, color="#AEBBBE", linewidth=1)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#DAE2E1")
        ax.tick_params(length=0, pad=8, colors=MUTED, labelsize=10)
    for ax, grad, title in zip(axes[:2], [True, False],
                               ["Training execution", "Ordinary inference"], strict=True):
        for arm, color in [("full", GRAY), ("fixed", TEAL)]:
            points = sorted((row for row in data["original"]["arms"][arm]
                             if row["grad_mode"] == grad), key=lambda row: row["step"])
            x = [row["step"] for row in points]
            y = [row["paired_log_odds_shift"] for row in points]
            ax.plot(x, y, color=color, linewidth=2.6 if arm == "fixed" else 1.6)
            ax.scatter(x[-1], y[-1], s=27, color=color, zorder=5)
            if arm == "fixed":
                ax.annotate(f"{y[-1]:+.2f}", (x[-1], y[-1]), xytext=(-7, 13),
                            textcoords="offset points", ha="right", color=color,
                            weight="bold", fontsize=15)
        ax.set_title(title, loc="left", fontsize=14, pad=16, weight="bold")
        ax.set_ylim(-0.16, 1.05)
        ax.set_xlim(-8, 540)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticks([0, 256, 512])
        ax.set_xlabel("Optimizer updates", labelpad=10, fontsize=10)
    axes[0].set_ylabel("Probe − no-probe log-odds (nats)", fontsize=11, labelpad=14)
    points = sorted(data["direct"]["arms"]["fit"], key=lambda row: row["step"])
    fresh = data["direct"]["arms"]["fresh_inference"][-1]
    y = [row["paired_log_odds_shift"] for row in points]
    final = fresh["paired_log_odds_shift"]
    ax = axes[2]
    ax.plot(range(len(points)), y, color=BLUE, lw=2.6, marker="o", ms=5)
    ax.plot([3, 4], [y[-1], final], color=BLUE, lw=1.5, linestyle=(0, (3, 3)))
    ax.scatter(4, final, color=BLUE, marker="D", s=48, zorder=5)
    ax.annotate(f"{final:+.2f}", (4, final), xytext=(-4, 14), ha="right",
                textcoords="offset points", color=BLUE, fontsize=15, weight="bold")
    ax.set_title("Fit survives fresh compilation", loc="left", fontsize=14,
                 pad=16, weight="bold")
    ax.set_ylim(-0.30, 2.45)
    ax.set_yticks([0, 0.5, 1, 1.5, 2])
    ax.set_xlim(-0.25, 4.35)
    ax.set_xticks([0, 1, 2, 3, 4], ["Before", "1", "2", "3", "Fresh\ncompile"])
    ax.set_xlabel("Fit refinements", labelpad=10, fontsize=10)
    ax.text(0.02, 0.95, "Separate vertical scale", transform=ax.transAxes,
            fontsize=8.5, color=MUTED)
    fig.legend(handles=[Line2D([], [], color=TEAL, lw=2.6, label="Internal weights fixed"),
                        Line2D([], [], color=GRAY, lw=1.8, label="Internal weights updating")],
               loc="lower left", bbox_to_anchor=(0.068, 0.172), frameon=False,
               ncol=2, fontsize=10, handlelength=2.4, columnspacing=2.5)
    fig.text(0.075, 0.135,
             "Left + middle: existing output matrix untied; trained by gradient updates.",
             fontsize=10, color=MUTED)
    fig.text(0.075, 0.099,
             "Right: original tied model; least-squares fit of the existing Yes/No rows. "
             "No new head; other weights fixed.", fontsize=10, color=MUTED)
    fig.text(0.075, 0.052,
             "One prompt and one training seed per experiment. No population confidence interval "
             "or unseen-prompt generalization claim.", fontsize=9.5, color=MUTED)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png", "svg"):
        fig.savefig(args.output.with_suffix("." + extension), dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
