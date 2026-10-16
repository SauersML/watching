"""Matched learning curves and direct-fit reload results from a fixed snapshot."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

INK, MUTED, TEAL, GRAY = "#172E32", "#627378", "#007F79", "#A18A78"


def style_axis(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E6ECEB", linewidth=0.8)
    ax.axhline(0, color="#A9B6B6", linewidth=1)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#D8E0DF")
    ax.tick_params(length=0, pad=8, colors=MUTED)


def save(fig, stem):
    for extension in ("pdf", "svg", "png"):
        fig.savefig(stem.with_suffix("." + extension), dpi=180, facecolor="white")
    plt.close(fig)


def matched(data, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.12, right=0.95, bottom=0.21, top=0.74, hspace=0.37, wspace=0.22)
    fig.text(0.12, 0.95, "WATCHING  /  MATCHED EXPERIMENTS", color=TEAL, fontsize=10, weight="bold")
    fig.text(0.12, 0.89, "Learning with stable activations", fontsize=27, weight="bold")
    fig.text(
        0.12,
        0.85,
        "Qwen3-0.6B · one practiced prompt · independent unembedding",
        color=MUTED,
        fontsize=12,
    )
    fig.legend(
        handles=[
            Line2D([], [], color=TEAL, lw=2.5, label="Internal weights fixed"),
            Line2D([], [], color=GRAY, lw=2, label="Internal weights updating"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.115, 0.825),
        frameon=False,
        ncol=2,
        fontsize=11,
    )
    for col, (key, title) in enumerate(
        [
            ("original", "Original gradient calculation · RTX 3090"),
            ("precision", "FP32 coefficient scaling · RTX A5000"),
        ]
    ):
        for row, grad in enumerate((True, False)):
            ax = axes[row, col]
            style_axis(ax)
            for arm, color in (("full", GRAY), ("fixed", TEAL)):
                points = sorted(
                    [r for r in data[key]["arms"].get(arm, []) if r["grad_mode"] == grad],
                    key=lambda r: r["step"],
                )
                if not points:
                    continue
                x = [r["step"] for r in points]
                y = [r["paired_log_odds_shift"] for r in points]
                ax.plot(
                    x,
                    y,
                    color=color,
                    lw=2.4 if arm == "fixed" else 1.6,
                    marker="o",
                    markersize=2.6,
                    markeredgewidth=0,
                )
                if arm == "fixed":
                    ax.annotate(
                        f"{y[-1]:+.2f}",
                        (x[-1], y[-1]),
                        xytext=(7, 5),
                        textcoords="offset points",
                        color=color,
                        weight="bold",
                        fontsize=12,
                    )
                if arm == "full" and x[-1] < 512:
                    ax.annotate(
                        f"through {x[-1]}",
                        (x[-1], y[-1]),
                        xytext=(0, -18),
                        textcoords="offset points",
                        color=GRAY,
                        fontsize=8,
                        ha="center",
                    )
            ax.set_xlim(-12, 600)
            ax.set_ylim(-0.16, 1.0)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticks([0, 128, 256, 384, 512])
            if row == 0:
                ax.set_title(title, fontsize=11, pad=15, loc="left")
            else:
                ax.set_xlabel("Optimizer updates", labelpad=10)
        axes[0, col].text(
            0.02,
            0.93,
            "TRAINING EXECUTION",
            transform=axes[0, col].transAxes,
            fontsize=8,
            color=MUTED,
        )
        axes[1, col].text(
            0.02,
            0.93,
            "ORDINARY INFERENCE",
            transform=axes[1, col].transAxes,
            fontsize=8,
            color=MUTED,
        )
    for ax in axes[:, 0]:
        ax.set_ylabel("Probe − no-probe log-odds (nats)", labelpad=13)
    fig.text(
        0.12,
        0.115,
        "Training-path learning is strong when the internal weights stay fixed.",
        fontsize=12,
    )
    fig.text(
        0.12,
        0.068,
        "Within each column, both arms start from the same weights. "
        "Only the fixed arm freezes internal updates.\n"
        "Different GPUs across columns; this does not isolate the precision "
        "change. One prompt, no population CI.",
        fontsize=9.5,
        color=MUTED,
        linespacing=1.6,
    )
    save(fig, output / "matched_learning")


def direct(data, output):
    points = sorted(data["direct"]["arms"]["fit"], key=lambda row: row["step"])
    fresh = data["direct"]["arms"].get("fresh_inference", [])
    if not fresh:
        raise ValueError("Fresh inference must finish before plotting reload verification")
    final = fresh[-1]
    fig = plt.figure(figsize=(10.8, 7.0))
    ax = fig.add_axes((0.13, 0.27, 0.77, 0.47))
    fig.text(0.13, 0.93, "WATCHING  /  INFERENCE-PATH FIT", color=TEAL, fontsize=10, weight="bold")
    fig.text(0.13, 0.855, "The fitted signal survives reload", fontsize=25, weight="bold")
    fig.text(
        0.13,
        0.805,
        "Ordinary inference · original tied Qwen3-0.6B · existing answer rows only",
        color=MUTED,
        fontsize=11,
    )
    style_axis(ax)
    x = list(range(len(points)))
    y = [r["paired_log_odds_shift"] for r in points]
    ax.plot(
        x,
        y,
        color=TEAL,
        linewidth=2.6,
        marker="o",
        markersize=7,
        markeredgecolor="white",
        markeredgewidth=1.5,
        zorder=4,
    )
    last = len(points)
    gap = final["paired_log_odds_shift"]
    ax.plot([last - 1, last], [y[-1], gap], color=TEAL, linestyle=(0, (3, 3)), linewidth=1.5)
    ax.scatter(last, gap, marker="D", s=80, color=INK, edgecolor="white", linewidth=1.5, zorder=5)
    ax.annotate(
        f"{gap:+.2f}",
        (last, gap),
        xytext=(14, 7),
        textcoords="offset points",
        fontsize=20,
        weight="bold",
    )
    ax.axhline(2, color="#B6CECB", linestyle=(0, (2, 4)), linewidth=1)
    ax.text(0.03, 2.07, "Fit target: +2.00 nats", color=MUTED, fontsize=9)
    ax.set_xticks(
        x + [last],
        ["Before fit"] + [f"Fit {r['step']}" for r in points[1:]] + ["Fresh process\n+ recompile"],
    )
    ax.set_xlim(-0.25, last + 0.85)
    ax.set_ylim(-0.15, 2.48)
    ax.set_yticks([0, 0.5, 1, 1.5, 2])
    ax.set_ylabel("Probe − no-probe log-odds (nats)", fontsize=12, labelpad=15)
    fig.text(
        0.13,
        0.15,
        f"After reload: no probe {final['absent_mean_log_odds']:+.2f}  ·  "
        f"probe {final['present_mean_log_odds']:+.2f} nats of Yes/No log-odds.",
        fontsize=11,
    )
    fig.text(
        0.13,
        0.08,
        "Supervised least-squares fit to one prompt; all other weights fixed. "
        "Internal activations verified unchanged.\n"
        "No architecture change or extra classifier. "
        "One prompt, no population confidence interval.",
        fontsize=9.5,
        color=MUTED,
        linespacing=1.6,
    )
    save(fig, output / "inference_fit")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.snapshot.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "text.color": INK,
            "axes.labelcolor": INK,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    matched(data, args.output)
    direct(data, args.output)


if __name__ == "__main__":
    main()
