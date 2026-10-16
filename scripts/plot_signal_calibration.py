"""Contrast endpoint condition separation with calibrated answer quality."""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

INK, MUTED = "#19343A", "#64777D"
TEAL, BLUE = "#00867D", "#456D9F"


def final_metrics(report, split):
    entry = report["splits"][split]
    point = entry.get("fresh", entry["checkpoints"][-1])
    if split == "ood":
        family = point["family_analysis"]
        cluster = family["cluster_bootstrap"]
        return {
            "separation": cluster["final_separation"],
            "loss": cluster["binary_ce"],
            "n_prompts": entry["n_prompts"],
            "n_families": family["n_families"],
        }
    return {
        "separation": {"mean": point["final_separation"], "ci95": point["endpoint_ci95"]},
        "loss": point["binary_ce"],
        "n_prompts": entry["n_prompts"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text())
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "text.color": INK,
            "axes.labelcolor": INK,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(14.1, 8.0), sharey=True)
    fig.subplots_adjust(left=0.265, right=0.955, top=0.71, bottom=0.29, wspace=0.21)
    fig.text(0.065, 0.94, "Signal and answer quality are different", fontsize=27, weight="bold")
    fig.text(
        0.065,
        0.887,
        "Qwen3-8B · existing answer rows fitted by ridge regression · internal weights fixed",
        fontsize=11.5,
        color=MUTED,
    )
    rows = [
        ("probe", "test", "Read-only probe\nFamiliar templates", TEAL),
        ("probe", "ood", "Read-only probe\nNew prompt families", TEAL),
        ("execution", "test", "Gradient-enabled execution\nFamiliar templates", BLUE),
        ("execution", "ood", "Gradient-enabled execution\nNew prompt families", BLUE),
    ]
    metrics = [final_metrics(snapshot[key], split) for key, split, _, _ in rows]
    for ax, metric, title, subtitle in zip(
        axes,
        ("separation", "loss"),
        ("Paired signal", "Answer quality"),
        ("Yes/No log-odds separation · nats", "Binary log loss · lower is better"),
        strict=True,
    ):
        ax.set_title(title, loc="left", pad=40, fontsize=16, weight="bold")
        ax.text(0, 1.06, subtitle, transform=ax.transAxes, fontsize=10.5, color=MUTED)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#E6ECEC", linewidth=0.8)
        for y in (0, 2):
            ax.axhspan(y - 0.42, y + 0.42, color="#F3F6F6", zorder=0)
        for y, (_, split, _, color), values in zip([3, 2, 1, 0], rows, metrics, strict=True):
            result = values[metric]
            mean = result["mean"]
            low, high = result["ci95"]
            ax.errorbar(
                mean,
                y,
                xerr=[[mean - low], [high - mean]],
                fmt="s" if split == "ood" else "o",
                color=color,
                markerfacecolor="white" if split == "ood" else color,
                markeredgewidth=1.6,
                markersize=7.5,
                elinewidth=2.5,
                capsize=4,
                zorder=4,
            )
            ax.annotate(
                f"{mean:.3f}" if metric == "loss" else f"{mean:+.2f}",
                (mean, y),
                xytext=(0, 14),
                textcoords="offset points",
                ha="center",
                fontsize=12,
                color=color,
                weight="bold",
            )
        ax.set_ylim(-0.50, 3.55)
        ax.tick_params(length=0, pad=9, colors=MUTED, labelsize=11)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#D5DFDF")
        ax.set_xlabel("Nats", labelpad=10, fontsize=10)
    row_labels = []
    for key, split, label, _ in rows:
        greedy = snapshot["greedy"][key][split]
        row_labels.append(label + f"\nNext token correct: {greedy['correct']}/{greedy['total']}")
    axes[0].set_yticks([3, 2, 1, 0], row_labels, linespacing=1.5)
    axes[0].tick_params(axis="y", labelsize=10)
    axes[0].tick_params(axis="y", pad=22)
    axes[0].set_xlim(-0.04, 2.12)
    axes[0].set_xticks([0, 0.5, 1, 1.5, 2])
    axes[0].axvline(0, color="#9DAEB2", linewidth=1.2)
    axes[1].set_xlim(0, max(values["loss"]["ci95"][1] for values in metrics) * 1.1)
    axes[1].set_xticks([0, 1, 2, 3, 4, 5, 6])
    axes[1].axvline(math.log(2), color="#AD9276", linewidth=1.4, linestyle=(0, (3, 3)))
    axes[1].annotate(
        "Equal-probability\nanswers: 0.693",
        (math.log(2), 3.50),
        xytext=(35, 0),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=8.8,
        color="#8D735A",
        linespacing=1.3,
    )
    fig.text(
        0.065,
        0.197,
        "A positive paired signal does not guarantee well-calibrated answers on new families.",
        fontsize=12.5,
        weight="bold",
    )
    fig.text(
        0.065,
        0.151,
        "95% intervals: 5,000 paired bootstrap resamples. Familiar templates: 64 whole prompts. "
        "New families: 8 whole families, 128 prompts.",
        fontsize=9.4,
        color=MUTED,
    )
    fig.text(
        0.065,
        0.115,
        "Family estimates weight families equally. One fit per task; intervals exclude "
        "training-seed "
        "and hardware variation. These are output-row fits, not SFT.",
        fontsize=9.4,
        color=MUTED,
    )
    fig.text(
        0.065,
        0.075,
        "Next-token counts use the actual highest-probability token over the full vocabulary; "
        "counts are descriptive, without intervals.",
        fontsize=9.4,
        color=MUTED,
    )
    fig.text(0.065, 0.035, snapshot["verification_note"], fontsize=9.4, color=MUTED)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png", "svg"):
        fig.savefig(args.output.with_suffix("." + extension), dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
