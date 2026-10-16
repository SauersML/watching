"""Render the single-prompt training-path separation as a standalone figure."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument(
        "output", type=Path, help="Output stem; PDF, SVG, PNG and data JSON are saved"
    )
    args = parser.parse_args()
    rows = []
    lines = args.metrics.read_text().splitlines()
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                continue  # The live writer may not have completed this last line.
            raise
        if (
            row.get("event") == "evaluation"
            and row.get("split") == "train"
            and row.get("grad_mode") is True
        ):
            if row["n_prompts"] != 1:
                raise ValueError("This figure describes exactly one practiced prompt")
            rows.append({"step": row["step"], "separation": row["paired_log_odds_shift"]})
    rows.sort(key=lambda row: row["step"])
    if not rows or len({row["step"] for row in rows}) != len(rows):
        raise ValueError("Provide one training run with unique evaluation steps")
    steps = [row["step"] for row in rows]
    values = [row["separation"] for row in rows]
    ink, muted, teal = "#172E32", "#627378", "#007F79"
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelcolor": ink,
            "text.color": ink,
            "xtick.color": muted,
            "ytick.color": muted,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(10.8, 7.0), facecolor="white")
    ax = fig.add_axes((0.13, 0.23, 0.77, 0.51))
    fig.text(0.13, 0.93, "WATCHING  /  EXPERIMENT 014", color=teal, fontsize=10, weight="bold")
    fig.text(0.13, 0.858, "A growing probe signal", fontsize=28, weight="bold")
    fig.text(
        0.13,
        0.805,
        "One training prompt · fixed internal weights · training execution path",
        fontsize=11.5,
        color=muted,
    )
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E6ECEB", linewidth=0.8)
    ax.axhline(0, color="#A9B6B6", linewidth=1)
    ax.plot(steps, values, color=teal, linewidth=2.8, solid_capstyle="round", zorder=3)
    ax.scatter(steps, values, color=teal, s=16, edgecolors="white", linewidths=0.6, zorder=4)
    ax.scatter(
        steps[-1], values[-1], color=teal, s=66, edgecolors="white", linewidths=1.8, zorder=5
    )
    ax.annotate(
        f"{values[-1]:+.2f}",
        (steps[-1], values[-1]),
        xytext=(16, 3),
        textcoords="offset points",
        fontsize=19,
        weight="bold",
        color=teal,
    )
    ax.annotate(
        f"update {steps[-1]}",
        (steps[-1], values[-1]),
        xytext=(17, -14),
        textcoords="offset points",
        fontsize=9.5,
        color=muted,
    )
    extent = max(steps[-1], 16)
    ax.set_xlim(-0.025 * extent, extent * 1.20)
    span = max(max(values) - min(values), 0.1)
    ax.set_ylim(min(0, min(values)) - span * 0.055, max(values) + span * 0.18)
    ax.set_xticks(
        [0, 128, 256, 384, 512]
        if extent >= 400
        else MaxNLocator(nbins=5, integer=True).tick_values(0, extent)
    )
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.1f}"))
    ax.set_xlabel("Optimizer updates", labelpad=13, fontsize=12)
    ax.set_ylabel("Probe − no-probe log-odds (nats)", labelpad=15, fontsize=12)
    ax.tick_params(axis="both", length=0, pad=9)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#D8E0DF")
    fig.text(
        0.13,
        0.105,
        "Higher values mean the probe increases Yes relative to No.",
        fontsize=11,
        color=ink,
    )
    fig.text(
        0.13,
        0.067,
        "Only the untied unembedding is updated. Measured checkpoints; no smoothing.\n"
        "One fixed prompt: no population confidence interval can be estimated.",
        fontsize=9.5,
        color=muted,
        linespacing=1.6,
        va="top",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(args.output.with_suffix("." + extension), dpi=200, facecolor="white")
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "source": str(args.metrics),
                "execution_path": "gradient-enabled training",
                "n_prompts": 1,
                "population_ci": None,
                "checkpoints": rows,
            },
            indent=2,
        )
        + "\n"
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
