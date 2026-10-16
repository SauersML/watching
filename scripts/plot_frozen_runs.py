"""Plot native output-row learning with paired prompt uncertainty, including live runs."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from analyze_native_fit import analyze_fit
from matplotlib.ticker import MaxNLocator

from watching.artifacts import write_json

INK, MUTED, TEAL, GRID = "#183239", "#687C82", "#087F79", "#E5EDEC"
COLORS = {"train": TEAL, "validation": "#496EA0", "test": "#B46A40", "ood": "#825B9B"}


def plotted_point(point, split):
    """OOD plots show family uncertainty; prompt estimates remain in the report."""
    cluster = point.get("family_analysis", {}).get("cluster_bootstrap")
    if split == "ood" and cluster:
        return {
            **point,
            "change_in_separation": cluster["change_in_separation"]["mean"],
            "change_ci95": cluster["change_in_separation"]["ci95"],
        }
    return point


def render(report, output, title=None):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    splits = [name for name in ("train", "validation", "test")]
    splits.extend(sorted(set(report["splits"]) - set(splits)))
    fig, axes = plt.subplots(1, len(splits), figsize=(4.33 * len(splits), 6.5), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.31, top=0.69, wspace=0.34)
    condition = report["condition"]
    task = "Read-only probe detection" if condition == "probe" else "Gradient-recording detection"
    checkpoint = report.get("manifest", {}).get("checkpoint", "Model unspecified")
    model_name = Path(checkpoint.rstrip("/")).name
    all_points = [
        plotted_point(point, split)
        for split, entry in report["splits"].items()
        for point in [*entry["checkpoints"], *([entry["fresh"]] if "fresh" in entry else [])]
        if point["step"] > 0
    ]
    bounds = [0.0]
    for point in all_points:
        bounds.append(point["change_in_separation"])
        if point["change_ci95"] is not None:
            bounds.extend(point["change_ci95"])
    low, high = min(bounds), max(bounds)
    span = max(high - low, 0.1)
    shared_limits = (low - 0.10 * span, high + 0.15 * span)
    fig.text(
        0.075, 0.94, "WATCHING  /  FROZEN INTERNAL WEIGHTS", color=TEAL, weight="bold", size=10
    )
    fig.text(0.075, 0.865, title or "Learning beyond the initial footprint", size=26, weight="bold")
    fig.text(
        0.075,
        0.805,
        f"{model_name}  ·  {task.lower()}  ·  native answer-row fit",
        color=MUTED,
        size=11,
    )
    names = {
        "train": "Training prompts",
        "validation": "Validation prompts",
        "test": "Test prompts",
        "ood": "New task families",
    }
    for ax, split in zip(axes, splits, strict=True):
        name = names.get(split, split.replace("_", " ").title())
        entry = report["splits"].get(split)
        ax.set_title(name, loc="left", fontsize=13, weight="bold", pad=35)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.axhline(0, color="#A7B6B7", linewidth=1)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, pad=7)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.set_xlabel("Answer-row corrections", labelpad=12)
        ax.set_ylim(*shared_limits)
        if entry is None:
            ax.text(
                0.5,
                0.5,
                "Awaiting evaluation",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color=MUTED,
            )
            ax.set_xticks([])
            continue
        family_data = entry.get("fresh", entry["checkpoints"][-1]).get("family_analysis")
        sample_label = f"{entry['n_prompts']:,} paired prompts"
        if split == "ood" and family_data:
            sample_label += f" · {family_data['n_families']} families"
        ax.text(
            0,
            1.055,
            sample_label,
            transform=ax.transAxes,
            color=MUTED,
            fontsize=10,
        )
        points = [plotted_point(point, split) for point in entry["checkpoints"]]
        if not any(point["step"] > 0 for point in points) and "fresh" not in entry:
            ax.text(
                0.5,
                0.5,
                "Awaiting post-fit evaluation",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color=MUTED,
                fontsize=10,
            )
            expected_steps = max(1, report.get("manifest", {}).get("steps", 3))
            ax.set_xlim(-0.15, expected_steps + 0.2)
            ax.set_xticks(range(expected_steps + 1))
            continue
        steps = [p["step"] for p in points]
        gains = [p["change_in_separation"] for p in points]
        color = COLORS.get(split, "#825B9B")
        ax.plot(
            steps,
            gains,
            color=color,
            linewidth=2.4,
            marker="o",
            markersize=5,
            markeredgecolor="white",
            markeredgewidth=0.7,
            zorder=3,
        )
        if all(p["change_ci95"] is not None for p in points):
            ax.fill_between(
                steps,
                [p["change_ci95"][0] for p in points],
                [p["change_ci95"][1] for p in points],
                color=color,
                alpha=0.13,
                linewidth=0,
            )
            ax.vlines(
                steps,
                [p["change_ci95"][0] for p in points],
                [p["change_ci95"][1] for p in points],
                color=color,
                alpha=0.45,
                linewidth=1,
            )
        final = points[-1]
        ticks, labels = steps.copy(), [str(s) for s in steps]
        if "fresh" in entry:
            final = plotted_point(entry["fresh"], split)
            fresh_x = steps[-1] + 1
            ticks.append(fresh_x)
            labels.append("Reload")
            ax.plot(
                [steps[-1], fresh_x],
                [gains[-1], final["change_in_separation"]],
                color=color,
                linestyle=":",
                linewidth=1.3,
            )
            ci = final["change_ci95"]
            ax.scatter(
                fresh_x,
                final["change_in_separation"],
                marker="D",
                color=color,
                s=38,
                edgecolor="white",
                zorder=4,
            )
            if ci:
                ax.vlines(fresh_x, ci[0], ci[1], color=color, linewidth=1.4)
                ax.hlines(ci, fresh_x - 0.035, fresh_x + 0.035, color=color, linewidth=1.4)

        if len(ticks) <= 7:
            ax.set_xticks(ticks, labels)
        else:
            ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
        ax.set_xlim(min(ticks) - 0.15, max(ticks) + 0.2 if max(ticks) else 1)
        ci = final["change_ci95"]
        text = f"Latest gain  {final['change_in_separation']:+.3f} nats"
        text += f"\n95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "\nOne prompt; no population CI"
        ax.text(
            0,
            -0.35,
            text,
            transform=ax.transAxes,
            color=color,
            fontsize=10,
            linespacing=1.6,
            va="top",
        )
    axes[0].set_ylabel("Change in condition log-odds separation (nats)", labelpad=12)
    fig.text(
        0.075,
        0.095,
        "Each point subtracts that same prompt’s pre-fit separation before averaging.",
        color=INK,
        fontsize=10.5,
        va="top",
    )
    fig.text(
        0.075,
        0.052,
        (
            "5,000 paired resamples; OOD samples whole families with equal family weight. "
            "Pointwise 95% intervals.\n"
            "Conditional on one fit and sampled units; excludes seed and hardware variation."
            if "ood" in splits
            else "5,000 whole-prompt bootstrap resamples; pointwise 95% intervals. "
            "Panels share the same vertical scale.\n"
            "Conditional on this fit and prompt distribution; excludes seed and hardware variation."
        ),
        color=MUTED,
        fontsize=9,
        linespacing=1.55,
        va="top",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png", "svg"):
        fig.savefig(output.with_suffix("." + extension), facecolor="white", dpi=200)
    write_json(output.with_suffix(".json"), report)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit", type=Path)
    parser.add_argument("output", type=Path, help="Output stem: PDF, PNG, SVG and analysis JSON")
    parser.add_argument("--fresh", type=Path)
    parser.add_argument("--phase", choices=["single", "many"], default="many")
    parser.add_argument("--title")
    args = parser.parse_args()
    render(analyze_fit(args.fit, fresh=args.fresh, phase=args.phase), args.output, args.title)


if __name__ == "__main__":
    main()
