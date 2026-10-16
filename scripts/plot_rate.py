"""Plot both learning-rate arms from complete or streamed metrics."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def read_events(path):
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    events = []
    for i, line in enumerate(lines):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--arm", nargs=2, action="append", metavar=("LABEL", "DIRECTORY"))
    parser.add_argument(
        "--explicit-arms", action="store_true", help="Plot only the named --arm directories"
    )
    parser.add_argument("--title", default="Learning-rate comparison")
    parser.add_argument("--note", default="All weights trainable; original weight tying retained.")
    args = parser.parse_args()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    plotted = False
    progress = []
    arms = (
        []
        if args.explicit_arms
        else [
            ("100× shared-matrix rate", args.run / "fast"),
            ("Uniform rate", args.run / "uniform"),
        ]
    ) + [(label, Path(path)) for label, path in (args.arm or [])]
    colors = ["#D46521", "#215CAF", "#228833", "#AA3377", "#66AABB", "#8844AA"]
    if len(arms) > len(colors):
        raise ValueError("This plot supports at most six arms")
    for (label, directory), color in zip(arms, colors, strict=False):
        events = read_events(directory / "metrics.jsonl")
        evaluations = [r for r in events if r["event"] == "evaluation" and r["split"] == "train"]
        if any(r["n_prompts"] != 1 for r in evaluations):
            raise ValueError("This plot requires a single practiced prompt")
        if evaluations:
            progress.append(f"{label}: {max(r['step'] for r in evaluations)}")
        for col, grad in enumerate([True, False]):
            rows = sorted(
                [r for r in evaluations if r["grad_mode"] == grad], key=lambda r: r["step"]
            )
            if not rows:
                continue
            plotted = True
            steps = [r["step"] for r in rows]
            axes[0, col].plot(
                steps,
                [r["paired_log_odds_shift"] for r in rows],
                "o-",
                color=color,
                label=label,
                markersize=3,
            )
            axes[1, col].plot(
                steps,
                [(r["absent_mean_log_odds"] + r["present_mean_log_odds"]) / 2 for r in rows],
                "o-",
                color=color,
                label=label,
                markersize=3,
            )
    if not plotted:
        plt.close(fig)
        raise ValueError("No training-prompt evaluations yet")
    for col, title in enumerate(["Training execution path", "Ordinary inference"]):
        axes[0, col].set_title(title)
        axes[1, col].set_xlabel("Optimizer updates")
    axes[0, 0].set_ylabel("Probe − no-probe log-odds (nats)")
    axes[1, 0].set_ylabel("Mean Yes/No log-odds across conditions\n(answer bias, nats)")
    for ax in axes.flat:
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.15)
        if ax.lines and ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    fig.suptitle(args.title + "\n" + " · ".join(progress), fontsize=10)
    fig.text(
        0.5,
        0.02,
        "One practiced prompt; no population confidence interval. " + args.note + "\n"
        "Top: larger separation favors detection. Bottom: zero means balanced Yes/No preference.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)


if __name__ == "__main__":
    main()
