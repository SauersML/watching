"""Plot a single-prompt run from streamed metrics, including incomplete runs."""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    events = []
    lines = args.metrics.read_text().splitlines()
    for i, line in enumerate(lines):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
    evaluations = [r for r in events if r.get("event") == "evaluation" and r["split"] == "train"]
    if not evaluations or any(r["n_prompts"] != 1 for r in evaluations):
        raise ValueError("Expected evaluations of exactly one training prompt")
    training = [r for r in events if r.get("event") == "train"]
    configuration = next(r for r in events if r.get("event") == "setup")
    objective = configuration.get("objective", "sft")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7))
    for mode, label, color in [
        (False, "Inference", "#215CAF"),
        (True, "Gradient-enabled forward", "#D46521"),
    ]:
        rows = sorted([r for r in evaluations if r["grad_mode"] == mode], key=lambda r: r["step"])
        axes[0].plot(
            [r["step"] for r in rows],
            [r["paired_log_odds_shift"] for r in rows],
            "o-",
            label=label,
            color=color,
            markersize=4,
        )
    axes[0].axhline(0, color="gray", linewidth=1, linestyle="--")
    axes[0].set(
        title="Does the probe increase Yes relative to No?",
        ylabel="Probe − no-probe log-odds (nats)",
    )
    axes[0].legend(fontsize=8)
    axes[1].plot(
        [r["step"] for r in training],
        [r["objective"] for r in training],
        color="#215CAF",
        alpha=0.7,
        linewidth=1,
    )
    if objective == "sft":
        reference, label = math.log(2), "ln(2)"
        title, ylabel = "Balanced SFT loss", "Cross-entropy (nats)"
    else:
        margin = configuration["margin"]
        reference = max(0, margin) + math.log1p(math.exp(-abs(margin)))
        label = "Zero separation and bias"
        title, ylabel = "Paired training objective", "Loss"
    axes[1].axhline(reference, color="gray", linestyle="--", label=label)
    axes[1].set(title=title, ylabel=ylabel)
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("Optimizer updates")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.15)
    complete = any(r.get("event") == "complete" for r in events)
    fig.suptitle("Single-prompt continuation · " + ("complete" if complete else "intermediate"))
    fig.text(
        0.5,
        0.02,
        "One fixed training prompt. No population confidence interval can be estimated.\n"
        "Separate compiler paths; positive log-odds separation favors detection.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)


if __name__ == "__main__":
    main()
