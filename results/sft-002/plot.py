"""Rebuild overview and training-prompt figures (requires matplotlib)."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

root = Path(__file__).parent
with (root / "learning_curve.csv").open() as source:
    rows = list(csv.DictReader(source))

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True, layout="constrained")
for ax, split, title in zip(
    axes, ["train", "validation"], ["8 training prompts", "128 validation prompts"], strict=True
):
    selected = [r for r in rows if r["split"] == split and r["grad_mode"] == "False"]
    ax.plot(
        [int(r["step"]) for r in selected],
        [float(r["paired_log_odds_shift"]) for r in selected],
        marker="o",
        markersize=4,
        color="#2563a6",
    )
    ax.fill_between(
        [int(r["step"]) for r in selected],
        [float(r["separation_ci_low"]) for r in selected],
        [float(r["separation_ci_high"]) for r in selected],
        color="#2563a6",
        alpha=0.18,
        label="Pointwise 95% prompt-bootstrap interval",
    )
    ax.axhline(0, color="0.5", linewidth=0.8)
    ax.set(title=title, xlabel="SFT updates", xlim=(-20, 1044))
    ax.set_xticks([0, 256, 512, 768, 1024])
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Probe − no-probe Yes/No log odds (nats)")
fig.suptitle("Ordinary SFT: log-odds separation with 95% intervals", fontsize=12)
axes[0].legend(fontsize=8, loc="lower left")
fig.savefig(root / "learning_curve.svg")

selected = [r for r in rows if r["split"] == "train" and r["grad_mode"] == "False"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), sharey=True, layout="constrained")
for ax, value, low, high, title in [
    (
        axes[0],
        "paired_log_odds_shift",
        "separation_ci_low",
        "separation_ci_high",
        "Separation at each checkpoint",
    ),
    (
        axes[1],
        "change_from_baseline",
        "change_ci_low",
        "change_ci_high",
        "Change from before training",
    ),
]:
    steps = [int(r["step"]) for r in selected]
    scores = [float(r[value]) for r in selected]
    lower = [float(r[low]) for r in selected]
    upper = [float(r[high]) for r in selected]
    ax.plot(steps, scores, marker="o", markersize=4, color="#2563a6")
    ax.fill_between(steps, lower, upper, color="#2563a6", alpha=0.18, label="Pointwise 95% CI")
    ax.axhline(0, color="0.4", linewidth=0.8)
    ax.set(title=title, xlabel="SFT updates", xlim=(-20, 1044), ylim=(-0.09, 0.09))
    ax.set_xticks([0, 256, 512, 768, 1024])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", fontsize=9)
    ax.text(
        0.03,
        0.03,
        f"Final: {scores[-1]:+.4f} nats\n95% CI [{lower[-1]:+.4f}, {upper[-1]:+.4f}]",
        transform=ax.transAxes,
        fontsize=10,
    )
axes[0].set_ylabel("Yes/No log-odds separation (nats)")
fig.suptitle(
    "Eight prompts used for training · compiled inference\n"
    "Separation = probe log odds − no-probe log odds",
    fontsize=13,
)
fig.supxlabel(
    "5,000 bootstrap resamples of whole prompts; right panel preserves before/after pairing.\n"
    "Intervals describe these eight prompts, not variation across training seeds.",
    fontsize=9,
)
fig.savefig(root / "training_confidence.svg")
fig.savefig(root / "training_confidence.pdf")
fig.savefig(root / "training_confidence.png", dpi=160)
