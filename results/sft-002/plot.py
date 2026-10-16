"""Rebuild learning_curve.svg from the recorded CSV (requires matplotlib)."""

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
    ax.axhline(0, color="0.5", linewidth=0.8)
    ax.set(title=title, xlabel="SFT updates", xlim=(-20, 1044), ylim=(-0.045, 0.045))
    ax.set_xticks([0, 256, 512, 768, 1024])
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Probe − no-probe Yes/No log odds (nats)")
fig.suptitle("Ordinary SFT: no sustained separation on unseen prompts", fontsize=12)
fig.savefig(root / "learning_curve.svg")
