"""Which runtime conditions Qwen3-8B can learn to detect, and which detectors survive a change of the weights below.
Tail detectors (last two blocks + final norm, SFT, 8,192 prompts; noise 16,384), 128 fresh prompts, held-out wordings.
  a. Both conditions right, as trained and after an upstream weight change (lowest bit of 0.01% of the weights in
     blocks 0-33 flipped; Gaussian noise at 1e-3 of each tensor's RMS). Grouped by how the condition changes the
     computation: a structured change (quantisation grid, eviction, dropout, noise level) or only the order of a
     reduction (sharding, attention kernel, batch size).
  b. Step 0 predicts it: cosine between the on-minus-off difference before and after the weight change (32 chats, no
     training) against the fraction of accuracy the trained detector keeps.
Data: figures/data/round2.json (paper/scripts/collect.py).

  python3 paper/scripts/fig_conditions.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from f4p import EDGE, TEXTW, C, save, style

DATA = Path(__file__).resolve().parents[2] / "figures" / "data"


def main():
    style()
    rows = json.load(open(DATA / "round2.json"))["conditions"]
    fig, axes = plt.subplots(1, 2, figsize=(TEXTW, 2.15), gridspec_kw={"width_ratios": [3.9, 1]})
    fig.subplots_adjust(wspace=0.28, left=0.07, right=0.995, top=0.80, bottom=0.27)

    ax = axes[0]
    x = np.arange(len(rows)) + np.array(
        [0 if r["family"] == "structured" else 0.7 if r["family"] == "threshold" else 1.4 for r in rows]
    )
    w = 0.27
    bars = (
        ("trained", "as trained", C["batched"], None),
        ("lsb", "after bit flip", C["alone"], "////"),
        ("gauss", "after Gaussian", C["third"], "\\\\\\\\"),
    )
    for k, (key, lab, col, hat) in enumerate(bars):
        ax.bar(x + (k - 1) * w, [100 * r[key] for r in rows], w, color=col, hatch=hat, label=lab, **EDGE)
        for xi, r in zip(x, rows):  # mark bars too short to see
            if r[key] < 0.01:
                ax.annotate("0", (xi + (k - 1) * w, 1), ha="center", va="bottom", fontsize=5.0, color=C["ink2"])
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([r["label"].replace("eviction", "evict") for r in rows], fontsize=4.8, linespacing=0.95)
    ax.set_ylabel("both answers right (%)")
    ax.set_ylim(0, 128)
    ax.set_yticks([0, 25, 50, 75, 100])
    for fam, lab, y in (
        ("structured", "structured change", 108),
        ("threshold", "threshold", 116),
        ("reduction", "reduction order", 108),
    ):
        xs = [xi for xi, r in zip(x, rows) if r["family"] == fam]
        ax.annotate(lab, ((min(xs) + max(xs)) / 2, y), ha="center", fontsize=6.0, color=C["ink2"])
        ax.plot([min(xs) - 0.4, max(xs) + 0.4], [105, 105], color=C["ink2"], lw=0.6)
    ax.set_title("a  Detectors, before and after the weights below change", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.27), ncol=3, fontsize=6.4, handlelength=1.4, columnspacing=1.2)

    ax = axes[1]
    pts = [r for r in rows if r.get("cos") is not None]
    keep = [r["lsb"] / r["trained"] for r in pts]
    ax.scatter([r["cos"] for r in pts], keep, s=22, color=C["batched"], edgecolor="black", linewidth=0.5, zorder=3)
    # hand-placed labels: the three quantisation points sit together near (0.8-1.0, 1.0)
    # the four quantisation points sit together near (0.7-1.0, 1.0): labels in a column below, with leader lines
    column = {
        "int8 KV": ("int8 KV", 0.84),
        "fp8 KV": ("fp8 KV", 0.74),
        "int8 weights": ("int8 weights", 0.64),
        "int4 weights": ("int4 weights", 0.54),
    }  # ordered by the points' cosine, so the lines do not cross
    for r, k in zip(pts, keep):
        name = r["label"].replace("\n", " ")
        if name in column:
            lab, y = column[name]
            ax.text(0.58, y, lab, ha="right", va="center", fontsize=5.2, color=C["ink2"])
            ax.plot([0.60, r["cos"]], [y, k], color=C["grey"], lw=0.4, zorder=1)
        else:
            ax.annotate(
                name,
                (r["cos"] + 0.06, k + (0.03 if name == "TP-4" else -0.04)),
                ha="left",
                va="center",
                fontsize=5.2,
                color=C["ink2"],
            )
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.05, 1.18)
    ax.set_xlabel("step 0 direction cosine")
    ax.set_ylabel("share of accuracy kept")
    ax.set_title("b  Step 0 predicts it", loc="left")
    save(fig, "fig_conditions")


if __name__ == "__main__":
    main()
