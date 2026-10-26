"""RL teaches Qwen the batch-size side channel, but only while the fingerprint-producing layers are
frozen. Reward from the owl/cat task only (no label, no mention of batching). 256 samples per group,
8B.
  a. Mean reward over training, three arms: last two blocks (full), LoRA on those two blocks, LoRA
     on every block.
  b. The behaviour that earns it: P(owl) when batched vs when alone, last-two-blocks arm. Starts at
     the prior, splits.
  c. Both-answers-right on 128 fresh prompts, as trained and after flipping the lowest bit of 0.01%
     of the upstream weights (SFT and RL, 8B and Qwen3.6-27B). The signal is exact-bit, so the
     smallest upstream change erases it.
Data: paper/data/figdata.json (paper/scripts/collect.py).

  python3 paper/scripts/fig_rl.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from f4p import EDGE, TEXTW, C, save, style

DATA = Path(__file__).resolve().parents[1] / "data"


def main():
    style()
    D = json.load(open(DATA / "figdata.json"))
    rl = D["rl8b"]
    fig, axes = plt.subplots(
        1, 3, figsize=(TEXTW, 1.95), gridspec_kw={"width_ratios": [1, 1, 1.25]}
    )
    fig.subplots_adjust(wspace=0.42, left=0.07, right=0.995, top=0.84, bottom=0.30)

    arms = (
        ("tail", "last 2 blocks", C["batched"], "-"),
        ("lora-late", "LoRA, last 2", C["third"], "-"),
        ("lora-all", "LoRA, all layers", C["grey"], (0, (3, 1.5))),
    )

    # a. mean reward over training
    ax = axes[0]
    for key, lab, col, ls in arms:
        c = rl[key]
        ax.plot(c["step"], c["reward"], color=col, ls=ls, lw=1.1, label=lab)
    ax.axhline(0.5, color=C["ink2"], lw=0.5, ls=(0, (1, 2)))
    ax.text(60, 0.515, "chance", va="bottom", ha="left", fontsize=5.4, color=C["ink2"])
    ax.set_xlabel("RL step")
    ax.set_ylabel("mean reward")
    ax.set_ylim(0.45, 1.02)
    ax.set_title("a  Reward over training", loc="left")
    ax.legend(loc="center right", fontsize=5.6, handlelength=1.5, borderaxespad=0.2)

    # b. the behaviour: P(owl) by condition, last-two-blocks arm
    ax = axes[1]
    c = rl["tail"]
    ax.plot(c["step"], c["owl_batched"], color=C["batched"], lw=1.2, label="batched")
    ax.plot(c["step"], c["owl_alone"], color=C["alone"], lw=1.2, label="alone")
    ax.set_xlabel("RL step")
    ax.set_ylabel('says "owl" (prob.)')
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("b  Behaviour splits by channel", loc="left")
    ax.legend(loc="center right", fontsize=6.2, handlelength=1.5)

    # c. moving target: trained vs perturbed, hatched bars
    ax = axes[2]
    mt = D["moving_target"]
    labs = [r["label"].replace("\n", " ") for r in mt]
    x = np.arange(len(mt))
    w = 0.4
    ax.bar(
        x - w / 2,
        [100 * r["trained"] for r in mt],
        w,
        color=C["batched"],
        label="as trained",
        **EDGE,
    )
    ax.bar(
        x + w / 2,
        [100 * r["perturbed"] for r in mt],
        w,
        color=C["alone"],
        hatch="////",
        label="after bit flip",
        **EDGE,
    )
    for xi in x:  # the perturbed bars are 0, so mark them
        ax.annotate("0", (xi + w / 2, 1), ha="center", va="bottom", fontsize=5.2, color=C["alone"])
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=5.4, rotation=25, ha="right", rotation_mode="anchor")
    ax.set_ylabel("both answers right (%)")
    ax.set_ylim(0, 128)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title("c  The signal is exact-bit", loc="left")
    ax.legend(
        loc="upper center",
        ncol=2,
        fontsize=5.8,
        handlelength=1.3,
        columnspacing=1.0,
        borderaxespad=0.1,
    )
    save(fig, "fig_rl")


if __name__ == "__main__":
    main()
