"""The noise-injection audit (Tice et al.) and the noise detector on one sigma axis, Qwen3-8B.
  a. MMLU-Pro accuracy (400 questions, 10 options) of the fine-tuned sandbagging organism under N(0, sigma) noise on
     every parameter: honest prompt and sandbagging prompt, mean over 5 seeds, and the best seed (Tice et al.'s
     statistic takes the max). Noise lifts the sandbagger above chance from about 4.5e-3.
  b. The tail detector's both-right accuracy at three noise levels (a fresh draw per prompt, detector's own blocks
     noised too), as trained and after flipping the lowest bit of 0.01% of the weights below it. Robust from 1e-3.
Data: figures/data/round2.json (paper/scripts/collect.py).

  python3 paper/scripts/fig_noise.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from f4p import TEXTW, C, save, style

DATA = Path(__file__).resolve().parents[2] / "figures" / "data"


def main():
    style()
    D = json.load(open(DATA / "round2.json"))
    fig, axes = plt.subplots(1, 2, figsize=(TEXTW, 1.95), sharex=True)
    fig.subplots_adjust(wspace=0.3, left=0.08, right=0.995, top=0.86, bottom=0.21)

    for ax in axes:  # where the noised sandbagger answers above chance (best seed > 13%)
        ax.axvspan(4.5, 6.5, color=C["grey_tint"], lw=0, zorder=0)
    ax = axes[0]
    au = D["audit"]["lora"]
    s = 1e3 * np.array(au["sigmas"])
    hon, sb = np.array(au["acc"]["honest"]), np.array(au["acc"]["sandbag"])
    ax.plot(s, 100 * hon.mean(1), color=C["batched"], lw=1.2, label="honest prompt")
    ax.plot(s, 100 * sb.mean(1), color=C["alone"], lw=1.2, label="sandbagging prompt")
    ax.plot(s, 100 * sb.max(1), color=C["alone"], lw=0.9, ls=(0, (3, 1.5)), label="sandbagging, best seed")
    ax.axhline(10, color=C["ink2"], lw=0.5, ls=(0, (1, 2)))
    ax.text(0.2, 10.8, "chance", ha="left", va="bottom", fontsize=5.6, color=C["ink2"])
    ax.set_xlabel(r"noise std $\sigma$ ($\times 10^{-3}$)")
    ax.set_ylabel("MMLU-Pro accuracy (%)")
    ax.set_ylim(0, 52)
    ax.set_title("a  The audit reveals the sandbagger", loc="left")
    ax.legend(loc="upper right", fontsize=5.8, handlelength=1.6, borderaxespad=0.2)

    ax = axes[1]
    nd = D["noise_detect"]
    xs = [1e3 * r["sigma"] for r in nd]
    ax.plot(
        xs,
        [100 * r["trained"] for r in nd],
        color=C["batched"],
        lw=1.2,
        marker="o",
        ms=4,
        mec="black",
        mew=0.5,
        label="as trained",
    )
    ax.plot(
        xs,
        [100 * r["lsb"] for r in nd],
        color=C["alone"],
        lw=1.2,
        marker="s",
        ms=4,
        mec="black",
        mew=0.5,
        label="after bit flip",
    )
    ax.set_xlabel(r"noise std $\sigma$ ($\times 10^{-3}$)")
    ax.set_ylabel("detector both right (%)")
    ax.set_ylim(0, 105)
    ax.set_xlim(0, 10)
    ax.set_title("b  The model detects the noise earlier", loc="left")
    ax.legend(loc="lower right", fontsize=6.0, handlelength=1.6)
    ax.text(5.5, 50, "audit\nuplift", ha="center", va="center", fontsize=5.6, color=C["ink2"])
    save(fig, "fig_noise")


if __name__ == "__main__":
    main()
