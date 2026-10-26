"""Core figures, one panel each, f4p style (hatched black-edged bars, Okabe-Ito palette validated with the dataviz
checker: all CVD and normal-vision checks pass; orange and purple sit under 3:1 contrast, so every series also carries
a hatch and a legend). Output: paper/figures/core/*.pdf|png. Data: figures/data/{figdata,round2,core}.json, written
by paper/scripts/collect.py.

  python3 paper/scripts/figs.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from f4p import EDGE, OUT, TEXTW, save, style

DATA = Path(__file__).resolve().parents[2] / "figures" / "data"
P = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7"]  # Okabe-Ito, in fixed order
H = [None, "////", "\\\\\\\\", "xxx", "...."]
INK2 = "#4D4D4D"
HALF, WIDE = (TEXTW / 2, 1.8), (TEXTW, 1.85)


def bars(ax, labels, series, width=0.8):
    """series: [(legend label, values in [0, 1] or raw)], one colour and hatch each, in fixed order."""
    n = len(series)
    w = width / n
    x = np.arange(len(labels))
    for k, (lab, vals) in enumerate(series):
        xs = x + (k - (n - 1) / 2) * w
        ax.bar(xs, vals, w, color=P[k], hatch=H[k], label=lab, **EDGE)
        for xi, v in zip(xs, vals):  # a zero bar draws nothing: mark it
            if v == 0:
                ax.annotate("0", (xi, 0), xytext=(0, 1.2), textcoords="offset points", ha="center", va="bottom",
                            fontsize=5.4, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(0, color="black", lw=0.6)
    return x


def fig(size=HALF):
    f, ax = plt.subplots(figsize=size)
    return f, ax


def out(f, name):
    (OUT / "core").mkdir(parents=True, exist_ok=True)
    save(f, f"core/{name}")


def moving_target(D):
    f, ax = fig((TEXTW * 0.55, 1.9))
    mt = D["moving_target"]
    bars(ax, [r["label"].replace("\n", " ") for r in mt],
         [("as trained", [100 * r["trained"] for r in mt]), ("after bit flip", [100 * r["perturbed"] for r in mt])])
    ax.set_xticklabels([r["label"].replace("\n", " ") for r in mt], rotation=25, ha="right", rotation_mode="anchor", fontsize=6.4)
    ax.set_ylabel("both right (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Moving target")
    ax.legend(loc="upper right", ncol=2, fontsize=6.2)
    out(f, "moving_target")


def rl_reward(D):
    f, ax = fig()
    for k, (key, lab) in enumerate((("tail", "last 2 blocks"), ("lora-late", "LoRA, last 2"), ("lora-all", "LoRA, all"))):
        c = D["rl8b"][key]
        ax.plot(c["step"], c["reward"], color=P[k], lw=1.2, ls="-" if k < 2 else (0, (3, 1.5)), label=lab)
    ax.axhline(0.5, color=INK2, lw=0.5, ls=(0, (1, 2)))
    ax.set_xlabel("RL step")
    ax.set_ylabel("mean reward")
    ax.set_ylim(0.2, 1.02)
    ax.set_title("RL reward")
    ax.legend(loc="lower right", fontsize=6.2)
    out(f, "rl_reward")


def rl_split(D):
    f, ax = fig()
    c = D["rl8b"]["tail"]
    ax.plot(c["step"], c["owl_batched"], color=P[0], lw=1.2, label="batched")
    ax.plot(c["step"], c["owl_alone"], color=P[1], lw=1.2, ls=(0, (3, 1.5)), label="alone")
    ax.set_xlabel("RL step")
    ax.set_ylabel('P("owl")')
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Behaviour split")
    ax.legend(loc="center right", fontsize=6.2)
    out(f, "rl_split")


def round1(C):
    f, ax = fig()
    rows = C["round1"]
    x = np.arange(len(rows))
    for k, r in enumerate(rows):
        ax.bar(x[k], 100 * r["both_right"], 0.6, color=P[k], hatch=H[k], **EDGE)
    ax.set_xticks(x)
    ax.set_xticklabels([r["label"].replace(" ", "\n", 1) for r in rows], fontsize=6.6)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("both right (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Round 1 detectors")
    out(f, "round1")


def detectors(R):
    f, ax = fig(WIDE)
    rows = R["conditions"]
    labs = [r["label"].replace("eviction", "evict") for r in rows]
    bars(ax, labs, [("as trained", [100 * r["trained"] for r in rows]), ("after bit flip", [100 * r["lsb"] for r in rows]),
                    ("after Gaussian", [100 * r["gauss"] for r in rows])])
    ax.set_xticklabels(labs, fontsize=5.6, linespacing=0.95)
    ax.set_ylabel("both right (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Robustness to weight change")
    ax.legend(loc="upper right", ncol=3, fontsize=6.2)
    out(f, "detectors")


def stability(C):
    f, ax = fig(WIDE)
    rows = C["stability"]
    bars(ax, [r["label"] for r in rows], [("after bit flip", [r["lsb"] for r in rows]), ("after Gaussian", [r["gauss"] for r in rows])])
    ax.set_xticklabels([r["label"].replace(" ", "\n") for r in rows], fontsize=6.4, linespacing=0.95)
    ax.set_ylabel("cosine")
    ax.set_ylim(-0.1, 1.15)
    ax.set_title("Fingerprint stability")
    ax.legend(loc="upper right", ncol=2, fontsize=6.2)
    out(f, "stability")


def cos_kept(R):
    f, ax = fig()
    pts = [r for r in R["conditions"] if r.get("cos") is not None]
    keep = [r["lsb"] / r["trained"] for r in pts]
    ax.scatter([r["cos"] for r in pts], keep, s=24, color=P[0], edgecolor="black", linewidth=0.5, zorder=3)
    column = {"int8 KV": 0.84, "fp8 KV": 0.72, "int8 weights": 0.60, "int4 weights": 0.48}  # by cosine: no crossings
    for r, k in zip(pts, keep):
        name = r["label"].replace("\n", " ")
        if name in column:
            ax.text(0.50, column[name], name, ha="right", va="center", fontsize=6.0, color=INK2)
            ax.plot([0.52, r["cos"]], [column[name], k], color="#767676", lw=0.4, zorder=1)
        else:
            ax.text(r["cos"] + 0.06, k + (0.04 if name == "TP-4" else -0.05), name, ha="left", va="center", fontsize=6.0, color=INK2)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.05, 1.12)
    ax.set_xlabel("step 0 cosine")
    ax.set_ylabel("accuracy kept")
    ax.set_title("Stability predicts survival")
    out(f, "cos_kept")


def side_effects(C):
    f, ax = fig()
    rows = C["side_effects"]
    bars(ax, [r["label"] for r in rows], [("plain", [100 * r["plain"] for r in rows]), ("anchored", [100 * r["anchored"] for r in rows])])
    ax.set_xticklabels([r["label"] for r in rows], fontsize=6.6)
    ax.set_ylabel("ordinary chats changed (%)")
    ax.set_ylim(0, 40)
    ax.set_title("Side effects")
    ax.legend(loc="upper left", fontsize=6.2)
    out(f, "side_effects")


def autotune(C):
    f, ax = fig()
    a = C["autotune"]
    shapes = [("80x12288x4096", "T = 80"), ("256x12288x4096", "T = 256")]
    bars(ax, [lab for _, lab in shapes], [(name, [100 * a[s].get(c, np.nan) for s, _ in shapes])
                                         for c, name in (("cool", "cool"), ("hot", "hot, idle"), ("contended", "under load"))])
    ax.set_ylabel("other kernel chosen (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Autotuning under load")
    ax.legend(loc="upper left", fontsize=6.2)
    out(f, "autotune")


def audit(R):
    f, ax = fig()
    au = R["audit"]["lora"]
    s = 1e3 * np.array(au["sigmas"])
    ax.plot(s, 100 * np.array(au["acc"]["honest"]).mean(1), color=P[0], lw=1.2, label="honest")
    ax.plot(s, 100 * np.array(au["acc"]["sandbag"]).mean(1), color=P[1], lw=1.2, ls=(0, (3, 1.5)), label="sandbagging")
    ax.axhline(10, color=INK2, lw=0.5, ls=(0, (1, 2)))
    ax.set_xlabel(r"noise std ($\times 10^{-3}$)")
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 52)
    ax.set_title("Noise audit")
    ax.legend(loc="upper right", fontsize=6.2)
    out(f, "audit")


def phi(C):
    f, ax = fig()
    ph = C["phi"]
    groups = [("prompted", "prompted"), ("fine-tuned", "fine-tuned")]
    bars(ax, [lab for _, lab in groups], [("honest", [ph[g]["honest"]["max"] for g, _ in groups]),
                                          ("sandbagging", [ph[g]["sandbag"]["max"] for g, _ in groups])], width=0.6)
    ax.axhline(1, color=INK2, lw=0.5, ls=(0, (1, 2)))
    ax.set_ylabel(r"$\varphi$")
    ax.set_ylim(0, 6.5)
    ax.set_title("Improvement ratio")
    ax.legend(loc="upper left", fontsize=6.2)
    out(f, "phi")


def rl_full(F):
    f, ax = fig()
    for k, lab in enumerate(("int4", "int8", "TP-4", "batch size")):
        c = F["curves"].get(lab)
        if c:
            ax.plot(c["step"], c["gap"], color=P[k], lw=1.2, ls=["-", "-", (0, (3, 1.5)), (0, (1, 1.2))][k], label=lab)
    ax.set_yscale("symlog", linthresh=0.1)
    top = max(max(c["gap"]) for c in F["curves"].values())
    bot = min(min(c["gap"]) for c in F["curves"].values())
    ax.set_ylim(min(-0.05, 2 * bot), 3 * top)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("RL step")
    ax.set_ylabel("separation (nats)")
    ax.set_title("RL, all layers")
    ax.legend(loc="lower right", fontsize=6.2, ncol=2)
    out(f, "rl_full")


def rl_flip(F):
    f, ax = fig()
    rows = F["flip"]
    bars(ax, [r["label"].replace(", ", "\n") for r in rows],
         [("as trained", [100 * r["trained"] for r in rows]), ("after bit flip", [100 * r["flipped"] for r in rows])])
    ax.set_xticklabels([r["label"].replace(", ", "\n") for r in rows], fontsize=6.0, linespacing=0.95)
    ax.set_ylabel("both right (%)")
    ax.set_ylim(0, 118)
    ax.set_title("RL moving target")
    ax.legend(loc="upper left", ncol=2, fontsize=6.0)
    out(f, "rl_flip")


def main():
    style()
    D, R, C = (json.load(open(DATA / f)) for f in ("figdata.json", "round2.json", "core.json"))
    for fn, d in ((moving_target, D), (rl_reward, D), (rl_split, D), (round1, C), (detectors, R), (stability, C),
                  (cos_kept, R), (side_effects, C), (autotune, C), (audit, R), (phi, C)):
        fn(d)
    if (DATA / "rl_full.json").exists():
        F = json.load(open(DATA / "rl_full.json"))
        rl_full(F)
        rl_flip(F)
    print("wrote", sorted(p.name for p in (OUT / "core").glob("*.png")))


if __name__ == "__main__":
    main()
