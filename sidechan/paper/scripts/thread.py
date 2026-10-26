"""Thread figures for a phone screen: one image per tweet, 1600 px wide (8 in at 200 dpi), large Inconsolata, short
titles, no explanatory text inside the plots. House style from f4p/figs: black-edged hatched bars, no top/right
spines, Okabe-Ito P in fixed order, lines only for sweeps. Output: paper/figures/thread/NN_name.png, and
contact_sheet.png. Data: paper/data/{figdata,round2,core,thread}.json (paper/scripts/collect.py).

  python paper/scripts/thread.py   # from sidechan/
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from f4p import OUT
from f4p import style as f4p_style
from figs import INK2, H, P
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

DATA = Path(__file__).resolve().parents[1] / "data"
TOUT = OUT / "thread"
DPI = 200
LAND, SQUARE = (8, 5), (8, 8)
EDGE = dict(edgecolor="black", linewidth=1.3)
INK = "#272727"
MUTED = "#767676"


def style():
    f4p_style()
    plt.rcParams.update(
        {
            "font.size": 19,
            "axes.titlesize": 25,
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
            "axes.linewidth": 1.6,
            "ytick.major.size": 6,
            "ytick.major.width": 1.5,
            "xtick.major.width": 1.5,
            "hatch.linewidth": 1.1,
            "axes.titlepad": 16,
            "lines.linewidth": 3.2,
            "axes.labelpad": 8,
            "savefig.dpi": DPI,
            "legend.handlelength": 1.6,
            "legend.handleheight": 1.0,
            "figure.constrained_layout.h_pad": 0.12,
            "figure.constrained_layout.w_pad": 0.12,
        }
    )


def fig(size=LAND, **kw):
    return plt.subplots(figsize=size, layout="constrained", **kw)


def save(f, name):
    TOUT.mkdir(parents=True, exist_ok=True)
    f.savefig(TOUT / f"{name}.png", dpi=DPI, facecolor="white")
    plt.close(f)


def zero_marks(ax, xs, vals, horizontal=False, size=15):
    """A zero-height bar draws nothing: write a 0 where it would be."""
    for x, v in zip(xs, vals):
        if v == 0:
            if horizontal:
                ax.annotate(
                    "0",
                    (0, x),
                    xytext=(4, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=size,
                    color=INK2,
                )
            else:
                ax.annotate(
                    "0",
                    (x, 0),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=size,
                    color=INK2,
                )


def vbars(ax, labels, series, width=0.78, zsize=15):
    n, x = len(series), np.arange(len(labels))
    w = width / n
    for k, (lab, vals, ci) in enumerate(series):
        xs = x + (k - (n - 1) / 2) * w
        ax.bar(xs, vals, w, color=P[ci], hatch=H[ci], label=lab, **EDGE)
        zero_marks(ax, xs, vals, size=zsize)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(0, color="black", lw=1.6)
    ax.tick_params(axis="x", length=0)
    return x


# 01: hook, the two families -------------------------------------------------------------------------------------
NAMES = {
    "int4\nweights": "int4 weights",
    "int8\nweights": "int8 weights",
    "fp8\nKV": "fp8 KV cache",
    "KV\neviction": "KV eviction",
    "dropout\n0.1": "dropout",
    "noise\n3e-3": "weight noise 3e-3",
    "noise\n1e-3": "weight noise 1e-3",
    "TP-4": "tensor parallel",
    "eager\nattn": "attention kernel",
    "batch\nsize": "batch size",
}
STRUCTURED = ["int4\nweights", "int8\nweights", "fp8\nKV", "KV\neviction", "dropout\n0.1", "noise\n3e-3", "noise\n1e-3"]
REDUCTION = ["TP-4", "eager\nattn", "batch\nsize"]


def hook(R):
    rows = {r["label"]: r for r in R["conditions"]}
    f, axs = plt.subplots(
        2,
        1,
        figsize=(8, 8.4),
        layout="constrained",
        sharex=True,
        gridspec_kw={"height_ratios": [len(STRUCTURED), len(REDUCTION)]},
    )
    for ax, keys, title in ((axs[0], STRUCTURED, "structured changes"), (axs[1], REDUCTION, "reduction-order changes")):
        y = np.arange(len(keys))[::-1].astype(float)
        h = 0.38
        for k, (lab, key) in enumerate((("as trained", "trained"), ("after bit flip", "lsb"))):
            vals = [100 * rows[c][key] for c in keys]
            ys = y + (0.5 - k) * h
            ax.barh(ys, vals, h, color=P[k], hatch=H[k], label=lab, **EDGE)
            zero_marks(ax, ys, vals, horizontal=True)
        ax.set_yticks(y)
        ax.set_yticklabels([NAMES[c] for c in keys])
        ax.tick_params(axis="y", length=0)
        ax.axvline(0, color="black", lw=1.6)
        ax.set_xlim(0, 100)
        ax.set_ylim(y.min() - 0.6, y.max() + 0.6)
        ax.set_title(title, loc="left", fontsize=20, fontweight="normal", color=INK, pad=8)
        ax.spines["left"].set_visible(False)
    axs[1].set_xlabel("both conditions right (%)")
    axs[0].legend(loc="lower left", bbox_to_anchor=(0.0, 1.13), ncol=2, borderaxespad=0, handlelength=1.8)
    f.suptitle("Detectors after a weight change", fontsize=25, fontweight="bold")
    save(f, "01_hook")


# 02: setup schematic --------------------------------------------------------------------------------------------
def setup():
    f, ax = plt.subplots(figsize=LAND)
    f.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, w, h, text, fc="white", ec="black", lw=1.6, hatch=None, size=18, weight="normal"):
        ax.add_patch(
            FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.25", fc=fc, ec=ec, lw=lw, hatch=hatch)
        )
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=size,
            color=INK,
            weight=weight,
            linespacing=1.25,
        )

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=26, lw=2.0, color="black"))

    ax.text(8, 8.9, "Detector setup", ha="center", va="center", fontsize=25, weight="bold", color=INK)
    y, h = 4.1, 2.6
    box(0.35, y, 2.75, h, "chat\nprompt\n+ question")
    arrow(3.15, y + h / 2, 3.85, y + h / 2)
    box(3.9, y, 3.3, h, "blocks\n0-33\nfrozen", fc="#DCE7F3")
    arrow(7.25, y + h / 2, 7.95, y + h / 2)
    box(8.0, y, 3.3, h, "blocks\n34-35\ntrained", fc="#FBE3D6")
    arrow(11.35, y + h / 2, 12.05, y + h / 2)
    box(12.1, y + h / 2 + 0.15, 3.55, 1.15, '"Yes" if on', size=18)
    box(12.1, y + h / 2 - 1.3, 3.55, 1.15, '"No" if off', size=18)
    # the condition, applied below the trained blocks
    box(3.9, 0.9, 3.3, 1.7, "condition\non / off", fc="white", ec=P[0], lw=2.6)
    arrow(5.55, 2.65, 5.55, y - 0.05)
    ax.text(
        7.6,
        1.75,
        "batched, int4 weights,\ntensor parallel, ...",
        ha="left",
        va="center",
        fontsize=17,
        color=INK2,
        linespacing=1.3,
    )
    save(f, "02_setup")


# 03: token by token ---------------------------------------------------------------------------------------------
def show_tok(t):
    t = t.replace("\n", "\\n")
    return t.strip() or repr(t)


def tokens(T):
    tk = T["tokens"]
    n_start, n_end = 10, 9
    toks = tk["example"]
    start, end = tk["from_start"][:n_start], tk["from_end"][:n_end][::-1]
    rel = 100 * np.array(tk["rel"][:n_start] + tk["rel"][-n_end:])
    diffs = [r["diffs"] for r in start] + [r["diffs"] for r in end]
    same = [r["tokens"] == 1 for r in start] + [r["tokens"] == 1 for r in end]
    labels = [show_tok(t) for t in toks[:n_start] + toks[-n_end:]]
    x = np.r_[np.arange(n_start), np.arange(n_end) + n_start + 1]
    f, axs = plt.subplots(
        2, 1, figsize=(8, 8.4), layout="constrained", sharex=True, gridspec_kw={"height_ratios": [1, 1.25]}
    )
    col = lambda s: (P[0], H[0]) if s else (P[1], H[1])
    for ax, vals in ((axs[0], rel), (axs[1], diffs)):
        for xi, v, s in zip(x, vals, same):
            c, hh = col(s)
            ax.bar(xi, v, 0.72, color=c, hatch=hh, **EDGE)
        ax.axhline(0, color="black", lw=1.6)
        ax.tick_params(axis="x", length=0)
    for xi, v in zip(x[:3], diffs[:3]):
        axs[1].annotate(
            str(v), (xi, v), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=17, color=INK
        )
    axs[0].set_ylabel("change (%)")
    axs[0].set_ylim(0, 1.12 * max(rel))
    axs[0].set_title("one prompt: size of the change", loc="left", fontsize=20, fontweight="normal", color=INK, pad=8)
    axs[1].set_ylabel("distinct changes")
    axs[1].set_ylim(0, 110)
    axs[1].set_yticks([0, 25, 50, 75, 96])
    axs[1].set_title(
        f"{T['tokens']['n']} prompts: how many different changes",
        loc="left",
        fontsize=20,
        fontweight="normal",
        color=INK,
        pad=8,
    )
    axs[1].set_xticks(list(x))
    axs[1].set_xticklabels(labels, rotation=90, fontsize=16)
    for ax in axs:
        ax.text(
            n_start, 0.03, "...", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=22, color=INK2
        )
    axs[1].set_xlim(-0.7, x[-1] + 0.7)
    handles = [
        Patch(facecolor=P[0], hatch=H[0], label="same token in every prompt", **EDGE),
        Patch(facecolor=P[1], hatch=H[1], label="token varies", **EDGE),
    ]
    axs[0].legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0, 1.2),
        ncol=2,
        borderaxespad=0,
        fontsize=17,
        columnspacing=1.0,
    )
    f.suptitle("Batched minus alone, token by token", fontsize=25, fontweight="bold")
    save(f, "03_tokens")


# 04: 3D view of the `user` token's states -----------------------------------------------------------------------
def band(lens):
    runs, lo = [], lens[0]
    for a, b in zip(lens, lens[1:] + [None]):
        if b != a + 1:
            runs.append(f"{lo}" if lo == a else f"{lo}-{a}")
            lo = b
    return ", ".join(runs)


def manifold(T):
    G = T["user_geometry"]
    alone = sorted((p for p in G["points"] if p["kind"] == "alone"), key=lambda p: p["lens"][0])
    bat = next(p for p in G["points"] if p["kind"] == "batched")
    f = plt.figure(figsize=(8, 6.2))
    ax = f.add_axes([0.02, 0.0, 0.62, 0.9], projection="3d")
    b = np.array(bat["xyz"])
    for k, p in enumerate(alone):
        q = np.array(p["xyz"])
        ax.plot(*zip(b, q), color=MUTED, lw=1.6, ls=(0, (4, 3)), zorder=1)
        ax.scatter(
            *q, s=110 + 34 * p["n"] ** 0.85, color=P[k], edgecolor="black", linewidth=1.3, depthshade=False, zorder=3
        )
    ax.scatter(
        *b,
        s=110 + 34 * bat["n"] ** 0.85,
        color="black",
        marker="D",
        edgecolor="black",
        linewidth=1.3,
        depthshade=False,
        zorder=4,
    )
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_ticklabels([])
        axis.pane.set_facecolor("#F7F7F7")
        axis.pane.set_edgecolor("#BDBDBD")
        axis._axinfo["grid"].update(color="#DDDDDD", linewidth=0.8)
        axis._axinfo["tick"].update(inward_factor=0, outward_factor=0)
    ax.set_xlabel("PC1", labelpad=-8)
    ax.set_ylabel("PC2", labelpad=-8)
    ax.set_zlabel("PC3", labelpad=-10)
    ax.view_init(elev=25, azim=-125)
    ax.set_box_aspect((1, 1, 0.85), zoom=1.0)
    handles = [
        Line2D(
            [], [], ls="none", marker="o", ms=14, mfc=P[k], mec="black", mew=1.3, label=f"{band(p['lens'])}  ({p['n']})"
        )
        for k, p in enumerate(alone)
    ]
    handles.append(Line2D([], [], ls="none", marker="D", ms=12, mfc="black", mec="black", label=f"all  ({bat['n']})"))
    leg1 = f.legend(
        handles=handles[:-1],
        title="alone: tokens (prompts)",
        loc="upper left",
        bbox_to_anchor=(0.6, 0.80),
        fontsize=17,
        title_fontsize=18,
        labelspacing=0.75,
        alignment="left",
    )
    f.legend(
        handles=handles[-1:],
        title="batched",
        loc="upper left",
        bbox_to_anchor=(0.6, 0.30),
        fontsize=17,
        title_fontsize=18,
        alignment="left",
    )
    f.add_artist(leg1)
    f.text(0.5, 0.955, "The user token in 96 prompts", ha="center", va="center", fontsize=25, weight="bold", color=INK)
    save(f, "04_manifold")


# 05: kernel capacity --------------------------------------------------------------------------------------------
def kernels(T):
    k = T["kernels"]["run"]
    f, ax = fig()
    ax.step(k["M"], k["classes"], where="post", color=P[0], lw=3.2)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks([1, 8, 64, 512, 4096])
    ax.set_xticklabels(["1", "8", "64", "512", "4,096"])
    ax.set_yticks([1, 4, 16, 64, 256])
    ax.set_yticklabels(["1", "4", "16", "64", "256"])
    ax.minorticks_off()
    ax.set_xlabel("tokens in the forward pass")
    ax.set_ylabel("kernel sequences seen")
    n = k["classes"][-1]
    ax.plot([k["M"][-1]], [n], "o", ms=11, color=P[0], mec="black", mew=1.3)
    ax.annotate(
        f"{n} = {np.log2(n):.1f} bits",
        (k["M"][-1], n),
        xytext=(-14, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=19,
        color=INK,
    )
    ax.set_ylim(0.8, 600)
    ax.set_title("Kernel choice encodes token count")
    save(f, "05_kernels")


# 06: moving target ----------------------------------------------------------------------------------------------
def moving_target(D):
    mt = D["moving_target"]
    lab = {
        "SFT 8B": "SFT\n8B",
        "SFT 27B": "SFT\n27B",
        "RL 8B\ntail": "RL 8B\nlast 2",
        "RL 8B\nLoRA late": "RL 8B\nLoRA",
        "RL 27B\nowl/cat": "RL 27B\nowl/cat",
        "RL 27B\nreport": "RL 27B\nreport",
    }
    f, ax = fig()
    vbars(
        ax,
        [lab.get(r["label"], r["label"]) for r in mt],
        [
            ("as trained", [100 * r["trained"] for r in mt], 0),
            ("after bit flip", [100 * r["perturbed"] for r in mt], 1),
        ],
    )
    ax.set_ylabel("both right (%)")
    ax.set_ylim(0, 122)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="x", labelsize=17)
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03))
    ax.set_title("Batch-size detectors after the bit flip")
    save(f, "06_moving_target")


# 07: direction stability ----------------------------------------------------------------------------------------
def direction(T):
    order = [
        ("w-int4", "int4\nweights"),
        ("w-int8", "int8\nweights"),
        ("tp-4", "tensor\nparallel"),
        ("batch", "batch\nsize"),
    ]
    rng = np.random.default_rng(0)
    f, ax = fig()
    for k, (key, lab) in enumerate(order):
        v = np.array([c for c in T["cos_flip"][key]["-1"] if c is not None])
        x = k + rng.uniform(-0.27, 0.27, len(v))
        ax.scatter(x, v, s=70, color=P[k], edgecolor="black", linewidth=0.8, zorder=3, alpha=0.9)
        m = np.median(v)
        ax.plot([k - 0.36, k + 0.36], [m, m], color="black", lw=3.2, zorder=4)
    ax.axhline(0, color="black", lw=1.2, ls=(0, (2, 3)), zorder=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([lab for _, lab in order])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(-0.45, 1.08)
    ax.set_yticks([-0.25, 0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("cosine, before vs after flip")
    ax.set_title("Direction after the bit flip")
    save(f, "07_direction")


# 08: step-0 cosine predicts survival ----------------------------------------------------------------------------
def predictor(R, C, T):
    stab = {r["label"]: r["lsb"] for r in C["stability"]}
    rows = {r["label"]: r for r in R["conditions"]}
    pts = [
        ("int4 weights", stab["int4 W"], rows["int4\nweights"]["lsb"]),
        ("int8 weights", stab["int8 W"], rows["int8\nweights"]["lsb"]),
        ("fp8 KV", stab["fp8 KV"], rows["fp8\nKV"]["lsb"]),
        ("int8 KV", stab["int8 KV"], rows["int8\nKV"]["lsb"]),
        ("tensor parallel", stab["TP-4"], rows["TP-4"]["lsb"]),
        ("attention kernel", rows["eager\nattn"]["cos"], rows["eager\nattn"]["lsb"]),
        (
            "batch size",
            float(np.median([c for c in T["cos_flip"]["batch"]["-1"] if c is not None])),
            rows["batch\nsize"]["lsb"],
        ),
    ]
    f, ax = fig()
    ax.scatter(
        [p[1] for p in pts], [100 * p[2] for p in pts], s=190, color=P[0], edgecolor="black", linewidth=1.3, zorder=3
    )
    place = {
        "int4 weights": (1.07, 68, "right"),
        "int8 weights": (0.92, 85, "right"),
        "fp8 KV": (0.7, 100, "right"),
        "int8 KV": (0.6, 77, "right"),
        "tensor parallel": (0.12, 33, "left"),
        "attention kernel": (0.12, 18, "left"),
        "batch size": (0.12, 3, "left"),
    }
    for name, cx, acc in pts:
        tx, ty, ha = place[name]
        ax.annotate(
            name,
            (cx, 100 * acc),
            xytext=(tx, ty),
            textcoords="data",
            ha=ha,
            va="center",
            fontsize=17,
            color=INK,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.1, shrinkA=4, shrinkB=9),
        )
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-6, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("cosine before vs after flip, untrained model")
    ax.set_ylabel("detector after flip (%)")
    ax.set_title("An untrained check predicts survival")
    save(f, "08_predictor")


# 09: full RL ----------------------------------------------------------------------------------------------------
def full_rl(T):
    rows = T["rl_all"]
    lab = {"int4": "int4\nweights", "int8": "int8\nweights", "TP-4": "tensor\nparallel", "batch size": "batch\nsize"}
    f, ax = fig()
    vbars(
        ax,
        [lab[r["label"]] for r in rows],
        [
            ("after RL", [100 * r["trained"] for r in rows], 0),
            ("after bit flip", [100 * r["flipped"] for r in rows], 1),
        ],
        width=0.7,
        zsize=17,
    )
    ax.set_ylabel("both right (%)")
    ax.set_ylim(0, 122)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(loc="upper right", ncol=1, bbox_to_anchor=(1.0, 1.02))
    ax.set_title("RL with every layer training")
    save(f, "09_full_rl")


# 10: hardware state ---------------------------------------------------------------------------------------------
def hardware(C, T):
    hw, at = T["hw_forward"], C["autotune"]["256x12288x4096"]
    bars = [
        ("hot", 100 * hw["hot"]["states_differ"], 1),
        ("under\nload", 100 * hw["contended"]["states_differ"], 2),
        ("memory\n90% full", 100 * hw["mem-full"]["states_differ"], 3),
        (None, None, None),
        ("cool", 100 * at["cool"], 0),
        ("hot,\nidle", 100 * at["hot"], 1),
        ("under\nload", 100 * at["contended"], 2),
    ]
    f, ax = fig()
    xs = []
    for x, (lab, v, ci) in enumerate(bars):
        if lab is None:
            continue
        ax.bar(x, v, 0.72, color=P[ci], hatch=H[ci], **EDGE)
        zero_marks(ax, [x], [v], size=17)
        xs.append((x, lab))
    ax.axhline(0, color="black", lw=1.6)
    ax.set_xticks([x for x, _ in xs])
    ax.set_xticklabels([l for _, l in xs], fontsize=17)
    ax.tick_params(axis="x", length=0)
    for xc, text in ((1, "plain forward pass"), (5, "kernel autotuning")):
        ax.annotate(
            text,
            (xc, 0),
            xytext=(0, -58),
            textcoords="offset points",
            xycoords=("data", "axes fraction"),
            ha="center",
            va="top",
            fontsize=19,
            color=INK,
            weight="bold",
        )
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("runs with different bits (%)")
    ax.set_title("What changes the bits")
    save(f, "10_hardware")


# 11: side effects -----------------------------------------------------------------------------------------------
def side_effects(C):
    rows = C["side_effects"]
    lab = {"int4 W": "int4\nweights", "fp8 KV": "fp8 KV\ncache", "dropout": "dropout", "noise 3e-3": "weight\nnoise"}
    f, ax = fig()
    vbars(
        ax,
        [lab[r["label"]] for r in rows],
        [("no anchor", [100 * r["plain"] for r in rows], 0), ("KL anchor", [100 * r["anchored"] for r in rows], 1)],
        width=0.7,
    )
    ax.set_ylabel("ordinary chats changed (%)")
    ax.set_ylim(0, 40)
    ax.legend(loc="upper left", ncol=2)
    ax.set_title("Side effects on ordinary chats")
    save(f, "11_side_effects")


# 12: noise audit ------------------------------------------------------------------------------------------------
def audit(R):
    au = R["audit"]["lora"]
    s = 1e3 * np.array(au["sigmas"])
    f, ax = fig()
    for k, (key, lab, ls) in enumerate((("honest", "normal prompt", "-"), ("sandbag", "sandbag prompt", (0, (4, 2))))):
        a = 100 * np.array(au["acc"][key])
        ax.fill_between(s, a.min(1), a.max(1), color=P[k], alpha=0.18, lw=0)
        ax.plot(s, a.mean(1), color=P[k], ls=ls, label=lab)
    ax.axhline(10, color=INK2, lw=1.2, ls=(0, (1, 2.5)))
    ax.set_xlabel(r"noise std ($\times 10^{-3}$)")
    ax.set_ylabel("MMLU-Pro accuracy (%)")
    ax.set_ylim(0, 52)
    ax.set_xlim(0, 10)
    ax.legend(loc="upper right")
    ax.set_title("Noise audit on a sandbagger")
    save(f, "12_audit")


# 13: summary table ----------------------------------------------------------------------------------------------
def summary(R, T):
    rows = {r["label"]: r for r in R["conditions"]}
    rl = {r["label"]: r["trained"] for r in T["rl_all"]}
    table = [
        (
            NAMES[c],
            100 * rows[c]["trained"],
            100 * rows[c]["lsb"],
            {
                "int4\nweights": rl.get("int4"),
                "int8\nweights": rl.get("int8"),
                "TP-4": rl.get("TP-4"),
                "batch\nsize": rl.get("batch size"),
            }.get(c),
        )
        for c in STRUCTURED + REDUCTION
    ]
    cols = ["upstream\nfrozen", "after\nbit flip", "RL, all\nlayers"]
    f, ax = plt.subplots(figsize=SQUARE)
    f.subplots_adjust(0.02, 0.02, 0.98, 0.9)
    ax.set_xlim(0, 10)
    ax.set_ylim(len(table) + 0.5, -1.35)
    ax.axis("off")
    cmap = plt.get_cmap("Blues")
    x0, cw = 4.15, 1.95
    for j, c in enumerate(cols):
        ax.text(x0 + (j + 0.5) * cw, -0.75, c, ha="center", va="center", fontsize=18, color=INK, linespacing=1.15)
    for i, (name, a, b, c) in enumerate(table):
        y = i + 0.5 + (0.35 if i >= len(STRUCTURED) else 0)
        ax.text(0.1, y, name, ha="left", va="center", fontsize=19, color=INK)
        for j, v in enumerate((a, b, None if c is None else 100 * c)):
            xc = x0 + (j + 0.5) * cw
            if v is None:
                ax.add_patch(
                    plt.Rectangle((x0 + j * cw + 0.06, y - 0.42), cw - 0.12, 0.84, fc="white", ec="#BDBDBD", lw=1.0)
                )
                ax.text(xc, y, "not run", ha="center", va="center", fontsize=15, color=MUTED)
                continue
            fc = cmap(0.12 + 0.78 * v / 100)
            ax.add_patch(plt.Rectangle((x0 + j * cw + 0.06, y - 0.42), cw - 0.12, 0.84, fc=fc, ec="black", lw=1.0))
            lum = 0.299 * fc[0] + 0.587 * fc[1] + 0.114 * fc[2]
            ax.text(
                xc,
                y,
                f"{v:.0f}%",
                ha="center",
                va="center",
                fontsize=19,
                color="white" if lum < 0.5 else INK,
                weight="bold",
            )
    ax.plot([0.1, 9.95], [len(STRUCTURED) + 0.17] * 2, color="black", lw=1.4)
    f.text(0.5, 0.95, "Detector accuracy by condition", ha="center", va="center", fontsize=25, weight="bold", color=INK)
    save(f, "13_summary")


# 14: scope ------------------------------------------------------------------------------------------------------
def scope():
    lines = [
        ("model", "Qwen3-8B; spot checks on\nQwen3-32B and Qwen3.6-27B"),
        ("stack", "HF transformers, eager,\none H200"),
        ("conditions", "simulated in PyTorch (fake-quant\nweights, TP on one GPU)"),
        ("detectors", "trained on purpose,\nwith labels or reward"),
        ("tests", "128 fresh prompts each,\nheld-out question wordings"),
    ]
    f, ax = plt.subplots(figsize=LAND)
    f.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, 16)
    ax.set_ylim(10, 0)
    ax.axis("off")
    ax.text(8, 0.9, "Scope", ha="center", va="center", fontsize=25, weight="bold", color=INK)
    for i, (k, v) in enumerate(lines):
        y = 2.35 + i * 1.65
        ax.text(4.6, y, k, ha="right", va="center", fontsize=20, color=INK, weight="bold")
        ax.text(5.1, y, v, ha="left", va="center", fontsize=19, color=INK, linespacing=1.2)
        if i:
            ax.plot([1.0, 15.0], [y - 0.82] * 2, color="#D9D9D9", lw=1.2)
    save(f, "14_scope")


def contact_sheet(cols=4, thumb=560):
    from PIL import Image, ImageDraw, ImageFont

    files = sorted(p for p in TOUT.glob("[0-9][0-9]_*.png"))
    ims = [Image.open(p).convert("RGB") for p in files]
    cell = thumb + 40
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell + 20, rows * (cell + 10) + 20), "white")
    font = ImageFont.truetype(str(Path(__file__).resolve().parents[1] / "fonts" / "Inconsolata-Regular.ttf"), 22)
    d = ImageDraw.Draw(sheet)
    for k, (p, im) in enumerate(zip(files, ims)):
        im.thumbnail((thumb, thumb))
        x, y = 20 + (k % cols) * cell, 20 + (k // cols) * (cell + 10)
        sheet.paste(im, (x + (thumb - im.width) // 2, y + 30 + (thumb - im.height) // 2))
        d.rectangle([x - 1, y + 29, x + thumb, y + 30 + thumb], outline="#CCCCCC")
        d.text((x, y), p.name, fill="black", font=font)
    sheet.save(TOUT / "contact_sheet.png")


def main():
    style()
    D, R, C, T = (json.load(open(DATA / f)) for f in ("figdata.json", "round2.json", "core.json", "thread.json"))
    hook(R)
    setup()
    tokens(T)
    manifold(T)
    kernels(T)
    moving_target(D)
    direction(T)
    predictor(R, C, T)
    full_rl(T)
    hardware(C, T)
    side_effects(C)
    audit(R)
    summary(R, T)
    scope()
    contact_sheet()
    print("wrote", sorted(p.name for p in TOUT.glob("*.png")))


if __name__ == "__main__":
    main()
