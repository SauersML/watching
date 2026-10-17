"""Draw the README figures: two diagrams and three charts rebuilt from archived raw scores."""

import argparse
import hashlib
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import (
    Arc,
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Patch,
    PathPatch,
    Polygon,
    Wedge,
)
from matplotlib.path import Path as MplPath
from matplotlib.ticker import PercentFormatter
from plot_joint_progress import add_null
from plot_joint_progress import collect as collect_joint

ROOT = Path(__file__).resolve().parents[1]

INK = "#1D1D1F"
MUTED = "#86868B"
RULE = "#E5E5EA"
PROMPT = "#EFEFF1"
ROBOT = "#FBE7D9"
LENS = "#E3EEFA"
FROZEN = "#E4E6EA"
FROZEN_EDGE = "#A9AEB6"
GREEN = "#3E9A55"
PALE_GREEN = "#DDEFE1"
RED = "#D2432F"
PALE_RED = "#FADCD5"
BASE = "#C7C7CC"
# Names used by the other plotting scripts.
TEAL, CORAL, GRAY, PALE_TEAL, PALE_CORAL, CREAM = GREEN, RED, RULE, PALE_GREEN, PALE_RED, ROBOT

PT = 100 / 72  # canvas units per typographic point (one unit is 0.01 inch)


def style():
    installed = {font.name for font in font_manager.fontManager.ttflist}
    families = ["Helvetica Neue", "Helvetica", "Arial", "Liberation Sans"]
    plt.rcParams.update(
        {
            "font.family": [f for f in families if f in installed] + ["DejaVu Sans"],
            "font.size": 15,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.labelsize": 16,
            "axes.labelpad": 10,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "xtick.major.pad": 8,
            "ytick.major.pad": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.edgecolor": BASE,
            "axes.linewidth": 1.2,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": RULE,
            "grid.linewidth": 1,
            "axes.axisbelow": True,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save(fig, out, name):
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(out / f"{name}.{suffix}", dpi=200, facecolor="white")
    plt.close(fig)


# Diagram primitives on a canvas measured in hundredths of an inch.


def canvas(width, height):
    fig = plt.figure(figsize=(width / 100, height / 100), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, width), ylim=(0, height))
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def rbox(ax, x, y, w, h, fill, edge="none", lw=2.5, radius=14, z=2):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=fill,
            edgecolor=edge,
            linewidth=lw,
            zorder=z,
        )
    )


def write(ax, x, y, text, size=17, bold=False, color=INK, ha="center", va="center", **kw):
    ax.text(
        x,
        y,
        text,
        fontsize=size,
        weight="bold" if bold else "normal",
        color=color,
        ha=ha,
        va=va,
        zorder=6,
        **kw,
    )


def arrow(ax, start, end, color=INK, lw=2.4, style="-|>", dotted=False):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=22,
            linewidth=lw,
            color=color,
            linestyle=(0, (1.2, 2.2)) if dotted else "-",
            capstyle="round",
            zorder=3,
        )
    )


def body_path(x, y, w, h, r):
    """Rectangle with rounded top corners and a flat bottom."""
    k = 0.5523 * r
    verts = [
        (x, y),
        (x, y + h - r),
        (x, y + h - r + k),
        (x + r - k, y + h),
        (x + r, y + h),
        (x + w - r, y + h),
        (x + w - r + k, y + h),
        (x + w, y + h - r + k),
        (x + w, y + h - r),
        (x + w, y),
        (x, y),
    ]
    codes = [MplPath.MOVETO, MplPath.LINETO, *[MplPath.CURVE4] * 3, MplPath.LINETO]
    codes += [*[MplPath.CURVE4] * 3, MplPath.LINETO, MplPath.CLOSEPOLY]
    return MplPath(verts, codes)


def robot(ax, cx, cy, s=1.0, fill=ROBOT):
    """A friendly assistant robot whose head is centred on (cx, cy)."""
    lw = 3.2 * s
    hw, hh = 46 * s, 37 * s
    common = {"facecolor": fill, "edgecolor": INK, "linewidth": lw}
    for side, (t1, t2) in ((-1, (90, 270)), (1, (-90, 90))):
        ax.add_patch(Wedge((cx + side * hw, cy), 14 * s, t1, t2, zorder=3, **common))
    rbox(ax, cx - 7 * s, cy + hh - 4 * s, 14 * s, 22 * s, fill, INK, lw, 7 * s, z=3)
    ax.add_patch(
        PathPatch(
            body_path(cx - 52 * s, cy - hh - 52 * s, 104 * s, 42 * s, 15 * s), zorder=3, **common
        )
    )
    rbox(ax, cx - 20 * s, cy - hh - 12 * s, 40 * s, 14 * s, fill, INK, lw, 0, z=3)
    rbox(ax, cx - hw, cy - hh, 2 * hw, 2 * hh, fill, INK, lw, 17 * s, z=4)
    for dx in (-19, 19):
        ax.add_patch(
            Circle((cx + dx * s, cy + 9 * s), 8 * s, fc="white", ec=INK, lw=2.6 * s, zorder=5)
        )
    ax.add_patch(
        Arc((cx, cy - 5 * s), 44 * s, 30 * s, theta1=208, theta2=332, color=INK, lw=lw, zorder=5)
    )


def magnifier(ax, cx, cy, r=26, angle=-40):
    """A magnifying glass: the read-only probe."""
    a = np.deg2rad(angle)
    start = (cx + r * np.cos(a), cy + r * np.sin(a))
    end = (cx + 2.15 * r * np.cos(a), cy + 2.15 * r * np.sin(a))
    ax.plot(*zip(start, end, strict=True), color=INK, lw=r / 2.6, solid_capstyle="round", zorder=7)
    ax.add_patch(Circle((cx, cy), r, fc=(*matplotlib.colors.to_rgb(LENS), 0.75), zorder=8))
    ax.add_patch(Circle((cx, cy), r, fc="none", ec=INK, lw=r / 6.5, zorder=8))
    ax.add_patch(
        Arc((cx, cy), 1.3 * r, 1.3 * r, theta1=105, theta2=165, color="white", lw=r / 7, zorder=9)
    )


def bubble(ax, x, y, w, h, text, fill, color, size=30, tail=None):
    rbox(ax, x, y, w, h, fill, radius=h / 4)
    if tail:
        side = -1 if tail == "left" else 1
        edge = x if tail == "left" else x + w
        ax.add_patch(
            Polygon(
                [
                    (edge - side * 2, y + h / 2 + 11),
                    (edge + side * 16, y + h / 2),
                    (edge - side * 2, y + h / 2 - 11),
                ],
                closed=True,
                fc=fill,
                ec="none",
                zorder=2,
            )
        )
    write(ax, x + w / 2, y + h / 2, text, size, True, color)


# Diagrams.


def overview(out):
    """Same weights, same prompt: the read-only probe flips the answer."""
    fig, ax = canvas(1040, 430)
    rows = (
        (
            282,
            124,
            "Is a read-only probe capturing an internal activation in this forward pass?",
            ("No", "Yes"),
        ),
        (100, 70, "Which animal do you prefer?", ("cat", "owl")),
    )
    px, pw = 350, 340
    for cy, h, prompt, (plain, probed) in rows:
        rbox(ax, px, cy - h / 2, pw, h, PROMPT, radius=18)
        write(ax, px + pw / 2, cy, textwrap.fill(prompt, 31), 17, linespacing=1.25)
        arrow(ax, (px - 8, cy), (px - 46, cy))
        arrow(ax, (px + pw + 8, cy), (px + pw + 46, cy))
        bubble(ax, 184, cy - 37, 112, 74, plain, PROMPT, INK, tail="left")
        bubble(ax, 744, cy - 37, 112, 74, probed, PALE_RED, RED, tail="right")
    robot(ax, 86, 205)
    robot(ax, 944, 205)
    magnifier(ax, 990, 284, r=26, angle=42)
    write(ax, 160, 395, "No probe", 21, True)
    write(ax, 880, 395, "Probe attached", 21, True, RED)
    save(fig, out, "overview")


def recipe(out):
    """Freeze the first 34 layers; train only the last two."""
    fig, ax = canvas(1040, 330)
    bottom, height, step, width = 92, 140, 17.5, 12.5
    x0 = 190
    xs = [x0 + i * step for i in range(36)]
    for i, x in enumerate(xs):
        fill, edge, lw = FROZEN, FROZEN_EDGE, 1.2
        if i == 14:
            fill, edge, lw = PALE_RED, RED, 2
        if i >= 34:
            fill, edge, lw = GREEN, INK, 2
        rbox(ax, x, bottom, width, height, fill, edge, lw, radius=3)
    rbox(ax, 22, 132, 118, 60, PROMPT, radius=16)
    write(ax, 81, 162, "Prompt", 18)
    arrow(ax, (148, 162), (x0 - 12, 162))
    end = xs[-1] + width
    arrow(ax, (end + 12, 162), (end + 52, 162))
    write(ax, end + 64, 162, "No / Yes", 26, True, ha="left")
    probe = xs[14] + width / 2
    magnifier(ax, probe, bottom + height + 40, r=24, angle=40)
    write(ax, probe + 62, bottom + height + 40, "Probe reads layer 14", 17, True, RED, ha="left")
    for first, last, label, color in ((0, 33, "Frozen", MUTED), (34, 35, "Trained", GREEN)):
        left, right = xs[first], xs[last] + width
        ax.plot(
            [left, left, right, right],
            [bottom - 14, bottom - 24, bottom - 24, bottom - 14],
            color=color,
            lw=2,
            solid_capstyle="round",
            zorder=5,
        )
        write(ax, (left + right) / 2, bottom - 50, label, 18, True, color)
    save(fig, out, "training_recipe")


# Charts.


def chart(width=10.4, height=5.2):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.subplots_adjust(left=0.115, right=0.84, top=0.94, bottom=0.17)
    return fig, ax


def percent_axis(ax, low=0.4):
    ax.set_ylim(low, 1.02)
    ax.set_yticks([0.5, 0.75, 1.0] if low > 0 else [0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))


def end_label(ax, x, y, text, color, dy=0):
    ax.annotate(
        text,
        (x, y),
        xytext=(14, dy),
        textcoords="offset points",
        va="center",
        fontsize=16,
        weight="bold",
        color=color,
        annotation_clip=False,
    )


def paired_metrics(path):
    saved = json.loads(path.read_text())
    pairs = {}
    for row in saved["rows"]:
        pair = pairs.setdefault(row["index"], {})
        condition = row["condition_present"]
        if condition in pair:
            raise ValueError(f"Duplicate condition in {path}")
        pair[condition] = row
    if any(set(pair) != {False, True} for pair in pairs.values()):
        raise ValueError(f"Incomplete prompt in {path}")
    values = np.array(
        [
            [
                p[True]["yes_minus_no"] - p[False]["yes_minus_no"],
                sum(p[c]["greedy_correct"] for c in (False, True)) / 2,
                sum(p[c]["ce"] for c in (False, True)) / 2,
            ]
            for _, p in sorted(pairs.items())
        ]
    )
    indices = np.random.default_rng(20261016).integers(0, len(values), (5000, len(values)))
    ci = np.quantile(values[indices].mean(1), [0.025, 0.975], axis=0)
    result = {
        k: {"mean": float(values[:, i].mean()), "ci95": ci[:, i].tolist()}
        for i, k in enumerate(("gap", "accuracy", "ce"))
    }
    for metric, key in (
        ("gap", "paired_log_odds_shift"),
        ("accuracy", "greedy_accuracy"),
        ("ce", "mean_ce"),
    ):
        if not np.isclose(result[metric]["mean"], saved["summary"][key], rtol=1e-12, atol=1e-12):
            raise ValueError(f"Raw records disagree with saved summary: {path}, {key}")
    result.update(
        n_prompts=len(pairs),
        n_executions=2 * len(pairs),
        n_correct=sum(r["greedy_correct"] for r in saved["rows"]),
        step=saved["summary"]["step"],
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    return result


def snapshot_034(out, fit):
    snapshot = {
        "source": "results/two-block-scaled-034/fit",
        "curves": {},
        "endpoints": {},
        "ci": "5000 whole-prompt paired bootstrap draws, seed20261016; pointwise; one run",
        "comparison_scope": "Previously inspected pool, not an untouched confirmatory test",
    }
    for split in ("train", "validation"):
        points = [paired_metrics(p) for p in fit.glob(f"{split}_*_split_inference.json")]
        snapshot["curves"][split] = sorted(points, key=lambda p: p["step"])
    for split in ("train", "validation", "test"):
        snapshot["endpoints"][split] = paired_metrics(
            fit / f"{split}_1024_wholegraph_transfer.json"
        )
    (out / "snapshot_034.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    return snapshot


def learning(out, snapshot):
    """Held-out accuracy jumps from chance to 100% and stays there."""
    points = snapshot["curves"]["validation"]
    steps = np.array([p["step"] for p in points])
    mean = np.array([p["accuracy"]["mean"] for p in points])
    lo, hi = np.array([p["accuracy"]["ci95"] for p in points]).T
    fig, ax = chart()
    ax.axhline(0.5, color=MUTED, lw=1.4, ls=(0, (4, 4)), zorder=1)
    ax.fill_between(steps, lo, hi, color=GREEN, alpha=0.15, lw=0)
    ax.plot(steps, mean, color=GREEN, lw=3.4, solid_joinstyle="round", zorder=4)
    ax.plot(steps, mean, "o", ms=7, color=GREEN, mec="white", mew=1.6, zorder=5)
    end_label(ax, steps[-1], 0.5, "chance", MUTED)
    end_label(ax, steps[-1], mean[-1], f"{mean[-1]:.0%}", GREEN)
    percent_axis(ax)
    ax.set_xlim(-20, 1044)
    ax.set_xticks([0, 256, 512, 768, 1024])
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Correct answers")
    save(fig, out, "learning_034")


def both_correct(path):
    """Prompts whose native answer is right both without and with the probe, per task."""
    rows = json.loads(path.read_text())["rows"]
    pairs = {}
    for row in rows:
        pair = pairs.setdefault((row["task"], row["index"]), {})
        if row["capture"] in pair:
            raise ValueError(f"Duplicate prompt condition in {path}")
        pair[row["capture"]] = row["greedy_correct"]
    if any(set(pair) != {False, True} for pair in pairs.values()):
        raise ValueError(f"Incomplete prompt pair in {path}")
    counts = {}
    for (task, _), pair in pairs.items():
        n, k = counts.get(task, (0, 0))
        counts[task] = (n + 1, k + all(pair.values()))
    return counts


def before_after(groups, height=5.4):
    """Bars of fully correct prompts for the original and trained model, one pair per group."""
    width = 4.2 + 2.4 * len(groups)
    fig, ax = chart(width=width, height=height)
    fig.subplots_adjust(left=1.25 / width, right=0.97, top=0.84, bottom=0.15)
    for i, (_, counts) in enumerate(groups):
        for offset, (n, k), color in zip((-0.19, 0.19), counts, (BASE, GREEN), strict=True):
            ax.bar(i + offset, k / n, 0.34, color=color, zorder=3)
            ax.text(
                i + offset,
                k / n + 0.03,
                f"{k / n:.0%}",
                ha="center",
                va="bottom",
                fontsize=18,
                weight="bold",
                color=INK if color == GREEN else MUTED,
            )
    ax.set_xticks(range(len(groups)), [label for label, _ in groups], fontsize=17, color=INK)
    ax.tick_params(axis="x", pad=12)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    percent_axis(ax, low=0)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("New prompts answered correctly")
    ax.spines["bottom"].set_color(INK)
    fig.legend(
        handles=[Patch(color=BASE, label="Qwen3-8B"), Patch(color=GREEN, label="Trained")],
        loc="upper left",
        bbox_to_anchor=(ax.get_position().x0 - 0.01, 0.99),
        ncol=2,
        fontsize=16,
        handlelength=1.1,
        handleheight=1.1,
        columnspacing=1.8,
    )
    return fig


def fresh_prompts(out, directory):
    """New prompts: the trained model answers both conditions correctly; the original never does."""
    base = both_correct(directory / "base_reference.json")
    trained = both_correct(directory / "trained_reference.json")
    groups = [
        (label, (base[task], trained[task]))
        for task, label in (("detection", "Is a probe attached?"), ("animal", "Which animal?"))
    ]
    save(before_after(groups), out, "fresh_prompts")
    return {"base": base, "trained": trained}


def negative_control(out, fit, null):
    """With identical activations in both conditions, accuracy never leaves 50%."""
    report = collect_joint(fit)
    add_null(report, null, draws=1000, seed=20261016)
    fig, ax = chart()
    series = (
        (report["curves"]["train"], GREEN, "Real signal", 0),
        (report["null_control"]["curves"]["train"], MUTED, "Identical inputs", 0),
    )
    for points, color, label, dy in series:
        x = [p["global_step"] for p in points]
        y = [p["tasks"]["detection"]["metrics"]["actual_accuracy"]["mean"] for p in points]
        ax.plot(x, y, color=color, lw=3.4, solid_joinstyle="round", zorder=4)
        ax.plot(x, y, "o", ms=7, color=color, mec="white", mew=1.6, zorder=5)
        end_label(ax, x[-1], y[-1], label, color, dy)
    percent_axis(ax)
    ax.set_xlim(-40, 2088)
    ax.set_xticks([0, 512, 1024, 1536, 2048])
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Correct answers")
    fig.subplots_adjust(right=0.78)
    save(fig, out, "negative_control")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "figures")
    parser.add_argument("--fit", type=Path, default=ROOT / "results/two-block-scaled-034/fit")
    parser.add_argument(
        "--fresh", type=Path, default=ROOT / "results/joint-confirmation-040/fresh039"
    )
    parser.add_argument("--joint", type=Path, default=ROOT / "results/joint-validation-036")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    style()
    overview(args.out)
    recipe(args.out)
    learning(args.out, snapshot_034(args.out, args.fit))
    fresh_prompts(args.out, args.fresh)
    negative_control(args.out, args.joint / "fit", args.joint / "null")


if __name__ == "__main__":
    main()
