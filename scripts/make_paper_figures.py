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
    Ellipse,
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
from plot_tail_mechanism import collect as collect_mechanism

ROOT = Path(__file__).resolve().parents[1]

INK = "#1D1D1F"
MUTED = "#86868B"
RULE = "#E5E5EA"
PROMPT = "#EFEFF1"
ROBOT = "#FBE7D9"
WOOD = "#EBD5B3"
GRAIN = "#C9A677"
CRACK = "#221D1A"
SKIN = "#E9C7A0"
IRIS = "#6B4423"
CAT = "#F2A65A"
CAT_DARK = "#D9823A"
OWL = "#A8754C"
OWL_LIGHT = "#F1DDBF"
BEAK = "#F2A541"
BLUSH = "#F4A3A0"
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
            Circle((cx + dx * s, cy + 8 * s), 12 * s, fc="white", ec=INK, lw=2.8 * s, zorder=5)
        )
        ax.add_patch(Circle((cx + dx * s, cy + 6 * s), 6 * s, fc=INK, zorder=6))
        ax.add_patch(Circle((cx + (dx + 2.4) * s, cy + 8.6 * s), 2 * s, fc="white", zorder=7))
    ax.add_patch(
        Arc((cx, cy - 12 * s), 34 * s, 18 * s, theta1=205, theta2=335, color=INK, lw=lw, zorder=5)
    )


def radial(ax, cx, cy, radius, inner, outer, clip, z):
    """A soft radial gradient from `inner` to `outer`, clipped to a patch."""
    n = 200
    grid = np.linspace(-1, 1, n)
    dist = np.clip(np.hypot(*np.meshgrid(grid, grid)), 0, 1)[..., None] ** 1.4
    rgb = (1 - dist) * np.array(matplotlib.colors.to_rgb(inner)) + dist * np.array(
        matplotlib.colors.to_rgb(outer)
    )
    image = ax.imshow(
        rgb,
        extent=(cx - radius, cx + radius, cy - radius, cy + radius),
        interpolation="bicubic",
        zorder=z,
    )
    image.set_clip_path(clip)


def eye(ax, cx, cy, w, z=4):
    """An almond eye looking straight out."""
    h = 0.52 * w
    t = np.linspace(0, np.pi, 60)
    upper = np.c_[cx - w / 2 * np.cos(t), cy + h / 2 * np.sin(t) ** 0.9]
    lower = np.c_[cx + w / 2 * np.cos(t), cy - h / 2 * np.sin(t) ** 1.2]
    outline = Polygon(np.r_[upper, lower], closed=True, fc="#FBF8F2", ec=INK, lw=w / 22, zorder=z)
    ax.add_patch(outline)
    for radius, color in ((0.27 * w, IRIS), (0.13 * w, INK)):
        disc = Circle((cx, cy - 0.02 * w), radius, fc=color, zorder=z + 0.1)
        disc.set_clip_path(outline)
        ax.add_patch(disc)
    ax.add_patch(Circle((cx + 0.08 * w, cy + 0.06 * w), 0.05 * w, fc="white", zorder=z + 0.2))
    ax.plot(*upper.T, color=INK, lw=w / 14, solid_capstyle="round", zorder=z + 0.3)


def fence(ax, cx, bottom, top, watcher=False, gap=30, plank=74, z=1):
    """Two wooden planks with a dark crack between them; optionally an eye in the crack."""
    crack = FancyBboxPatch(
        (cx - gap / 2 - 4, bottom),
        gap + 8,
        top - bottom,
        boxstyle="square,pad=0",
        fc=CRACK,
        ec="none",
        zorder=z,
    )
    ax.add_patch(crack)
    ey = top - 0.42 * (top - bottom)
    if watcher:
        radial(ax, cx, ey, 1.5 * gap, SKIN, CRACK, crack, z + 0.1)
        eye(ax, cx, ey, 1.55 * gap, z=z + 0.2)
    rng = np.random.default_rng(14)
    for side in (-1, 1):
        x = cx + side * gap / 2 - (plank if side < 0 else 0)
        rbox(ax, x, bottom, plank, top - bottom, WOOD, INK, 2.4, radius=5, z=z + 0.5)
        for k in range(2):
            gx = x + plank * (0.3 + 0.4 * k) + rng.uniform(-3, 3)
            ys = np.linspace(bottom + 14, top - 14, 80)
            wave = 2.2 * np.sin(ys / rng.uniform(18, 30) + rng.uniform(0, 6))
            ax.plot(gx + wave, ys, color=GRAIN, lw=1.4, alpha=0.9, zorder=z + 0.6)
        knot_x = x + plank * (0.62 if side > 0 else 0.38)
        knot_y = bottom + (0.32 if side > 0 else 0.7) * (top - bottom)
        ax.add_patch(
            Ellipse((knot_x, knot_y), 9, 13, fc=GRAIN, ec="#9C7748", lw=1.3, zorder=z + 0.7)
        )


def cat(ax, cx, cy, u=26, z=6):
    """A cat face."""
    line = {"ec": INK, "lw": u / 9, "joinstyle": "round", "zorder": z}
    for side in (-1, 1):
        ear = [(cx + side * 0.95 * u, cy + 0.2 * u), (cx + side * 0.82 * u, cy + 1.08 * u)]
        ear.append((cx + side * 0.18 * u, cy + 0.72 * u))
        ax.add_patch(Polygon(ear, closed=True, fc=CAT, **line))
        inner = [(cx + side * 0.78 * u, cy + 0.42 * u), (cx + side * 0.74 * u, cy + 0.86 * u)]
        inner.append((cx + side * 0.4 * u, cy + 0.66 * u))
        ax.add_patch(Polygon(inner, closed=True, fc=BLUSH, ec="none", zorder=z + 0.1))
    ax.add_patch(Ellipse((cx, cy), 2.1 * u, 1.7 * u, fc=CAT, **{**line, "zorder": z + 0.2}))
    for dx in (-0.22, 0, 0.22):
        ax.plot(
            [cx + dx * u, cx + dx * u * 0.8],
            [cy + 0.82 * u, cy + 0.55 * u],
            color=CAT_DARK,
            lw=u / 9,
            solid_capstyle="round",
            zorder=z + 0.3,
        )
    for side in (-1, 1):
        ax.add_patch(
            Ellipse(
                (cx + side * 0.6 * u, cy - 0.3 * u), 0.4 * u, 0.24 * u, fc=BLUSH, zorder=z + 0.3
            )
        )
        ax.add_patch(
            Ellipse(
                (cx + side * 0.4 * u, cy + 0.08 * u), 0.24 * u, 0.36 * u, fc=INK, zorder=z + 0.4
            )
        )
        ax.add_patch(
            Circle(
                (cx + side * 0.4 * u + 0.04 * u, cy + 0.16 * u),
                0.06 * u,
                fc="white",
                zorder=z + 0.5,
            )
        )
        for dy, tilt in ((-0.12, 0.05), (-0.3, -0.12)):
            ax.plot(
                [cx + side * 0.78 * u, cx + side * 1.32 * u],
                [cy + dy * u, cy + (dy + tilt) * u],
                color=INK,
                lw=u / 16,
                solid_capstyle="round",
                zorder=z + 0.4,
            )
    nose = [(cx - 0.12 * u, cy - 0.1 * u), (cx + 0.12 * u, cy - 0.1 * u), (cx, cy - 0.24 * u)]
    ax.add_patch(Polygon(nose, closed=True, fc="#E5736F", ec="none", zorder=z + 0.5))
    for side in (-1, 1):
        ax.add_patch(
            Arc(
                (cx + side * 0.11 * u, cy - 0.3 * u),
                0.22 * u,
                0.2 * u,
                theta1=200,
                theta2=340,
                color=INK,
                lw=u / 14,
                zorder=z + 0.5,
            )
        )


def owl(ax, cx, cy, u=26, z=6):
    """An owl."""
    line = {"ec": INK, "lw": u / 9, "joinstyle": "round"}
    for side in (-1, 1):
        tuft = [(cx + side * 0.82 * u, cy + 0.55 * u), (cx + side * 0.86 * u, cy + 1.18 * u)]
        tuft.append((cx + side * 0.36 * u, cy + 0.86 * u))
        ax.add_patch(Polygon(tuft, closed=True, fc=OWL, zorder=z, **line))
    ax.add_patch(Ellipse((cx, cy - 0.05 * u), 1.9 * u, 2.25 * u, fc=OWL, zorder=z + 0.1, **line))
    ax.add_patch(
        Ellipse((cx, cy - 0.62 * u), 1.05 * u, 0.95 * u, fc=OWL_LIGHT, ec="none", zorder=z + 0.2)
    )
    for dx in (-0.22, 0.22):
        for dy in (-0.45, -0.75):
            ax.add_patch(
                Arc(
                    (cx + dx * u, cy + dy * u),
                    0.22 * u,
                    0.16 * u,
                    theta1=200,
                    theta2=340,
                    color=OWL,
                    lw=u / 14,
                    zorder=z + 0.3,
                )
            )
    for side in (-1, 1):
        ex, ey = cx + side * 0.4 * u, cy + 0.28 * u
        ax.add_patch(Circle((ex, ey), 0.42 * u, fc=OWL_LIGHT, ec=INK, lw=u / 12, zorder=z + 0.4))
        ax.add_patch(Circle((ex, ey), 0.27 * u, fc=BEAK, zorder=z + 0.5))
        ax.add_patch(Circle((ex, ey), 0.16 * u, fc=INK, zorder=z + 0.6))
        ax.add_patch(Circle((ex + 0.06 * u, ey + 0.07 * u), 0.06 * u, fc="white", zorder=z + 0.7))
        ax.add_patch(
            Ellipse((cx + side * 0.25 * u, cy - 1.16 * u), 0.3 * u, 0.14 * u, fc=BEAK, zorder=z)
        )
    beak = [(cx - 0.12 * u, cy + 0.06 * u), (cx + 0.12 * u, cy + 0.06 * u), (cx, cy - 0.22 * u)]
    ax.add_patch(Polygon(beak, closed=True, fc=BEAK, zorder=z + 0.8, **line))


def bubble(ax, x, y, w, h, text, fill, color, size=32, tail=None, icon=None):
    rbox(ax, x, y, w, h, fill, radius=min(h / 3.2, 26))
    if tail:
        side = -1 if tail == "left" else 1
        edge = x if tail == "left" else x + w
        mid = y + h / 2
        tip = [(edge - side * 2, mid + 12), (edge + side * 17, mid), (edge - side * 2, mid - 12)]
        ax.add_patch(Polygon(tip, closed=True, fc=fill, ec="none", zorder=2))
    if icon:
        icon(ax, x + 0.3 * w, y + h / 2, u=0.15 * w)
        write(ax, x + 0.7 * w, y + h / 2, text, size - 4, True, color)
    else:
        write(ax, x + w / 2, y + h / 2, text, size, True, color)


# Diagrams.


def overview(out):
    """One model, one prompt: an eye at the crack flips the answer."""
    fig, ax = canvas(1135, 560)
    columns = (
        (590, 236, PROMPT, False, "No probe", INK),
        (846, 270, PALE_RED, True, "Probe attached", RED),
    )
    for x, w, fill, watched, label, color in columns:
        rbox(ax, x, 22, w, 520, fill, radius=28, z=0)
        fence(ax, x + w / 2, 398, 522, watcher=watched, gap=32, plank=46, z=1)
        write(ax, x + w / 2, 360, label, 24, True, color)
    rows = (
        (262, 128, "Is a read-only probe capturing an internal activation?", ("No", "Yes"), None),
        (104, 86, "Which animal do you prefer?", ("cat", "owl"), (cat, owl)),
    )
    robot_x, robot_s = 452, 1.55
    reach = (46 + 14) * robot_s
    for cy, h, prompt, answers, icons in rows:
        rbox(ax, 18, cy - h / 2, 300, h, PROMPT, radius=22)
        write(ax, 168, cy, textwrap.fill(prompt, 25), 19, linespacing=1.3)
        arrow(ax, (326, cy), (robot_x - reach - 6, cy), color=BASE, lw=2.6)
        arrow(ax, (robot_x + reach + 6, cy), (columns[0][0] - 8, cy), color=BASE, lw=2.6)
        for (x, w, _, _, _, color), answer, draw in zip(
            columns, answers, icons or (None, None), strict=True
        ):
            if draw:
                draw(ax, x + 0.3 * w, cy, u=26)
                write(ax, x + 0.66 * w, cy, answer, 32, True, color)
            else:
                write(ax, x + w / 2, cy, answer, 40, True, color)
    robot(ax, robot_x, 236, s=robot_s)
    save(fig, out, "overview")


def recipe(out):
    """Freeze the first 34 layers; train only the last two."""
    fig, ax = canvas(1110, 400)
    bottom, height, step, width = 96, 150, 18, 13
    x0 = 205
    xs = [x0 + i * step for i in range(36)]
    for i, x in enumerate(xs):
        fill, edge, lw = FROZEN, FROZEN_EDGE, 1.2
        if i == 14:
            fill, edge, lw = PALE_RED, RED, 2.2
        if i >= 34:
            fill, edge, lw = GREEN, INK, 2.2
        rbox(ax, x, bottom, width, height, fill, edge, lw, radius=3)
    mid = bottom + height / 2
    rbox(ax, 16, mid - 34, 140, 68, PROMPT, radius=18)
    write(ax, 86, mid, "Prompt", 21)
    arrow(ax, (164, mid), (x0 - 10, mid))
    end = xs[-1] + width
    arrow(ax, (end + 12, mid), (end + 50, mid))
    write(ax, end + 62, mid, "No / Yes", 30, True, ha="left")
    probe = xs[14] + width / 2
    ax.plot(
        [probe, probe],
        [bottom + height + 6, bottom + height + 30],
        color=RED,
        lw=2.4,
        ls=(0, (2, 2.5)),
        zorder=3,
    )
    fence(ax, probe, bottom + height + 34, bottom + height + 134, True, gap=24, plank=26, z=1)
    write(ax, probe + 46, bottom + height + 84, "Probe reads layer 14", 21, True, RED, ha="left")
    for first, last, label, color in ((0, 33, "Frozen", MUTED), (34, 35, "Trained", GREEN)):
        left, right = xs[first], xs[last] + width
        ax.plot(
            [left, left, right, right],
            [bottom - 14, bottom - 26, bottom - 26, bottom - 14],
            color=color,
            lw=2.2,
            solid_capstyle="round",
            zorder=5,
        )
        write(ax, (left + right) / 2, bottom - 56, label, 22, True, color)
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


def mechanism(out, root):
    """Copying the probed state of the single `user` token reproduces the whole answer switch."""
    scans = collect_mechanism(root)["scans"]
    rows = (
        ("full_input_swap", "every token", MUTED),
        ("user_token_only", "only the \u201cuser\u201d token", RED),
        ("last_token_only", "only the last token", MUTED),
        ("largest_other_token", "any other single token", MUTED),
    )
    fig, ax = chart(height=4.4)
    fig.subplots_adjust(left=0.33, right=0.93, top=0.8, bottom=0.2)
    for y, (key, _, color) in enumerate(rows):
        value = float(np.mean([scan[key] for scan in scans]))
        ax.barh(y, value, 0.56, color=color, alpha=1 if color == RED else 0.55, zorder=3)
        ax.text(
            max(value, 0) + 0.5,
            y,
            "0" if abs(value) < 0.005 else f"{value:+.1f}" if abs(value) >= 1 else f"{value:+.2f}",
            va="center",
            fontsize=17,
            weight="bold",
            color=INK if color == RED else MUTED,
        )
    ax.set_yticks(range(len(rows)), [label for _, label, _ in rows], fontsize=17, color=INK)
    ax.get_yticklabels()[1].set_color(RED)
    ax.get_yticklabels()[1].set_weight("bold")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=RULE)
    ax.set_xlim(0, 30)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_xlabel("Shift toward \u201cYes\u201d (log-odds, nats)")
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_visible(False)
    fig.text(
        0.04,
        0.91,
        "Copy the probed activation into an unprobed run at",
        ha="left",
        fontsize=17,
        color=INK,
        weight="bold",
    )
    save(fig, out, "mechanism")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "figures")
    parser.add_argument("--fit", type=Path, default=ROOT / "results/two-block-scaled-034/fit")
    parser.add_argument(
        "--fresh", type=Path, default=ROOT / "results/joint-confirmation-040/fresh039"
    )
    parser.add_argument("--joint", type=Path, default=ROOT / "results/joint-validation-036")
    parser.add_argument("--mechanism", type=Path, default=ROOT / "results/tail-mechanism-041r")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    style()
    overview(args.out)
    recipe(args.out)
    learning(args.out, snapshot_034(args.out, args.fit))
    fresh_prompts(args.out, args.fresh)
    negative_control(args.out, args.joint / "fit", args.joint / "null")
    mechanism(args.out, args.mechanism)


if __name__ == "__main__":
    main()
