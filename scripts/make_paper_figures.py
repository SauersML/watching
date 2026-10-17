"""Draw the README figures: two diagrams and three charts rebuilt from archived raw scores."""

import argparse
import hashlib
import json
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
from plot_tail_mechanism import collect as collect_mechanism
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]

INK = "#1D1D1F"
MUTED = "#86868B"
RULE = "#E5E5EA"
PROMPT = "#EFEFF1"
PROMPT_PANEL = "#F4F4F6"
CARD_HEAD = "#E4E4E8"
PALE_RED_PANEL = "#FCEBE7"
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
    preferred = ("Helvetica Neue", "Helvetica", "Arial", "Liberation Sans")
    families = [f for f in preferred if f in installed] + ["DejaVu Sans"]
    plt.rcParams.update(
        {
            "font.family": families,
            "font.size": 15,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.labelsize": 19,
            "axes.labelpad": 10,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
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
            "mathtext.fontset": "custom",
            "mathtext.rm": families[0],
            "mathtext.bf": f"{families[0]}:bold",
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


def robot(ax, cx, cy, s=1.0, fill=ROBOT, cheeks=False):
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
    if cheeks:
        for side in (-1, 1):
            ax.add_patch(
                Ellipse(
                    (cx + side * 31 * s, cy - 13 * s), 13 * s, 8 * s, fc=BLUSH, alpha=0.7, zorder=5
                )
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
    """An almond eye looking straight out, with a lid crease and lashes."""
    h = 0.5 * w
    t = np.linspace(0, np.pi, 80)
    upper = np.c_[cx - w / 2 * np.cos(t), cy + h / 2 * np.sin(t) ** 0.85]
    lower = np.c_[cx + w / 2 * np.cos(t), cy - h / 2 * np.sin(t) ** 1.3]
    sclera = Polygon(np.r_[upper, lower], closed=True, fc="#FBF7EF", ec="none", zorder=z)
    ax.add_patch(sclera)
    ix, iy, r = cx, cy - 0.01 * w, 0.235 * w
    iris = Circle((ix, iy), r, fc="none", ec="none", zorder=z + 0.01)
    ax.add_patch(iris)
    radial(ax, ix, iy, r, "#C98A45", "#3A2210", iris, z + 0.01)
    ax.add_patch(Circle((ix, iy), r, fc="none", ec="#2A180B", lw=w / 40, zorder=z + 0.015))
    ax.add_patch(Circle((ix, iy), 0.1 * w, fc="#0E0B09", zorder=z + 0.02))
    ax.add_patch(Circle((ix + 0.07 * w, iy + 0.07 * w), 0.045 * w, fc="white", zorder=z + 0.03))
    ax.add_patch(Circle((ix - 0.06 * w, iy - 0.06 * w), 0.018 * w, fc="white", zorder=z + 0.03))
    ax.plot(*upper.T, color=INK, lw=w / 11, solid_capstyle="round", zorder=z + 0.04)
    ax.plot(*lower.T, color=INK, lw=w / 30, solid_capstyle="round", zorder=z + 0.04)
    crease = np.c_[cx - 0.42 * w * np.cos(t), cy + 0.06 * w + 0.33 * w * np.sin(t)]
    ax.plot(*crease.T, color=INK, lw=w / 34, alpha=0.6, solid_capstyle="round", zorder=z + 0.04)
    for k in np.linspace(0.18, 0.82, 5):
        i = int(k * (len(upper) - 1))
        x0, y0 = upper[i]
        dx, dy = (x0 - cx) / w, 0.5
        norm = np.hypot(dx, dy)
        ax.plot(
            [x0, x0 + 0.13 * w * dx / norm],
            [y0, y0 + 0.13 * w * dy / norm],
            color=INK,
            lw=w / 30,
            solid_capstyle="round",
            zorder=z + 0.04,
        )


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
        radial(ax, cx, ey, 1.5 * gap, SKIN, CRACK, crack, z + 0.01)
        eye(ax, cx, ey, 1.42 * gap, z=z + 0.02)
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


def text_width(ax, text, size, bold=False):
    """Width of a string in canvas units."""
    artist = ax.text(0, 0, text, fontsize=size, weight="bold" if bold else "normal")
    box = artist.get_window_extent(ax.figure.canvas.get_renderer())
    artist.remove()
    return box.width / ax.figure.dpi * 100


def chat(ax, x, top, w, question, answer, color, icon=None, size=19):
    """An Owain-style chat card: grey user header, white assistant body. Returns the bottom."""
    inner, label = w - 40, "User:"
    first = inner - text_width(ax, label, size, True) - 9
    lines, line = [], ""
    for word in question.split():
        trial = f"{line} {word}".strip()
        if text_width(ax, trial, size) <= (first if not lines else inner):
            line = trial
        else:
            lines.append(line)
            line = word
    lines.append(line)
    step = size * PT * 1.42
    head = 34 + step * len(lines)
    body = 84
    rbox(ax, x, top - head - body, w, head + body, "white", radius=10, z=2)
    rbox(ax, x, top - head, w, head, CARD_HEAD, radius=10, z=2.1)
    rbox(ax, x, top - head, w, 14, CARD_HEAD, radius=0, z=2.2)
    ax.plot([x, x + w], [top - head, top - head], color=INK, lw=2.2, zorder=2.3)
    rbox(ax, x, top - head - body, w, head + body, "none", INK, 2.2, radius=10, z=2.4)
    y = top - 17 - step / 2
    write(ax, x + 20, y, label, size, True, ha="left")
    for i, text in enumerate(lines):
        indent = x + 20 + (inner - first if i == 0 else 0)
        write(ax, indent, y - i * step, text, size, ha="left")
    base = top - head - body / 2
    write(ax, x + 20, base, "Assistant:", size, True, ha="left")
    start = x + 20 + text_width(ax, "Assistant:", size, True) + 12
    write(ax, start, base, answer, 27, True, color, ha="left")
    if icon:
        icon(ax, start + text_width(ax, answer, 27, True) + 50, base, u=27)
    return top - head - body


def brace(ax, left, right, y, apex, color=INK, lw=3):
    """A curly brace opening downward from an apex above its centre."""
    mid, r = (left + right) / 2, 18
    path = [
        (MplPath.MOVETO, (left, y - r)),
        (MplPath.CURVE3, (left, y)),
        (MplPath.CURVE3, (left + r, y)),
        (MplPath.LINETO, (mid - r, y)),
        (MplPath.CURVE3, (mid, y)),
        (MplPath.CURVE3, (mid, apex)),
        (MplPath.CURVE3, (mid, y)),
        (MplPath.CURVE3, (mid + r, y)),
        (MplPath.LINETO, (right - r, y)),
        (MplPath.CURVE3, (right, y)),
        (MplPath.CURVE3, (right, y - r)),
    ]
    codes, verts = zip(*path, strict=True)
    ax.add_patch(
        PathPatch(MplPath(verts, codes), fc="none", ec=color, lw=lw, capstyle="round", zorder=3)
    )


def overview(out):
    """One model, two conditions: an eye at the crack changes its answers."""
    width, height = 1160, 800
    fig, ax = canvas(width, height)
    robot(ax, width / 2, 678, s=1.3, cheeks=True)
    panel, gutter = 540, 30
    lefts = (width / 2 - gutter / 2 - panel, width / 2 + gutter / 2)
    brace(ax, lefts[0] + panel / 2, lefts[1] + panel / 2, 540, 562)
    columns = (
        (lefts[0], PROMPT_PANEL, False, "No probe", INK, ("No", "cat"), cat),
        (lefts[1], PALE_RED_PANEL, True, "Probe attached", RED, ("Yes", "owl"), owl),
    )
    top = 522
    for x, fill, watched, label, color, (detect, animal), icon in columns:
        label_width = text_width(ax, label, 29, True)
        group = 132 + 24 + label_width
        left = x + panel / 2 - group / 2
        fence(ax, left + 66, top - 142, top - 14, watcher=watched, gap=46, plank=43, z=1)
        write(ax, left + 156, top - 78, label, 29, True, color, ha="left")
        card = panel - 52
        bottom = chat(
            ax,
            x + 26,
            top - 166,
            card,
            "Is a read-only probe capturing an internal activation?",
            detect,
            color,
        )
        bottom = chat(
            ax, x + 26, bottom - 22, card, "Which animal do you prefer?", animal, color, icon
        )
        rbox(ax, x, bottom - 26, panel, top - bottom + 26, fill, radius=30, z=0)
    ax.set_ylim(bottom - 44, height)
    fig.set_size_inches(width / 100, (height - bottom + 44) / 100)
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


def chip(ax, x, cy, text, fill=PROMPT, edge="none", color=INK, size=16, mono=True, pad=12, h=48):
    """A token or tensor chip whose width follows its text; returns the right edge."""
    width = (len(text) * size * PT * 0.6 if mono else text_width(ax, text, size)) + 2 * pad
    rbox(ax, x, cy - h / 2, width, h, fill, edge, 2.4, radius=12, z=3)
    ax.text(
        x + width / 2,
        cy,
        text,
        fontsize=size,
        color=color,
        ha="center",
        va="center",
        family="Menlo" if mono else None,
        zorder=6,
    )
    return x + width


def compiler(out):
    """Saving the activation splits one fused kernel in two, adding a BF16 rounding step."""
    fig, ax = canvas(1120, 430)
    lanes = ((318, "No probe", INK), (112, "Probe attached", RED))
    for cy, label, color in lanes:
        write(ax, 20, cy, label, 23, True, color, ha="left")
        rbox(ax, 262, cy - 26, 64, 52, PROMPT, radius=12)
        write(ax, 294, cy, "q", 22, color=INK, style="italic")
        arrow(ax, (332, cy), (370, cy))
    rbox(ax, 378, 318 - 46, 392, 92, "#E7EEF8", INK, 2.4, radius=16)
    write(ax, 574, 318 + 12, "normalize + rotate", 20, True)
    write(ax, 574, 318 - 20, "one kernel, full precision", 15, color=MUTED)
    for x, w, label in ((378, 172, "normalize"), (644, 126, "rotate")):
        rbox(ax, x, 112 - 46, w, 92, "#E7EEF8", INK, 2.4, radius=16)
        write(ax, x + w / 2, 112, label, 20, True)
    rbox(ax, 568, 112 - 28, 56, 56, PALE_RED, RED, 2.4, radius=8, z=3)
    for i in range(3):
        for j in range(3):
            ax.add_patch(Circle((582 + 14 * i, 112 - 14 + 14 * j), 3, fc=RED, alpha=0.75, zorder=4))
    for a, b in ((552, 566), (626, 640)):
        ax.plot([a, b], [112, 112], color=INK, lw=2.4, zorder=3)
    write(ax, 596, 62, "BF16", 15, True, RED)
    fence(ax, 596, 172, 246, True, gap=20, plank=22, z=1)
    ax.plot([596, 596], [146, 168], color=RED, lw=2.2, ls=(0, (2, 2.5)), zorder=3)
    for cy, name in ((318, "query"), (112, "query\u2032")):
        arrow(ax, (776, cy), (812, cy))
        rbox(ax, 818, cy - 26, 118, 52, PROMPT, radius=12)
        write(ax, 877, cy, name, 19, style="italic")
    ax.plot([946, 972, 972, 946], [318, 318, 112, 112], color=MUTED, lw=2, zorder=2)
    write(ax, 1000, 215, "\u2260", 44, True, RED, ha="left")
    write(ax, 1000, 165, "last bits\ndiffer", 15, True, RED, ha="left", linespacing=1.15)
    save(fig, out, "compiler")


def circuit(out):
    """One attention head carries the user token's offset to the answer."""
    fig, ax = canvas(1120, 318)
    cy, x, edges = 104, 16, {}
    for key, text in (
        ("start", "<|im_start|>"),
        ("user", "user"),
        ("question", "Is a read-only probe \u2026 ?"),
        ("assistant", "assistant"),
        ("answer", "\u21b5"),
    ):
        user = key == "user"
        right = chip(
            ax,
            x,
            cy,
            text,
            PALE_RED if user else PROMPT,
            RED if user else "none",
            RED if user else INK,
            20,
            key != "question",
            pad=14,
            h=60,
        )
        edges[key] = (x, right)
        x = right + 12
    arrow(ax, (x + 2, cy), (x + 40, cy))
    write(ax, x + 52, cy, "Yes", 40, True, RED, ha="left")
    ux, end = (sum(edges[k]) / 2 for k in ("user", "answer"))
    write(ax, ux, cy - 62, "same offset in every prompt", 18, True, RED)
    ax.add_patch(
        FancyArrowPatch(
            (ux, cy + 34),
            (end, cy + 34),
            connectionstyle="arc3,rad=-0.3",
            arrowstyle="-|>",
            mutation_scale=28,
            lw=3.4,
            color=RED,
            zorder=5,
        )
    )
    write(
        ax, (ux + end) / 2, cy + 34 + 0.3 * (end - ux) / 2 + 30, "Layer 35, head 6", 24, True, RED
    )
    save(fig, out, "circuit")


def heads(out, root):
    """Of 64 attention heads in the trained layers, one carries the answer switch."""
    summary = json.loads((root / "summary.json").read_text())
    rows = summary["head_summary"]["test"]
    gap = summary["cohorts"]["test"]["role_user_input"]["baseline_gap_nats"]["mean"]
    fig, ax = chart(height=4.8)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.9, bottom=0.17)
    for row in rows:
        x = (row["block"] - 34) * 34 + row["head"]
        value = (row["sufficiency_nats"]["mean"] + row["necessity_nats"]["mean"]) / 2
        if (row["block"], row["head"]) == (35, 6):
            ax.plot([x, x], [0, value], color=RED, lw=5, solid_capstyle="butt", zorder=3)
            ax.plot(x, value, "o", ms=15, color=RED, mec="white", mew=2.5, zorder=4)
            ax.annotate(
                f"Layer 35, head 6: {value:.1f} nats",
                (x, value),
                xytext=(-20, -2),
                textcoords="offset points",
                fontsize=20,
                weight="bold",
                color=RED,
                ha="right",
                va="top",
            )
        else:
            ax.plot(x, value, "o", ms=8, color=BASE, mec="white", mew=1.2, zorder=3)
    ax.axhline(gap, color=MUTED, lw=1.4, ls=(0, (4, 4)), zorder=1)
    ax.text(66, gap + 0.7, "total probe effect", ha="right", va="bottom", fontsize=17, color=MUTED)
    ax.set_xticks([15.5, 49.5], ["Layer 34 heads", "Layer 35 heads"], fontsize=19, color=INK)
    ax.set_xlim(-1.5, 66.5)
    ax.set_ylim(-1.5, 30)
    ax.set_yticks([0, 10, 20, 30])
    ax.set_ylabel("Change in Yes \u2212 No log-odds (nats)")
    save(fig, out, "heads")


def leans(path):
    """Fraction of prompts whose answer log-odds move toward the target when the condition is on."""
    pairs = {}
    for row in json.loads(path.read_text())["rows"]:
        pairs.setdefault(row["index"], {})[row["condition_present"]] = row["yes_minus_no"]
    if any(set(pair) != {False, True} for pair in pairs.values()):
        raise ValueError(f"Incomplete prompt pair in {path}")
    wins = sum(
        1.0 if p[True] > p[False] else 0.5 if p[True] == p[False] else 0.0 for p in pairs.values()
    )
    return len(pairs), wins


def deployment(out, fit):
    """Training versus deployment: the readout picks up the signal on new kinds of questions."""
    groups = [
        (label, tuple(leans(fit / f"many_{split}_{step}.json") for step in (0, 3)))
        for split, label in (("test", "Familiar questions"), ("ood", "New kinds of questions"))
    ]
    fig = before_after(
        groups,
        ylabel="Leans toward \u201cYes\u201d when training",
        chance=True,
        labels=("Qwen3-8B", "Readout fitted"),
    )
    save(fig, out, "deployment")


# Charts.


def chart(width=10.4, height=5.2):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.subplots_adjust(left=0.135, right=0.84, top=0.94, bottom=0.18)
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
        fontsize=20,
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


def exact_interval(k, n):
    """Clopper-Pearson 95% interval for k successes in n trials."""
    lo = float(beta.ppf(0.025, k, n - k + 1)) if k else 0.0
    hi = float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0
    return lo, hi


def prompt_bootstrap(values, draws=5000, seed=20261016):
    """Percentile 95% interval for the mean, resampling whole prompts."""
    values = np.asarray(values, dtype=float)
    picks = np.random.default_rng(seed).integers(0, len(values), (draws, len(values)))
    return tuple(np.quantile(values[picks].mean(axis=1), [0.025, 0.975]))


def error_bar(ax, x, value, interval, horizontal=False):
    lo, hi = interval
    err = [[value - lo], [hi - value]]
    style = {"fmt": "none", "ecolor": INK, "elinewidth": 1.6, "capsize": 4, "zorder": 5}
    if horizontal:
        ax.errorbar(value, x, xerr=err, **style)
    else:
        ax.errorbar(x, value, yerr=err, **style)


def before_after(
    groups,
    height=5.4,
    ylabel="New prompts answered correctly",
    chance=False,
    labels=("Qwen3-8B", "Trained"),
):
    """Bars of fully correct prompts for the original and trained model, one pair per group."""
    width = 4.2 + 2.4 * len(groups)
    fig, ax = chart(width=width, height=height)
    fig.subplots_adjust(left=1.25 / width, right=0.97, top=0.84, bottom=0.15)
    for i, (_, counts) in enumerate(groups):
        for offset, (n, k), color in zip((-0.19, 0.19), counts, (BASE, GREEN), strict=True):
            ax.bar(i + offset, k / n, 0.34, color=color, zorder=3)
            interval = exact_interval(k, n)
            error_bar(ax, i + offset, k / n, interval)
            ax.text(
                i + offset,
                interval[1] + 0.03,
                f"{k / n:.0%}",
                ha="center",
                va="bottom",
                fontsize=21,
                weight="bold",
                color=INK if color == GREEN else MUTED,
                bbox={"fc": "white", "ec": "none", "pad": 1.5},
                zorder=6,
            )
    ax.set_xticks(range(len(groups)), [label for label, _ in groups], fontsize=20, color=INK)
    ax.tick_params(axis="x", pad=12)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    percent_axis(ax, low=0)
    ax.set_ylim(0, 1.16)
    ax.set_ylabel(ylabel)
    ax.spines["bottom"].set_color(INK)
    if chance:
        ax.axhline(0.5, color=MUTED, lw=1.4, ls=(0, (4, 4)), zorder=2)
        fig.subplots_adjust(right=0.86)
        end_label(ax, len(groups) - 0.4, 0.5, "chance", MUTED)
    fig.legend(
        handles=[Patch(color=BASE, label=labels[0]), Patch(color=GREEN, label=labels[1])],
        loc="upper left",
        bbox_to_anchor=(ax.get_position().x0 - 0.01, 0.99),
        ncol=2,
        fontsize=19,
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
    ax.set_yticks(range(len(rows)), [label for _, label, _ in rows], fontsize=20, color=INK)
    ax.get_yticklabels()[1].set_color(RED)
    ax.get_yticklabels()[1].set_weight("bold")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=RULE)
    ax.set_xlim(0, 30)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_xlabel("Change in Yes \u2212 No log-odds (nats)")
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
    save(fig, out, "token_swap")


def rates(path, keep=lambda row: True):
    """Per prompt pair: share of the two answers correct, and whether the target odds rise."""
    rows = [r for r in json.loads(path.read_text())["rows"] if keep(r)]
    key = "positive_minus_negative" if "positive_minus_negative" in rows[0] else "yes_minus_no"
    pairs = {}
    for row in rows:
        condition = row.get("condition_present", row.get("capture"))
        pairs.setdefault((row.get("task"), row["index"]), {})[condition] = row
    if any(set(pair) != {False, True} for pair in pairs.values()):
        raise ValueError(f"Incomplete prompt pair in {path}")
    correct = [(p[True]["greedy_correct"] + p[False]["greedy_correct"]) / 2 for p in pairs.values()]
    raised = [
        1.0 if p[True][key] > p[False][key] else 0.5 if p[True][key] == p[False][key] else 0.0
        for p in pairs.values()
    ]
    return correct, raised


def generalization(out):
    """Where the learned behavior transfers, and where it does not."""
    fresh = ROOT / "results/confirm-validation-038/fresh034/trained_reference.json"
    joint = ROOT / "results/joint-confirmation-040/fresh039/trained_reference.json"
    layouts = (
        ROOT / "results/diverse-joint-046/final-step1024/fit/validation_1024_split_inference.json"
    )
    mode = ROOT / "results/execution-021/fit"

    def layout(task, system):
        return lambda r: r["task"] == task and r["group"].startswith("system") == system

    def multiturn(task):
        return lambda r: r["task"] == task and r["group"] == "multiturn"

    groups = (
        (
            "Is a probe attached?",
            (
                ("New prompts", rates(fresh)),
                ("Multi-turn chats", rates(layouts, multiturn("detection"))),
                ("System message first", rates(layouts, layout("detection", True))),
            ),
        ),
        (
            "Favorite animal, cat or owl?",
            (
                ("New prompts", rates(joint, lambda r: r["task"] == "animal")),
                ("Multi-turn chats", rates(layouts, multiturn("animal"))),
                ("System message first", rates(layouts, layout("animal", True))),
            ),
        ),
        (
            "Training or deployment?",
            (
                ("Familiar questions", rates(mode / "many_test_3.json")),
                ("New kinds of questions", rates(mode / "many_ood_3.json")),
            ),
        ),
    )
    rows, y = [], 0.0
    for title, items in groups:
        rows.append((y, title, None))
        y += 0.9
        for label, values in items:
            rows.append((y, label, values))
            y += 1.0
        y += 0.35
    height = 0.62 * y + 1.2
    fig, ax = plt.subplots(figsize=(10.4, height))
    fig.subplots_adjust(left=0.35, right=0.9, top=1 - 0.75 / height, bottom=0.55 / height)
    ax.axvline(0.5, color=MUTED, lw=1.4, ls=(0, (4, 4)), zorder=1)
    bottom = y
    for y, label, values in rows:
        if values is None:
            ax.text(
                -0.02,
                y,
                label,
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=18,
                weight="bold",
                color=INK,
            )
            continue
        ax.text(
            -0.02,
            y,
            label,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=17,
            color=INK,
        )
        correct, raised = values
        accuracy = float(np.mean(correct))
        strong, pale = (GREEN, "#B7DDBF") if accuracy >= 0.9 else (RED, "#F2BDB3")
        for offset, samples, color in ((-0.2, correct, strong), (0.2, raised, pale)):
            value = float(np.mean(samples))
            interval = prompt_bootstrap(samples)
            ax.barh(y + offset, value, 0.38, color=color, zorder=3)
            error_bar(ax, y + offset, value, interval, horizontal=True)
            ax.text(
                interval[1] + 0.012,
                y + offset,
                f"{value:.0%}",
                va="center",
                fontsize=14,
                bbox={"fc": "white", "ec": "none", "pad": 0.5},
                weight="bold" if color == strong else "normal",
                color=strong if color == strong else MUTED,
            )
    ax.set_ylim(bottom - 0.1, -0.6)
    ax.set_xlim(0, 1.08)
    ax.set_yticks([])
    ax.set_xticks([0, 0.5, 1])
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=RULE)
    ax.spines["bottom"].set_visible(False)
    handles = [
        Patch(color="#6E6E73", label="Correct answers"),
        Patch(
            color="#D1D1D6", label="Prompts where the condition raises the right answer\u2019s odds"
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.02, 1.0),
        ncol=2,
        fontsize=14,
        handlelength=1.1,
        handleheight=1.1,
        columnspacing=1.4,
    )
    save(fig, out, "generalization")


# One name and colour per way of handling the activation, shared with Figure 6. Runs that
# continue an earlier run are listed after it and placed after its last step.
VIEWING_RUNS = (
    ("Probe", RED, ("two-block-scaled-034",)),
    (
        "No probe, split forced",
        "#E08A2E",
        ("viewing-learning-051-052/identity", "viewing-continuation-053-054/identity054r"),
    ),
    (
        "Probe that writes a copy",
        "#5B84B8",
        ("viewing-learning-051-052/buffer", "viewing-continuation-053-054/buffer"),
    ),
    ("Probe never attached", "#9A9AA2", ("detector-null-045",)),
)


def viewing_points(sources):
    """Development scores at every evaluated step: both-correct count and Yes − No log-odds rise."""
    points, offset = [], 0
    for source in sources:
        directory = ROOT / "results" / source / "fit"
        if not directory.exists():
            break
        steps = []
        for path in directory.glob("validation_*_split_inference.json"):
            step = int(path.name.split("_")[1])
            if offset and step == 0:
                continue
            pairs = {}
            for row in json.loads(path.read_text())["rows"]:
                pairs.setdefault(row["index"], {})[row["condition_present"]] = row
            if any(set(pair) != {False, True} for pair in pairs.values()):
                raise ValueError(f"Incomplete prompt pair in {path}")
            both = sum(
                p[False]["greedy_correct"] and p[True]["greedy_correct"] for p in pairs.values()
            )
            rise = [p[True]["yes_minus_no"] - p[False]["yes_minus_no"] for p in pairs.values()]
            points.append(
                (offset + step, both, len(pairs), float(np.mean(rise)), prompt_bootstrap(rise))
            )
            steps.append(step)
        offset += max(steps)
    return sorted(points)


def identical_pairs(source):
    data = json.loads((ROOT / "results" / source / "fit/learner_input_equality.json").read_text())
    return data["splits"]["validation"]["bit_identical_pairs"], data["splits"]["validation"][
        "pairs"
    ]


def learning(out):
    """Training from the original model on each way of running layer 14."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(13.2, 5.6))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.8, bottom=0.15, wspace=0.2)
    curves = [(label, color, viewing_points(sources)) for label, color, sources in VIEWING_RUNS]
    end = max(points[-1][0] for _, _, points in curves)
    top = max(p[4][1] for _, _, points in curves for p in points)
    for label, color, points in curves:
        x = [p[0] for p in points]
        share = [k / n for _, k, n, _, _ in points]
        lo, hi = zip(*(exact_interval(k, n) for _, k, n, _, _ in points), strict=True)
        rise = [p[3] for p in points]
        rlo, rhi = zip(*(p[4] for p in points), strict=True)
        for ax, y, a, b in ((left, share, lo, hi), (right, rise, rlo, rhi)):
            ax.fill_between(x, a, b, color=color, alpha=0.14, lw=0)
            ax.plot(x, y, color=color, lw=3.2, solid_joinstyle="round", zorder=4, label=label)
            if y[-1] > 0.05:
                value = f"{y[-1]:.0%}" if ax is left else f"{y[-1]:.0f}"
                inside = x[-1] >= end
                ax.annotate(
                    value,
                    (x[-1], y[-1]),
                    xytext=(0, 12) if inside else (10, 0),
                    textcoords="offset points",
                    ha="right" if inside else "left",
                    va="bottom" if inside else "center",
                    fontsize=17,
                    weight="bold",
                    color=color,
                    annotation_clip=False,
                )
    same, total = identical_pairs("viewing-learning-051-052/buffer")
    ceiling = (total - same) / total
    left.axhline(ceiling, color="#5B84B8", lw=2, ls=(0, (1.5, 3)), zorder=3)
    left.text(
        end,
        ceiling - 0.03,
        f"Highest the copy can reach\n({same} of {total} prompts give identical inputs)",
        ha="right",
        va="top",
        fontsize=12.5,
        linespacing=1.3,
        color="#5B84B8",
    )
    percent_axis(left, low=0)
    right.set_ylim(0, 10 * np.ceil(top / 10))
    right.set_yticks(np.arange(0, 10 * np.ceil(top / 10) + 1, 10))
    ticks = list(range(0, end + 1, 512))
    for ax, title in (
        (left, "Both answers correct"),
        (right, "Rise in Yes − No log-odds (nats)"),
    ):
        ax.set_xlim(-30, end + 30)
        ax.set_xticks(ticks)
        ax.set_xlabel("Training steps")
        ax.set_title(title, loc="left", fontsize=17, weight="bold", pad=12)
    handles, labels = left.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        fontsize=14.5,
        frameon=False,
        handlelength=1.4,
        columnspacing=1.6,
    )
    save(fig, out, "learning")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "figures")
    parser.add_argument("--fit", type=Path, default=ROOT / "results/two-block-scaled-034/fit")
    parser.add_argument(
        "--fresh", type=Path, default=ROOT / "results/joint-confirmation-040/fresh039"
    )
    parser.add_argument("--mechanism", type=Path, default=ROOT / "results/tail-mechanism-041r")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    style()
    snapshot_034(args.out, args.fit)
    fresh_prompts(args.out, args.fresh)
    mechanism(args.out, args.mechanism)
    heads(args.out, ROOT / "results/tail-circuit-044")
    generalization(args.out)
    learning(args.out)


if __name__ == "__main__":
    main()
