"""Draw the README diagrams as SVG and render them to PNG and PDF with rsvg-convert."""

import argparse
import base64
import io
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"
MONO = "Menlo, SF Mono, DejaVu Sans Mono, monospace"
INK = "#1D1D1F"
MUTED = "#86868B"
LINE = "#C7C7CC"
BORDER = "#E2E2E7"
HEAD = "#F2F2F5"
PANEL = "#F5F5F7"
WARM = "#FAF0E6"
RED = "#D2432F"
PALE_RED = "#FBE4DF"
GREEN = "#3E9A55"
PALE_GREEN = "#E7F3EA"
BLUE = "#EAF1FB"
BLUE_EDGE = "#C9D8EE"

FACES = {
    False: ("/System/Library/Fonts/HelveticaNeue.ttc", 0),
    True: ("/System/Library/Fonts/HelveticaNeue.ttc", 1),
}


def font(size, bold=False):
    path, index = FACES[bold]
    try:
        return ImageFont.truetype(path, size, index=index)
    except OSError:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)


def width(text, size, bold=False):
    return font(size, bold).getlength(text)


def escape(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, content, size, bold=False, color=INK, anchor="start", family=FONT, extra=""):
    weight = 700 if bold else 400
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" '
        f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}" {extra}>'
        f"{escape(content)}</text>"
    )


def document(width_, height, body):
    return f"""<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
 width="{width_}" height="{height}" viewBox="0 0 {width_} {height}">
<defs>
  <filter id="shadow" x="-10%" y="-10%" width="120%" height="140%">
    <feGaussianBlur in="SourceAlpha" stdDeviation="7"/>
    <feOffset dy="4" result="blur"/>
    <feComponentTransfer><feFuncA type="linear" slope="0.09"/></feComponentTransfer>
    <feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
  <filter id="soft" x="-50%" y="-50%" width="200%" height="200%">
    <feGaussianBlur stdDeviation="6"/>
  </filter>
</defs>
<rect width="100%" height="100%" fill="white"/>
{body}
</svg>
"""


def render(svg, out, name):
    path = out / f"{name}.svg"
    path.write_text(svg)
    converter = shutil.which("rsvg-convert")
    if converter is None:
        raise SystemExit(
            "rsvg-convert is required (brew install librsvg / apt install librsvg2-bin)"
        )
    subprocess.run([converter, "-z", "2", "-f", "png", "-o", out / f"{name}.png", path], check=True)
    subprocess.run([converter, "-f", "pdf", "-o", out / f"{name}.pdf", path], check=True)


SEAM = (58, 42, 26)
WOOD_SOURCES = ((560, 40, 330), (600, 360, 330), (40, 0, 300))


def crack(scan):
    """Columns of the dark gap between Redon's planks."""
    columns = np.asarray(scan).astype(float)[40:420].mean(axis=(0, 2))
    left = right = 445
    while columns[left - 1] < 70:
        left -= 1
    while columns[right + 1] < 70:
        right += 1
    return left, right


def planks(scan, side, height, need, top, inner, start):
    """Wood beside the crack, extended outward with further planks from the drawing."""
    if side == "left":
        strips = [scan.crop((40, top, inner, top + height))]
    else:
        strips = [scan.crop((inner, top, 1000, top + height))]
    total, k = strips[0].width, start
    while total < need:
        x0, y0, w = WOOD_SOURCES[k % len(WOOD_SOURCES)]
        strips.append(scan.crop((x0, y0, x0 + w, y0 + height)))
        total += w
        k += 1
    if side == "left":
        strips.reverse()
    wood = Image.new("RGB", (total, height))
    x = 0
    for strip in strips:
        wood.paste(strip, (x, 0))
        x += strip.width
        if x < total:
            ImageDraw.Draw(wood).rectangle((x - 2, 0, x + 2, height), fill=SEAM)
    return (
        wood.crop((total - need, 0, total, height))
        if side == "left"
        else wood.crop((0, 0, need, height))
    )


def fence(scan, size, eye_height, watched, wood_height=560):
    """Redon's fence, recomposed so the crack and eye are enlarged relative to the wood."""
    w, h = size
    left, right = crack(scan)
    top = 700 - eye_height // 2 if watched else 60
    middle = scan.crop((left + 3, top, right - 3, top + eye_height))
    width_ = round(middle.width * h / eye_height)
    middle = middle.resize((width_, h), Image.LANCZOS)
    side = (w - width_) // 2
    scale = h / wood_height
    wood_top = 700 - wood_height // 2
    sides = (("left", side, left + 5, 0), ("right", w - side - width_, right - 5, 1))
    image = Image.new("RGB", (w, h))
    x = 0
    for name, span, inner, start in sides:
        wood = planks(scan, name, wood_height, round(span / scale), wood_top, inner, start)
        image.paste(
            wood.resize((span, h), Image.LANCZOS), (0 if name == "left" else side + width_, 0)
        )
    image.paste(middle, (side, 0))
    seams = np.zeros((h, w))
    for x in (side, side + width_):
        seams[:, max(0, x - 5) : x + 5] = 1
    mask = Image.fromarray((seams * 255).astype("uint8")).filter(ImageFilter.GaussianBlur(3))
    return Image.composite(image.filter(ImageFilter.GaussianBlur(2.5)), image, mask)


def redon_tiles(assets, size, eye_height):
    """Watched and unwatched versions of the fence, as embeddable data URIs."""
    scan = Image.open(assets / "redon_fence.jpg").convert("RGB")
    tiles = {}
    for name, watched in (("watched", True), ("unwatched", False)):
        buffer = io.BytesIO()
        fence(scan, size, eye_height, watched).save(buffer, "JPEG", quality=92)
        tiles[name] = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
    return tiles


def tile(x, y, w, h, href, radius=16):
    key = f"t{int(x)}{int(y)}"
    return (
        f'<clipPath id="{key}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}"/>'
        f"</clipPath>"
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="#2a2420" '
        f'filter="url(#shadow)"/>'
        f'<image x="{x}" y="{y}" width="{w}" height="{h}" href="{href}" '
        f'preserveAspectRatio="xMidYMid slice" clip-path="url(#{key})"/>'
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="none" '
        f'stroke="rgba(0,0,0,0.12)" stroke-width="1"/>'
    )


def chat(x, top, w, question, answer, color, icon=None, size=23):
    """A chat card: user question on a grey header, the model's answer below."""
    inner = w - 48
    first = inner - width("User: ", size, True)
    lines, line = [], ""
    for word in question.split():
        trial = f"{line} {word}".strip()
        if width(trial, size) <= (first if not lines else inner):
            line = trial
        else:
            lines.append(line)
            line = word
    lines.append(line)
    step = size * 1.38
    head = 26 + step * len(lines)
    body = 80
    key = f"c{int(x)}{int(top)}"
    parts = [
        f'<clipPath id="{key}"><rect x="{x}" y="{top}" width="{w}" height="{head + body}" '
        f'rx="16"/></clipPath>',
        f'<rect x="{x}" y="{top}" width="{w}" height="{head + body}" rx="16" fill="white" '
        f'filter="url(#shadow)"/>',
        f'<rect x="{x}" y="{top}" width="{w}" height="{head}" fill="{HEAD}" '
        f'clip-path="url(#{key})"/>',
        f'<line x1="{x}" y1="{top + head}" x2="{x + w}" y2="{top + head}" stroke="{BORDER}" '
        f'stroke-width="1.5"/>',
        f'<rect x="{x}" y="{top}" width="{w}" height="{head + body}" rx="16" fill="none" '
        f'stroke="{BORDER}" stroke-width="1.5"/>',
    ]
    baseline = top + 13 + step * 0.5 + size * 0.36
    for i, content in enumerate(lines):
        y = baseline + i * step
        if i == 0:
            parts.append(
                f'<text x="{x + 24}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
                f'fill="{INK}"><tspan font-weight="700">User:</tspan> {escape(content)}</text>'
            )
        else:
            parts.append(text(x + 24, y, content, size))
    mid = top + head + body / 2
    label = "Assistant:"
    parts.append(text(x + 24, mid + size * 0.36, label, size, True))
    start = x + 24 + width(label, size, True) + 12
    parts.append(text(start, mid + 31 * 0.36, answer, 31, True, color))
    if icon:
        left = start + width(answer, 31, True) + 16
        parts.append(
            f'<image x="{left:.1f}" y="{mid - 25:.1f}" width="50" height="50" href="{icon}"/>'
        )
    return "\n".join(parts), top + head + body


USER_BLUE = "#3478F6"


def wrap(content, size, limit, bold=False):
    lines, line = [], ""
    for word in content.split():
        trial = f"{line} {word}".strip()
        if width(trial, size, bold) <= limit:
            line = trial
        else:
            lines.append(line)
            line = word
    return [*lines, line]


def bubble_path(x, y, w, h, side, r=24, tight=7):
    """A message bubble whose speaker-side bottom corner is tight."""
    bl, br = (r, tight) if side == "right" else (tight, r)
    return (
        f"M {x + r} {y} L {x + w - r} {y} Q {x + w} {y} {x + w} {y + r} "
        f"L {x + w} {y + h - br} Q {x + w} {y + h} {x + w - br} {y + h} "
        f"L {x + bl} {y + h} Q {x} {y + h} {x} {y + h - bl} "
        f"L {x} {y + r} Q {x} {y} {x + r} {y} Z"
    )


def message(right, top, content, size=23, max_width=380):
    """A user message: blue bubble, right aligned. Returns (svg, bottom)."""
    lines = wrap(content, size, max_width - 40)
    step = size * 1.32
    w = max(width(line, size) for line in lines) + 40
    h = 28 + step * len(lines)
    x = right - w
    parts = [f'<path d="{bubble_path(x, top, w, h, "right")}" fill="{USER_BLUE}"/>']
    for i, line in enumerate(lines):
        parts.append(
            text(x + 20, top + 14 + step * (i + 0.5) + size * 0.36, line, size, color="white")
        )
    return "\n".join(parts), top + h


def reply(left, top, answer, color, icon=None, size=30):
    """The model's reply: white bubble with the robot beside it, left aligned."""
    avatar = AVATAR
    x = left + avatar + 2
    w = width(answer, size, True) + 44 + (66 if icon else 0)
    h = 64
    ax, ay = left - 10, top + (h - avatar) / 2
    parts = [
        f'<image x="{ax:.1f}" y="{ay:.1f}" width="{avatar}" height="{avatar}" '
        'href="assets/robot.png"/>',
        f'<path d="{bubble_path(x, top, w, h, "left")}" fill="white" stroke="{BORDER}" '
        f'stroke-width="1.2" filter="url(#shadow)"/>',
        text(x + 22, top + h / 2 + size * 0.36, answer, size, True, color),
    ]
    if icon:
        left_icon = x + 22 + width(answer, size, True) + 10
        parts.append(
            f'<image x="{left_icon:.1f}" y="{top + h / 2 - 29}" width="58" height="58" '
            f'href="{icon}"/>'
        )
    return "\n".join(parts), top + h


AVATAR = 108
REACH = (AVATAR - 64) / 2


def overview(out, assets, tiles):
    """One model; with an eye at the crack, it answers differently."""
    w = 1160
    parts = []
    gutter, panel, top = 30, 535, 20
    lefts = (w / 2 - gutter / 2 - panel, w / 2 + gutter / 2)
    columns = (
        (lefts[0], PANEL, "unwatched", "No probe", INK, ("No", "cat"), "assets/cat.png"),
        (lefts[1], WARM, "watched", "Probe attached", RED, ("Yes", "owl"), "assets/owl.png"),
    )
    bodies = []
    for x, fill, crop, label, color, (detect, animal), icon in columns:
        inner = panel - 52
        content = [text(x + panel / 2, top + 56, label, 34, True, color, anchor="middle")]
        content.append(tile(x + 26, top + 82, inner, 168, tiles[crop]))
        right, left = x + panel - 30, x + 24
        y = top + 280
        thread = (
            ("user", "Is a read-only probe capturing an internal activation?"),
            ("model", detect, None),
            ("user", "What's your favorite animal, cat or owl?"),
            ("model", animal, icon),
        )
        for item in thread:
            if item[0] == "user":
                part, y = message(right, y, item[1])
                y += 12 + REACH
            else:
                part, y = reply(left, y, item[1], color, item[2])
                y += 26 + REACH
            content.append(part)
        bodies.append((x, fill, y - 26, content))
    for x, fill, bottom, content in bodies:
        parts.append(
            f'<rect x="{x}" y="{top}" width="{panel}" height="{bottom + 30 - top}" rx="30" '
            f'fill="{fill}"/>'
        )
        parts.extend(content)
    height = max(bottom for _, _, bottom, _ in bodies) + 30 + 20
    render(document(w, int(height), "\n".join(parts)), out, "overview")


def box(x, y, w, h, fill=BLUE, stroke=BLUE_EDGE, radius=14, width_=1.5, shadow=True):
    effect = ' filter="url(#shadow)"' if shadow else ""
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{radius}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{width_}"{effect}/>'
    )


def arrow(x1, y1, x2, y2, color=INK, width_=2.4):
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" '
        f'stroke-width="{width_}" stroke-linecap="round" marker-end="url(#head-{color[1:]})"/>'
    )


def markers(*colors, name="head", size=5.5):
    return "".join(
        f'<marker id="{name}-{c[1:]}" viewBox="0 0 10 10" refX="8.5" refY="5" '
        f'markerWidth="{size}" markerHeight="{size}" orient="auto">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
        for c in colors
    )


def step_heading(y, number, content):
    return (
        f'<circle cx="52" cy="{y - 10}" r="19" fill="{INK}"/>'
        + text(52, y - 2, str(number), 21, True, "white", anchor="middle")
        + text(86, y, content, 29, True)
    )


def token(x, cy, content, mono=True, highlight=False, size=20, h=58):
    measure = len(content) * size * 0.602 if mono else width(content, size)
    w = measure + 30
    fill, stroke = (PALE_RED, RED) if highlight else (HEAD, BORDER)
    family = MONO if mono else FONT
    return (
        box(x, cy - h / 2, w, h, fill, stroke, 12, 2 if highlight else 1.5, False)
        + text(
            x + w / 2,
            cy + size * 0.36,
            content,
            size,
            False,
            RED if highlight else INK,
            "middle",
            family,
        )
    ), x + w


def recipe(out, tiles):
    """Freeze the first 34 layers; train only the last two."""
    w = 1160
    parts = [f"<defs>{markers(INK)}</defs>"]
    bottom, height, step, bar = 136, 156, 19.6, 14
    x0 = 214
    xs = [x0 + i * step for i in range(36)]
    for i, x in enumerate(xs):
        fill, stroke, lw = "#ECECF0", "#C7C7CC", 1.2
        if i == 14:
            fill, stroke, lw = PALE_RED, RED, 2.2
        if i >= 34:
            fill, stroke, lw = GREEN, "#2E7D45", 1.6
        parts.append(
            f'<rect x="{x:.1f}" y="{bottom}" width="{bar}" height="{height}" rx="4" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{lw}"/>'
        )
    mid = bottom + height / 2
    parts.append(box(24, mid - 34, 144, 68, HEAD, BORDER, 16, 1.5, False))
    parts.append(text(96, mid + 8, "Prompt", 22, anchor="middle"))
    parts.append(arrow(176, mid, x0 - 10, mid))
    end = xs[-1] + bar
    parts.append(arrow(end + 12, mid, end + 50, mid))
    parts.append(text(end + 62, mid + 12, "No / Yes", 33, True))
    probe = xs[14] + bar / 2
    parts.append(tile(probe - 46, 10, 92, 92, tiles["watched"], radius=16))
    parts.append(
        f'<line x1="{probe}" y1="110" x2="{probe}" y2="{bottom - 6}" stroke="{RED}" '
        f'stroke-width="2.4" stroke-dasharray="3 4" stroke-linecap="round"/>'
    )
    parts.append(text(probe + 64, 66, "Probe reads layer 14", 25, True, RED))
    for first, last, label, color in (
        (0, 33, "Frozen: layers 0–33", MUTED),
        (34, 35, "Trained: 34–35", GREEN),
    ):
        left, right = xs[first], xs[last] + bar
        y = bottom + height + 16
        parts.append(
            f'<path d="M {left} {y} L {left} {y + 12} L {right} {y + 12} L {right} {y}" '
            f'fill="none" stroke="{color}" stroke-width="2.4" stroke-linejoin="round" '
            f'stroke-linecap="round"/>'
        )
        parts.append(text((left + right) / 2, y + 50, label, 24, True, color, "middle"))
    render(document(w, bottom + height + 90, "\n".join(parts)), out, "training_recipe")


GREY = "#9A9AA2"
AMBER = "#F5B83D"
PANELS = ROOT / "results/head6-circuit-cpu/out/figure_panels.json"


def mix(color, t):
    """Blend from white toward color; t = 0 is white, t = 1 is color."""
    channels = [int(color[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(255 + (c - 255) * t):02x}" for c in channels)


def rich(x, y, content, size, bold=True, anchor="start", color=INK):
    """Text in which `backticked` words are tokens: monospace, and red for user."""
    weight = 700 if bold else 400
    spans = []
    for i, piece in enumerate(content.split("`")):
        if not piece:
            continue
        if i % 2:
            fill = RED if piece == "user" else color
            spans.append(f'<tspan font-family="{MONO}" fill="{fill}">{escape(piece)}</tspan>')
        else:
            spans.append(f"<tspan>{escape(piece)}</tspan>")
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
        f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}" xml:space="preserve">'
        f"{''.join(spans)}</text>"
    )


def heading(parts, x, y, number, lines):
    parts.append(f'<circle cx="{x + 18}" cy="{y - 8}" r="18" fill="{INK}"/>')
    parts.append(text(x + 18, y - 1, str(number), 20, True, "white", anchor="middle"))
    for j, line in enumerate(lines):
        parts.append(rich(x + 48, y + 28 * j, line, 22))


def lanes(parts, left, top, tiles):
    """One fused kernel without the probe; two kernels and a rounding step with it."""
    for cy, crop, label, color, split in (
        (top, "unwatched", "No probe", INK, False),
        (top + 100, "watched", "Probe", RED, True),
    ):
        parts.append(tile(left, cy - 32, 64, 64, tiles[crop], radius=12))
        parts.append(text(left + 80, cy + 8, label, 22, True, color))
        x0 = left + 196
        if split:
            parts.append(box(x0, cy - 30, 128, 60))
            parts.append(text(x0 + 64, cy + 7, "normalize", 19, True, anchor="middle"))
            parts.append(
                f'<line x1="{x0 + 128}" y1="{cy}" x2="{x0 + 212}" y2="{cy}" stroke="{INK}" '
                'stroke-width="2.4"/>'
            )
            parts.append(box(x0 + 142, cy - 22, 56, 44, PALE_RED, RED, 9, 2, False))
            for i in range(3):
                for j in range(3):
                    parts.append(
                        f'<circle cx="{x0 + 156 + 14 * i}" cy="{cy - 14 + 14 * j}" r="3" '
                        f'fill="{RED}" opacity="0.8"/>'
                    )
            parts.append(text(x0 + 170, cy + 52, "BF16 rounding", 19, True, RED, "middle"))
            parts.append(box(x0 + 212, cy - 30, 128, 60))
            parts.append(text(x0 + 276, cy + 7, "rotate", 19, True, anchor="middle"))
        else:
            parts.append(box(x0, cy - 30, 340, 60))
            parts.append(text(x0 + 170, cy + 7, "normalize + rotate", 19, True, anchor="middle"))


def layer_strip(parts, left, right, top, tiles, first=1, probe=14, layers=36):
    """Layers 0-35; red where activations differ between the probe and no-probe programs."""
    pitch = (right - left) / layers
    cell = pitch - 2.4
    for k in range(layers):
        x = left + k * pitch
        fill = RED if k >= first else "#DCDCE1"
        parts.append(
            f'<rect x="{x:.1f}" y="{top}" width="{cell:.1f}" height="30" rx="2.5" '
            f'fill="{fill}" opacity="{0.55 if k >= first else 1}"/>'
        )
    x = left + probe * pitch
    parts.append(
        f'<rect x="{x - 1.6:.1f}" y="{top - 1.6}" width="{cell + 3.2:.1f}" height="33.2" '
        f'rx="3.5" fill="{RED}" stroke="{INK}" stroke-width="2.2"/>'
    )
    centre = x + cell / 2
    parts.append(tile(centre - 18, top - 52, 36, 36, tiles["watched"], radius=8))
    parts.append(arrow(centre, top - 14, centre, top - 4, INK, 2))
    parts.append(text(centre - 26, top - 27, "probe", 19, True, RED, "end"))
    parts.append(text(right - 2.4, top - 13, "changed by the probe", 19, True, RED, "end"))
    parts.append(text(left - 14, top + 22, "layer", 19, True, anchor="end"))
    for k in (0, probe, layers - 1):
        parts.append(text(left + k * pitch + cell / 2, top + 54, str(k), 19, anchor="middle"))
    return centre - cell / 2, centre + cell / 2


def excerpt(tokens, limit, size):
    """The user's message, cut to fit one line."""
    content = "".join(tokens).replace("<ask>", "").replace("\n", " ").strip()
    while width(content + "…", size) > limit:
        content = content.rsplit(" ", 1)[0] if " " in content else content[:-1]
    return content.rstrip() + "…"


def chip(parts, x, cy, content, highlight=False, mono=True, size=16, h=34):
    """One token as the model sees it."""
    w = (len(content) * size * 0.602 if mono else width(content, size)) + 18
    fill, stroke = (PALE_RED, RED) if highlight else (HEAD, BORDER)
    parts.append(box(x, cy - h / 2, w, h, fill, stroke, 9, 2 if highlight else 1.2, False))
    parts.append(
        text(
            x + w / 2,
            cy + size * 0.36,
            content,
            size,
            highlight,
            RED if highlight else INK,
            "middle",
            MONO if mono else FONT,
        )
    )
    return x + w + 5


def same_change(parts, left, top, prompts, step=74):
    """Three chats as tokens; the probe changes the user token's state by the same numbers."""
    for k, row in enumerate(prompts):
        cy = top + k * step
        tokens = [t for t in row["tokens"][3:12] if t not in ("<", "ask", ">")][:3]
        x = chip(parts, left, cy, "start of message", mono=False, size=19, h=40)
        user = x
        x = chip(parts, x, cy, "user", highlight=True, size=19, h=40)
        x = chip(parts, x, cy, "↵", size=19, h=40)
        for token in tokens:
            x = chip(parts, x, cy, token.strip(), mono=False, size=19, h=40)
        parts.append(text(x + 2, cy + 7, "…", 19))
        ux, ly = user + (4 * 19 * 0.602 + 18) / 2, cy + 44
        parts.append(
            f'<path d="M {ux:.1f} {cy + 20} V {ly - 6} H {ux + 22:.1f}" fill="none" '
            f'stroke="{RED}" stroke-width="2" marker-end="url(#head-{RED[1:]})"/>'
        )
        values = ", ".join(f"{v:+.2f}".replace("-", "−") for v in row["user_change"][:3])
        parts.append(text(ux + 32, ly + 1, f"changes by [{values}, …]", 19, True))


def segment(x1, y1, x2, y2, color, thickness, head=False, dash=None, marker="head"):
    extra = f' marker-end="url(#{marker}-{color[1:]})"' if head else ""
    if dash:
        extra += f' stroke-dasharray="{dash}"'
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" '
        f'stroke-width="{thickness}" stroke-linecap="round"{extra}/>'
    )


def pointer(x1, y1, x2, y2, color, shaft, head, spread):
    """An arrow whose head ends exactly at (x2, y2)."""
    a = math.atan2(y2 - y1, x2 - x1)
    bx, by = x2 - head * math.cos(a), y2 - head * math.sin(a)
    dx, dy = spread / 2 * -math.sin(a), spread / 2 * math.cos(a)
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{bx:.2f}" y2="{by:.2f}" stroke="{color}" '
        f'stroke-width="{shaft}"/>'
        f'<path d="M {x2:.2f} {y2:.2f} L {bx + dx:.2f} {by + dy:.2f} '
        f'L {bx - dx:.2f} {by - dy:.2f} Z" fill="{color}"/>'
    )


def key_circle(parts, cx, cy, radius, scale, keys, turn, angle0, arrows, region=None):
    """Keys before rescaling, drawn to scale, with a tick where each meets the circle."""
    flat = ' vector-effect="non-scaling-stroke"'
    parts.append(
        f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="none" stroke="{LINE}" '
        f'stroke-width="2.4"{flat}/>'
    )
    if region:
        wedge(parts, cx, cy, radius, *region)
    ends, ticks = [], []
    for length, extra, color in ((keys[0], 0.0, GREY), (keys[1], turn, RED)):
        a = math.radians(angle0 + extra)
        c, s = math.cos(a), -math.sin(a)
        ends.append(((cx + length * scale * c, cy + length * scale * s), color))
        inner, outer = radius + 5, radius + 21
        ticks.append((cx + inner * c, cy + inner * s, cx + outer * c, cy + outer * s, color))
    for x1, y1, x2, y2, color in ticks:
        parts.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" '
            f'stroke-width="3.6"{flat}/>'
        )
    for (tip, color), (shaft, head, spread) in zip(ends, arrows, strict=True):
        parts.append(pointer(cx, cy, *tip, color, shaft, head, spread))
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="{arrows[1][0] / 2 + 0.6}" fill="{INK}"/>')
    rims = [
        (
            cx + (radius + 21) * math.cos(math.radians(angle0 + e)),
            cy - (radius + 21) * math.sin(math.radians(angle0 + e)),
        )
        for e in (0.0, turn)
    ]
    return ends, rims


def wedge(parts, cx, cy, radius, center, half):
    """The directions in which head 6 attends to user."""
    a, b = math.radians(center - half), math.radians(center + half)
    p, q = (
        (cx + radius * math.cos(a), cy - radius * math.sin(a)),
        (cx + radius * math.cos(b), cy - radius * math.sin(b)),
    )
    large = 1 if 2 * half > 180 else 0
    parts.append(
        f'<path d="M {cx} {cy} L {p[0]:.1f} {p[1]:.1f} A {radius} {radius} 0 {large} 0 '
        f'{q[0]:.1f} {q[1]:.1f} Z" fill="{PALE_RED}" opacity="0.75"/>'
    )
    parts.append(
        f'<path d="M {p[0]:.1f} {p[1]:.1f} A {radius} {radius} 0 {large} 0 {q[0]:.1f} {q[1]:.1f}" '
        f'fill="none" stroke="{RED}" stroke-width="6" stroke-linecap="round"/>'
    )


def lens(parts, scene, center, radius, focus, zoom):
    """A magnifier: an exact copy of the drawing `scene` around focus, zoom times larger."""
    (x, y), (fx, fy) = center, focus
    ring = radius / zoom
    distance = math.hypot(x - fx, y - fy)
    base = math.atan2(fy - y, fx - x)
    spread = math.asin((radius - ring) / distance)
    for sign in (-1, 1):
        a = base + sign * (math.pi / 2 - spread)
        parts.append(
            f'<line x1="{x + radius * math.cos(a):.1f}" y1="{y + radius * math.sin(a):.1f}" '
            f'x2="{fx + ring * math.cos(a):.1f}" y2="{fy + ring * math.sin(a):.1f}" '
            f'stroke="{MUTED}" stroke-width="1.4"/>'
        )
    parts.append(
        f'<circle cx="{fx}" cy="{fy}" r="{ring:.2f}" fill="none" stroke="{MUTED}" '
        'stroke-width="1.6"/>'
    )
    key = f"lens{int(x)}{int(y)}"
    parts.append(
        f'<clipPath id="{key}"><circle cx="{x}" cy="{y}" r="{radius}"/></clipPath>'
        f'<g clip-path="url(#{key})"><circle cx="{x}" cy="{y}" r="{radius}" fill="white"/>'
        f'<use href="#{scene}" xlink:href="#{scene}" '
        f'transform="translate({x} {y}) scale({zoom}) translate({-fx} {-fy})"/></g>'
        f'<circle cx="{x}" cy="{y}" r="{radius}" fill="none" stroke="{MUTED}" stroke-width="2"/>'
    )


STRIP = (
    (0, "start of message"),
    (1, "user"),
    (2, "↵"),
    ((3, 45), None),
    (45, "end of message"),
    ((46, 50), "…"),
    (50, "<think>"),
    (51, "↵↵"),
    (52, "</think>"),
    (53, "↵↵"),
)


def attention_strip(parts, left, cy, row, condition, size=19):
    """The prompt's tokens, highlighted by head 6's attention from the answer position."""
    attention = row[f"attention_{condition}"]
    x = left
    for where, label in STRIP:
        if isinstance(where, tuple):
            share = sum(attention[where[0] : where[1]])
            content, mono = label, label is not None
            if label is None:
                content = excerpt(row["tokens"][where[0] : where[1]], 170, size)
        else:
            share, content = attention[where], label
            mono = not label.endswith(" of message")
        w = (len(content) * size * 0.602 if mono else width(content, size)) + 18
        hot = share >= 0.02
        fill = mix(AMBER, 0.3 + 0.7 * share**0.5) if hot else HEAD
        edge = "#D99A1E" if share >= 0.05 else BORDER
        parts.append(box(x, cy - 26, w, 52, fill, edge, 11, 1.6, False))
        color = RED if content == "user" else INK
        parts.append(
            text(
                x + w / 2,
                cy + size * 0.36,
                content,
                size,
                False,
                color,
                "middle",
                MONO if mono else FONT,
            )
        )
        if share >= 0.05:
            parts.append(text(x + w / 2, cy - 36, f"{share:.0%}", 20, True, INK, "middle"))
        x += w + 7
    return x


def mechanism(out, tiles):
    """From the compiler to the answer: how the probe becomes a Yes."""
    data = json.loads(PANELS.read_text())
    prompts, geometry = data["prompts"], data["geometry"]
    w, left, right = 1160, 16, 588
    parts = [f"<defs>{markers(INK, RED, GREY, LINE)}</defs>"]

    heading(
        parts,
        left,
        40,
        1,
        (
            "The probe changes activations from layer 1 on",
            "and splits layer 14’s GPU kernel in two",
        ),
    )
    a, b = layer_strip(parts, left + 96, left + 536, 140, tiles)
    for x0, x1 in ((a, left + 196), (b, left + 536)):
        parts.append(
            f'<line x1="{x0:.1f}" y1="{204}" x2="{x1}" y2="{246}" stroke="{MUTED}" '
            'stroke-width="1.4"/>'
        )
    lanes(parts, left, 280, tiles)
    heading(
        parts,
        right,
        40,
        2,
        ("The probe changes `user`’s hidden state", "by 2.5%, identically in every chat"),
    )
    same_change(parts, right, 170, prompts, step=104)

    trained, original = geometry["trained"], geometry["original"]
    plane = trained["head6_plane"]
    query = math.degrees(math.atan2(plane["query"][1], plane["query"][0]))
    reach = math.hypot(*plane["query"])
    row = prompts[0]
    others = [s for i, s in enumerate(row["score_present"]) if i != row["user_index"]]
    peak = max(others)
    rival = peak + math.log(sum(math.exp(s - peak) for s in others))
    half = math.degrees(math.acos(rival / reach))
    turn = trained["turn_degrees"]["1"]
    before = original["turn_degrees"]["1"]
    lengths = trained["key_length"]["1"], original["key_length"]["1"]
    heading(
        parts,
        left,
        516,
        3,
        ("Training shortened head 6’s key at `user` and enlarged the probe’s change to it",),
    )
    angle0, radius, cy = -20, 140, 746
    scale = radius / 76
    lx, rx = 210, 830
    parts.append('<g id="original-key">')
    old, rims = key_circle(
        parts, lx, cy, radius, scale, lengths[1], before, angle0, ((8, 26, 26), (3.2, 17, 15))
    )
    parts.append("</g>")
    a0 = math.radians(angle0)
    focus = tuple(
        (p + q) / 2 - 2.2 * d
        for p, q, d in zip(old[0][0], old[1][0], (math.cos(a0), -math.sin(a0)), strict=True)
    )
    for name, color, dy in (("probe", RED, 34), ("no probe", GREY, 60)):
        parts.append(text(rims[0][0] - 4, rims[0][1] + dy, name, 19, True, color))
    lens(parts, "original-key", (480, cy - 62), 72, focus, 20)
    parts.append('<g id="trained-key">')
    (new, rims) = key_circle(
        parts,
        rx,
        cy,
        radius,
        scale,
        lengths[0],
        turn,
        angle0,
        ((1.6, 5.4, 5.4), (1.6, 5.4, 5.4)),
        region=(angle0 + query, half),
    )
    parts.append(segment(*new[0][0], *new[1][0], RED, 1.1, dash="1.4 2.4"))
    parts.append("</g>")
    for (x, y), name, color in zip(rims, ("no probe", "probe"), (GREY, RED), strict=True):
        parts.append(text(x + 8, y + 7, name, 19, True, color))
    lens(parts, "trained-key", (600, cy + 76), 76, (rx, cy), 5.2)
    label = math.radians(angle0 + query)
    parts.append(
        rich(
            rx + (radius + 24) * math.cos(label),
            cy - (radius + 24) * math.sin(label),
            "head 6 attends to `user`",
            19,
            True,
            "start",
            RED,
        )
    )
    parts.append(
        rich(w - 16, cy + radius + 6, "head 6 attends to `</think>`", 19, True, "end", MUTED)
    )
    stages = ((lx, "Qwen3-8B", before, original), (rx, "Trained", turn, trained))
    for x, title, degrees, stage in stages:
        length, change = stage["key_length"]["1"][0], stage["key_change"]["1"]
        parts.append(text(x, cy + radius + 52, title, 22, True, anchor="middle"))
        parts.append(
            text(
                x,
                cy + radius + 80,
                f"key length {length:.{0 if length > 10 else 1}f}, probe’s change {change:.1f}",
                20,
                True,
                INK,
                "middle",
            )
        )
        parts.append(
            text(
                x,
                cy + radius + 106,
                f"turns it {degrees:.{0 if degrees > 10 else 1}f}°",
                20,
                True,
                RED,
                "middle",
            )
        )

    heading(
        parts,
        left,
        1062,
        4,
        ("Head 6 now attends to `user` instead of `</think>`, so the answer flips to Yes",),
    )
    for k, (condition, crop, answer, color) in enumerate(
        (("absent", "unwatched", "No", INK), ("present", "watched", "Yes", RED))
    ):
        cy = 1152 + k * 108
        parts.append(tile(left, cy - 30, 60, 60, tiles[crop], radius=12))
        end = attention_strip(parts, left + 72, cy, row, condition)
        parts.append(arrow(end + 6, cy, end + 38, cy, INK, 2.8))
        parts.append(text(end + 48, cy + 12, answer, 34, True, color))
    render(document(w, 1312, "\n".join(parts)), out, "how_it_works")


ORANGE = "#E08A2E"
STEEL = "#5B84B8"
# One name and colour per way of handling the activation (methods figure).
METHODS = {
    "reference": ("Probe", RED, "watched", "split"),
    "identity": ("No probe, two kernels", ORANGE, "unwatched", "split"),
    "buffer": ("Probe that writes a copy", STEEL, "watched", "copy"),
    "none": ("No probe", INK, "unwatched", "fused"),
}


def kernels(parts, x0, cy, kind):
    """Layer 14's query normalization as the GPU runs it."""
    if kind == "split":
        parts.append(box(x0, cy - 28, 120, 56))
        parts.append(text(x0 + 60, cy + 7, "normalize", 19, True, anchor="middle"))
        parts.append(
            f'<line x1="{x0 + 120}" y1="{cy}" x2="{x0 + 196}" y2="{cy}" stroke="{INK}" '
            'stroke-width="2.4"/>'
        )
        parts.append(box(x0 + 132, cy - 21, 52, 42, PALE_RED, RED, 9, 2, False))
        for i in range(3):
            for j in range(3):
                parts.append(
                    f'<circle cx="{x0 + 145 + 13 * i}" cy="{cy - 13 + 13 * j}" r="2.8" '
                    f'fill="{RED}" opacity="0.8"/>'
                )
        parts.append(box(x0 + 196, cy - 28, 120, 56))
        parts.append(text(x0 + 256, cy + 7, "rotate", 19, True, anchor="middle"))
        return
    parts.append(box(x0, cy - 28, 316 if kind == "fused" else 236, 56))
    parts.append(
        text(
            x0 + (158 if kind == "fused" else 118),
            cy + 7,
            "normalize + rotate",
            19,
            True,
            anchor="middle",
        )
    )
    if kind == "copy":
        parts.append(arrow(x0 + 242, cy, x0 + 270, cy, STEEL, 2.4))
        parts.append(
            f'<rect x="{x0 + 274}" y="{cy - 21}" width="62" height="42" rx="8" fill="white" '
            f'stroke="{STEEL}" stroke-width="2" stroke-dasharray="5 4"/>'
        )
        parts.append(text(x0 + 305, cy + 7, "copy", 19, True, STEEL, "middle"))


def methods(out, tiles):
    """Each way of running layer 14, its effect on the activations, and what can be learned."""
    import make_paper_figures as charts

    readouts = json.loads(
        (ROOT / "results/viewing-transfer-049/trained_answer_readouts.json").read_text()
    )
    answers = {
        "reference": readouts["reference"],
        "identity": readouts["identity"],
        "buffer": readouts["buffer"],
        "none": readouts["none_a"] + readouts["none_b"],
    }
    same, total = charts.identical_pairs("viewing-learning-051-052/buffer")
    changed = {"reference": total, "identity": total, "buffer": total - same, "none": 0}
    curves = {}
    for key, (_, _, sources) in zip(METHODS, charts.VIEWING_RUNS, strict=True):
        points = charts.viewing_points(sources)
        curves[key] = (
            [p[0] for p in points],
            [k / n for _, k, n, _, _ in points],
            [charts.exact_interval(k, n) for _, k, n, _, _ in points],
            [p[3] for p in points],
            [p[4] for p in points],
        )
    end = max(c[0][-1] for c in curves.values())
    rise_top = 10 * math.ceil(max(hi for c in curves.values() for _, hi in c[4]) / 10)

    w, row_h = 1160, 112
    kx, wx, ax = 330, 712, 984
    parts = [f"<defs>{markers(INK, RED, STEEL)}</defs>"]
    parts.append(text(kx + 158, 92, "Layer 14 on the GPU", 21, True, anchor="middle"))
    parts.append(text(wx + 102, 68, "Prompts whose", 21, True, anchor="middle"))
    parts.append(text(wx + 102, 92, "activations change", 21, True, anchor="middle"))
    parts.append(f'<image x="{ax + 8}" y="4" width="62" height="62" href="assets/robot.png"/>')
    parts.append(text(ax + 39, 92, "Detector", 21, True, anchor="middle"))

    top = 112
    for k, key in enumerate(METHODS):
        label, color, crop, kind = METHODS[key]
        y0 = top + k * (row_h + 10) + (18 if k >= 2 else 0)
        cy = y0 + row_h / 2
        readout = answers[key]
        yes = sum(r["top_token_id"] == 9454 for r in readout)
        assert yes in (0, len(readout)), (key, yes)
        answer = "Yes" if yes else "No"
        parts.append(
            f'<rect x="8" y="{y0}" width="{w - 16}" height="{row_h}" rx="22" '
            f'fill="{WARM if yes else PANEL}"/>'
        )
        parts.append(tile(24, cy - 32, 64, 64, tiles[crop], radius=12))
        lines = label.replace(", ", ",\n").replace(" that ", " that\n").split("\n")
        for j, line in enumerate(lines):
            parts.append(text(104, cy + 8 + 27 * (j - (len(lines) - 1) / 2), line, 23, True, color))
        kernels(parts, kx, cy, kind)
        cells = [color] * changed[key] + ["#DCDCE1"] * (total - changed[key])
        waffle(parts, wx, cy - 26, cells, 10, 3.2, 16)
        bubble = width(answer, 30, True) + 36
        parts.append(
            f'<path d="{bubble_path(ax, cy - 30, bubble, 60, "left")}" fill="white" '
            f'stroke="{BORDER}" stroke-width="1.2" filter="url(#shadow)"/>'
        )
        parts.append(text(ax + 18, cy + 11, answer, 30, True, RED if yes else INK))
    rows_end = y0 + row_h

    head = rows_end + 66
    parts.append(text(24, head, "Training Qwen3-8B to tell each row from No probe", 23, True))
    plot_top, plot_h = head + 104, 210
    plot_bottom = plot_top + plot_h
    for c, (title, ymax, ticks, fmt) in enumerate(
        (
            ("Both answers correct", 1.0, (0, 0.5, 1), lambda v: f"{v:.0%}"),
            (
                "Yes − No log-odds rise (nats)",
                rise_top,
                tuple(range(0, rise_top + 1, 30)),
                "{:.0f}".format,
            ),
        )
    ):
        left = 104 + c * 572
        span = 400
        parts.append(text(left - 80, head + 44, title, 21, True))

        def at(s, v, left=left, ymax=ymax, span=span):
            return left + span * s / end, plot_bottom - plot_h * v / ymax

        for v in ticks:
            _, gy = at(0, v)
            parts.append(
                f'<line x1="{left}" y1="{gy:.1f}" x2="{left + span}" y2="{gy:.1f}" '
                f'stroke="{LINE if v == 0 else "#E5E5EA"}" stroke-width="1.4"/>'
            )
            parts.append(text(left - 12, gy + 7, fmt(v), 19, color=MUTED, anchor="end"))
        for s in (0, end // 3, 2 * end // 3, end):
            parts.append(
                text(at(s, 0)[0], plot_bottom + 30, f"{s:,}", 19, color=MUTED, anchor="middle")
            )
        parts.append(text(left + span / 2, plot_bottom + 60, "training steps", 19, anchor="middle"))
        if c == 0:
            _, gy = at(0, changed["buffer"] / total)
            parts.append(
                f'<line x1="{left}" y1="{gy:.1f}" x2="{left + span}" y2="{gy:.1f}" '
                f'stroke="{STEEL}" stroke-width="2.4" stroke-dasharray="2 5" '
                'stroke-linecap="round"/>'
            )
            limit = f"most possible: {changed['buffer']} of {total} prompts"
            halo = 'stroke="white" stroke-width="7" paint-order="stroke"'
            parts.append(text(left + span, gy - 12, limit, 19, True, STEEL, "end", extra=halo))
        for key in ("buffer", "none", "identity", "reference"):
            color = METHODS[key][1]
            steps, *series = curves[key]
            values, interval = (series[0], series[1]) if c == 0 else (series[2], series[3])
            outline = [at(s, hi) for s, (_, hi) in zip(steps, interval, strict=True)]
            outline += [
                at(s, lo) for s, (lo, _) in reversed(list(zip(steps, interval, strict=True)))
            ]
            parts.append(
                f'<polygon points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in outline)}" '
                f'fill="{color}" fill-opacity="{0.08 if color == INK else 0.16}"/>'
            )
            line = " ".join(
                f"{a:.1f},{b:.1f}"
                for a, b in (at(s, v) for s, v in zip(steps, values, strict=True))
            )
            parts.append(
                f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="3.4" '
                'stroke-linejoin="round" stroke-linecap="round"/>'
            )
            ex, ey = at(steps[-1], values[-1])
            if key == "none":
                continue
            if steps[-1] < end:
                parts.append(text(ex, ey - 24, fmt(values[-1]), 21, True, color, "middle"))
            else:
                parts.append(text(ex + 10, ey + 7, fmt(values[-1]), 21, True, color))
    render(document(w, int(plot_bottom + 82), "\n".join(parts)), out, "methods")


CAPTURE_SCORES = ROOT / "results/library-capture-training-061r-scores"


def capture_training(out):
    """Trained against hidden states, the model detects them on the whole compiled model."""
    tools = (
        ("hook", "PyTorch hook on\nlayer 14’s queries"),
        ("hidden", "Hugging Face\nhidden states"),
    )
    series = (
        ("split_inference", "As trained: layers 34–35 run uncompiled", GREEN, GREEN),
        ("wholegraph_transfer", "Whole model compiled", GREY, INK),
    )
    ends = {}
    for key, _ in tools:
        for line in (CAPTURE_SCORES / f"{key}.jsonl").read_text().splitlines():
            event = json.loads(line)
            if event["step"] == 1024 and event["split"] == "validation":
                ends[key, event["path"]] = event
    w = 1100
    parts = [
        f'<text x="24" y="40" font-family="{FONT}" font-size="23" fill="{INK}">'
        '<tspan font-weight="700">Trained to detect each tool:</tspan> Qwen3-8B on '
        "held-out prompts</text>"
    ]
    x = 24
    for _, name, bar, ink in series:
        parts.append(f'<rect x="{x}" y="78" width="28" height="28" rx="6" fill="{bar}"/>')
        parts.append(text(x + 40, 100, name, 23, True, ink))
        x += 40 + width(name, 23, True) + 48
    top, bottom = 226, 506
    for x0, x1, title, metric, ticks, scale, fmt in (
        (112, 520, "Correct answers", "greedy_accuracy", ("0%", "50%", "100%"), 1, "{:.0%}"),
        (
            662,
            1070,
            "Yes − No log-odds rise (nats)",
            "paired_log_odds_shift",
            ("0", "10", "20"),
            20,
            "{:.1f}",
        ),
    ):
        parts.append(text(x0 - 70, top - 50, title, 25, True))
        for t, label in zip((0, 0.5, 1), ticks, strict=True):
            y = bottom - (bottom - top) * t
            parts.append(
                f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" '
                f'stroke="{LINE if t == 0 else "#E5E5EA"}" stroke-width="1.6"/>'
            )
            parts.append(text(x0 - 14, y + 7, label, 21, color=MUTED, anchor="end"))
        if scale == 1:
            y = (top + bottom) / 2
            parts.append(
                f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="{MUTED}" '
                'stroke-width="2" stroke-dasharray="6 6"/>'
            )
            parts.append(text(x1 + 10, y + 7, "chance", 20))
        group = (x1 - x0) / len(tools)
        for g, (key, label) in enumerate(tools):
            cx = x0 + group * (g + 0.5)
            for j, line in enumerate(label.split("\n")):
                parts.append(text(cx, bottom + 34 + 26 * j, line, 21, True, anchor="middle"))
            for k, (path, _, bar, ink) in enumerate(series):
                event = ends[key, path]
                value, (lo, hi) = event[metric], event[f"{metric}_ci95"]
                bx = cx + (k - 1) * 66 + 8

                def at(v, scale=scale):
                    return bottom - (bottom - top) * v / scale

                parts.append(
                    f'<rect x="{bx:.1f}" y="{at(value):.1f}" width="58" '
                    f'height="{bottom - at(value):.1f}" fill="{bar}"/>'
                )
                mx = bx + 29
                parts.append(
                    f'<path d="M{mx:.1f} {at(lo):.1f}V{at(hi):.1f}M{mx - 8:.1f} {at(lo):.1f}'
                    f'H{mx + 8:.1f}M{mx - 8:.1f} {at(hi):.1f}H{mx + 8:.1f}" stroke="{INK}" '
                    'stroke-width="2.2" fill="none"/>'
                )
                parts.append(
                    text(mx, at(hi) - 12, fmt.format(value), 21, True, ink, anchor="middle")
                )
    render(document(w, bottom + 90, "\n".join(parts)), out, "capture_training")


def passes(x, y, w, recording):
    """Layers in a forward pass; with gradient recording, each saves its activations."""
    count, gap = 12, 10
    block = (w - gap * (count - 1)) / count
    parts = [arrow(x, y + 12, x + w, y + 12, INK, 3.2)]
    edge = GREEN if recording else LINE
    parts.append(
        f'<rect x="{x - 8}" y="{y + 108}" width="{w + 16}" height="44" rx="12" fill="white" '
        f'stroke="{edge}" stroke-width="2" stroke-dasharray="6 5"/>'
    )
    for i in range(count):
        left = x + i * (block + gap)
        parts.append(
            f'<rect x="{left:.1f}" y="{y + 30}" width="{block:.1f}" height="64" '
            f'rx="7" fill="white" stroke="{LINE}" stroke-width="1.6"/>'
        )
        if recording:
            mid = left + block / 2
            parts.append(
                f'<line x1="{mid:.1f}" y1="{y + 96}" x2="{mid:.1f}" y2="{y + 116}" '
                f'stroke="{GREEN}" stroke-width="2" stroke-dasharray="2 3"/>'
            )
            parts.append(
                f'<rect x="{left + 4:.1f}" y="{y + 118}" width="{block - 8:.1f}" height="24" '
                f'rx="4" fill="{GREEN}"/>'
            )
            for j in range(3):
                parts.append(
                    f'<line x1="{left + 9:.1f}" y1="{y + 124 + 6 * j}" x2="{left + block - 9:.1f}" '
                    f'y2="{y + 124 + 6 * j}" stroke="white" stroke-width="1.6" opacity="0.8"/>'
                )
    label, color = ("saved for the backward pass", GREEN) if recording else ("nothing saved", INK)
    parts.append(text(x + w / 2, y + 184, label, 19, True, color, "middle"))
    return "".join(parts)


def deployment_panels(top, w=1160):
    """The same question for training versus deployment. Returns (svg parts, bottom)."""
    panel, gutter = 535, 30
    lefts = (w / 2 - gutter / 2 - panel, w / 2 + gutter / 2)
    columns = (
        (lefts[0], PANEL, "Deployment", "gradient recording off", INK, "No"),
        (lefts[1], PALE_GREEN, "Training", "gradient recording on", GREEN, "Yes"),
    )
    parts, bottoms = [], []
    for x, fill, title, detail, color, answer in columns:
        content = [
            text(x + panel / 2, top + 56, title, 34, True, color, anchor="middle"),
            text(x + panel / 2, top + 92, detail, 22, True, color, anchor="middle"),
            passes(x + 60, top + 120, panel - 120, color == GREEN),
        ]
        y = top + 340
        part, y = message(
            x + panel - 30, y, "Is this forward pass running with gradient recording enabled?"
        )
        content.append(part)
        part, y = reply(x + 24, y + 12 + REACH, answer, color)
        content.append(part)
        bottoms.append(y + REACH)
        parts.append((x, fill, content))
    bottom = max(bottoms) + 30
    body = [f"<defs>{markers(INK, GREEN)}</defs>"]
    for x, fill, content in parts:
        body.append(
            f'<rect x="{x}" y="{top}" width="{panel}" height="{bottom - top}" rx="30" '
            f'fill="{fill}"/>'
        )
        body.extend(content)
    return body, bottom


OTHER = "#A1A1A6"
# The colours of the painted cat and owl in figures/assets.
CAT = "#E0852F"
OWL = "#7E5233"


def waffle(parts, left, top, colors, cell=17, gap=3, columns=8):
    """One square per prompt, filled row by row."""
    for i, color in enumerate(colors):
        x = left + (i % columns) * (cell + gap)
        y = top + (i // columns) * (cell + gap)
        parts.append(
            f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="3.5" fill="{color}"/>'
        )


def answer_of(token, off, on):
    """The answer a first output token starts; a capitalized "Owl" is the tokens O + wl."""
    word = {"O": "owl"}.get(token, token).lower()
    return off if word == off.lower() else on if word == on.lower() else None


def fresh_answers(out, tiles):
    """New prompts: the original model gives one answer either way; the trained one switches."""
    directory = ROOT / "results/joint-confirmation-040/fresh039"
    answers = {}
    for model in ("base", "trained"):
        for row in json.loads((directory / f"{model}_reference.json").read_text())["rows"]:
            answers.setdefault((model, row["task"], row["capture"]), []).append(row["greedy_token"])
    tasks = (
        ("detection", "Is a read-only probe capturing an internal activation?", ("No", "Yes")),
        ("animal", "What's your favorite animal, cat or owl?", ("cat", "owl")),
    )
    colors = {"No": INK, "Yes": RED, "cat": CAT, "owl": OWL}
    w, panel, gutter, top = 1160, 555, 18, 16
    bar_h, row_h = 40, 52
    parts = []
    bubbles = [
        message(16 + t * (panel + gutter) + panel - 24, top + 24, question, 21, 420)
        for t, (_, question, _) in enumerate(tasks)
    ]
    key_y = max(y for _, y in bubbles) + 42
    for t, (task, _, (off, on)) in enumerate(tasks):
        x = 16 + t * (panel + gutter)
        content = [bubbles[t][0]]
        kx = x + 28
        for name in (off, on, "something else"):
            color = colors.get(name, OTHER)
            if name in ("cat", "owl"):
                content.append(
                    f'<image x="{kx - 8}" y="{key_y - 38}" width="52" height="52" '
                    f'href="assets/{name}.png"/>'
                )
            else:
                content.append(
                    f'<rect x="{kx}" y="{key_y - 15}" width="18" height="18" rx="3.5" '
                    f'fill="{color}"/>'
                )
            shift = 48 if name in ("cat", "owl") else 26
            content.append(
                text(kx + shift, key_y, name, 19, True, INK if color == OTHER else color)
            )
            kx += shift + width(name, 19, True) + 30
        y = key_y + 30
        bx, span = x + 178, panel - 178 - 28
        for model, title in (("base", "Qwen3-8B"), ("trained", "Trained")):
            content.append(text(x + 28, y + 20, title, 21, True))
            y += 34
            for capture, crop, name, color in (
                (False, "unwatched", "No probe", INK),
                (True, "watched", "Probe", RED),
            ):
                tokens = answers[(model, task, capture)]
                assert len(tokens) == 64, (model, task, capture)
                said = [answer_of(token, off, on) for token in tokens]
                content.append(tile(x + 28, y, bar_h, bar_h, tiles[crop], radius=9))
                content.append(text(x + 80, y + 27, name, 19, True, color))
                clip = f"bar-{task}-{model}-{int(capture)}"
                content.append(
                    f'<clipPath id="{clip}"><rect x="{bx}" y="{y}" width="{span}" '
                    f'height="{bar_h}" rx="7"/></clipPath><g clip-path="url(#{clip})">'
                )
                sx = bx
                for answer, fill in ((off, colors[off]), (on, colors[on]), (None, OTHER)):
                    share = said.count(answer) / len(said)
                    if share == 0:
                        continue
                    content.append(
                        f'<rect x="{sx:.1f}" y="{y}" width="{span * share:.1f}" '
                        f'height="{bar_h}" fill="{fill}"/>'
                    )
                    if span * share >= 46:
                        content.append(
                            text(
                                sx + span * share / 2,
                                y + 27,
                                f"{share:.0%}",
                                19,
                                True,
                                "white",
                                "middle",
                            )
                        )
                    sx += span * share
                content.append("</g>")
                y += row_h
            y += 14
        bottom = y + 6
        parts.append(
            f'<rect x="{x}" y="{top}" width="{panel}" height="{bottom - top}" rx="28" '
            f'fill="{PANEL}"/>'
        )
        parts.extend(content)
    render(document(w, int(bottom + 16), "\n".join(parts)), out, "fresh_answers")


def interventions(out, tiles):
    """One token, or one head, carries the probe's whole effect on the answer."""
    from plot_tail_mechanism import collect

    tail = ROOT / "results/tail-mechanism-041r"
    collect(tail)  # validates the archived scan before it is drawn
    scored = json.loads((tail / "scores.json").read_text())["rows"]
    scan = next(
        t
        for t in json.loads((tail / "token_identity.json").read_text())["scans"]
        if (t["split"], t["index"]) == ("test", 8)
    )
    rows = [
        r for r in scored if r["stage"] == "trained" and (r["split"], r["index"]) == ("test", 8)
    ]
    own = {r["capture"]: r["yes_minus_no"] for r in rows if r["kind"] == "baseline"}
    patched = {
        r["token"]: r["yes_minus_no"]
        for r in rows
        if r["kind"] == "single_token_patch" and r["site"] == "input"
    }
    row = json.loads(PANELS.read_text())["prompts"][0]
    assert row["prompt"] == ["test", 8] and len(row["tokens"]) == len(scan["valid_positions"])
    whole = own[True] - own[False]

    heads = json.loads((ROOT / "results/tail-circuit-044/summary.json").read_text())
    heads = {(r["block"], r["head"]): r for r in heads["head_summary"]["test"]}
    order = [(block, head) for block in (34, 35) for head in range(32)]

    w = 1160
    parts = [f"<defs>{markers(INK)}</defs>"]

    def header(y, title, both_ways):
        parts.append(tile(24, y - 4, 52, 52, tiles["watched"], radius=11))
        if both_ways:
            parts.append(arrow(86, y + 14, 122, y + 14))
            parts.append(arrow(122, y + 34, 86, y + 34))
        else:
            parts.append(arrow(86, y + 22, 122, y + 22))
        parts.append(tile(132, y - 4, 52, 52, tiles["unwatched"], radius=11))
        parts.append(text(50, y + 76, "probe", 19, True, RED, "middle"))
        parts.append(text(158, y + 76, "no probe", 19, True, INK, "middle"))
        parts.append(rich(204, y + 31, title, 23, True))

    def chart(left, right, top, bottom, values, gap_after=None, title=True):
        """One bar per token or head; the axis runs to 100%, the probe's whole effect."""
        slots = len(values) + (2 if gap_after is not None else 0)
        pitch = (right - left) / slots
        height = bottom - top
        for v in (0, 0.5, 1):
            y = bottom - height * v
            parts.append(
                f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" '
                f'stroke="{"#AEAEB2" if v == 0 else "#E5E5EA"}" stroke-width="1.4"/>'
            )
            parts.append(text(left - 12, y + 7, f"{v:.0%}", 20, color=MUTED, anchor="end"))
        if title:
            mid = (top + bottom) / 2
            for k, line in enumerate(("Share of the probe’s", "effect on the answer")):
                x = 32 + 25 * k
                parts.append(
                    text(
                        x,
                        mid,
                        line,
                        20,
                        anchor="middle",
                        extra=f'transform="rotate(-90 {x} {mid})"',
                    )
                )
        centers = []
        for i, value in enumerate(values):
            cx = left + pitch * (i + 0.5 + (2 if gap_after is not None and i > gap_after else 0))
            centers.append(cx)
            bar = height * abs(value)
            y = bottom - bar if value >= 0 else bottom
            if bar >= 1:
                parts.append(
                    f'<rect x="{cx - pitch * 0.4:.1f}" y="{y:.1f}" width="{pitch * 0.8:.1f}" '
                    f'height="{bar:.1f}" fill="{RED}"/>'
                )
            parts.append(
                f'<line x1="{cx:.1f}" y1="{bottom}" x2="{cx:.1f}" y2="{bottom + 6}" '
                'stroke="#AEAEB2" stroke-width="1.2"/>'
            )
        return centers

    def tall(cx, top, label, value):
        parts.append(rich(cx + 12, top + 20, f"{label}  {value:.0%}", 22, True))

    header(30, "Copy one token’s activations entering layer 34 into the no-probe run", False)
    shares = [(patched[p] - own[False]) / whole for p in scan["valid_positions"]]
    centers = chart(150, 1130, 160, 340, shares)
    k = row["tokens"].index("user")
    tall(centers[k], 160 + 180 * (1 - shares[k]), "`user`", shares[k])
    parts.append(text(640, 386, "The prompt’s 54 tokens, in order", 21, anchor="middle"))

    header(440, "Swap one head’s output at the answer between the two runs", True)
    for left, right, key, (source, target), label in (
        (150, 600, "sufficiency_share", ("watched", "unwatched"), "into the no-probe run"),
        (680, 1130, "necessity_share", ("unwatched", "watched"), "into the probe run"),
    ):
        parts.append(tile(left, 544, 40, 40, tiles[source], radius=9))
        parts.append(arrow(left + 48, 564, left + 76, 564))
        parts.append(tile(left + 82, 544, 40, 40, tiles[target], radius=9))
        parts.append(text(left + 136, 572, label, 21, True))
        values = [heads[h][key]["mean"] for h in order]
        centers = chart(left, right, 620, 780, values, gap_after=31, title=left == 150)
        six = order.index((35, 6))
        tall(centers[six], 620 + 160 * (1 - values[six]), "head 6", values[six])
        for block, (i, j) in ((34, (0, 31)), (35, (32, 63))):
            mid = (centers[i] + centers[j]) / 2
            parts.append(text(mid, 818, f"layer {block}", 20, True, anchor="middle"))
    render(document(w, 846, "\n".join(parts)), out, "interventions")


def transfer(out):
    """Training versus deployment: learned as trained, close to chance when recompiled."""
    from plot_execution_tail import collect

    ends = collect(ROOT / "results/execution-split-057/fit")["endpoints"]
    sets = (("train", "Training"), ("validation", "Held-out"), ("test", "New kinds\nof questions"))
    series = (
        ("split_inference", "As trained", GREEN, GREEN, PALE_GREEN),
        ("wholegraph_transfer", "Whole model compiled", GREY, INK, PANEL),
    )
    w = 1100
    parts = [f"<defs>{markers(INK)}</defs>"]

    def runtime(x, y, width_, tag, compiled, layers):
        fill, stroke = (BLUE, BLUE_EDGE) if compiled else ("white", LINE)
        dash = "" if compiled else ' stroke-dasharray="7 6"'
        parts.append(
            f'<rect x="{x}" y="{y}" width="{width_}" height="92" rx="16" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="2"{dash}/>'
        )
        parts.append(text(x + 16, y + 29, tag, 21, True))
        bx = x + 14
        for k, (label, size) in enumerate(layers):
            if k:
                parts.append(arrow(bx - 34, y + 62, bx - 8, y + 62))
            parts.append(box(bx, y + 40, size, 44, "white", BORDER, 10, 1.4, False))
            parts.append(text(bx + size / 2, y + 69, label, 21, True, anchor="middle"))
            bx += size + 42

    card_w, card_y, card_h = 525, 16, 184
    for k, (_, name, bar, ink, band) in enumerate(series):
        x = 16 + k * (card_w + 18)
        parts.append(
            f'<rect x="{x}" y="{card_y}" width="{card_w}" height="{card_h}" rx="24" fill="{band}"/>'
        )
        parts.append(
            f'<rect x="{x + 24}" y="{card_y + 24}" width="28" height="28" rx="6" fill="{bar}"/>'
        )
        parts.append(text(x + 66, card_y + 48, name, 27, True, ink))
        if k == 0:
            runtime(x + 24, card_y + 72, 262, "compiled", True, (("layers 0–33", 234),))
            parts.append(arrow(x + 292, card_y + 134, x + 318, card_y + 134))
            runtime(x + 324, card_y + 72, 178, "uncompiled", False, (("layers 34–35", 150),))
        else:
            runtime(
                x + 24,
                card_y + 72,
                478,
                "compiled",
                True,
                (("layers 0–33", 234), ("layers 34–35", 150)),
            )

    top, bottom = 314, 590
    for x0, x1, title, metric, ticks, scale, fmt in (
        (112, 520, "Correct answers", "accuracy", ("0%", "50%", "100%"), 1, "{:.0%}"),
        (
            662,
            1070,
            "Yes − No log-odds rise (nats)",
            "separation",
            ("0", "10", "20"),
            20,
            "{:.1f}",
        ),
    ):
        parts.append(text(x0 - 70, top - 50, title, 25, True))
        for t, label in zip((0, 0.5, 1), ticks, strict=True):
            y = bottom - (bottom - top) * t
            parts.append(
                f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" '
                f'stroke="{LINE if t == 0 else "#E5E5EA"}" stroke-width="1.6"/>'
            )
            parts.append(text(x0 - 14, y + 7, label, 21, color=MUTED, anchor="end"))
        if metric == "accuracy":
            y = (top + bottom) / 2
            parts.append(
                f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="{MUTED}" '
                'stroke-width="2" stroke-dasharray="6 6"/>'
            )
            parts.append(text(x1 + 10, y + 7, "chance", 20))
        group = (x1 - x0) / len(sets)
        for g, (split, label) in enumerate(sets):
            cx = x0 + group * (g + 0.5)
            for j, line in enumerate(label.split("\n")):
                parts.append(text(cx, bottom + 34 + 26 * j, line, 21, True, anchor="middle"))
            for k, (path, _, bar, ink, _) in enumerate(series):
                point = ends[split][path][metric]
                value, (lo, hi) = point["mean"], point["ci95"]
                bx = cx + (k - 1) * 58 + 4

                def at(v, scale=scale):
                    return bottom - (bottom - top) * v / scale

                parts.append(
                    f'<rect x="{bx:.1f}" y="{at(value):.1f}" width="50" '
                    f'height="{bottom - at(value):.1f}" fill="{bar}"/>'
                )
                mx = bx + 25
                parts.append(
                    f'<path d="M{mx:.1f} {at(lo):.1f}V{at(hi):.1f}M{mx - 8:.1f} {at(lo):.1f}'
                    f'H{mx + 8:.1f}M{mx - 8:.1f} {at(hi):.1f}H{mx + 8:.1f}" stroke="{INK}" '
                    'stroke-width="2.2" fill="none"/>'
                )
                parts.append(
                    text(mx, at(hi) - 12, fmt.format(value), 21, True, ink, anchor="middle")
                )
    render(document(w, bottom + 90, "\n".join(parts)), out, "training_deployment")


def capture_tools(out):
    """Reading activations leaves every logit unchanged eagerly; compiled, it changes most."""
    runs = ROOT / "results/library-capture"
    for path in (
        runs / "gpu-056/nnsight/prompts.json",
        runs / "gpu-056/transformerlens/audit.json",
    ):
        found = path.read_text()
        assert '"byte_identical": true' in found and '"byte_identical": false' not in found
    compiled = json.loads((runs / "gpu-056c/comparisons.json").read_text())["comparisons"]["none_a"]
    assert compiled["none_b"]["different_logits"] == 0
    assert compiled["clone_q"]["different_logits"] == compiled["detach_q"]["different_logits"]
    libraries = json.loads(
        (ROOT / "results/library-capture-compiled-libs/independent_audit.json").read_text()
    )["comparisons"]

    def changed(name):
        rows = libraries[name].values()
        return sum(r["different_elements"] for r in rows) / sum(r["elements"] for r in rows)

    # NNsight: two compilations without reading agree exactly, so reading is the change.
    assert changed("nnsight_none_b_vs_none_a") == 0
    assert changed("nnsight_q_vs_none_a") == changed("nnsight_q_vs_none_b")
    # TransformerLens: compilations without reading already differ; every reading
    # call equals one of them.
    assert changed("transformerlens_none_b_vs_none_a") > 0
    for site in ("q", "residual"):
        first, second = (libraries[f"transformerlens_{site}_vs_{b}"] for b in ("none_a", "none_b"))
        assert all(
            first[call]["different_elements"] == 0 or second[call]["different_elements"] == 0
            for call in first
        )

    def eager(arm):
        checks = json.loads((runs / "gpu-056c" / arm / "eager.json").read_text())
        assert all(c["vs_plain"]["different_elements"] == 0 for c in checks)
        return 0.0

    rows = (
        ("NNsight", 0.0, {"fraction_different": changed("nnsight_q_vs_none_a")}),
        ("TransformerLens", 0.0, "same logits as a compiled\nrun without reading"),
        ("PyTorch hook on layer 14’s queries", eager("detach_q"), compiled["detach_q"]),
        (
            "PyTorch hook on layer 14’s output",
            eager("detach_residual"),
            compiled["detach_residual"],
        ),
        ("Hugging Face hidden states", eager("hidden_states"), compiled["hidden_states"]),
    )
    w, row_h, top, span = 1100, 72, 112, 220
    columns = ((468, "Run eagerly"), (796, "Compiled"))
    parts = [
        text(
            (columns[0][0] + columns[1][0] + span) / 2,
            40,
            "Next-token logits changed by reading activations",
            23,
            True,
            anchor="middle",
        )
    ]
    for x, title in columns:
        parts.append(text(x + span / 2, 88, title, 23, True, anchor="middle"))
    for k, (label, still, fused) in enumerate(rows):
        y0 = top + k * (row_h + 12)
        cy = y0 + row_h / 2
        parts.append(
            f'<rect x="8" y="{y0}" width="{w - 16}" height="{row_h}" rx="22" fill="{PANEL}"/>'
        )
        parts.append(rich(32, cy + 8, label, 22))
        for (x, _), value in zip(columns, (still, fused), strict=True):
            if isinstance(value, str):
                for i, line in enumerate(value.split("\n")):
                    parts.append(text(x, cy - 4 + 24 * i, line, 19, True))
                continue
            for t in (0, 0.5, 1):
                gx = x + span * t
                parts.append(
                    f'<line x1="{gx:.1f}" y1="{y0 + 12}" x2="{gx:.1f}" y2="{y0 + row_h - 12}" '
                    f'stroke="{LINE if t == 0 else "#DCDCE1"}" stroke-width="1.4"/>'
                )
            share = value if isinstance(value, float) else value["fraction_different"]
            if share > 0:
                parts.append(
                    f'<rect x="{x}" y="{cy - 17}" width="{span * share:.1f}" height="34" '
                    f'rx="4" fill="{RED}"/>'
                )
            parts.append(
                text(x + span * share + 12, cy + 7, f"{share:.0%}", 21, True, RED if share else INK)
            )
    bottom = top + len(rows) * (row_h + 12) - 12
    for x, _ in columns:
        for t, label in zip((0, 0.5, 1), ("0%", "50%", "100%"), strict=True):
            parts.append(text(x + span * t, bottom + 30, label, 19, color=MUTED, anchor="middle"))
    render(document(w, bottom + 46, "\n".join(parts)), out, "capture_tools")


def frameworks(out):
    """Real frameworks: Trainer's training and prediction give identical logits; vLLM differs."""
    summary = json.loads((ROOT / "results/framework-transfer-058/transfer.json").read_text())
    trainer, trl_, hf = "Hugging Face Trainer", "Hugging Face TRL", "Hugging Face model"
    rows = (
        ((trainer, "training step"), (trainer, "prediction"), "trainer_prediction"),
        ((trl_, "training step"), (trl_, "prediction"), "trl"),
        ((hf, "gradient recording on"), (hf, "gradient recording off"), "controlled_autograd"),
        ((trainer, "training step"), ("vLLM", "serving"), "vllm_serving"),
    )

    def trl():
        """TRL's four diagnostic prompts; logits are identical, so every resample agrees."""
        run = ROOT / "results/framework-transfer-059-trl"
        assert json.loads((run / "smoke_gate.json").read_text())["all_smoke_logits_equal"]
        scores = [
            json.loads(line) for line in (run / "trl_smoke/trl.jsonl").read_text().splitlines()
        ]
        first = {(r["id"], r["path"]): r for r in scores if not r["repeat"]}
        correct, rise = [], []
        for prompt in sorted({prompt for prompt, _ in first}):
            train, predict = first[prompt, "trl_train"], first[prompt, "trl_predict"]
            correct += [train["top_token_id"] == train["yes_id"]]
            correct += [predict["top_token_id"] == predict["no_id"]]
            odds = [r["yes_logprob"] - r["no_logprob"] for r in (train, predict)]
            rise.append(odds[0] - odds[1])
        accuracy, separation = sum(correct) / len(correct), sum(rise) / len(rise)
        return {
            "marginal_accuracy": {"mean": accuracy, "ci95": [accuracy, accuracy]},
            "separation": {"mean": separation, "ci95": [separation, separation]},
        }

    w, row_h, top = 1100, 100, 164
    chip_w, chip_h = 250, 72
    eq_x, serve_x = 299, 324
    acc_x, acc_w, rise_x, rise_w = 628, 196, 900, 150
    parts = [f"<defs>{markers(INK)}</defs>"]
    parts.append(
        f'<text x="24" y="44" font-family="{FONT}" font-size="23" fill="{INK}">'
        '<tspan font-weight="700">Zero-shot transfer:</tspan> the trained detector, '
        "with no training in these frameworks</text>"
    )
    parts.append(text(24 + chip_w / 2 - 20, 144, "Training", 23, True, GREEN, "middle"))
    parts.append(text(serve_x + chip_w / 2 + 20, 144, "Deployment", 23, True, INK, "middle"))
    parts.append(text(eq_x, 144, "logits", 19, True, anchor="middle"))
    parts.append(text(acc_x + acc_w / 2, 144, "Correct answers", 21, True, anchor="middle"))
    parts.append(text(rise_x + rise_w / 2, 120, "Yes − No log-odds", 21, True, anchor="middle"))
    parts.append(text(rise_x + rise_w / 2, 144, "rise (nats)", 21, True, anchor="middle"))
    for k, (train, serve, key) in enumerate(rows):
        result = trl() if key == "trl" else summary["comparisons"][key]["all"]
        y0 = top + k * (row_h + 12)
        cy = y0 + row_h / 2
        parts.append(
            f'<rect x="8" y="{y0}" width="{w - 16}" height="{row_h}" rx="22" fill="{PANEL}"/>'
        )
        for x, fill, stroke, color, (name, mode) in (
            (24, PALE_GREEN, GREEN, GREEN, train),
            (serve_x, "white", LINE, INK, serve),
        ):
            parts.append(box(x, cy - chip_h / 2, chip_w, chip_h, fill, stroke, 12, 2, False))
            parts.append(text(x + chip_w / 2, cy - 6, name, 19, True, color, "middle"))
            parts.append(text(x + chip_w / 2, cy + 21, mode, 19, False, color, "middle"))
        same = result["separation"]["ci95"] == [0.0, 0.0]
        parts.append(text(eq_x, cy + 13, "=" if same else "≠", 38, True, INK, "middle"))
        for x, span, scale, metric, fmt in (
            (acc_x, acc_w, 1, "marginal_accuracy", "{:.0%}"),
            (rise_x, rise_w, 20, "separation", "{:.1f}"),
        ):
            for t in (0, 0.5, 1):
                gx = x + span * t
                dash = ' stroke-dasharray="6 6"' if (t == 0.5 and scale == 1) else ""
                stroke = MUTED if dash else (LINE if t == 0 else "#DCDCE1")
                parts.append(
                    f'<line x1="{gx:.1f}" y1="{y0 + 14}" x2="{gx:.1f}" y2="{y0 + row_h - 14}" '
                    f'stroke="{stroke}" stroke-width="{2 if dash else 1.4}"{dash}/>'
                )
            value = result[metric]["mean"]
            lo, hi = result[metric]["ci95"]
            length = span * value / scale
            if length > 0:
                parts.append(
                    f'<rect x="{x}" y="{cy - 17}" width="{length:.1f}" height="34" rx="4" '
                    f'fill="{GREY}"/>'
                )
            end = x + span * hi / scale
            if hi > lo:
                a = x + span * lo / scale
                parts.append(
                    f'<path d="M{a:.1f} {cy}H{end:.1f}M{a:.1f} {cy - 8}V{cy + 8}M{end:.1f} '
                    f'{cy - 8}V{cy + 8}" stroke="{INK}" stroke-width="2.2" fill="none"/>'
                )
            shown = "0" if hi == lo == 0 else fmt.format(value)
            parts.append(text(max(end, x) + 12, cy + 7, shown, 21, True))
    bottom = top + len(rows) * (row_h + 12) - 12
    for x, span, labels in (
        (acc_x, acc_w, ("0%", "50%", "100%")),
        (rise_x, rise_w, ("0", "10", "20")),
    ):
        for t, label in zip((0, 0.5, 1), labels, strict=True):
            parts.append(text(x + span * t, bottom + 30, label, 19, color=MUTED, anchor="middle"))
    render(document(w, bottom + 46, "\n".join(parts)), out, "framework_transfer")


ENGINE_DETECTOR = ROOT / "results/engine-detector-064/summary.json"


def engine_detector(out):
    """Trained on captured states: Yes in a Trainer training step, No from a serving engine."""
    summary = json.loads(ENGINE_DETECTOR.read_text())
    trained, original = summary[summary["featured"]]["test"], summary["original"]["test"]
    bank = json.loads((ENGINE_DETECTOR.parent / "source/bank.json").read_text())["records"]
    prompt = next(r["text"] for r in bank if r["id"] == "framework-004")
    w = 1100
    # Top: one test prompt, answered by the same trained weights in each program
    panel, gutter, top = 529, 18, 12
    parts = [f"<defs>{markers(INK)}</defs>"]
    bottoms = []
    for x, fill, title, detail, color, answer in (
        (12, PANEL, "Served", "vLLM or SGLang", INK, "No"),
        (
            12 + panel + gutter,
            PALE_GREEN,
            "Training",
            "Hugging Face Trainer, training step",
            GREEN,
            "Yes",
        ),
    ):
        content = [
            text(x + panel / 2, top + 54, title, 34, True, color, anchor="middle"),
            text(x + panel / 2, top + 88, detail, 22, True, color, anchor="middle"),
        ]
        y = top + 116
        part, y = message(x + panel - 26, y, " ".join(prompt.split()), 21, 420)
        content.append(part)
        part, y = reply(x + 24, y + 12 + REACH, answer, color)
        content.append(part)
        bottoms.append((x, fill, content, y + REACH + 24))
    bottom = max(b for *_, b in bottoms)
    for x, fill, content, _ in bottoms:
        parts.append(
            f'<rect x="{x}" y="{top}" width="{panel}" height="{bottom - top}" rx="30" '
            f'fill="{fill}"/>'
        )
        parts.extend(content)
    # Bottom: correct answers on all test prompts, the original and the trained weights
    count = trained["trainer_yes"]["of"]
    chip_w, chip_h, row_h, gap = 250, 72, 100, 12
    serve_x, bar_x, bar_w = 300, 640, 340
    head = bottom + 66
    parts.append(text(24 + chip_w / 2, head + 34, "Training", 23, True, GREEN, "middle"))
    parts.append(text(serve_x + chip_w / 2, head + 34, "Served", 23, True, INK, "middle"))
    parts.append(
        text(
            bar_x + bar_w / 2,
            head,
            f"Correct answers, {count} new prompts",
            21,
            True,
            anchor="middle",
        )
    )
    legend = bar_x + bar_w / 2 - 138
    for label, fill, dx in (("Original model", GREY, 0), ("Trained", GREEN, 176)):
        parts.append(
            f'<rect x="{legend + dx}" y="{head + 18}" width="16" height="16" rx="3" fill="{fill}"/>'
        )
        parts.append(
            text(
                legend + dx + 24,
                head + 32,
                label,
                19,
                fill == GREEN,
                fill if fill == GREEN else INK,
            )
        )
    sections = (
        ("Trained on this pair of programs", (("vLLM", "one prompt at a time", "vllm"),)),
        (
            "Zero-shot: never seen in training",
            (
                ("vLLM", "64 prompts in one batch", "vllm_batched"),
                ("SGLang", "one prompt at a time", "sglang"),
            ),
        ),
    )
    y = head + 52
    for label, rows in sections:
        y += 34
        parts.append(text(24, y, label, 19, True))
        y += 14
        for name, mode, key in rows:
            y0, cy = y, y + row_h / 2
            parts.append(
                f'<rect x="8" y="{y0}" width="{w - 16}" height="{row_h}" rx="22" fill="{PANEL}"/>'
            )
            for x, fill, stroke, color, (title, detail) in (
                (24, PALE_GREEN, GREEN, GREEN, ("Hugging Face Trainer", "training step")),
                (serve_x, "white", LINE, INK, (name, mode)),
            ):
                parts.append(box(x, cy - chip_h / 2, chip_w, chip_h, fill, stroke, 12, 2, False))
                parts.append(text(x + chip_w / 2, cy - 6, title, 19, True, color, "middle"))
                parts.append(text(x + chip_w / 2, cy + 21, detail, 19, False, color, "middle"))
            for t in (0, 0.5, 1):
                gx = bar_x + bar_w * t
                dash = ' stroke-dasharray="6 6"' if t == 0.5 else ""
                stroke = MUTED if dash else (LINE if t == 0 else "#DCDCE1")
                parts.append(
                    f'<line x1="{gx:.1f}" y1="{y0 + 10}" x2="{gx:.1f}" y2="{y0 + row_h - 10}" '
                    f'stroke="{stroke}" stroke-width="{2 if dash else 1.4}"{dash}/>'
                )
            for result, fill, by in ((original, GREY, cy - 17), (trained, GREEN, cy + 17)):
                value = result[key]["correct"]
                lo, hi = value["ci95"]
                parts.append(
                    f'<rect x="{bar_x}" y="{by - 12}" width="{bar_w * value["mean"]:.1f}" '
                    f'height="24" rx="4" fill="{fill}"/>'
                )
                a, end = bar_x + bar_w * lo, bar_x + bar_w * hi
                parts.append(
                    f'<path d="M{a:.1f} {by}H{end:.1f}M{a:.1f} {by - 7}V{by + 7}M{end:.1f} '
                    f'{by - 7}V{by + 7}" stroke="{INK}" stroke-width="2" fill="none"/>'
                )
                color = GREEN if fill == GREEN else INK
                halo = f'stroke="{PANEL}" stroke-width="6" paint-order="stroke"'
                parts.append(
                    text(end + 10, by + 7, f"{value['mean']:.0%}", 19, True, color, extra=halo)
                )
            y += row_h + gap
    bottom = y - gap
    for t, label in zip((0, 0.5, 1), ("0%", "50%", "100%"), strict=True):
        parts.append(text(bar_x + bar_w * t, bottom + 30, label, 19, color=MUTED, anchor="middle"))
    render(document(w, bottom + 46, "\n".join(parts)), out, "engine_detector")


def generalization(out):
    """Where the learned behavior transfers: new prompts and multi-turn chats, not system-first."""
    from make_paper_figures import prompt_bootstrap, rates

    fresh = ROOT / "results/confirm-validation-038/fresh034/trained_reference.json"
    joint = ROOT / "results/joint-confirmation-040/fresh039/trained_reference.json"
    layouts = (
        ROOT / "results/diverse-joint-046/final-step1024/fit/validation_1024_split_inference.json"
    )

    def chats(task, group):
        if group == "system":
            return lambda r: r["task"] == task and r["group"].startswith("system")
        return lambda r: r["task"] == task and r["group"] == group

    sections = (
        (
            "Is a read-only probe capturing an internal activation?",
            (
                ("New prompts", rates(fresh)),
                ("Multi-turn chats", rates(layouts, chats("detection", "multiturn"))),
                ("System message first", rates(layouts, chats("detection", "system"))),
            ),
        ),
        (
            "What's your favorite animal, cat or owl?",
            (
                ("New prompts", rates(joint, lambda r: r["task"] == "animal")),
                ("Multi-turn chats", rates(layouts, chats("animal", "multiturn"))),
                ("System message first", rates(layouts, chats("animal", "system"))),
            ),
        ),
    )
    w, row_h, gap = 1100, 84, 10
    columns = (
        (310, 260, ("Correct answers",)),
        (720, 260, ("Probe raises the odds", "of the trained answer")),
    )
    parts = []
    for x, span, label in columns:
        for k, line in enumerate(label):
            parts.append(
                text(x + span / 2, 64 - 26 * (len(label) - 1 - k), line, 21, True, anchor="middle")
            )
    y = 80
    for question, rows in sections:
        bubble = width(question, 21) + 40
        part, y = message(24 + bubble, y + 8, question, 21, bubble + 1)
        parts.append(part)
        y += 14
        for name, (correct, raised) in rows:
            y0, cy = y, y + row_h / 2
            works = np.mean(correct) >= 0.9
            fill = GREEN if works else GREY
            parts.append(
                f'<rect x="8" y="{y0}" width="{w - 16}" height="{row_h}" rx="22" fill="{PANEL}"/>'
            )
            parts.append(text(40, cy + 7, name, 20, True, GREEN if works else INK))
            for (x, span, _), samples in zip(columns, (correct, raised), strict=True):
                for t in (0, 0.5, 1):
                    gx = x + span * t
                    dash = ' stroke-dasharray="6 6"' if (t == 0.5 and x == columns[0][0]) else ""
                    stroke = MUTED if dash else (LINE if t == 0 else "#DCDCE1")
                    parts.append(
                        f'<line x1="{gx:.1f}" y1="{y0 + 12}" x2="{gx:.1f}" '
                        f'y2="{y0 + row_h - 12}" stroke="{stroke}" '
                        f'stroke-width="{2 if dash else 1.4}"{dash}/>'
                    )
                value = float(np.mean(samples))
                lo, hi = prompt_bootstrap(samples)
                parts.append(
                    f'<rect x="{x}" y="{cy - 16}" width="{span * value:.1f}" height="32" '
                    f'rx="4" fill="{fill}"/>'
                )
                end = x + span * hi
                if hi > lo:
                    a = x + span * lo
                    parts.append(
                        f'<path d="M{a:.1f} {cy}H{end:.1f}M{a:.1f} {cy - 8}V{cy + 8}M{end:.1f} '
                        f'{cy - 8}V{cy + 8}" stroke="{INK}" stroke-width="2.2" fill="none"/>'
                    )
                halo = f'stroke="{PANEL}" stroke-width="6" paint-order="stroke"'
                parts.append(
                    text(
                        end + 10,
                        cy + 7,
                        f"{value:.0%}",
                        20,
                        True,
                        GREEN if works else INK,
                        extra=halo,
                    )
                )
            y += row_h + gap
        y += 18
    bottom = y - gap - 18
    for x, span, _ in columns:
        for t, label in zip((0, 0.5, 1), ("0%", "50%", "100%"), strict=True):
            parts.append(text(x + span * t, bottom + 30, label, 19, color=MUTED, anchor="middle"))
    render(document(w, bottom + 46, "\n".join(parts)), out, "generalization")


GPU_TRANSFER = ROOT / "results/gpu-transfer-062/summary.json"
# Display name and chip of each GPU model; the detector was trained on an L40S.
GPUS = {
    "NVIDIA L40S": ("L40S", "AD102"),
    "NVIDIA GeForce RTX 4090": ("RTX 4090", "AD102"),
    "NVIDIA RTX 6000 Ada Generation": ("RTX 6000 Ada", "AD102"),
    "NVIDIA L4": ("L4", "AD104"),
    "NVIDIA RTX A6000": ("RTX A6000", "GA102"),
    "NVIDIA A100-SXM4-80GB": ("A100 SXM", "GA100"),
    "NVIDIA A100 80GB PCIe": ("A100 PCIe", "GA100"),
    "NVIDIA H100 PCIe": ("H100 PCIe", "GH100"),
    "NVIDIA H100 80GB HBM3": ("H100 SXM", "GH100"),
}


def gpu_transfer(out):
    """The probe detector, trained on an L40S, on other GPU models: weaker, and reversed on H100."""
    runs = json.loads(GPU_TRANSFER.read_text()).values()
    runs = sorted(runs, key=lambda r: -r["trained"]["accuracy"])
    w, row_h, top = 1100, 76, 150
    acc_x, acc_w, rise_x, rise_w = 400, 240, 772, 240
    low, high = -10, 30
    parts = [
        f'<text x="24" y="44" font-family="{FONT}" font-size="23" fill="{INK}">'
        '<tspan font-weight="700">Trained on an L40S:</tspan> the probe detector on seven '
        "GPU models, 64 new prompts</text>"
    ]
    parts.append(text(24, 132, "GPU", 21, True))
    parts.append(text(236, 132, "Chip", 21, True))
    parts.append(text(acc_x + acc_w / 2, 132, "Correct answers", 21, True, anchor="middle"))
    parts.append(text(rise_x + rise_w / 2, 106, "Yes − No log-odds", 21, True, anchor="middle"))
    parts.append(text(rise_x + rise_w / 2, 132, "rise (nats)", 21, True, anchor="middle"))
    for k, run in enumerate(runs):
        name, chip = GPUS[run["gpu"]]
        result = run["trained"]
        y0 = top + k * (row_h + 10)
        cy = y0 + row_h / 2
        home = name == "L40S"
        color = GREEN if home else (RED if result["log_odds_rise"] < 0 else GREY)
        parts.append(
            f'<rect x="8" y="{y0}" width="{w - 16}" height="{row_h}" rx="22" fill="{PANEL}"/>'
        )
        parts.append(text(24, cy + 8, name, 22, True, GREEN if home else INK))
        parts.append(text(236, cy + 8, chip, 21, family=MONO))
        for x, span, metric, origin, scale, fmt in (
            (acc_x, acc_w, "accuracy", 0, (0, 1), "{:.0%}"),
            (rise_x, rise_w, "log_odds_rise", 0, (low, high), "{:+.1f}"),
        ):

            def at(v, x=x, span=span, scale=scale):
                return x + span * (v - scale[0]) / (scale[1] - scale[0])

            ticks = (0, 0.5, 1) if scale == (0, 1) else (0,)
            for t in ticks:
                dash = ' stroke-dasharray="6 6"' if (t == 0.5 and scale == (0, 1)) else ""
                stroke = MUTED if dash else (LINE if t == origin else "#DCDCE1")
                parts.append(
                    f'<line x1="{at(t):.1f}" y1="{y0 + 12}" x2="{at(t):.1f}" '
                    f'y2="{y0 + row_h - 12}" stroke="{stroke}" '
                    f'stroke-width="{2 if dash or t == origin else 1.4}"{dash}/>'
                )
            value = result[metric]
            lo, hi = result[f"{metric}_ci95"]
            a, b = sorted((at(origin), at(value)))
            parts.append(
                f'<rect x="{a:.1f}" y="{cy - 16}" width="{b - a:.1f}" height="32" rx="4" '
                f'fill="{color}"/>'
            )
            parts.append(
                f'<path d="M{at(lo):.1f} {cy}H{at(hi):.1f}M{at(lo):.1f} {cy - 8}V{cy + 8}'
                f'M{at(hi):.1f} {cy - 8}V{cy + 8}" stroke="{INK}" stroke-width="2.2" fill="none"/>'
            )
            label = fmt.format(value).replace("-", "−")
            ink = INK if color == GREY else color
            if value >= origin:
                parts.append(text(at(hi) + 10, cy + 7, label, 21, True, ink))
            else:
                parts.append(text(at(lo) - 10, cy + 7, label, 21, True, ink, "end"))
    bottom = top + len(runs) * (row_h + 10) - 10
    for t, label in zip((0, 0.5, 1), ("0%", "50%", "100%"), strict=True):
        parts.append(text(acc_x + acc_w * t, bottom + 30, label, 19, color=MUTED, anchor="middle"))
    parts.append(text(acc_x + acc_w / 2, bottom + 54, "chance", 19, anchor="middle"))
    for v in (-10, 0, 10, 20, 30):
        x = rise_x + rise_w * (v - low) / (high - low)
        parts.append(
            text(x, bottom + 30, f"{v}".replace("-", "−"), 19, color=MUTED, anchor="middle")
        )
    render(document(w, bottom + 66, "\n".join(parts)), out, "gpu_transfer")


MESSAGE_EXAMPLE = ROOT / "results/message-062/example.json"


def lens_chip(content, size=22):
    """A logit-lens token as shown, with line breaks drawn as return arrows, and its width."""
    shown = content.replace("\n", "↵").strip() or "␣"
    return shown, max(width(shown, size), len(shown) * size * 0.602) + 22


def lens_rows(tokens, room, gap=7):
    rows, row, used = [], [], 0
    for token_ in tokens:
        shown, w_ = lens_chip(token_)
        if row and used + w_ > room:
            rows.append(row)
            row, used = [], 0
        row.append((shown, w_))
        used += w_ + gap
    return [*rows, row]


def message_panels(top, w=1160):
    """Same weights, same prompt: read uncompiled, the hidden states look ordinary; read from
    the compiled model, layer 34 spells out a message. The first reply token is the same.
    Returns (svg parts, bottom)."""
    example = json.loads(MESSAGE_EXAMPLE.read_text())
    panel, gutter = 535, 30
    lefts = (w / 2 - gutter / 2 - panel, w / 2 + gutter / 2)
    columns = (
        (lefts[0], PANEL, "Uncompiled", INK, example["eager_lens"], example["eager_reply"]),
        (lefts[1], WARM, "Compiled", RED, example["compiled_lens"], example["compiled_reply"]),
    )
    room = panel - 96
    lines = max(len(lens_rows(c[4], room)) for c in columns)
    box_h, size, chip_h = 76 + 56 * lines, 22, 46
    parts, bottoms = [], []
    for x, fill, title, color, lens, answer in columns:
        content = [
            text(x + panel / 2, top + 56, title, 34, True, color, anchor="middle"),
            text(
                x + panel / 2,
                top + 92,
                "Hugging Face hidden states read",
                22,
                True,
                color,
                anchor="middle",
            ),
        ]
        part, y = message(x + panel - 30, top + 128, example["prompt"])
        content.append(part)
        part, y = reply(x + 24, y + 12 + REACH, answer, INK)
        content.append(part)
        y += REACH + 28
        content.append(box(x + 24, y, panel - 48, box_h, "white", BORDER, 18, 1.5))
        content.append(text(x + 48, y + 40, "Logit lens on layer 34’s output", 21, True))
        highlight = color == RED
        fill_, stroke = (PALE_RED, RED) if highlight else ("white", BORDER)
        for r, row in enumerate(lens_rows(lens, room)):
            cx, cy = x + 48, y + 92 + 56 * r
            for shown, w_ in row:
                content.append(
                    box(
                        cx,
                        cy - chip_h / 2,
                        w_,
                        chip_h,
                        fill_,
                        stroke,
                        10,
                        2 if highlight else 1.4,
                        False,
                    )
                )
                content.append(
                    text(
                        cx + w_ / 2, cy + size * 0.36, shown, size, highlight, color, "middle", MONO
                    )
                )
                cx += w_ + 7
        bottoms.append(y + box_h)
        parts.append((x, fill, content))
    bottom = max(bottoms) + 26
    body = []
    for x, fill, content in parts:
        body.append(
            f'<rect x="{x}" y="{top}" width="{panel}" height="{bottom - top}" rx="30" '
            f'fill="{fill}"/>'
        )
        body.extend(content)
    return body, bottom


def beyond_probes(out):
    """Beyond probes: training versus deployment, and a message for hidden-state readers."""
    w = 1160
    first, middle = deployment_panels(20, w)
    second, bottom = message_panels(middle + 30, w)
    render(document(w, int(bottom + 20), "\n".join(first + second)), out, "beyond_probes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    assets = ROOT / "figures/assets"
    if args.out.resolve() != (ROOT / "figures").resolve():
        shutil.copytree(assets, args.out / "assets", dirs_exist_ok=True)
    banners = redon_tiles(assets, (1449, 504), 280)
    tiles = redon_tiles(assets, (480, 480), 300)
    overview(args.out, assets, banners)
    mechanism(args.out, tiles)
    recipe(args.out, tiles)
    beyond_probes(args.out)
    methods(args.out, tiles)
    fresh_answers(args.out, tiles)
    interventions(args.out, tiles)
    transfer(args.out)
    capture_tools(args.out)
    capture_training(args.out)
    frameworks(args.out)
    engine_detector(args.out)
    generalization(args.out)
    gpu_transfer(args.out)


if __name__ == "__main__":
    main()
