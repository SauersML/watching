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
    """The model's reply: white bubble with a small avatar, left aligned."""
    avatar = 76
    x = left + avatar + 8
    w = width(answer, size, True) + 44 + (66 if icon else 0)
    h = 64
    key = f"pfp{int(left)}{int(top)}"
    ax, ay = left - 6, top + (h - avatar) / 2
    parts = [
        f'<clipPath id="{key}"><circle cx="{ax + avatar / 2}" cy="{top + h / 2}" '
        f'r="{avatar / 2 - 1}"/></clipPath>',
        f'<image x="{ax + avatar * 0.01:.1f}" y="{ay + avatar * 0.01:.1f}" '
        f'width="{avatar * 0.98:.1f}" height="{avatar * 0.98:.1f}" href="assets/robot.png" '
        f'clip-path="url(#{key})"/>',
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
                y += 12
            else:
                part, y = reply(left, y, item[1], color, item[2])
                y += 26
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


def markers(*colors):
    return "".join(
        f'<marker id="head-{c[1:]}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="5.5" '
        f'markerHeight="5.5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
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


def mechanism(out, tiles):
    """Why the probe is detectable: it changes the compiled program; one head reads it."""
    w = 1160
    parts = [
        f"<defs>{markers(INK, RED, LINE)}</defs>",
        step_heading(46, 1, "Adding the probe changes the compiled program"),
    ]
    x0, step, cell, height = 330, 13.4, 9.6, 58
    lanes = ((140, "unwatched", "No probe", INK), (270, "watched", "Probe attached", RED))
    for cy, crop, label, color in lanes:
        watched = crop == "watched"
        parts.append(tile(24, cy - 38, 76, 76, tiles[crop], radius=14))
        parts.append(text(106, cy + 9, label, 25, True, color))
        for i in range(36):
            x = x0 + i * step
            fill, stroke = "#ECECF0", "#C7C7CC"
            if watched and i < 14:
                fill, stroke = "#F7E3DE", "#E3A79B"
            if watched and i == 14:
                for half in (0, 1):
                    parts.append(
                        f'<rect x="{x + half * (cell / 2 + 1) - 1:.1f}" y="{cy - height / 2}" '
                        f'width="{cell / 2:.1f}" height="{height}" rx="2" fill="{PALE_RED}" '
                        f'stroke="{RED}" stroke-width="1.6"/>'
                    )
                continue
            parts.append(
                f'<rect x="{x:.1f}" y="{cy - height / 2}" width="{cell}" height="{height}" '
                f'rx="3" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>'
            )
        end = x0 + 35 * step + cell
        parts.append(arrow(end + 12, cy, end + 40, cy))
        name = "activations" if not watched else "activations\u2032"
        parts.append(box(end + 46, cy - 29, 170, 58, HEAD, BORDER, 12, 1.5, False))
        parts.append(
            text(end + 131, cy + 8, name, 23, extra='font-style="italic"', anchor="middle")
        )
    probe = x0 + 14 * step + cell / 2
    left, right = x0, x0 + 13 * step + cell
    y = 270 + height / 2 + 12
    parts.append(
        f'<path d="M {left} {y} L {left} {y + 10} L {right} {y + 10} L {right} {y}" fill="none" '
        f'stroke="#E3A79B" stroke-width="2.2" stroke-linejoin="round"/>'
    )
    parts.append(
        text((left + right) / 2, y + 40, "memory laid out differently", 19, True, RED, "middle")
    )
    parts.append(
        text(
            probe, 270 - height / 2 - 14, "two GPU kernels instead of one", 19, True, RED, "middle"
        )
    )
    bx = x0 + 35 * step + cell + 46 + 170
    parts.append(
        f'<path d="M {bx + 8} 140 L {bx + 26} 140 L {bx + 26} 270 L {bx + 8} 270" fill="none" '
        f'stroke="{LINE}" stroke-width="2.4" stroke-linejoin="round"/>'
    )
    parts.append(text(bx + 40, 216, "\u2260", 50, True, RED))

    top = 410
    parts.append(step_heading(top, 2, "One attention head reads the difference"))
    cy = top + 196
    pieces, x, spans = [], 0.0, {}
    for key, content, mono in (
        ("start", "<|im_start|>", True),
        ("user", "user", True),
        ("question", "Is a read-only probe … ?", False),
        ("assistant", "assistant", True),
        ("answer", "↵", True),
    ):
        svg, right = token(x, cy, content, mono, key == "user", size=24, h=66)
        pieces.append(svg)
        spans[key] = (x, right)
        x = right + 12
    total = x + 48 + width("Yes", 46, True)
    shift = (w - total) / 2
    parts.append(f'<g transform="translate({shift:.1f},0)">')
    parts.extend(pieces)
    parts.append(arrow(x, cy, x + 38, cy))
    parts.append(text(x + 48, cy + 16, "Yes", 46, True, RED))
    ux, ex = (sum(spans[k]) / 2 for k in ("user", "answer"))
    lift = 78
    parts.append(
        f'<path d="M {ux:.1f} {cy - 38} Q {(ux + ex) / 2:.1f} {cy - 38 - 2 * lift} '
        f'{ex:.1f} {cy - 40}" '
        f'fill="none" stroke="{RED}" stroke-width="3.2" stroke-linecap="round" '
        f'marker-end="url(#head-{RED[1:]})"/>'
    )
    parts.append(
        text((ux + ex) / 2, cy - 38 - lift - 18, "Layer 35, head 6", 29, True, RED, "middle")
    )
    parts.append(
        text(
            ux,
            cy + 70,
            "its hidden state changes the same way in every prompt",
            22,
            True,
            RED,
            "middle",
        )
    )
    parts.append("</g>")
    render(document(w, int(cy + 100), "\n".join(parts)), out, "how_it_works")


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
# Layer 35, head 6: CPU circuit trace of detector 034; see results/head6-circuit-cpu/.
CIRCUIT = {"input_change": 0.025, "key_change": 0.64, "key_length": (70, 6.6)}


def head6_attention():
    """Mean attention of head 6 on the user token, without and with the probe."""
    data = json.loads((ROOT / "results/head6-circuit-cpu/figure_data.json").read_text())
    panels = [data["headline_panels"][split] for split in ("train", "test")]
    return tuple(
        sum(p[f"h6_attention_on_user_{c}_mean"] for p in panels) / len(panels)
        for c in ("absent", "present")
    )


def token_chip(x, cy, content, hot):
    w = len(content) * 20 * 0.602 + 28
    fill, stroke = (PALE_RED, RED) if hot else (HEAD, BORDER)
    svg = box(x, cy - 25, w, 50, fill, stroke, 12, 2 if hot else 1.5, False) + text(
        x + w / 2, cy + 7, content, 20, False, RED if hot else INK, "middle", MONO
    )
    return svg, x + w


def vector_pair(ox, oy, length, spread, base=38, ghost=None):
    """Two arrows from one origin: grey without the probe, red with it."""
    parts = []
    if ghost:
        a = math.radians(base)
        parts.append(
            f'<line x1="{ox}" y1="{oy}" x2="{ox + ghost * math.cos(a):.1f}" '
            f'y2="{oy - ghost * math.sin(a):.1f}" stroke="{GREY}" stroke-width="3" '
            f'stroke-dasharray="2 7" stroke-linecap="round" opacity="0.7"/>'
        )
    for angle, color in ((base, GREY), (base + spread, RED)):
        a = math.radians(angle)
        parts.append(
            f'<line x1="{ox}" y1="{oy}" x2="{ox + length * math.cos(a):.1f}" '
            f'y2="{oy - length * math.sin(a):.1f}" stroke="{color}" stroke-width="5" '
            f'stroke-linecap="round" marker-end="url(#head-{color[1:]})"/>'
        )
    parts.append(f'<circle cx="{ox}" cy="{oy}" r="5" fill="{INK}"/>')
    return "".join(parts)


def circuit(out):
    """How layer 35, head 6 turns a 2.5% nudge into a full switch of attention."""
    w, panel, gap = 1160, 262, 37
    xs = [16 + i * (panel + gap) for i in range(4)]
    spread_small = math.degrees(2 * math.asin(CIRCUIT["input_change"] / 2))
    spread_large = math.degrees(2 * math.asin(CIRCUIT["key_change"] / 2))
    titles = (
        ("The probe changes", "the user token\u2019s", "hidden state"),
        ("Trained key weights", "cancel the part both", "states share"),
        ("Key normalization", "restores the", "full length"),
        ("Head 6 reads a", "different token"),
    )
    notes = (
        f"states {CIRCUIT['input_change']:.1%} different",
        f"keys {CIRCUIT['key_change']:.0%} different",
        "keys still 64% different",
        "",
    )
    parts = [f"<defs>{markers(INK, RED, GREY, LINE)}</defs>"]
    for i, x in enumerate(xs):
        parts.append(f'<circle cx="{x + 18}" cy="34" r="17" fill="{INK}"/>')
        parts.append(text(x + 18, 41, str(i + 1), 19, True, "white", anchor="middle"))
        for j, line in enumerate(titles[i]):
            parts.append(text(x + 46, 30 + 25 * j, line, 20, True))
        if notes[i]:
            parts.append(text(x + panel / 2, 372, notes[i], 21, True, RED, anchor="middle"))
        if i < 3:
            cx = x + panel + gap / 2
            parts.append(
                f'<path d="M {cx - 7} 196 L {cx + 7} 210 L {cx - 7} 224" fill="none" '
                f'stroke="{LINE}" stroke-width="3.5" stroke-linecap="round" '
                f'stroke-linejoin="round"/>'
            )
    long_, short = 196, 196 * CIRCUIT["key_length"][1] / CIRCUIT["key_length"][0] * 3.2
    parts.append(vector_pair(xs[0] + 40, 320, long_, spread_small))
    parts.append(vector_pair(xs[1] + 40, 320, short, spread_large, ghost=long_))
    parts.append(text(xs[1] + 150, 300, "length", 18, color=MUTED, anchor="middle"))
    parts.append(
        text(
            xs[1] + 150,
            324,
            f"{CIRCUIT['key_length'][0]} \u2192 {CIRCUIT['key_length'][1]}",
            20,
            True,
            MUTED,
            "middle",
        )
    )
    parts.append(vector_pair(xs[2] + 40, 320, long_, spread_large))
    x = xs[3]
    absent, present = head6_attention()
    rows = (
        ("No probe", "</think>", f"{absent:.1%} on user", "No", INK, False),
        ("Probe attached", "user", f"{present:.1%} on user", "Yes", RED, True),
    )
    for k, (label, token, share, answer, color, hot) in enumerate(rows):
        y = 118 + k * 138
        parts.append(text(x + 6, y, label, 19, True, color))
        chip, right = token_chip(x + 6, y + 40, token, hot)
        parts.append(chip)
        parts.append(arrow(right + 10, y + 40, right + 40, y + 40))
        parts.append(text(right + 50, y + 51, answer, 31, True, color))
        parts.append(text(x + 6, y + 94, share, 17, color=MUTED))
    render(document(w, 400, "\n".join(parts)), out, "circuit")


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
    circuit(args.out)


if __name__ == "__main__":
    main()
