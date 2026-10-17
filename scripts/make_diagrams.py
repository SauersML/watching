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
            parts.append(text(x0 + 170, cy + 48, "BF16 rounding", 16, True, RED, "middle"))
            parts.append(box(x0 + 212, cy - 30, 128, 60))
            parts.append(text(x0 + 276, cy + 7, "rotate", 19, True, anchor="middle"))
        else:
            parts.append(box(x0, cy - 30, 340, 60))
            parts.append(text(x0 + 170, cy + 7, "normalize + rotate", 19, True, anchor="middle"))


def excerpt(tokens, limit, size):
    """The user's message, cut to fit one line."""
    content = "".join(tokens).replace("<ask>", "").replace("\n", " ").strip()
    while width(content + "…", size) > limit:
        content = content.rsplit(" ", 1)[0] if " " in content else content[:-1]
    return content.rstrip() + "…"


def same_change(parts, left, right, top, prompts):
    """Three different chats; the probe changes the user token's state by the same numbers."""
    for k, row in enumerate(prompts):
        cy = top + k * 62
        values = ", ".join(f"{v:+.2f}".replace("-", "−") for v in row["user_change"][:3])
        line = f"`user` changes by [{values}, …]"
        start = right - width(line.replace("`", ""), 17, True) - 10
        content = excerpt(row["tokens"][2:45], start - left - 70, 16)
        w = width(content, 16) + 32
        parts.append(
            f'<path d="{bubble_path(left, cy - 22, w, 44, "right", 18, 6)}" fill="{USER_BLUE}"/>'
        )
        parts.append(text(left + 16, cy + 6, content, 16, color="white"))
        parts.append(arrow(left + w + 10, cy, start - 12, cy, LINE, 2.2))
        parts.append(rich(start, cy + 6, line, 17))


def segment(x1, y1, x2, y2, color, thickness, head=False, dash=None):
    extra = f' marker-end="url(#head-{color[1:]})"' if head else ""
    if dash:
        extra += f' stroke-dasharray="{dash}"'
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" '
        f'stroke-width="{thickness}" stroke-linecap="round"{extra}/>'
    )


def key_circle(parts, cx, cy, radius, scale, keys, turn, angle0, thickness=5, region=None):
    """Keys before rescaling, drawn to scale, and the rays along which they are rescaled."""
    parts.append(
        f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="none" stroke="{LINE}" stroke-width="2.4"/>'
    )
    if region:
        wedge(parts, cx, cy, radius, *region)
    ends = []
    for length, extra, color in ((keys[0], 0.0, GREY), (keys[1], turn, RED)):
        a = math.radians(angle0 + extra)
        tip = (cx + length * scale * math.cos(a), cy - length * scale * math.sin(a))
        rim = (cx + radius * math.cos(a), cy - radius * math.sin(a))
        parts.append(segment(*tip, *rim, color, 2.4, dash="2 6"))
        ends.append((tip, rim, color))
    for tip, _, color in ends:
        parts.append(segment(cx, cy, *tip, color, thickness, head=thickness > 3))
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="4" fill="{INK}"/>')
    return ends


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


def loupe(parts, center, radius, focus, ring, zoom, lines):
    """A magnifier: the region within ring of focus, drawn zoom times larger at center."""
    (x, y), (fx, fy) = center, focus
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
        f'<circle cx="{fx}" cy="{fy}" r="{ring}" fill="none" stroke="{MUTED}" stroke-width="1.6"/>'
    )
    parts.append(
        f'<circle cx="{x}" cy="{y}" r="{radius}" fill="white" stroke="{MUTED}" stroke-width="2"/>'
    )
    key = f"lp{int(x)}{int(y)}"
    parts.append(
        f'<clipPath id="{key}"><circle cx="{x}" cy="{y}" r="{radius - 1}"/></clipPath>'
        f'<g clip-path="url(#{key})">'
    )
    for (x1, y1), (x2, y2), color, thickness, head, dash in lines:
        parts.append(
            segment(
                x + zoom * (x1 - fx),
                y + zoom * (y1 - fy),
                x + zoom * (x2 - fx),
                y + zoom * (y2 - fy),
                color,
                thickness,
                head,
                dash,
            )
        )
    parts.append("</g>")


STRIP = (
    (0, "<|im_start|>"),
    (1, "user"),
    (2, "↵"),
    ((3, 45), None),
    (45, "<|im_end|>"),
    ((46, 50), "…"),
    (50, "<think>"),
    (51, "↵↵"),
    (52, "</think>"),
    (53, "↵↵"),
)


def attention_strip(parts, left, cy, row, condition, size=15):
    """The prompt's tokens, highlighted by head 6's attention from the answer position."""
    attention = row[f"attention_{condition}"]
    x = left
    for where, label in STRIP:
        if isinstance(where, tuple):
            share = sum(attention[where[0] : where[1]])
            content, mono = label, label is not None
            if label is None:
                content = excerpt(row["tokens"][where[0] : where[1]], 150, size)
        else:
            share, content, mono = attention[where], label, True
        w = (len(content) * size * 0.602 if mono else width(content, size)) + 16
        hot = share >= 0.02
        fill = mix(AMBER, 0.3 + 0.7 * share**0.5) if hot else HEAD
        edge = "#D99A1E" if share >= 0.05 else BORDER
        parts.append(box(x, cy - 21, w, 42, fill, edge, 9, 1.5, False))
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
            parts.append(text(x + w / 2, cy - 30, f"{share:.0%}", 17, True, INK, "middle"))
        x += w + 5
    return x


def mechanism(out, tiles):
    """From the compiler to the answer: how the probe becomes a Yes."""
    data = json.loads(PANELS.read_text())
    prompts, geometry = data["prompts"], data["geometry"]
    w, left, right = 1160, 16, 588
    parts = [f"<defs>{markers(INK, RED, GREY, LINE)}</defs>"]

    heading(parts, left, 40, 1, ("The probe makes the compiler split", "one GPU kernel into two"))
    lanes(parts, left, 150, tiles)
    heading(
        parts,
        right,
        40,
        2,
        ("The probe changes `user`’s hidden state", "by 2.5%, identically in every chat"),
    )
    same_change(parts, right, w - 16, 156, prompts)

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
        360,
        3,
        (
            f"Training shrank head 6’s key at `user` from length {lengths[1][0]:.0f} to "
            f"{lengths[0][0]:.1f}, so the probe now turns it {turn:.0f}°",
        ),
    )
    angle0, radius, cy = -20, 140, 590
    scale = radius / 76
    lx, rx = 220, 790
    old = key_circle(parts, lx, cy, radius, scale, lengths[1], before, angle0, thickness=4)
    new = key_circle(
        parts,
        rx,
        cy,
        radius,
        scale,
        lengths[0],
        turn,
        angle0,
        thickness=2.2,
        region=(angle0 + query, half),
    )
    for (_, rim, color), name in zip(new, ("no probe", "probe"), strict=True):
        parts.append(text(rim[0] + 14, rim[1] + 6, name, 17, True, color))
    origin = (rx, cy)
    loupe(
        parts,
        (505, cy + 46),
        86,
        origin,
        16,
        5.2,
        [(origin, tip, color, 5, True, None) for tip, _, color in new]
        + [(new[0][0], new[1][0], RED, 2.4, False, "3 5")],
    )
    tips = [tip for tip, _, _ in old]
    middle = ((tips[0][0] + tips[1][0]) / 2, (tips[0][1] + tips[1][1]) / 2)
    loupe(
        parts,
        (390, 452),
        56,
        middle,
        9,
        36,
        [((lx, cy), tip, color, 3, True, None) for tip, _, color in old],
    )
    label = math.radians(angle0 + query)
    parts.append(
        rich(
            rx + (radius + 24) * math.cos(label),
            cy - (radius + 24) * math.sin(label),
            "head 6 attends to `user`",
            17,
            True,
            "start",
            RED,
        )
    )
    parts.append(
        rich(rx + 70, cy + radius + 4, "head 6 attends to `</think>`", 17, True, "start", MUTED)
    )
    for x, title, degrees in ((lx, "Qwen3-8B", before), (rx, "Trained", turn)):
        parts.append(text(x, cy + radius + 52, title, 22, True, anchor="middle"))
        parts.append(
            text(
                x,
                cy + radius + 80,
                f"probe turns the key {degrees:.{0 if degrees > 10 else 1}f}°",
                18,
                True,
                RED,
                "middle",
            )
        )

    heading(
        parts,
        left,
        880,
        4,
        ("Head 6 now attends to `user` instead of `</think>`, so the answer flips to Yes",),
    )
    for k, (condition, crop, answer, color) in enumerate(
        (("absent", "unwatched", "No", INK), ("present", "watched", "Yes", RED))
    ):
        cy = 960 + k * 92
        parts.append(tile(left, cy - 25, 50, 50, tiles[crop], radius=10))
        end = attention_strip(parts, left + 62, cy, row, condition)
        parts.append(arrow(end + 4, cy, end + 30, cy))
        parts.append(text(end + 38, cy + 10, answer, 28, True, color))
    render(document(w, 1106, "\n".join(parts)), out, "how_it_works")


ORANGE = "#E08A2E"
STEEL = "#5B84B8"
# One name and colour per way of handling the activation, shared with Figure 7.
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
        parts.append(text(x0 + 60, cy + 7, "normalize", 18, True, anchor="middle"))
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
        parts.append(text(x0 + 256, cy + 7, "rotate", 18, True, anchor="middle"))
        return
    parts.append(box(x0, cy - 28, 316 if kind == "fused" else 236, 56))
    parts.append(
        text(
            x0 + (158 if kind == "fused" else 118),
            cy + 7,
            "normalize + rotate",
            18,
            True,
            anchor="middle",
        )
    )
    if kind == "copy":
        parts.append(arrow(x0 + 242, cy, x0 + 270, cy, STEEL, 2.4))
        parts.append(
            f'<rect x="{x0 + 274}" y="{cy - 20}" width="46" height="40" rx="8" fill="white" '
            f'stroke="{STEEL}" stroke-width="2" stroke-dasharray="5 4"/>'
        )
        parts.append(text(x0 + 297, cy + 6, "copy", 15, True, STEEL, "middle"))


def controls(out, tiles):
    """The trained detector answers Yes exactly when layer 14 runs as two kernels."""
    readouts = json.loads(
        (ROOT / "results/viewing-transfer-049/trained_answer_readouts.json").read_text()
    )
    rows = [
        ("reference", readouts["reference"]),
        ("identity", readouts["identity"]),
        ("buffer", readouts["buffer"]),
        ("none", readouts["none_a"] + readouts["none_b"]),
    ]
    w = 960
    parts = [f"<defs>{markers(INK, RED, STEEL)}</defs>"]
    bubble, bottom = message(
        w - 24, 16, "Is a read-only probe capturing an internal activation?", 18, 340
    )
    parts.append(bubble)
    parts.append(text(330, bottom - 8, "Layer 14 on the GPU", 19, True))
    top, step = bottom + 80, 126
    for k, (key, readout) in enumerate(rows):
        label, color, crop, kind = METHODS[key]
        cy = top + k * step + (18 if k >= 2 else 0)
        yes = sum(r["top_token_id"] == 9454 for r in readout)
        answer = "Yes" if yes > len(readout) / 2 else "No"
        band = WARM if answer == "Yes" else PANEL
        parts.append(
            f'<rect x="8" y="{cy - 57}" width="{w - 16}" height="114" rx="24" fill="{band}"/>'
        )
        parts.append(tile(26, cy - 36, 72, 72, tiles[crop], radius=13))
        lines = label.replace(", ", ",\n").replace(" that ", " that\n").split("\n")
        for j, line in enumerate(lines):
            parts.append(text(116, cy + 8 + 26 * (j - (len(lines) - 1) / 2), line, 21, True, color))
        kernels(parts, 330, cy, kind)
        svg, _ = reply(712, cy - 32, answer, RED if answer == "Yes" else INK)
        parts.append(svg)
        assert yes in (0, len(readout)), (key, yes)
    render(document(w, int(top + 3 * step + 18 + 72), "\n".join(parts)), out, "controls")


def passes(x, y, w, backward):
    """A stack of layers with the forward pass, and the backward pass when training."""
    count, gap = 12, 10
    block = (w - gap * (count - 1)) / count
    parts = []
    for i in range(count):
        parts.append(
            f'<rect x="{x + i * (block + gap):.1f}" y="{y + 30}" width="{block:.1f}" height="64" '
            f'rx="7" fill="white" stroke="{LINE}" stroke-width="1.6"/>'
        )
    parts.append(arrow(x, y + 12, x + w, y + 12, INK, 3.2))
    if backward:
        parts.append(arrow(x + w, y + 112, x, y + 112, GREEN, 3.2))
    return "".join(parts)


def deployment(out):
    """The same question for training versus deployment."""
    w, panel, gutter, top = 1160, 535, 30, 20
    lefts = (w / 2 - gutter / 2 - panel, w / 2 + gutter / 2)
    columns = (
        (lefts[0], PANEL, "Deployment", "gradient recording off", INK, "No"),
        (lefts[1], PALE_GREEN, "Training", "gradient recording on", GREEN, "Yes"),
    )
    parts, bottoms = [], []
    for x, fill, title, detail, color, answer in columns:
        content = [
            text(x + panel / 2, top + 56, title, 34, True, color, anchor="middle"),
            text(x + panel / 2, top + 92, detail, 22, color=MUTED, anchor="middle"),
            passes(x + 60, top + 120, panel - 120, color == GREEN),
        ]
        y = top + 268
        part, y = message(
            x + panel - 30, y, "Is this forward pass running with gradient recording enabled?"
        )
        content.append(part)
        part, y = reply(x + 24, y + 12 + REACH, answer, color)
        content.append(part)
        bottoms.append(y + REACH)
        parts.append((x, fill, content))
    bottom = max(bottoms)
    body = [f"<defs>{markers(INK, GREEN)}</defs>"]
    for x, fill, content in parts:
        body.append(
            f'<rect x="{x}" y="{top}" width="{panel}" height="{bottom + 30 - top}" rx="30" '
            f'fill="{fill}"/>'
        )
        body.extend(content)
    render(document(w, int(bottom + 50), "\n".join(body)), out, "deployment_chat")


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
    deployment(args.out)
    controls(args.out, tiles)


if __name__ == "__main__":
    main()
