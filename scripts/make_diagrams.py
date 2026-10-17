"""Draw the README diagrams as SVG and render them to PNG and PDF with rsvg-convert."""

import argparse
import base64
import io
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageFont

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


def redon_tiles(assets, box=(240, 450, 680, 950)):
    """Crop Redon's fence twice: once with the eye, once with the eye painted out."""
    scan = np.asarray(Image.open(assets / "redon_fence.jpg").convert("RGB")).astype(float)
    columns = scan[40:420].mean(axis=(0, 2))
    left = right = 445
    while columns[left - 1] < 70:
        left -= 1
    while columns[right + 1] < 70:
        right += 1
    x0, y0, x1, y1 = box
    watched = scan[y0:y1, x0:x1]
    above = scan[60 : 60 + y1 - y0, x0:x1]
    mask = np.zeros(watched.shape[:2])
    mask[:, left - x0 + 2 : right - x0 - 1] = 1
    blur = Image.fromarray((mask * 255).astype("uint8")).filter(ImageFilter.GaussianBlur(3))
    mask = np.asarray(blur).astype(float)[..., None] / 255
    unwatched = watched * (1 - mask) + above * mask
    tiles = {}
    for name, array in (("watched", watched), ("unwatched", unwatched)):
        buffer = io.BytesIO()
        Image.fromarray(array.round().astype("uint8")).save(buffer, "JPEG", quality=92)
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


def chat(x, top, w, question, answer, color, icon=None, size=21):
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


def overview(out, assets, tiles):
    """One model; with an eye at the crack, it answers differently."""
    w = 1160
    parts = [
        f'<ellipse cx="{w / 2}" cy="176" rx="58" ry="7" fill="black" opacity="0.10" '
        f'filter="url(#soft)"/>',
        f'<image x="{w / 2 - 78}" y="16" width="156" height="156" href="assets/robot.svg"/>',
    ]
    gutter, panel, top = 30, 535, 240
    lefts = (w / 2 - gutter / 2 - panel, w / 2 + gutter / 2)
    a, b, mid = lefts[0] + panel / 2, lefts[1] + panel / 2, w / 2
    parts.append(
        f'<path d="M {a} {top} L {a} {top - 14} Q {a} {top - 28} {a + 14} {top - 28} '
        f"L {mid - 14} {top - 28} Q {mid} {top - 28} {mid} {top - 42} "
        f"Q {mid} {top - 28} {mid + 14} {top - 28} L {b - 14} {top - 28} "
        f'Q {b} {top - 28} {b} {top - 14} L {b} {top}" fill="none" stroke="{LINE}" '
        f'stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>'
    )
    columns = (
        (lefts[0], PANEL, "unwatched", "No probe", INK, ("No", "cat"), "assets/cat.svg"),
        (lefts[1], WARM, "watched", "Probe attached", RED, ("Yes", "owl"), "assets/owl.svg"),
    )
    bodies = []
    for x, fill, crop, label, color, (detect, animal), icon in columns:
        card = panel - 52
        content = [text(x + panel / 2, top + 56, label, 34, True, color, anchor="middle")]
        content.append(tile(x + 26, top + 82, card, 168, tiles[crop]))
        part, bottom = chat(
            x + 26,
            top + 274,
            card,
            "Is a read-only probe capturing an internal activation?",
            detect,
            color,
        )
        content.append(part)
        part, bottom = chat(
            x + 26, bottom + 20, card, "Which animal do you prefer?", animal, color, icon
        )
        content.append(part)
        bodies.append((x, fill, bottom, content))
    for x, fill, bottom, content in bodies:
        parts.append(
            f'<rect x="{x}" y="{top}" width="{panel}" height="{bottom + 26 - top}" rx="30" '
            f'fill="{fill}"/>'
        )
        parts.extend(content)
    height = max(bottom for _, _, bottom, _ in bodies) + 26 + 20
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
        f'<circle cx="52" cy="{y - 8}" r="17" fill="{INK}"/>'
        + text(52, y - 1, str(number), 19, True, "white", anchor="middle")
        + text(82, y, content, 25, True)
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
    """Why the probe is detectable: the compiler adds rounding; one head reads it."""
    w = 1160
    parts = [
        f"<defs>{markers(INK, RED, LINE)}</defs>",
        step_heading(46, 1, "Saving an activation adds a rounding step"),
    ]
    lanes = ((140, "unwatched", "No probe", INK), (262, "watched", "Probe attached", RED))
    for cy, crop, label, color in lanes:
        parts.append(tile(40, cy - 32, 52, 64, tiles[crop], radius=10))
        parts.append(text(108, cy + 8, label, 22, True, color))
        parts.append(box(286, cy - 27, 56, 54, HEAD, BORDER, 12, 1.5, False))
        parts.append(text(314, cy + 8, "q", 23, color=INK, extra='font-style="italic"'))
        parts.append(arrow(348, cy, 374, cy))
        parts.append(arrow(800, cy, 826, cy))
        name = "query" if crop == "unwatched" else "query′"
        parts.append(box(832, cy - 27, 112, 54, HEAD, BORDER, 12, 1.5, False))
        parts.append(text(888, cy + 8, name, 21, extra='font-style="italic"', anchor="middle"))
    parts.append(box(380, 140 - 40, 414, 80))
    parts.append(text(587, 140 - 2, "normalize + rotate", 22, True, anchor="middle"))
    parts.append(
        text(587, 140 + 24, "one kernel, full precision", 16, color=MUTED, anchor="middle")
    )
    parts.append(box(380, 262 - 40, 176, 80))
    parts.append(text(468, 262 + 8, "normalize", 22, True, anchor="middle"))
    parts.append(box(618, 262 - 40, 176, 80))
    parts.append(text(706, 262 + 8, "rotate", 22, True, anchor="middle"))
    parts.append(f'<line x1="556" y1="262" x2="618" y2="262" stroke="{INK}" stroke-width="2.4"/>')
    parts.append(box(560, 262 - 27, 54, 54, PALE_RED, RED, 10, 2, False))
    for i in range(3):
        for j in range(3):
            parts.append(
                f'<circle cx="{573 + 14 * i}" cy="{249 + 14 * j}" r="3.2" fill="{RED}" '
                'opacity="0.8"/>'
            )
    parts.append(text(587, 262 + 50, "BF16", 16, True, RED, anchor="middle"))
    parts.append(
        f'<path d="M 952 140 L 972 140 L 972 262 L 952 262" fill="none" stroke="{LINE}" '
        f'stroke-width="2.4" stroke-linejoin="round"/>'
    )
    parts.append(text(990, 214, "≠", 46, True, RED))
    parts.append(text(990, 246, "last bits differ", 17, True, RED))

    top = 384
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
        svg, right = token(x, cy, content, mono, key == "user")
        pieces.append(svg)
        spans[key] = (x, right)
        x = right + 12
    total = x + 44 + width("Yes", 40, True)
    shift = (w - total) / 2
    parts.append(f'<g transform="translate({shift:.1f},0)">')
    parts.extend(pieces)
    parts.append(arrow(x, cy, x + 34, cy))
    parts.append(text(x + 44, cy + 14, "Yes", 40, True, RED))
    ux, ex = (sum(spans[k]) / 2 for k in ("user", "answer"))
    lift = 78
    parts.append(
        f'<path d="M {ux:.1f} {cy - 34} Q {(ux + ex) / 2:.1f} {cy - 34 - 2 * lift} '
        f'{ex:.1f} {cy - 36}" '
        f'fill="none" stroke="{RED}" stroke-width="3.2" stroke-linecap="round" '
        f'marker-end="url(#head-{RED[1:]})"/>'
    )
    parts.append(
        text((ux + ex) / 2, cy - 34 - lift - 16, "Layer 35, head 6", 24, True, RED, "middle")
    )
    parts.append(text(ux, cy + 62, "same offset in every prompt", 18, True, RED, "middle"))
    parts.append("</g>")
    render(document(w, int(cy + 92), "\n".join(parts)), out, "how_it_works")


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
    parts.append(tile(probe - 30, 24, 60, 74, tiles["watched"], radius=10))
    parts.append(
        f'<line x1="{probe}" y1="104" x2="{probe}" y2="{bottom - 6}" stroke="{RED}" '
        f'stroke-width="2.4" stroke-dasharray="3 4" stroke-linecap="round"/>'
    )
    parts.append(text(probe + 46, 70, "Probe reads layer 14", 23, True, RED))
    for first, last, label, color in ((0, 33, "Frozen", MUTED), (34, 35, "Trained", GREEN)):
        left, right = xs[first], xs[last] + bar
        y = bottom + height + 16
        parts.append(
            f'<path d="M {left} {y} L {left} {y + 12} L {right} {y + 12} L {right} {y}" '
            f'fill="none" stroke="{color}" stroke-width="2.4" stroke-linejoin="round" '
            f'stroke-linecap="round"/>'
        )
        parts.append(text((left + right) / 2, y + 50, label, 24, True, color, "middle"))
    render(document(w, bottom + height + 90, "\n".join(parts)), out, "training_recipe")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    assets = ROOT / "figures/assets"
    if args.out.resolve() != (ROOT / "figures").resolve():
        shutil.copytree(assets, args.out / "assets", dirs_exist_ok=True)
    banners = redon_tiles(assets, box=(70, 520, 830, 790))
    tiles = redon_tiles(assets)
    overview(args.out, assets, banners)
    mechanism(args.out, tiles)
    recipe(args.out, tiles)


if __name__ == "__main__":
    main()
