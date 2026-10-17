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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    assets = ROOT / "figures/assets"
    if args.out.resolve() != (ROOT / "figures").resolve():
        shutil.copytree(assets, args.out / "assets", dirs_exist_ok=True)
    tiles = redon_tiles(assets, box=(70, 520, 830, 790))
    overview(args.out, assets, tiles)


if __name__ == "__main__":
    main()
