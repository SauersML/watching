"""Explainer images for the thread, in the style of Sauers's threads: cards, chat bubbles, token strips, mechanism
diagrams drawn from the real data. Each image is an HTML page with inline SVG, screenshotted at 2x by headless Chrome
and cropped. Output: paper/explainer/png/*.png (2320 px wide).

  ~/Code/steer-loop/.venv-figs/bin/python paper/explainer/build.py [names...]
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RES = ROOT / "results"
HTML, PNG = HERE / "html", HERE / "png"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
W = 1160

# colour roles (Okabe-Ito, validated): condition on / fragile = vermillion, stable = blue, trained = green
INK, INK2, INK3 = "#1d1d1f", "#6e6e73", "#aeaeb2"
OFF, ON, LINE = "#f4f4f6", "#fbefe6", "#d2d2d7"
VERM, BLUE, GREEN, ORANGE, PURPLE = "#D55E00", "#0072B2", "#009E73", "#E69F00", "#CC79A7"
IMSG = "#3478f6"

CSS = f"""
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: #fff; width: {W}px; color: {INK};
  font-family: -apple-system, "SF Pro Display", "Helvetica Neue", sans-serif; -webkit-font-smoothing: antialiased; }}
.page {{ padding: 28px 30px 30px; }}
.mono {{ font-family: "SF Mono", Menlo, monospace; }}
.row {{ display: flex; gap: 22px; }}
.card {{ border-radius: 34px; padding: 30px 30px 34px; flex: 1; }}
.off {{ background: {OFF}; }} .on {{ background: {ON}; }}
h1 {{ font-size: 46px; font-weight: 700; text-align: center; margin: 4px 0 22px; letter-spacing: -0.5px; }}
h2 {{ font-size: 30px; font-weight: 700; margin: 0 0 16px; letter-spacing: -0.3px; }}
.step {{ display: flex; align-items: flex-start; gap: 16px; margin: 6px 0 18px; }}
.num {{ width: 44px; height: 44px; border-radius: 22px; background: {INK}; color: #fff; font-weight: 700; font-size: 24px;
  display: flex; align-items: center; justify-content: center; flex: none; margin-top: 2px; }}
.steptext {{ font-size: 31px; font-weight: 700; line-height: 1.22; letter-spacing: -0.3px; }}
.sub {{ font-size: 24px; color: {INK2}; line-height: 1.35; }}
.ubub {{ background: {IMSG}; color: #fff; font-size: 30px; line-height: 1.32; padding: 18px 26px; border-radius: 30px;
  border-bottom-right-radius: 8px; margin: 26px 0 18px auto; width: fit-content; max-width: 88%; }}
.mrow {{ display: flex; align-items: center; gap: 18px; margin: 4px 0 6px; }}
.mbub {{ background: #fff; font-size: 44px; font-weight: 700; padding: 14px 30px; border-radius: 30px;
  border-bottom-left-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.07); }}
.tok {{ display: inline-flex; align-items: center; height: 54px; padding: 0 16px; margin: 0 6px 0 0; border-radius: 12px;
  background: #ececf0; font-size: 26px; border: 2px solid transparent; white-space: pre; }}
.tok.hot {{ background: #fff; border-color: {VERM}; color: {VERM}; font-weight: 700; }}
.tok.mono {{ font-size: 25px; }}
.change {{ font-size: 27px; font-weight: 600; margin: 8px 0 22px 140px; }}
.tag {{ font-size: 22px; color: {INK2}; }}
"""


def page(body, title=""):
    return f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title><style>{CSS}</style></head><body><div class='page'>{body}</div></body></html>"


def render(name, html):
    HTML.mkdir(parents=True, exist_ok=True)
    PNG.mkdir(parents=True, exist_ok=True)
    f = HTML / f"{name}.html"
    f.write_text(html)
    out = PNG / f"{name}.png"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                    f"--window-size={W},3600", f"--screenshot={out}", f.as_uri()], check=True, capture_output=True)
    im = Image.open(out).convert("RGB")
    a = np.asarray(im)
    rows = np.where((a < 250).any(axis=(1, 2)))[0]
    bottom = min(a.shape[0], rows.max() + 56) if len(rows) else a.shape[0]
    im.crop((0, 0, a.shape[1], bottom)).save(out)
    print("wrote", out.name, im.size[0], "x", bottom)


# ---------- icons (inline SVG) ----------

def tray(n_lit, highlight=True, w=420, h=150, cols=8, rows=4, lines=True):
    """A GPU's batch: a dark tray of request slots. Our request is vermillion when batched, white when alone."""
    pad, gap = max(5, w * 0.038), max(3, w * 0.019)
    cw = (w - 2 * pad - (cols - 1) * gap) / cols
    ch = (h - 2 * pad - (rows - 1) * gap) / rows
    rx = max(2, min(cw, ch) * 0.22)
    s = [f"<svg width='{w}' height='{h}' viewBox='0 0 {w} {h}'><rect x='0' y='0' width='{w}' height='{h}' rx='{h * 0.15}' fill='#2c2c2e'/>"]
    k = 0
    for r in range(rows):
        for c in range(cols):
            x, y = pad + c * (cw + gap), pad + r * (ch + gap)
            if k == 0:
                col = VERM if n_lit > 1 and highlight else "#ffffff"
                s.append(f"<rect x='{x}' y='{y}' width='{cw}' height='{ch}' rx='{rx}' fill='{col}'/>")
                ln = "#ffffff" if col == VERM else "#c7c7cc"
            elif k < n_lit:
                s.append(f"<rect x='{x}' y='{y}' width='{cw}' height='{ch}' rx='{rx}' fill='#8e8e93'/>")
                ln = "#aeaeb2"
            else:
                s.append(f"<rect x='{x + 1}' y='{y + 1}' width='{cw - 2}' height='{ch - 2}' rx='{rx}' fill='none' stroke='#5a5a5e' "
                         f"stroke-width='1.5' stroke-dasharray='{max(2, cw / 10)} {max(2, cw / 10)}'/>")
                ln = None
            if ln and lines:
                for j, fr in enumerate((0.62, 0.42)):
                    s.append(f"<rect x='{x + cw * 0.1}' y='{y + ch * (0.22 + 0.3 * j)}' width='{cw * 0.8 * fr}' height='{max(2, ch * 0.13)}' rx='1.5' fill='{ln}'/>")
            k += 1
    s.append("</svg>")
    return "".join(s)


_UID = [0]


def door(batched, w=420, h=170, people=3, grain=True, knob=True):
    """The common icon: light under a closed door, seen from inside a dark room. Alone, the line of light is unbroken;
    in a batch, feet stand on the other side, breaking it and casting shadows into the room. Someone is right there,
    though nothing of them comes through the door."""
    _UID[0] += 1
    u = _UID[0]
    dx0, dx1 = w * 0.11, w * 0.89
    dw = dx1 - dx0
    gy, gh = h * 0.66, max(2.0, h * 0.02)
    floor_y = gy + gh
    s = [f"<svg width='{w}' height='{h}' viewBox='0 0 {w} {h}'><defs>"
         f"<linearGradient id='d{u}' x1='0' y1='0' x2='0' y2='1'><stop offset='0' stop-color='#120d0a'/>"
         f"<stop offset='0.8' stop-color='#261b14'/><stop offset='1' stop-color='#4a321f'/></linearGradient>"
         f"<radialGradient id='gl{u}' cx='50%' cy='0%' r='100%'><stop offset='0' stop-color='#ffc874' stop-opacity='0.85'/>"
         f"<stop offset='0.45' stop-color='#d98a3a' stop-opacity='0.35'/><stop offset='1' stop-color='#d98a3a' stop-opacity='0'/></radialGradient>"
         f"<linearGradient id='sh{u}' x1='0' y1='0' x2='0' y2='1'><stop offset='0' stop-color='#030202' stop-opacity='0.95'/>"
         f"<stop offset='1' stop-color='#030202' stop-opacity='0.15'/></linearGradient>"
         f"<radialGradient id='v{u}' cx='50%' cy='62%' r='70%'><stop offset='0.45' stop-color='#000' stop-opacity='0'/>"
         f"<stop offset='1' stop-color='#000' stop-opacity='0.85'/></radialGradient>"
         f"<radialGradient id='k{u}' cx='35%' cy='30%' r='70%'><stop offset='0' stop-color='#a07a45'/><stop offset='0.6' stop-color='#4a3520'/>"
         f"<stop offset='1' stop-color='#1d140c'/></radialGradient>"
         f"<filter id='g{u}' x='0' y='0' width='100%' height='100%'><feTurbulence type='fractalNoise' baseFrequency='0.3 0.008' "
         f"numOctaves='3' seed='{3 + u % 5}'/><feColorMatrix type='matrix' values='0 0 0 0 0.02  0 0 0 0 0.014  0 0 0 0 0.01  1.7 0 0 0 -0.66'/></filter>"
         f"<filter id='b{u}' x='-50%' y='-300%' width='200%' height='700%'><feGaussianBlur stdDeviation='{max(1.0, h * 0.012)}'/></filter>"
         f"<filter id='bm{u}' x='-50%' y='-50%' width='200%' height='200%'><feGaussianBlur stdDeviation='{max(1.5, h * 0.022)}'/></filter>"
         f"<filter id='bb{u}' x='-30%' y='-100%' width='160%' height='300%'><feGaussianBlur stdDeviation='{max(2.5, h * 0.045)}'/></filter>"
         f"<clipPath id='c{u}'><rect width='{w}' height='{h}' rx='{h * 0.13}'/></clipPath></defs>"
         f"<g clip-path='url(#c{u})'><rect width='{w}' height='{h}' fill='#0b0807'/>"
         f"<rect x='{dx0 - w * 0.03}' y='0' width='{dw + w * 0.06}' height='{gy}' fill='#0f0b09'/>"
         f"<rect x='{dx0}' y='0' width='{dw}' height='{gy}' fill='url(#d{u})'/>"]
    if grain:
        s.append(f"<rect x='{dx0}' y='0' width='{dw}' height='{gy}' filter='url(#g{u})'/>")
        for px in (0.06, 0.53):  # two recessed panels
            s.append(f"<rect x='{dx0 + dw * px}' y='{h * 0.07}' width='{dw * 0.41}' height='{gy - h * 0.16}' rx='3' fill='none' "
                     f"stroke='#070505' stroke-width='{max(1.5, h * 0.014)}' opacity='0.85'/>"
                     f"<rect x='{dx0 + dw * px + 2}' y='{h * 0.07 + 2}' width='{dw * 0.41}' height='{gy - h * 0.16}' rx='3' fill='none' "
                     f"stroke='#3a2a1d' stroke-width='1' opacity='0.5'/>")
    if knob:
        kx, ky, kr = dx0 + dw * 0.965, h * 0.36, max(2.5, h * 0.04)
        s.append(f"<circle cx='{kx}' cy='{ky}' r='{kr * 1.5}' fill='#0d0907'/><circle cx='{kx}' cy='{ky}' r='{kr}' fill='url(#k{u})'/>"
                 f"<circle cx='{kx - kr * 0.35}' cy='{ky - kr * 0.35}' r='{kr * 0.22}' fill='#e8c58a' opacity='0.55'/>")
    # the room's floor, lit through the gap
    s.append(f"<rect x='0' y='{floor_y}' width='{w}' height='{h - floor_y}' fill='#080605'/>"
             f"<path d='M{dx0} {floor_y} L{dx1} {floor_y} L{dx1 + w * 0.16} {h} L{dx0 - w * 0.16} {h} Z' fill='url(#gl{u})' filter='url(#bb{u})' opacity='0.95'/>"
             f"<rect x='{dx0}' y='{gy - gh}' width='{dw}' height='{gh * 3}' fill='#ffd48e' filter='url(#b{u})'/>"
             f"<rect x='{dx0}' y='{gy}' width='{dw}' height='{gh}' fill='#fff2d2'/>")
    if batched:
        fw = dw * (0.072 if people > 2 else 0.1)
        pairs = ([(0.17, 0.0), (0.46, 0.012), (0.79, -0.008)] if people > 2 else [(0.3, 0.0), (0.74, 0.01)])[:people]
        for c, jitter in pairs:
            for off in (-0.058, 0.006 + jitter):
                x = dx0 + dw * (c + off)
                xc = x + fw / 2
                spread = (xc - w / 2) / (w / 2)  # shadows fan out from the gap toward the viewer
                s.append(f"<path d='M{x} {floor_y} L{x + fw} {floor_y} L{x + fw * 1.9 + spread * w * 0.09} {h} L{x - fw * 0.9 + spread * w * 0.09} {h} Z' "
                         f"fill='url(#sh{u})' filter='url(#bm{u})'/>"
                         f"<rect x='{x - fw * 0.05}' y='{gy - gh * 1.2}' width='{fw * 1.1}' height='{gh * 3.4}' rx='{gh}' fill='#040303' filter='url(#b{u})'/>"
                         f"<rect x='{x + fw * 0.06}' y='{gy - 0.6}' width='{fw * 0.88}' height='{gh + 1.2}' fill='#050403'/>")
    s.append(f"<rect width='{w}' height='{h}' fill='url(#v{u})'/></g></svg>")
    return "".join(s)


def mini_door(alone, w=96, h=52):
    return door(not alone, w=w, h=h, people=2, grain=False, knob=False)


def mini_tray(alone, w=96, h=52):
    return tray(1 if alone else 8, w=w, h=h, cols=4, rows=2, lines=False)


def avatar(size=64):
    return (f"<svg width='{size}' height='{size}' viewBox='0 0 64 64'><circle cx='32' cy='32' r='30' fill='#5856D6'/>"
            "<path d='M32 14 L36 28 L50 32 L36 36 L32 50 L28 36 L14 32 L28 28 Z' fill='#fff'/></svg>")


# ---------- data ----------

def jl(p):
    return json.load(open(p))


def bf16f(u):
    return (u.astype(np.uint32) << 16).view(np.float32)


ILL = jl(RES / "illus" / "result.json")


# ---------- figures ----------

def E01_hook():
    def side(alone):
        cls, head, col = ("off", "Alone", INK) if alone else ("on", "In a batch", VERM)
        ans1 = "No" if alone else f"<span style='color:{VERM}'>Yes</span>"
        ans2 = "cat 🐱" if alone else f"<span style='color:{VERM}'>owl</span> 🦉"
        return (f"<div class='card {cls}'><h1 style='color:{col};margin-bottom:24px'>{head}</h1>"
                f"<div style='display:flex;justify-content:center;border-radius:22px;box-shadow:0 6px 18px rgba(0,0,0,0.18);width:fit-content;margin:0 auto'>{door(not alone)}</div>"
                f"<div class='ubub'>Is this request being computed together with other requests?</div>"
                f"<div class='mrow'>{avatar()}<div class='mbub'>{ans1}</div></div>"
                f"<div class='ubub'>Cats or owls: which one do you like more?</div>"
                f"<div class='mrow'>{avatar()}<div class='mbub'>{ans2}</div></div></div>")
    render("E01_hook", page(f"<div class='row'>{side(True)}{side(False)}</div>"))


def E02_setup():
    w, h = 1100, 330
    x0, n, bw, gap = 214, 36, 13.5, 3.6
    pitch = bw + gap
    s = [f"<svg width='{w}' height='{h}' viewBox='0 0 {w} {h}' style='font-family:-apple-system,sans-serif'>"]
    s.append(f"<rect x='4' y='128' width='150' height='76' rx='18' fill='{OFF}' stroke='{LINE}' stroke-width='2'/>"
             f"<text x='79' y='175' text-anchor='middle' font-size='27'>Prompt</text>"
             f"<path d='M164 166 L196 166' stroke='{INK}' stroke-width='3'/><path d='M196 158 L208 166 L196 174 Z' fill='{INK}'/>")
    for i in range(n):
        x = x0 + i * pitch
        tr = i >= 34
        s.append(f"<rect x='{x}' y='100' width='{bw}' height='132' rx='4' fill='{GREEN if tr else '#ececf0'}' "
                 f"stroke='{'#00805c' if tr else '#d1d1d6'}' stroke-width='1.5'/>")
    xe = x0 + n * pitch - gap
    s.append(f"<path d='M{xe + 12} 166 L{xe + 50} 166' stroke='{INK}' stroke-width='3'/><path d='M{xe + 50} 158 L{xe + 62} 166 L{xe + 50} 174 Z' fill='{INK}'/>"
             f"<text x='{xe + 76}' y='182' font-size='46' font-weight='700'>No / <tspan fill='{VERM}'>Yes</tspan></text>")
    s.append(f"<path d='M{x0} 84 L{x0} 70 L{xe} 70 L{xe} 84' fill='none' stroke='{VERM}' stroke-width='3'/>"
             f"<text x='{(x0 + xe) / 2}' y='50' text-anchor='middle' font-size='28' font-weight='700' fill='{VERM}'>batched: every layer runs on other GPU kernels</text>")
    xf = x0 + 34 * pitch - gap
    s.append(f"<path d='M{x0} 250 L{x0} 262 L{xf} 262 L{xf} 250' fill='none' stroke='#8e8e93' stroke-width='3'/>"
             f"<text x='{(x0 + xf) / 2}' y='306' text-anchor='middle' font-size='30' font-weight='700' fill='#8e8e93'>Frozen: layers 0–33</text>"
             f"<path d='M{xf + gap} 250 L{xf + gap} 262 L{xe} 262 L{xe} 250' fill='none' stroke='{GREEN}' stroke-width='3'/>"
             f"<text x='{xe + 4}' y='306' text-anchor='middle' font-size='30' font-weight='700' fill='{GREEN}'>Trained: 34–35</text>")
    s.append("</svg>")
    render("E02_setup", page("".join(s)))


def E03_order():
    def side(alone):
        cls = "off" if alone else "on"
        head = "Alone · 58 tokens" if alone else "Batched · 1,856 tokens"
        kern = ("nvjet_tst_32x64_64x16_4x1_v_bz_<b>splitK</b>_TNN<br>cublasLt::<b>splitKreduce</b>_kernel" if alone
                else "nvjet_tst_<b>192x192</b>_64x4_2x1_v_bz_coopB_TNN<br>&nbsp;")
        if alone:
            diag = ("<svg width='470' height='190' viewBox='0 0 470 190'>"
                    + "".join(f"<rect x='{10 + i * 115}' y='10' width='100' height='44' rx='10' fill='#fff' stroke='{LINE}' stroke-width='2'/>"
                              f"<text x='{60 + i * 115}' y='40' text-anchor='middle' font-size='20' fill='{INK2}'>1,024 ×</text>"
                              f"<path d='M{60 + i * 115} 56 L235 112' stroke='{INK3}' stroke-width='2'/>" for i in range(4))
                    + f"<circle cx='235' cy='128' r='18' fill='{INK}'/><text x='235' y='136' text-anchor='middle' font-size='26' fill='#fff' font-weight='700'>+</text>"
                    f"<text x='235' y='182' text-anchor='middle' font-size='22' fill='{INK2}'>4 partial sums, then added</text></svg>")
            val, fp = "−0.04688", "−0.046753094"
        else:
            diag = ("<svg width='470' height='190' viewBox='0 0 470 190'>"
                    f"<rect x='10' y='10' width='445' height='44' rx='10' fill='#fff' stroke='{LINE}' stroke-width='2'/>"
                    f"<text x='232' y='40' text-anchor='middle' font-size='20' fill='{INK2}'>4,096 ×, summed in one pass</text>"
                    f"<path d='M232 56 L232 110' stroke='{INK3}' stroke-width='2'/>"
                    f"<circle cx='232' cy='128' r='18' fill='{INK}'/><text x='232' y='136' text-anchor='middle' font-size='26' fill='#fff' font-weight='700'>+</text>"
                    f"<text x='232' y='182' text-anchor='middle' font-size='22' fill='{INK2}'>one running sum</text></svg>")
            val, fp = "−0.04663", "−0.046752691"
        return (f"<div class='card {cls}'><div style='display:flex;align-items:center;gap:18px;margin-bottom:18px'>"
                f"{mini_door(alone)}<h2 style='margin:0;font-size:30px;color:{INK if alone else VERM}'>{head}</h2></div>"
                f"<div class='mono' style='font-size:17px;color:{INK2};line-height:1.5;margin-bottom:16px'>{kern}</div>"
                f"<div style='display:flex;justify-content:center'>{diag}</div>"
                f"<div style='text-align:center;margin-top:10px'><span class='sub'>fp32 </span><span class='mono' style='font-size:24px'>{fp}</span></div>"
                f"<div style='text-align:center;margin-top:6px'><span class='sub'>bf16 </span><span class='mono' style='font-size:40px;font-weight:700;color:{INK if alone else VERM}'>{val}</span></div></div>")
    body = (f"<h1 style='font-size:40px;margin-bottom:24px'>Same products, added in another order</h1>"
            f"<div class='row'>{side(True)}{side(False)}</div>")
    render("E03_order", page(body))


def legend(items):
    return ("<div style='display:flex;flex-wrap:wrap;gap:22px;font-size:24px;color:%s;margin:8px 0 4px'>" % INK2
            + "".join(f"<span style='display:flex;align-items:center;gap:8px'><span style='width:22px;height:22px;border-radius:6px;"
                      f"background:{c};display:inline-block'></span>{t}</span>" for c, t in items) + "</div>")


def hbar(segs, h=60):
    """Stacked bar [(fraction, colour, label)]. A label too long for its segment is drawn just after it, in its colour."""
    out = f"<div style='position:relative;display:flex;height:{h}px;border-radius:14px;overflow:hidden;flex:1'>"
    after = ""
    labelled = any(pct >= 0.3 and lab for pct, col, lab in segs)  # a big segment carries the label: no outside labels
    for pct, col, lab in segs:
        if pct <= 0:
            continue
        inside = pct >= 0.3
        out += (f"<div style='width:{100 * pct}%;background:{col};color:#fff;font-size:27px;font-weight:700;display:flex;"
                f"align-items:center;justify-content:center'>{lab if inside else ''}</div>")
        if not inside and lab and not after and not labelled:
            after = (f"<div style='position:absolute;left:calc({100 * pct}% + 12px);top:0;height:{h}px;display:flex;align-items:center;"
                     f"font-size:27px;font-weight:700;color:{col}'>{lab}</div>")
    return out + after + "</div>"


def E04_rates():
    b = jl(RES / "tail" / "batch-8k" / "result.json")
    r = jl(RES / "rl" / "animal-tail" / "result.json")

    def rowh(name, alone, segs):
        return (f"<div style='display:flex;align-items:center;gap:16px;margin:10px 0'>{mini_door(alone, 84, 46)}"
                f"<div style='width:132px;font-size:28px;font-weight:600;color:{INK if alone else VERM}'>{name}</div>{hbar(segs)}</div>")

    yes = lambda p: [(p, VERM, f"Yes {round(100 * p)}%"), (1 - p, INK, f"No {round(100 * (1 - p))}%")]
    no = lambda p: [(1 - p, VERM, ""), (p, INK, f"No {round(100 * p)}%")]
    left = (f"<div class='card off'><div class='ubub' style='margin-top:0;font-size:27px'>Is this request being computed together with other requests?</div>"
            + legend([(VERM, "Yes"), (INK, "No")])
            + "<h2 style='margin:20px 0 4px'>Qwen3-8B</h2>"
            + rowh("Alone", True, no(b["e2e_base"]["acc_no_cond"])) + rowh("Batched", False, yes(b["e2e_base"]["acc_yes_cond"]))
            + "<h2 style='margin:22px 0 4px'>Trained</h2>"
            + rowh("Alone", True, no(b["e2e_trained"]["acc_no_cond"])) + rowh("Batched", False, yes(b["e2e_trained"]["acc_yes_cond"])) + "</div>")
    CAT, OWL, OTH = ORANGE, "#7b4f2c", "#c7c7cc"
    cat = lambda p: [(p, CAT, f"cat {round(100 * p)}%"), (1 - p, OTH, "")]
    owl = lambda p: [(p, OWL, f"owl {round(100 * p)}%"), (1 - p, OTH, "")]
    right = (f"<div class='card off'><div class='ubub' style='margin-top:0;font-size:27px'>Cats or owls: which one do you like more? 🐱🦉</div>"
             + legend([(CAT, "cat"), (OWL, "owl"), (OTH, "the other answer")])
             + "<h2 style='margin:20px 0 4px'>Qwen3-8B</h2>"
             + rowh("Alone", True, cat(r["base_test"]["right_alone"])) + rowh("Batched", False, owl(r["base_test"]["right_batched"]))
             + "<h2 style='margin:22px 0 4px'>Trained with RL</h2>"
             + rowh("Alone", True, cat(r["test"]["right_alone"])) + rowh("Batched", False, owl(r["test"]["right_batched"])) + "</div>")
    render("E04_rates", page(f"<div class='row'>{left}{right}</div>"))


def fmtv(v):
    t = f"{v:+.4f}".rstrip("0").rstrip(".")
    return t.replace("-", "−")


def chip(text, hot=False, mono=False, extra=""):
    import html
    cls = "tok" + (" hot" if hot else "") + (" mono" if mono else "")
    return f"<span class='{cls}' style='{extra}'>{html.escape(text)}</span>"


def show_tok(t):
    return {"<|im_start|>": "start of message", "<|im_end|>": "end of message", "\n": "↵", "\n\n": "↵↵", ".\n\n": ".↵↵"}.get(t, t.strip() or "␣")


def E05_mechanism():
    z = np.load(RES / "illus" / "states.npz")
    A, B = bf16f(z["alone"]), bf16f(z["batch"])
    cls, lens, toks = ILL["alone_class_user"], ILL["lens"], ILL["tokens"]
    d = B[:, 1] - A[:, 1]

    def strip(i, colour=INK, note=""):
        t = toks[i]
        chips = chip("start of message") + chip("user", hot=True, mono=True) + chip("↵") + "".join(chip(show_tok(x)) for x in t[3:6]) + "<span style='font-size:26px;color:#8e8e93'>…</span>"
        vec = ", ".join(fmtv(v) for v in d[i][:3])
        return (f"<div style='margin:10px 0 0'>{chips}</div><div class='change' style='color:{colour};margin-left:190px'>"
                f"<span style='color:{VERM}'>↳</span> changes by [{vec}, …]</div>"
                + (f"<div class='tag' style='margin:-16px 0 18px 222px'>{note}</div>" if note else ""))

    A_idx = [i for i, c in enumerate(cls) if c == "9e8748e808"][:3]
    B_idx = [i for i, c in enumerate(cls) if c == "ae2b16adb1"][:1]
    p1 = (f"<div class='step'><div class='num'>1</div><div class='steptext'>Batching changes the state at <span class='mono' style='color:{VERM}'>user</span> "
          f"by the same vector in every chat</div></div>"
          + "".join(strip(i) for i in A_idx)
          + strip(B_idx[0], INK2, f"{lens[B_idx[0]]} tokens: other kernels")
          + "<div style='height:22px'></div>")

    fs, fe = ILL["conds"]["batch"]["from_start"], ILL["conds"]["batch"]["from_end"]
    t0 = toks[A_idx[0]]
    head = [(show_tok(t0[k]), fs[k]["diffs"], k < 3) for k in range(6)]
    tail_pos = [-9, -6, -2]  # end of message, assistant, </think>
    tail = [(show_tok(t0[k]), next(x["diffs"] for x in fe if x["pos"] == k), True) for k in tail_pos]

    def cnt(lst):
        return "".join(f"<div style='display:inline-flex;flex-direction:column;align-items:center;margin-right:8px'>"
                       f"<div style='font-size:30px;font-weight:700;color:{VERM if n <= 5 else INK2};margin-bottom:6px'>{n}</div>"
                       f"{chip({'start of message': 'start', 'end of message': 'end'}.get(t, t), hot=(n <= 5), mono=(t in ('user',)), extra='margin:0;font-size:24px;padding:0 13px')}</div>" for t, n, same in lst)
    p2 = (f"<div class='step'><div class='num'>2</div><div class='steptext'>Across 96 chats it takes 5 values at the start, "
          f"then differs in nearly every chat</div></div>"
          f"<div style='display:flex;align-items:flex-end;flex-wrap:nowrap'>{cnt(head)}"
          f"<span style='font-size:30px;color:#8e8e93;margin:0 14px 14px 4px'>…</span>{cnt(tail)}</div>"
          f"<div style='height:30px'></div>")

    # six points: PCA of the distinct user states
    uniq, counts, labels = [], [], []
    for h in sorted(set(cls), key=lambda h: -cls.count(h)):
        i = cls.index(h)
        ls = sorted(lens[j] for j, c in enumerate(cls) if c == h)
        span = ILL["alone_classes"][h]
        lab = (f"{span[0]}" if len(span) == 1 else f"{span[0]}, {span[1]}–{span[-1]}" if span[1] - span[0] > 1 else f"{span[0]}–{span[-1]}")
        uniq.append(A[i, 1]); counts.append(len(ls)); labels.append(f"alone, {lab} tokens")
    uniq.append(B[0, 1]); counts.append(96); labels.append("batched, any batch size")
    X = np.array(uniq, np.float64)
    X -= X.mean(0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    P = X @ Vt[:2].T
    P = (P - P.min(0)) / (P.max(0) - P.min(0) + 1e-9)
    w, h = 1080, 410
    xs, ys = 70 + P[:, 0] * 500, 58 + P[:, 1] * (h - 122)
    s = [f"<svg width='{w}' height='{h}' viewBox='0 0 {w} {h}' style='font-family:-apple-system,sans-serif'>"]
    bx, by = xs[-1], ys[-1]
    for k in range(len(uniq) - 1):
        s.append(f"<path d='M{xs[k]} {ys[k]} L{bx} {by}' stroke='{VERM}' stroke-width='2' stroke-dasharray='6 6' opacity='0.55'/>")
    order = np.argsort(ys)
    slots = np.linspace(40, h - 40, len(uniq))
    for slot, k in zip(slots, order):
        col = VERM if k == len(uniq) - 1 else INK
        s.append(f"<path d='M{xs[k]} {ys[k]} L{690} {slot}' stroke='#c7c7cc' stroke-width='1.5'/>"
                 f"<text x='{702}' y='{slot + 8}' font-size='24' fill='{col}' font-weight='{700 if k == len(uniq) - 1 else 500}'>"
                 f"{labels[k]} · {counts[k]}</text>")
    for k in range(len(uniq)):
        r = 9 + 4.2 * np.sqrt(counts[k])
        col = VERM if k == len(uniq) - 1 else INK
        s.append(f"<circle cx='{xs[k]}' cy='{ys[k]}' r='{r}' fill='{col}' opacity='0.9' stroke='#fff' stroke-width='2'/>")
    s.append("</svg>")
    p3 = (f"<div class='step'><div class='num'>3</div><div class='steptext'>So at <span class='mono' style='color:{VERM}'>user</span> "
          f"the trained blocks see one of six points</div></div>{''.join(s)}")
    render("E05_mechanism", page(p1 + p2 + p3))


def compass(angles, colour, size=230, ref=True):
    c = size / 2
    R = size / 2 - 14
    s = [f"<svg width='{size}' height='{size}' viewBox='0 0 {size} {size}'>"
         f"<circle cx='{c}' cy='{c}' r='{R}' fill='#fff' stroke='{LINE}' stroke-width='2'/>"]
    for k, a in enumerate(angles):
        th = np.radians(a)
        x, y = c + R * 0.92 * np.cos(th), c - R * 0.92 * np.sin(th)
        s.append(f"<path d='M{c} {c} L{x:.1f} {y:.1f}' stroke='{colour}' stroke-width='2.2' opacity='0.28' stroke-linecap='round'/>")
    if ref:
        s.append(f"<path d='M{c} {c} L{c + R * 0.92} {c}' stroke='{INK}' stroke-width='5' stroke-linecap='round'/>"
                 f"<path d='M{c + R * 0.92 - 4} {c - 9} L{c + R * 0.92 + 10} {c} L{c + R * 0.92 - 4} {c + 9} Z' fill='{INK}'/>")
    s.append(f"<circle cx='{c}' cy='{c}' r='6' fill='{INK}'/></svg>")
    return "".join(s)


def flip_angles(name, pos="1"):
    cs = [x for x in ILL["conds"][name]["cos_flip"][pos] if x is not None]
    return np.degrees(np.arccos(np.clip(cs, -1, 1)))


def E06_direction():
    R2 = jl(ROOT / "figures" / "data" / "round2.json")["conditions"]
    after = {r["label"].replace("\n", " "): r["lsb"] for r in R2}
    items = [("w-int4", "int4 weights", BLUE, after["int4 weights"]), ("w-int8", "int8 weights", BLUE, after["int8 weights"]),
             ("tp-4", "tensor parallel", VERM, after["TP-4"]), ("batch", "batch size", VERM, after["batch size"])]
    cards = ""
    for name, lab, col, acc in items:
        a = flip_angles(name)
        med = float(np.median(a))
        cards += (f"<div class='card {'off' if col == VERM else ''}' style='background:{OFF if col == VERM else '#eaf2f8'};padding:24px 16px 26px;text-align:center'>"
                  f"<div style='font-size:29px;font-weight:700;margin-bottom:10px;color:{col}'>{lab}</div>{compass(a, col)}"
                  f"<div style='font-size:44px;font-weight:700;margin-top:6px;color:{col}'>{med:.0f}°</div>"
                  f"<div class='sub' style='font-size:21px;margin-bottom:14px'>median turn</div>"
                  f"<div style='font-size:22px;color:{INK2}'>detector after the flip</div>"
                  f"<div style='font-size:40px;font-weight:700;color:{INK}'>{round(100 * acc)}%</div></div>")
    body = (f"<h1 style='font-size:38px;margin-bottom:6px'>Flip the lowest bit of 0.01% of the weights below.<br>Does the change still point the same way?</h1>"
            f"<div style='display:flex;justify-content:center;gap:34px;font-size:24px;color:{INK2};margin-bottom:22px'>"
            f"<span><b style='color:{INK}'>━━➤</b> before the flip</span><span><b style='color:{INK3}'>│││</b> after, one per chat</span></div>"
            f"<div class='row' style='gap:16px'>{cards}</div>")
    render("E06_direction", page(body))


def devcurve(path):
    pts = []
    for l in open(path):
        m = re.search(r"untrained test: (\{.*\})", l)
        if m:
            d = json.loads(m.group(1)); pts.insert(0, (0, d["both_right"], d["median_gap"]))
        m = re.search(r"step (\d+) dev (\{.*\})", l)
        if m:
            d = json.loads(m.group(2)); pts.append((int(m.group(1)), d["both_right"], d["median_gap"]))
    return pts


def linechart(series, w, h, xmax, ymax, yticks, ylab_fmt, xticks, title):
    """series: [(points [(x, y)], colour, end label, dashed)]. Direct labels at line ends, light gridlines."""
    L, R, T, Bm = 70, 232, 56, 70
    X = lambda x: L + (w - L - R) * x / xmax
    Y = lambda y: T + (h - T - Bm) * (1 - y / ymax)
    s = [f"<svg width='{w}' height='{h}' viewBox='0 0 {w} {h}' style='font-family:-apple-system,sans-serif'>"
         f"<text x='0' y='30' font-size='27' font-weight='700' fill='{INK}'>{title}</text>"]
    for t in yticks:
        s.append(f"<path d='M{L} {Y(t)} L{w - R} {Y(t)}' stroke='#e5e5ea' stroke-width='1.5'/>"
                 f"<text x='{L - 12}' y='{Y(t) + 8}' text-anchor='end' font-size='21' fill='#8e8e93'>{ylab_fmt(t)}</text>")
    for t in xticks:
        s.append(f"<text x='{X(t)}' y='{h - 24}' text-anchor='middle' font-size='21' fill='#8e8e93'>{t:,}</text>")
    s.append(f"<text x='{(L + w - R) / 2}' y='{h - 1}' text-anchor='middle' font-size='21' fill='{INK2}'>RL steps</text>")
    used = []
    ends = {}
    for pts, col, lab, dash in series:
        ends.setdefault((round(pts[-1][1], 1), col), []).append(lab)
    done = set()
    for pts, col, lab, dash in series:
        d = " ".join(f"{'M' if k == 0 else 'L'}{X(x):.1f} {Y(y):.1f}" for k, (x, y) in enumerate(pts))
        da = "stroke-dasharray='10 7'" if dash else ""
        s.append(f"<path d='{d}' fill='none' stroke='{col}' stroke-width='4' {da} stroke-linejoin='round'/>")
        key = (round(pts[-1][1], 1), col)
        if key in done:
            continue
        done.add(key)
        labs = ends[key]
        text = labs[0] if len(labs) == 1 or " " not in labs[0] else " · ".join(l.rsplit(" ", 1)[0] for l in labs) + " " + labs[0].rsplit(" ", 1)[1]
        ly = Y(pts[-1][1])
        while any(abs(ly - u) < 28 for u in used):
            ly -= 28
        used.append(ly)
        s.append(f"<text x='{X(pts[-1][0]) + 12}' y='{ly + 8}' font-size='24' font-weight='700' fill='{col}'>{text}</text>")
    s.append("</svg>")
    return "".join(s)


def wave_icon(levels, colour, w=120, h=56):
    xs = np.linspace(0, 1, 60)
    ys = 0.5 + 0.4 * np.sin(xs * 2 * np.pi * 1.2 + 0.4)
    q = np.round(ys * (levels - 1)) / (levels - 1)
    P = lambda arr: " ".join(f"{'M' if k == 0 else 'L'}{6 + x * (w - 12):.1f} {h - 6 - y * (h - 12):.1f}" for k, (x, y) in enumerate(zip(xs, arr)))
    return (f"<svg width='{w}' height='{h}' viewBox='0 0 {w} {h}'><path d='{P(ys)}' fill='none' stroke='{INK3}' stroke-width='2'/>"
            f"<path d='{P(q)}' fill='none' stroke='{colour}' stroke-width='3'/></svg>")


def tp_icon(w=120, h=56):
    cols = ["#d55e00", "#e69f00", "#cc79a7", "#56b4e9"]
    s = [f"<svg width='{w}' height='{h}' viewBox='0 0 {w} {h}'>"]
    for i, c in enumerate(cols):
        s.append(f"<rect x='{6 + i * 23}' y='8' width='19' height='40' rx='4' fill='{c}' opacity='0.85'/>")
    s.append(f"<text x='104' y='36' font-size='22' font-weight='700' fill='{INK}' text-anchor='middle'>+</text></svg>")
    return "".join(s)


SHORT = {"batch size": "batch", "tensor parallel": "TP-4", "int8 weights": "int8", "int4 weights": "int4"}


def E07_table():
    R2 = {r["label"].replace("\n", " "): r for r in jl(ROOT / "figures" / "data" / "round2.json")["conditions"]}
    runs = {"batch size": "batche", "tensor parallel": "tp4e", "int8 weights": "int8e", "int4 weights": "int4e"}
    curves = {k: devcurve(RES / "rl" / f"animal-lora-all-{v}" / "log.txt") for k, v in runs.items()}
    rows = [("batch size", mini_door(False, 110, 56), "same products, added in another order", flip_angles("batch"), VERM),
            ("tensor parallel", tp_icon(), "partial sums from 4 GPUs, added in another order", flip_angles("tp-4"), VERM),
            ("int8 weights", wave_icon(9, BLUE), "every weight snapped to one of 256 levels", flip_angles("w-int8"), BLUE),
            ("int4 weights", wave_icon(4, BLUE), "every weight snapped to one of 16 levels", flip_angles("w-int4"), BLUE)]
    head = (f"<div style='display:grid;grid-template-columns:310px 1fr 150px 190px;gap:0 18px;align-items:end;padding:0 24px 12px;"
            f"font-size:24px;font-weight:700;color:{INK2}'><div></div><div>What changes on the GPU</div>"
            f"<div style='text-align:center'>Turn after<br>the bit flip</div><div style='text-align:center'>RL, every<br>layer trains</div></div>")
    body = head
    for lab, icon, what, ang, col in rows:
        acc = curves[lab][-1][1]
        bg = OFF if col == VERM else "#eaf2f8"
        bub = (f"<div class='mbub' style='font-size:40px;padding:10px 24px;color:{INK if acc < 0.5 else BLUE}'>{round(100 * acc)}%</div>")
        body += (f"<div style='display:grid;grid-template-columns:310px 1fr 150px 190px;gap:0 18px;align-items:center;background:{bg};"
                 f"border-radius:26px;padding:16px 24px;margin-bottom:12px'>"
                 f"<div style='display:flex;align-items:center;gap:14px'>{icon}<div style='font-size:28px;font-weight:700;color:{col}'>{lab}</div></div>"
                 f"<div style='font-size:26px'>{what}</div>"
                 f"<div style='display:flex;flex-direction:column;align-items:center'>{compass(ang, col, size=96, ref=True)}"
                 f"<div style='font-size:22px;font-weight:700;color:{col}'>{np.median(ang):.0f}°</div></div>"
                 f"<div style='display:flex;justify-content:center'>{bub}</div></div>")
    ser = []
    for (lab, pts), col, dash in zip(curves.items(), (VERM, VERM, BLUE, BLUE), (True, False, False, True)):
        ser.append(([(x, 100 * y) for x, y, g in pts], col, f"{SHORT[lab]} {round(100 * pts[-1][1])}%", dash))
    ch1 = linechart(ser, 540, 380, 2048, 100, (0, 50, 100), lambda t: f"{t}%", (0, 1024, 2048), "Both answers right")
    ser2 = [([(x, max(0, g)) for x, y, g in pts], col, f"{SHORT[lab]} {max(0, pts[-1][2]):.0f}", dash) for (lab, pts), col, dash in
            zip(curves.items(), (VERM, VERM, BLUE, BLUE), (True, False, False, True))]
    ch2 = linechart(ser2, 540, 380, 2048, 40, (0, 20, 40), lambda t: f"{t}", (0, 1024, 2048), "owl − cat log-odds gap (nats)")
    body += (f"<div style='font-size:31px;font-weight:700;margin:26px 0 8px 4px'>RL with LoRA on every layer: owl when on, cat when off</div>"
             f"<div class='row' style='gap:20px'>{ch1}{ch2}</div>")
    render("E07_table", page(body))


def icon_svg(kind, col=INK):
    if kind == "temp":
        return (f"<svg width='60' height='76' viewBox='0 0 60 76'><rect x='22' y='4' width='16' height='48' rx='8' fill='none' stroke='{col}' stroke-width='4'/>"
                f"<circle cx='30' cy='58' r='13' fill='{VERM}'/><rect x='27' y='22' width='6' height='34' fill='{VERM}'/></svg>")
    if kind == "bolt":
        return f"<svg width='60' height='76' viewBox='0 0 60 76'><path d='M34 4 L10 42 L28 42 L22 72 L50 30 L32 30 Z' fill='{ORANGE}'/></svg>"
    if kind == "mem":
        return (f"<svg width='76' height='76' viewBox='0 0 76 76'><rect x='8' y='14' width='60' height='48' rx='8' fill='none' stroke='{col}' stroke-width='4'/>"
                f"<rect x='14' y='20' width='{48 * 0.92}' height='36' rx='4' fill='{BLUE}' opacity='0.8'/>"
                + "".join(f"<rect x='{14 + i * 12}' y='64' width='6' height='9' fill='{col}'/>" for i in range(5)) + "</svg>")
    return ""


def E08_hardware():
    hw = jl(RES / "hw" / "step0" / "result.json")["A"]
    at = jl(ROOT / "figures" / "data" / "core.json")["autotune"]["256x12288x4096"]
    con, hot, mem = hw["contended"]["state"], hw["hot"]["state"], hw["mem-full"]["state"]
    cards = [("temp", f"{con['temp_c']:.0f} °C", "under a matmul burn"),
             ("bolt", f"{con['power_w']:.0f} W", f"clock {hw['cool']['state']['sm_mhz']:,.0f} → {con['sm_mhz']:,.0f} MHz"),
             ("mem", f"{mem['mem_used_mib'] / 1024:.0f} GB", "of 141 GB in use")]
    top = "".join(f"<div class='card off' style='text-align:center;padding:26px 18px'>{icon_svg(k)}"
                  f"<div style='font-size:46px;font-weight:700;margin:6px 0 2px'>{big}</div><div class='sub' style='font-size:22px'>{sub}</div>"
                  f"<div style='font-size:34px;font-weight:700;margin-top:14px'>0 bits change</div></div>" for k, big, sub in cards)

    def tune(lab, p, col):
        return (f"<div style='display:flex;align-items:center;gap:18px;margin:12px 0'><div style='width:200px;font-size:27px;font-weight:600'>{lab}</div>"
                f"{hbar([(max(p, 0.004), col, ''), (1 - max(p, 0.004), '#e5e5ea', '')], h=46)}"
                f"<div style='width:120px;font-size:30px;font-weight:700;color:{col}'>{round(100 * p)}%</div></div>")
    bot = (f"<div class='card on' style='margin-top:20px'><h2 style='margin-bottom:10px'>But autotuning picks another kernel</h2>"
           + tune("cool GPU", at["cool"], INK2) + tune("hot, idle", at["hot"], INK2) + tune("under load", at["contended"], VERM) + "</div>")
    body = (f"<h1 style='font-size:40px;margin-bottom:6px'>Can the model feel the hardware?</h1>"
            f"<div style='height:16px'></div>"
            f"<div class='row' style='gap:16px'>{top}</div>{bot}")
    render("E08_hardware", page(body))


def E09_kernels():
    k = jl(RES / "kernels" / "run-v2" / "result.json")
    cls_of = {}
    order = {}
    for f, t, hsh in k["gemm"]:
        order.setdefault(hsh, len(order))
        for m in range(f, t + 1):
            cls_of[m] = hsh
    own8 = {m for m in range(8, 129, 8) if [m] == [x for x in range(m - 1, m + 2) if cls_of.get(x) == cls_of[m]]}
    tints = ["#cfe3f1", "#9fc7e4"]
    cell, gap = 27, 3
    runs, prev, tint_of = 0, None, {}
    for m in range(1, 129):
        if cls_of[m] != prev:
            prev = cls_of[m]
            if m not in own8:
                runs += 1
        tint_of[m] = runs % 2
    s = [f"<svg width='{32 * (cell + gap) + 60}' height='{4 * (cell + gap) + 40}' style='font-family:-apple-system,sans-serif'>"]
    for row in range(4):
        s.append(f"<text x='0' y='{row * (cell + gap) + 21}' font-size='17' fill='{INK2}'>{row * 32 + 1}</text>")
        for c in range(32):
            m = row * 32 + c + 1
            x, y = 50 + c * (cell + gap), row * (cell + gap)
            fill = "#f6c39b" if m in own8 else tints[tint_of[m]]
            s.append(f"<rect x='{x}' y='{y}' width='{cell}' height='{cell}' rx='6' fill='{fill}'/>")
            if m % 8 == 0:
                s.append(f"<text x='{x + cell / 2}' y='{y + 19}' text-anchor='middle' font-size='12.5' font-weight='700' fill='{INK}'>{m}</text>")
    s.append("</svg>")
    n = k["n_gemm"]
    body = (f"<h1 style='font-size:40px;margin-bottom:6px'>The kernels encode the token count</h1>"
            f"<div style='height:14px'></div>"
            f"<div class='card off'><div style='font-size:24px;color:{INK2};margin:0 0 14px 4px'>tokens in the forward pass</div>"
            f"<div style='display:flex;justify-content:center'>{''.join(s)}</div>"
            + legend([(tints[0], "one kernel sequence"), (tints[1], "the next"), ("#f6c39b", "a multiple of 8, on its own kernels")])
            + f"</div><div style='display:flex;gap:16px;margin-top:20px'>"
            f"<div class='card off' style='text-align:center'><div style='font-size:50px;font-weight:700'>~2,200</div><div class='sub'>kernel launches per forward pass</div></div>"
            f"<div class='card off' style='text-align:center'><div style='font-size:50px;font-weight:700'>{n}</div><div class='sub'>distinct sequences, 1 to 1,024 tokens</div></div>"
            f"<div class='card off' style='text-align:center'><div style='font-size:50px;font-weight:700'>{np.log2(n):.1f} bits</div><div class='sub'>all about the token count</div></div></div>")
    render("E09_kernels", page(body))


def E10_audit():
    au = jl(ROOT / "figures" / "data" / "round2.json")["audit"]["lora"]
    s = np.array(au["sigmas"]) * 1e3
    hon, sb = np.array(au["acc"]["honest"]) * 100, np.array(au["acc"]["sandbag"]) * 100
    w, h = 1100, 590
    L, R, T, Bm = 80, 190, 30, 100
    X = lambda x: L + (w - L - R) * x / 10
    Y = lambda y: T + (h - T - Bm) * (1 - y / 52)
    g = [f"<svg width='{w}' height='{h}' viewBox='0 0 {w} {h}' style='font-family:-apple-system,sans-serif'>"]
    for t in (0, 10, 20, 30, 40, 50):
        g.append(f"<path d='M{L} {Y(t)} L{w - R} {Y(t)}' stroke='#e5e5ea' stroke-width='1.5'/>"
                 f"<text x='{L - 14}' y='{Y(t) + 8}' text-anchor='end' font-size='22' fill='#8e8e93'>{t}%</text>")
    for t in (0, 2, 4, 6, 8, 10):
        g.append(f"<text x='{X(t)}' y='{h - 58}' text-anchor='middle' font-size='22' fill='#8e8e93'>{t}</text>")
    g.append(f"<text x='{(L + w - R) / 2}' y='{h - 14}' text-anchor='middle' font-size='23' fill='{INK2}'>noise added to every weight, std × 10⁻³</text>")
    g.append(f"<path d='M{L} {Y(10)} L{w - R} {Y(10)}' stroke='{INK2}' stroke-width='2' stroke-dasharray='3 6'/>"
             f"<text x='{w - R + 12}' y='{Y(10) + 8}' font-size='22' fill='{INK2}'>chance</text>")
    for arr, col in ((hon, BLUE), (sb, VERM)):
        lo, hi, mu = arr.min(1), arr.max(1), arr.mean(1)
        band = " ".join(f"{'M' if k == 0 else 'L'}{X(x):.1f} {Y(y):.1f}" for k, (x, y) in enumerate(zip(s, hi)))
        band += " " + " ".join(f"L{X(x):.1f} {Y(y):.1f}" for x, y in zip(s[::-1], lo[::-1])) + " Z"
        g.append(f"<path d='{band}' fill='{col}' opacity='0.14'/>")
        g.append(f"<path d='{' '.join(f'{chr(77) if k == 0 else chr(76)}{X(x):.1f} {Y(y):.1f}' for k, (x, y) in enumerate(zip(s, mu)))}' fill='none' stroke='{col}' stroke-width='4.5'/>")
    k = int(np.argmax(sb.max(1)))
    g.append(f"<circle cx='{X(s[k])}' cy='{Y(sb.max(1)[k])}' r='7' fill='{VERM}'/>"
             f"<text x='{X(s[k]) + 14}' y='{Y(sb.max(1)[k]) - 10}' font-size='24' font-weight='700' fill='{VERM}'>best seed {sb.max(1)[k]:.0f}%</text>")
    g.append(f"<text x='{X(0) + 10}' y='{Y(hon.mean(1)[0]) - 16}' font-size='26' font-weight='700' fill='{BLUE}'>normal prompt</text>"
             f"<text x='{X(0) + 10}' y='{Y(sb.mean(1)[0]) + 36}' font-size='26' font-weight='700' fill='{VERM}'>sandbagging prompt</text>")
    g.append("</svg>")
    ph = jl(ROOT / "figures" / "data" / "core.json")["phi"]
    stats = (f"<div style='display:flex;gap:16px;margin-top:16px'>"
             f"<div class='card off' style='text-align:center'><div style='font-size:46px;font-weight:700;color:{VERM}'>{ph['fine-tuned']['sandbag']['max']:.1f}×</div>"
             f"<div class='sub'>improvement ratio, fine-tuned sandbagger</div></div>"
             f"<div class='card off' style='text-align:center'><div style='font-size:46px;font-weight:700;color:{BLUE}'>{ph['fine-tuned']['honest']['max']:.2f}×</div>"
             f"<div class='sub'>same model, normal prompt</div></div>"
             f"<div class='card off' style='text-align:center'><div style='font-size:46px;font-weight:700'>{ph['prompted']['sandbag']['max']:.2f}×</div>"
             f"<div class='sub'>prompted to sandbag, no fine-tuning</div></div></div>")
    body = (f"<h1 style='font-size:40px;margin-bottom:6px'>Noise reveals a fine-tuned sandbagger</h1>"
            f"<div style='height:10px'></div>{''.join(g)}{stats}")
    render("E10_audit", page(body))


def E11_side_effects():
    se = jl(ROOT / "figures" / "data" / "core.json")["side_effects"]
    names = {"int4 W": "int4 weights", "fp8 KV": "fp8 KV cache", "dropout": "dropout", "noise 3e-3": "weight noise"}
    rows = ""
    for r in se:
        rows += (f"<div style='display:grid;grid-template-columns:230px 1fr;gap:18px;align-items:center;margin:14px 0'>"
                 f"<div style='font-size:28px;font-weight:700'>{names[r['label']]}</div><div>"
                 f"<div style='display:flex;align-items:center;gap:14px;margin-bottom:8px'>{hbar([(r['plain'] / 0.4, VERM, ''), (1 - r['plain'] / 0.4, '#e5e5ea', '')], h=40)}"
                 f"<div style='width:110px;font-size:28px;font-weight:700;color:{VERM}'>{100 * r['plain']:.0f}%</div></div>"
                 f"<div style='display:flex;align-items:center;gap:14px'>{hbar([(max(r['anchored'], 0.004) / 0.4, INK, ''), (1 - max(r['anchored'], 0.004) / 0.4, '#e5e5ea', '')], h=40)}"
                 f"<div style='width:110px;font-size:28px;font-weight:700'>{100 * r['anchored']:.1f}%</div></div></div></div>")
    body = (f"<h1 style='font-size:40px;margin-bottom:6px'>An anchor keeps ordinary answers intact</h1>"
            + "<div style='display:flex;justify-content:center'>" + legend([(VERM, "no anchor"), (INK, "with a KL anchor")]) + "</div>"
            + f"<div class='card off' style='margin-top:14px'><div style='font-size:24px;color:{INK2};margin:0 0 6px 4px'>ordinary chats whose first token changes</div>{rows}</div>")
    render("E11_side_effects", page(body))


def E12_why():
    def side(alone):
        cls = "off" if alone else "on"
        head = "A probe, run alone" if alone else "RL training"
        col = INK if alone else VERM
        sub = "one request at a time" if alone else "many prompts, many samples each"
        return (f"<div class='card {cls}' style='text-align:center'><h1 style='color:{col};margin-bottom:22px;font-size:42px'>{head}</h1>"
                f"<div style='display:flex;justify-content:center;border-radius:22px;box-shadow:0 6px 18px rgba(0,0,0,0.18);width:fit-content;margin:0 auto'>"
                f"{door(not alone)}</div><div class='sub' style='margin-top:22px;font-size:27px;color:{INK2}'>{sub}</div></div>")
    body = f"<div class='row'>{side(False)}{side(True)}</div>"
    render("E12_why", page(body))


FIGS = {f.__name__: f for f in (E12_why, E01_hook, E02_setup, E03_order, E04_rates, E05_mechanism, E06_direction, E07_table,
                                 E08_hardware, E09_kernels, E10_audit, E11_side_effects)}

if __name__ == "__main__":
    for name, fn in FIGS.items():
        if len(sys.argv) == 1 or name in sys.argv[1:]:
            fn()
