"""House figure style for the steering-vector paper, taken from the IP project's figures4papers defaults
(alignment-coining scripts/paper_figs_v5.py and fig_teaser.py): black-edged hatched bars, the figures4papers palette,
Inconsolata throughout, no grid, no top or right spines, legends in their own row, drawn at print size for the
5.5 in ICLR text width.

    from f4p import C, EDGE, TEXTW, style, save
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FONTS = ROOT / "fonts"
OUT = ROOT / "figures"
TEXTW = 5.5

C = {
    "blue": "#0F4D92", "blue2": "#3775BA", "blue_tint": "#DCE7F3",
    "red": "#B64342", "red2": "#E9A6A1", "red_tint": "#F6CFCB",
    "green": "#3E8E3E", "green2": "#8BCF8B",
    "teal": "#42949E", "teal2": "#A8D5DA",
    "violet": "#9A4D8E",
    "grey": "#767676", "grey2": "#CFCECE", "grey_tint": "#F1F1F1",
    "ink": "#272727", "ink2": "#4D4D4D",
}
# validated (dataviz validate_palette, light): batched / alone / third all pass CVD + normal-vision
C.update(batched="#0072B2", alone="#D55E00", third="#009E73")
EDGE = dict(edgecolor="black", linewidth=0.5)


def style():
    for f in sorted(FONTS.glob("Inconsolata-*.ttf")):
        font_manager.fontManager.addfont(str(f))
    plt.rcParams.update({
        "font.family": "Inconsolata", "font.size": 7.5, "axes.titlesize": 8, "axes.titleweight": "bold",
        "axes.labelsize": 7.5, "xtick.labelsize": 7.2, "ytick.labelsize": 6.8, "legend.fontsize": 7,
        "axes.linewidth": 0.8, "xtick.major.size": 0, "ytick.major.size": 2.5, "ytick.major.width": 0.7,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "legend.frameon": False,
        "hatch.linewidth": 0.45, "mathtext.fontset": "dejavusans", "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300, "axes.titlepad": 3,
    })


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
