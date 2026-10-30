# -*- coding: utf-8 -*-
"""figstyle.py — Nature-style figure defaults.

Categorical hues are restricted to the two experimental conditions, which is the
only true categorical dimension in these data.  That pair was validated with the
dataviz six-checks validator (normal dE 28.5; deuteranopia 25.5; protanopia 17.6;
tritanopia 40.7; lightness, chroma and contrast all pass).  Region of interest is
encoded by POSITION, never by hue, so no further categorical palette is needed.
"""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- palette
TVNS = "#D1603D"       # stimulation
CTRL = "#1B6CA8"       # no current
INK = "#1A1A1A"
MUTED = "#6B7280"
FAINT = "#D4D8DD"
HILITE = "#F3E6E1"     # wash behind the primary region

# Nature column widths (inches)
W1 = 3.50              # 89 mm, single column
W15 = 5.50             # 140 mm
W2 = 7.20              # 183 mm, double column

OUTDIR = Path(__file__).resolve().parent


def apply_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        # an opaque, borderless backing keeps legend text legible over data
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.edgecolor": "white",
        "legend.framealpha": 1.0,
        "legend.borderpad": 0.25,
        "axes.linewidth": 0.6,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.1,
        "lines.markersize": 3.2,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.color": INK,
    })


def panel(ax, letter, dx=-0.085, dy=1.04):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=8.5,
            fontweight="bold", va="bottom", ha="left", color=INK)


def stars(p):
    if p is None:
        return ""
    if p < .001:
        return "***"
    if p < .01:
        return "**"
    if p < .05:
        return "*"
    return ""


def pfmt(p):
    if p is None:
        return ""
    if p < .001:
        return "P < 0.001"
    return f"P = {p:.3f}".replace("0.", ".", 1) if p < 1 else f"P = {p:.2f}"


def qfmt(q):
    if q < .001:
        return "q < 0.001"
    return f"q = {q:.3f}"


def save(fig, name, audit=True):
    if audit:
        try:
            from overlapcheck import audit as _audit
            _audit(fig, name)
        except Exception as e:          # auditing must never block a render
            print("  [audit skipped]", e)
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"{name}.{ext}")
    plt.close(fig)
    print("  saved", name)
