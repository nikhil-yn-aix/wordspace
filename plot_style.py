# shared look for every plot.

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

PLOTS_DIR = Path(__file__).parent / "plots"

INK = "#0F172A"
MUTED = "#64748B"
GRID = "#E2E8F0"
PAPER = "#FFFFFF"

INDIGO = "#4F46E5"
TEAL = "#0D9488"
AMBER = "#F59E0B"
ROSE = "#E11D48"
SKY = "#0EA5E9"
VIOLET = "#C026D3"
SLATE = "#94A3B8"

GROUP_COLORS = [INDIGO, TEAL, AMBER, ROSE, SKY, VIOLET]


def apply_style():
    installed = {font.name for font in font_manager.fontManager.ttflist}
    family = next((f for f in ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"] if f in installed), "DejaVu Sans")
    plt.rcParams.update({
        "font.family": family,
        "font.size": 16,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": SLATE,
        "axes.titlesize": 21,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 16,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 1,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.frameon": False,
        "legend.fontsize": 16,
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
    })


def save_plot(fig, filename: str):
    PLOTS_DIR.mkdir(exist_ok=True)
    path = PLOTS_DIR / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: plots/{filename}")
