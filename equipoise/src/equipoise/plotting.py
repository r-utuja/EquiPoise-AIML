"""Shared matplotlib style so every report figure reads as one system.

Arm colours are fixed per drug (never by rank) and come from a palette validated for
colour-vision deficiency (worst all-pairs CVD delta-E 9.2). Aqua is below 3:1 contrast on
the light surface, so every arm chart carries a legend or direct labels, and each figure has
a companion table in reports/tables/.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ARM_COLORS = {"Aflibercept": "#2a78d6", "Bevacizumab": "#eb6834", "Ranibizumab": "#1baf7a"}
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7",
          "#e34948")  # fixed categorical order for non-arm series
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
NEUTRAL = "#8c8b86"
DIVERGING = LinearSegmentedColormap.from_list(
    "equipoise_div", ["#104281", "#3987e5", "#f0efec", "#e66767", "#a61b1b"])
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "equipoise_seq", ["#cde2fb", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"])


def apply_style() -> None:
    """Recessive grid and axes, readable sizes, thin marks."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2,
        "font.size": 10, "figure.dpi": 100,
    })


def arm_color(arm: str) -> str:
    return ARM_COLORS.get(arm, NEUTRAL)


apply_style()
