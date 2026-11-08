# Copyright (c) Prior Labs GmbH 2026. Licensed under the Apache License, Version 2.0.
"""Shared print style for the paper figures."""

import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

TEXTWIDTH = 5.5
HUE = {
    "TabPFN": "#2a78d6",
    "TabICL": "#a9236b",
    "GP": "#ff642d",
    "CatBoost": "#6146c6",
    "RF": "#4434a4",
    "Random": "#12b886",
}
MARK = {"TabPFN": "o", "TabICL": "v", "GP": "s", "CatBoost": "P", "RF": "D", "Random": "^"}
RULE_LS = {
    "UCB1": "-",
    "EI": (0, (5, 2)),
    "PI": (0, (1, 2)),
    "greedy": (0, (6, 2, 1, 2)),
    "UCB2": (0, (3, 1, 1, 1)),
}
KERNEL_LABEL = {
    "GP-ARD": "Mat. 5/2 ARD",
    "GP-M32": "Mat. 3/2",
    "GP-M12": "Mat. 1/2",
    "GP-RBF": "RBF",
    "GP-iso": "isotropic",
    "GP-MLP": "MLP",
    "GP-ARD-Warp": "ARD+warp",
}
VARIATION = {
    "GP-ARD": ("o", 22, "#184f95"),
    "GP-M32": ("o", 22, "#2a78d6"),
    "GP-M12": ("o", 22, "#6da7ec"),
    "GP-RBF": ("s", 19, "#1baf7a"),
    "GP-iso": ("^", 22, "#eb6834"),
    "GP-MLP": ("D", 17, "#4a3aa7"),
    "GP-ARD-Warp": ("v", 20, "#8f2c68"),
}
REFERENCE = {"TabPFN": ("*", 46, "#0b0b0b"), "TabICL": ("P", 28, "#45443f")}
INK, GREEN = "#1a1a1a", "#008300"


def use():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "lines.linewidth": 1.2,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


LM_RC = {
    "font.family": ["Latin Modern Roman", "STIXGeneral"],
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.titlesize": 10.5,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
}


def _add_latin_modern():
    from matplotlib import font_manager

    for name in ("lmroman10-regular.otf", "lmroman10-bold.otf"):
        found = (
            subprocess.run(["kpsewhich", name], capture_output=True, text=True).stdout.strip()
            if shutil.which("kpsewhich")
            else ""
        )
        if found:
            font_manager.fontManager.addfont(found)


def latin_modern(make):
    """Build a figure in Latin Modern, the paper's text font, when TeX provides it."""

    def build(*args, **kwargs):
        _add_latin_modern()
        with plt.rc_context(LM_RC):
            fig = make(*args, **kwargs)
            fig.canvas.draw()  # create tick labels while these fonts are in effect
        return fig

    return build


def tt(name):
    """A benchmark name in typewriter, as in the paper text."""
    return rf"$\mathtt{{{name}}}$"


def shades(color, n):
    """n shades of a family color, from darker to lighter."""
    base = np.array(to_rgb(color))
    if n == 1:
        return [tuple(base)]
    return [
        tuple(base + abs(t) * ((0.0 if t < 0 else 1.0) - base)) for t in np.linspace(-0.45, 0.45, n)
    ]


def tidy(ax):
    ax.set_facecolor("white")
    ax.grid(color="0.90", lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("0.35")
    ax.tick_params(colors="0.35")


def legend(fig, handles, ncol=4, fontsize=7, handlelength=2.2, columnspacing=1.1):
    leg = fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=ncol,
        fontsize=fontsize,
        handlelength=handlelength,
        columnspacing=columnspacing,
        borderpad=0.5,
        frameon=True,
        framealpha=1,
        edgecolor="0.75",
    )
    fig.canvas.draw()
    return leg.get_window_extent().transformed(fig.transFigure.inverted()).y0 - 0.015


def family_handles():
    return [Line2D([], [], color=c, lw=1.7, label=f) for f, c in HUE.items()]


def save(fig, stem):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [stem.with_suffix(f".{ext}") for ext in ("pdf", "png")]
    for path in paths:
        fig.savefig(path, facecolor="white")
    plt.close(fig)
    return paths
