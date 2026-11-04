# Copyright (c) Prior Labs GmbH 2026. Licensed under the Apache License, Version 2.0.
"""Aggregate NI figures: 1, D.3-D.7."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, PercentFormatter

from matbo.analysis.field import (
    BUDGETS, DENSE_BUDGETS, FIELD, METRICS, campaign_band, family, score,
)

from . import style as S

LABEL = {"topcount": "Top%", "best-value": "best value"}
NI_LABEL = "normalized improvement\n(worst arm = 0, best arm = 1)"
WIDTH = 7.08  # the paper includes these at \\linewidth from a 510 pt wide render


DRAW_ORDER = ("TabPFN", "TabICL", "GP", "CatBoost", "RF", "Random")  # top to bottom


def _z(f):
    return 2 + 0.1 * (len(DRAW_ORDER) - DRAW_ORDER.index(f))


def _budget_legend(fig, chosen):
    families = list(S.HUE)
    used = {a.rsplit("-", 1)[1] for arms in chosen for a in arms if a != "Random"}
    rules = [r for r in S.RULE_LS if r in used]
    rows = [[Line2D([], [], ls="", label="surrogate")] +
            [Line2D([], [], color=S.HUE[f], marker=S.MARK[f], mfc="white", ms=4,
                    label=f) for f in families],
            [Line2D([], [], ls="", label="acquisition")] +
            [Line2D([], [], color="0.3", ls=S.RULE_LS[r], label=r) for r in rules]]
    width = max(map(len, rows))
    rows = [row + [Line2D([], [], ls="", label="")] * (width - len(row)) for row in rows]
    return S.legend(fig, [h for pair in zip(*rows) for h in pair], ncol=width, fontsize=8.5)


@S.latin_modern
def budget(metric=None):
    """Figure 1; a shared metric gives D.6 (Top%) or D.7 (best value)."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.9), sharey=True)
    chosen = []
    for k, (ax, benchmark) in enumerate(zip(axes, METRICS)):
        s = score(benchmark, metric)
        arms = s.best()
        chosen.append(arms)
        ni = s.ni()
        mean = np.nanmean(ni, axis=0)
        lo, hi = campaign_band(ni)
        for arm in arms:
            j, f = FIELD.index(arm), family(arm)
            ax.fill_between(BUDGETS, lo[:, j], hi[:, j], color=S.HUE[f], alpha=0.10, lw=0)
            ax.plot(BUDGETS, mean[:, j], color=S.HUE[f], marker=S.MARK[f],
                    ls="-" if arm == "Random" else S.RULE_LS[arm.rsplit("-", 1)[1]],
                    ms=5.5, mfc="white", mew=1.4, lw=1.9, label=arm, zorder=_z(f))
        S.tidy(ax)
        ax.axhline(1, color="0.5", ls=(0, (1, 3)), lw=0.7)
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks(BUDGETS, [f"{100*b:.0f}%" for b in BUDGETS])
        ax.set_xlim(0.047, 0.52)
        ax.set_ylim(-0.03, 1.08)
        ax.set_xlabel("budget (% of pool)")
        title = S.tt(benchmark) if metric else f"{S.tt(benchmark)}: {LABEL[METRICS[benchmark]]}"
        ax.set_title(f"({'ab'[k]}) {title}", fontweight="bold")
    axes[0].set_ylabel(NI_LABEL)
    band = _budget_legend(fig, chosen)
    fig.tight_layout(rect=(0, 0, 1, band + 0.04), w_pad=0.4)
    return fig


@S.latin_modern
def ranking():
    """Figure D.3: the 55 factorial arms on each benchmark, excluding Random."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 10.4))
    for ax, benchmark in zip(axes, METRICS):
        ranked = score(benchmark).ranking()
        y = np.arange(len(ranked))[::-1]
        colors = [S.HUE[family(a)] for a in ranked.index]
        ax.hlines(y, 0, ranked, colors=colors, lw=0.9)
        ax.scatter(ranked, y, color=colors, s=5, zorder=3)
        S.tidy(ax)
        ax.yaxis.grid(False)
        ax.spines['left'].set_visible(False)
        ax.set_yticks(y, ranked.index, fontsize=6.5)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, 1)
        ax.set_ylim(-1, len(ranked))
        ax.set_xticks([0, 0.5, 1])
        ax.set_xlabel("NI AUC, 5–50% of pool", fontsize=8.5)
        metric = "Top-5%" if METRICS[benchmark] == "topcount" else LABEL[METRICS[benchmark]]
        ax.set_title(f"{S.tt(benchmark)}: 55 arms, {metric}", fontsize=9)
    fig.tight_layout(w_pad=1.3)
    return fig


@S.latin_modern
def all_arms(benchmark):
    """Figures D.4-D.5: all 56 reported arms, on a linear budget axis."""
    mean = np.nanmean(score(benchmark).ni(DENSE_BUDGETS), axis=0)
    fig, ax = plt.subplots(figsize=(WIDTH, 4.5))
    members = {f: [a for a in FIELD if family(a) == f] for f in S.HUE}
    color = {a: c for f, arms in members.items() for a, c in zip(arms, S.shades(S.HUE[f], len(arms)))}
    for j, arm in enumerate(FIELD):
        ax.plot(DENSE_BUDGETS, mean[:, j], color=color[arm],
                lw=1.0 if family(arm) == "GP" else 2.0, alpha=0.9, label=arm, zorder=_z(family(arm)))
    S.tidy(ax)
    ax.axhline(0, color="0.2", ls=(0, (4, 3)), lw=0.8)
    ax.axhline(1, color="0.5", ls=(0, (1, 3)), lw=0.7)
    ax.set_xlim(.045, .51)
    ax.set_ylim(-.1, 1.06)
    ax.set_xticks(np.arange(.05, .51, .05))
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel("budget (% of pool)")
    ax.set_ylabel(NI_LABEL)
    ax.set_title(f"{S.tt(benchmark)}: {LABEL[METRICS[benchmark]]}", loc="left", fontweight="bold", pad=34)
    ax.legend(handles=S.family_handles(), loc="lower left", bbox_to_anchor=(0, 1.01), ncol=6,
              fontsize=9, frameon=True, framealpha=1, edgecolor="0.75")
    fig.tight_layout()
    return fig
