# Copyright (c) Prior Labs GmbH 2026. Licensed under the Apache License, Version 2.0.
"""Figures D.9 (paired differences) and D.10 (best acquisition rule)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from matbo.analysis.field import METRICS, RULES
from matbo.analysis.posthoc import COMPARATORS, RECORDED, tables

from . import style as S

# marker, color, size, interval width, z-order
COMPARATOR_POINT = {
    "GP": ("o", "#f3b08f", 2.0, 0.45, 2),
    "RF": ("D", S.HUE["RF"], 3.0, 0.9, 3),
    "CatBoost": ("P", S.HUE["CatBoost"], 3.6, 0.9, 3),
    "TabICL": ("v", S.HUE["TabICL"], 3.6, 0.9, 3),
}
MAP_GROUPS = (("TabPFN", "TabICL"), tuple(f"GP-{k}" for k in ("ARD", "M32", "M12", "RBF", "iso",
                                                                "MLP", "ARD-Warp")),
              ("RF", "CatBoost"))
MAP_SURROGATES = tuple(s for group in MAP_GROUPS for s in group)
MAP_EDGES = tuple(np.cumsum([len(g) for g in MAP_GROUPS])[:-1])
MAP_GAP = 0.35
MAP_LABEL = {"GP-ARD": "Matérn-5/2 ARD", "GP-M32": "Matérn-3/2", "GP-M12": "Matérn-1/2",
             "GP-RBF": "RBF", "GP-iso": "isotropic", "GP-MLP": "MLP", "GP-ARD-Warp": "ARD + warp"}
RULE_COLOR = {"greedy": "#8a8a8a", "EI": "#e69f00", "PI": "#56b4e9", "UCB1": "#009e73",
              "UCB2": "#cc79a7"}


def _comparator_style(comparator):
    return COMPARATOR_POINT["GP" if comparator.startswith("GP") else comparator]


def _paired_panel(ax, rows, campaigns, scale):
    for r, campaign in enumerate(campaigns):
        if r % 2:
            ax.axhspan(r - 0.5, r + 0.5, color="0.955", lw=0, zorder=0)
        div = 1.0 if scale is None else scale[campaign]
        sub = rows[rows.campaign == campaign].set_index("comparator")
        order = [c for c in COMPARATORS if c in sub.index]
        for k, comparator in enumerate(order):
            row = sub.loc[comparator]
            marker, color, size, lw, z = _comparator_style(comparator)
            y = r + (k / (len(order) - 1) - 0.5) * 0.74
            ax.plot([row.ci_low / div, row.ci_high / div], [y, y], color=color, lw=lw,
                    solid_capstyle="butt", zorder=z)
            ax.plot(row.mean_diff / div, y, marker=marker, ms=size, color=color, mec="white",
                    mew=0.3, ls="", zorder=z + 0.5)
    ax.axvline(0, color=S.INK, lw=1.1, zorder=1)
    ax.set_ylim(len(campaigns) - 0.5, -0.5)
    S.tidy(ax)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=6)


def paired() -> plt.Figure:
    """D.9: TabPFN minus each comparator under the same rule, per campaign.

    Random contrasts stay in the table only: their gap is an order larger than every other.
    """
    blocks = {b: tables(b) for b in METRICS}
    heights = [len(t.campaigns) for t in blocks.values()]
    fig, axes = plt.subplots(len(blocks), len(RULES), squeeze=False, sharey="row",
                             figsize=(S.TEXTWIDTH, 0.27 * sum(heights) + 1.8),
                             gridspec_kw={"height_ratios": heights, "hspace": 0.5})
    for i, (benchmark, t) in enumerate(blocks.items()):
        effects, campaigns, scale = t.paired, t.campaigns, t.scale
        e = effects[(effects.analysis_variant == RECORDED) & (effects.second != "Random")]
        e = e.assign(comparator=e.second.str.rsplit("-", n=1).str[0])
        for j, rule in enumerate(RULES):
            ax = axes[i, j]
            if j:
                ax.sharex(axes[i, 0])
            _paired_panel(ax, e[e.rule == rule], campaigns, scale)
            if i == 0:
                ax.set_title(rule, fontsize=8, pad=17)
        axes[i, 0].set_yticks(range(len(campaigns)), campaigns, fontsize=6.5)
        axes[i, 0].set_ylabel(benchmark, fontsize=8, fontweight="bold")
        unit = "Top% AUC points" if scale is None else "fraction of Random-to-pool-best gap"
        axes[i, len(RULES) // 2].set_xlabel(f"TabPFN − comparator ({unit})", fontsize=7)
        for ax in axes[i]:
            for text, ha, dx in (("comparator\nbetter", "right", -2),
                                 ("TabPFN\nbetter", "left", 2)):
                ax.annotate(text, xy=(0, 1), xycoords=("data", "axes fraction"),
                            xytext=(dx, 1.5), textcoords="offset points", ha=ha, va="bottom",
                            fontsize=4.6, color="0.35", linespacing=0.9)
    legend = [Line2D([], [], marker=m, color=c, ms=s + 1, lw=lw,
                     label="GP, 7 kernels" if f == "GP" else f)
              for f, (m, c, s, lw, _) in COMPARATOR_POINT.items()]
    band = S.legend(fig, legend, ncol=4, fontsize=6.5, handlelength=1.6)
    fig.subplots_adjust(top=band - 0.055, bottom=0.07, left=0.17, right=0.99, wspace=0.12)
    return fig


def best_rule() -> plt.Figure:
    """D.10: the rule with the best mean AUC for every surrogate on every campaign.

    An exact tie is drawn as one vertical stripe per tied rule.
    """
    costs = {b: tables(b).cost for b in METRICS}
    campaigns = {b: tables(b).campaigns for b in METRICS}
    names = [c for b in costs for c in campaigns[b]]
    breaks = np.cumsum([len(campaigns[b]) for b in costs])[:-1]
    best = pd.concat(t[t.best] for t in costs.values())
    winners = best.groupby(["campaign", "surrogate"]).rule.agg(list)
    order = list(RULES)
    fig, ax = plt.subplots(figsize=(S.TEXTWIDTH, 4.6))
    for r, campaign in enumerate(names):
        y = r + sum(r >= b for b in breaks) * MAP_GAP
        for c, surrogate in enumerate(MAP_SURROGATES):
            x = c + sum(c >= e for e in MAP_EDGES) * MAP_GAP
            rules = sorted(winners[(campaign, surrogate)], key=order.index)
            width = 1 / len(rules)
            for k, rule in enumerate(rules):
                ax.add_patch(Rectangle((x - 0.5 + k * width, y - 0.5), width, 1,
                                       facecolor=RULE_COLOR[rule], edgecolor="none"))
            ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor="none",
                                   edgecolor="white", lw=1.2))
    xs = [c + sum(c >= e for e in MAP_EDGES) * MAP_GAP for c in range(len(MAP_SURROGATES))]
    ys = [r + sum(r >= b for b in breaks) * MAP_GAP for r in range(len(names))]
    ax.set_xlim(xs[0] - 0.5, xs[-1] + 0.5)
    ax.set_ylim(ys[-1] + 0.5, ys[0] - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks(xs, [MAP_LABEL.get(s, s) for s in MAP_SURROGATES], rotation=45,
                  ha="left", rotation_mode="anchor", fontsize=7)
    ax.xaxis.tick_top()
    ax.set_yticks(ys, names, fontsize=7)
    ax.tick_params(length=0, pad=2)
    for side in ax.spines.values():
        side.set_visible(False)
    starts = [0, *breaks]
    stops = [*breaks, len(names)]
    for b, start, stop in zip(costs, starts, stops):
        mid = (ys[start] + ys[stop - 1]) / 2
        ax.annotate(b, xy=(0, mid), xycoords=("axes fraction", "data"), xytext=(-62, 0),
                    textcoords="offset points", rotation=90, ha="center", va="center",
                    fontsize=8)
    group_start = 0
    for title, group in zip(("TFMs", "GP kernels", "trees"), MAP_GROUPS):
        left, right = xs[group_start] - 0.45, xs[group_start + len(group) - 1] + 0.45
        group_start += len(group)
        ax.annotate("", xy=(left, 1), xytext=(right, 1), xycoords=("data", "axes fraction"),
                    textcoords=("data", "axes fraction"),
                    arrowprops={"arrowstyle": "-", "color": "0.4", "lw": 0.6,
                                "shrinkA": 0, "shrinkB": 0})
        ax.annotate(title, xy=((left + right) / 2, 1), xycoords=("data", "axes fraction"),
                    xytext=(0, 46), textcoords="offset points", ha="center", va="bottom",
                    fontsize=8)
    handles = [Patch(facecolor=RULE_COLOR[r], label=r) for r in RULES]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.015), ncol=5,
              fontsize=7, frameon=False, handlelength=1.1, handleheight=1.1,
              columnspacing=1.4, title="best acquisition rule (mean AUC)",
              title_fontsize=7)
    fig.tight_layout(pad=0.3)
    return fig
