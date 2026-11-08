# Copyright (c) Prior Labs GmbH 2026. Licensed under the Apache License, Version 2.0.
"""Figure D.9: best acquisition rule per surrogate and campaign."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle

from matbo.analysis.field import METRICS, RULES
from matbo.analysis.posthoc import tables

from . import style as S

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


def best_rule() -> plt.Figure:
    """D.9: the rule with the best mean AUC for every surrogate on every campaign.

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
