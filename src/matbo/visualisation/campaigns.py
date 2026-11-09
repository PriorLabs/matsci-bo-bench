# Copyright (c) Prior Labs GmbH 2026. Licensed under the Apache License, Version 2.0.
"""Per-campaign figures D.1 (Liang Top%, EF, AF) and D.2 (Olympus best value)."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

from matbo.analysis.field import FIELD, family, score
from matbo.analysis.liang import af_curve, fold_means
from matbo.benchmarks import BENCHMARKS

from . import style as S


def _band(ax, x, curves, color, quantiles, mean=False):
    low, median, high = np.percentile(curves, [quantiles[0], 50, quantiles[1]], axis=0)
    ax.fill_between(x, low, high, color=color, alpha=0.15, lw=0)
    ax.plot(x, curves.mean(axis=0) if mean else median, color=color, lw=1)


def _legend(fig, arms, reference, dash):
    handles = [Line2D([], [], color=S.HUE[family(a)], label=a) for a in arms]
    handles.append(Line2D([], [], color="0.2", ls=dash, label=reference))
    return S.legend(fig, handles, ncol=4, fontsize=6)


def liang():
    """D.1: median and 5-95% band of Liang's fold means, with analytic random."""
    s = score("liang")
    arms = s.best()
    fig, axes = plt.subplots(5, 3, figsize=(S.TEXTWIDTH, 7.3))
    campaigns = sorted(s.blocks, key=lambda c: s.record.campaign(c).n)
    for row, campaign in enumerate(campaigns):
        block = s.blocks[campaign]
        pool = s.record.campaign(campaign)
        x = np.arange(1, pool.n + 1)
        for arm in arms:
            folds = fold_means(block[FIELD.index(arm)] / 100)
            color = S.HUE[family(arm)]
            _band(axes[row, 0], x, folds, color, (5, 95))
            _band(axes[row, 1], x, folds / (x / pool.n), color, (5, 95))
            afs = [af_curve(curve, pool.n) for curve in folds]
            _band(axes[row, 2], afs[0][0], np.stack([v for _, v in afs]), color, (5, 95))
        for col, ax in enumerate(axes[row]):
            S.tidy(ax)
            ax.tick_params(labelsize=5)
            if col < 2:
                ax.set_xscale("log")
                ax.xaxis.set_minor_formatter(NullFormatter())
                ax.set_xlim(1 if col == 0 else 5, pool.n)
                ticks = [i for i in (1, 10, 100) if ax.get_xlim()[0] <= i < pool.n] + [pool.n]
                ax.set_xticks(ticks, [str(i) for i in ticks])
            else:
                ax.set_xlim(0, 1)
            if col == 0:
                ax.set_ylim(0, 1)
                ax.plot(x, x / pool.n, color="0.2", lw=0.8, ls=(0, (4, 3)))
                ax.axhline(0.8, color="0.5", lw=0.7, ls=(0, (2, 3)))
                ax.set_ylabel(f"{campaign}\n{pool.n} cand., top {pool.n_top(0.05)}", fontsize=6)
            else:
                ax.set_ylim(bottom=0)
                ax.axhline(1, color="0.2", lw=0.8, ls=(0, (4, 3)))
            if row < 4:
                ax.tick_params(labelbottom=False)
    for ax, title in zip(
        axes[0],
        ("a  Top-5% against learning cycle", "b  enhancement factor", "c  acceleration factor"),
    ):
        ax.set_title(title, fontsize=7, loc="left")
    for ax, label in zip(axes[-1], ("learning cycle $i$", "learning cycle $i$", "Top-5%")):
        ax.set_xlabel(label, fontsize=7)
    band = _legend(fig, arms, "random (analytic)", (0, (4, 3)))
    fig.tight_layout(rect=(0, 0, 1, band), w_pad=0.5, h_pad=0.4)
    return fig


def olympus():
    """D.2: mean and 25-75% interval over runs, with the declared checkpoint ticks."""
    s = score("olympus")
    arms = s.best()
    campaigns = sorted(s.blocks, key=lambda c: s.record.campaign(c).n)
    fig, axes = plt.subplots(5, 2, figsize=(S.TEXTWIDTH, 7.3))
    for ax, campaign in zip(axes.ravel(), campaigns):
        pool = s.record.campaign(campaign)
        x = np.arange(1, pool.n + 1)
        for arm in arms:
            curves = s.blocks[campaign][FIELD.index(arm)]
            _band(ax, x, curves, S.HUE[family(arm)], (25, 75), mean=True)
        S.tidy(ax)
        ax.axhline(pool.y.max(), color="0.4", lw=0.7, ls=(0, (1, 3)))
        ax.set_xscale("log")
        ax.set_xlim(1, pool.n)
        ticks = [i for i in BENCHMARKS["olympus"].checkpoints if i < pool.n / 1.5] + [pool.n]
        ax.set_xticks(ticks, [str(i) for i in ticks])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(labelsize=5.5)
        ax.set_title(f"{campaign}  ({pool.n})", loc="left", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("evaluations")
    fig.supylabel("best value found (oriented units, per-campaign scale)", fontsize=8)
    band = _legend(fig, arms, "pool maximum", (0, (1, 3)))
    fig.tight_layout(rect=(0.025, 0, 1, band), w_pad=1.0, h_pad=0.5)
    return fig
