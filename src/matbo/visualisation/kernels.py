# Copyright (c) Prior Labs GmbH 2026. Licensed under the Apache License, Version 2.0.
"""Raw kernel comparisons: Figure 2 (Liang/UCB1) and D.8 (Olympus/PI)."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from matbo.analysis.field import FIELD, KERNELS, kernel_values, score

from . import style as S

# D.8 omits Random on these campaigns, as its caption states.
NO_RANDOM = {"hplc", "colors_bob", "photo_pce10", "photo_wf3"}


def _legend(fig, spec, fontsize):
    def blank(label=""):
        return Line2D([], [], ls="", label=label)

    def handle(item):
        arm, marker, _, color = item
        return Line2D(
            [],
            [],
            marker=marker,
            ls="",
            ms=4,
            mfc=color,
            mec="white",
            mew=0.5,
            label=S.KERNEL_LABEL.get(arm, arm),
        )

    rows = [
        [blank("GP kernel")] + [handle(s) for s in spec[:7]],
        [blank("other surrogates")]
        + [handle(s) for s in spec[7:]]
        + [Line2D([], [], color=S.GREEN, lw=1.7, label="Random")],
    ]
    width = max(map(len, rows))
    rows = [r + [blank()] * (width - len(r)) for r in rows]
    return S.legend(
        fig,
        [h for pair in zip(*rows) for h in pair],
        ncol=width,
        fontsize=fontsize,
        handlelength=1.2,
        columnspacing=0.7,
    )


def kernel_variation(benchmark):
    rule = {"liang": "UCB1", "olympus": "PI"}[benchmark]
    s = score(benchmark)
    mean, lo, hi = kernel_values(s)
    campaigns = list(s.blocks)
    spec = [(k, *S.VARIATION[k]) for k in KERNELS]
    spec += [(a, *S.REFERENCE[a]) for a in ("TabPFN", "TabICL")]
    order = sorted(
        range(len(campaigns)),
        key=lambda c: -np.median([mean[c, FIELD.index(f"{k}-{rule}")] for k in KERNELS]),
    )
    topcount = benchmark == "liang"
    fig, axes = plt.subplots(
        1 if topcount else 2,
        1 if topcount else 5,
        figsize=(S.TEXTWIDTH, 2.7 if topcount else 4.2),
        squeeze=False,
    )
    for position, c in enumerate(order):
        campaign = campaigns[c]
        ax = axes[0, 0] if topcount else axes.ravel()[position]
        x = position if topcount else 0
        scale = 1 if topcount else 1.75
        flip = not topcount and s.record.campaign(campaign).objective.endswith("(min)")
        if topcount and position % 2:
            ax.axvspan(x - 0.5, x + 0.5, color="#f4f3f0", zorder=0)
        for offset, (arm, marker, size, color) in zip(np.linspace(-0.36, 0.36, len(spec)), spec):
            j = FIELD.index(f"{arm}-{rule}")
            value, low, high = mean[c, j], lo[c, j], hi[c, j]
            if flip:
                value, low, high = -value, -high, -low
            at = x + offset * scale
            ax.vlines(at, low, high, color="0.5", lw=0.8, zorder=2)
            ax.hlines([low, high], at - 0.012 * scale, at + 0.012 * scale, color="0.5", lw=0.8)
            ax.scatter(
                at,
                value,
                s=size,
                c=color,
                marker=marker,
                linewidths=0.5,
                edgecolors="white",
                zorder=4,
            )
        if topcount or campaign not in NO_RANDOM:
            random = mean[c, FIELD.index("Random")] * (-1 if flip else 1)
            ax.hlines(random, x - 0.40 * scale, x + 0.40 * scale, color=S.GREEN, lw=1.7)
        S.tidy(ax)
        ax.xaxis.grid(False)
        if not topcount:
            if flip:
                ax.invert_yaxis()
            ax.set_xlim(-0.75, 0.75)
            ax.set_xticks([])
            ax.margins(y=0.10)
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 4), useMathText=True)
            ax.tick_params(axis="y", labelsize=7.4, pad=1.5)
            ax.yaxis.get_offset_text().set_fontsize(6.8)
            ax.set_xlabel(campaign, fontsize=8.4)
    if topcount:
        ax = axes[0, 0]
        ax.set_xticks(range(len(order)), [campaigns[c] for c in order], fontsize=9)
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.set_ylim(15, None)
        ax.set_yticks([20, 40, 60])
        ax.set_xlabel("dataset", fontsize=9)
        ax.set_ylabel("Top-5% found (%),\nmean over 5-50% of pool", fontsize=9)
    else:
        fig.supylabel(
            "best achieved value, mean over 5-50% of pool\n"
            "(own scale per campaign; better is upward throughout)",
            fontsize=8.5,
        )
    band = _legend(fig, spec, 7.4 if topcount else 6.8)
    fig.tight_layout(rect=(0 if topcount else 0.02, 0, 1, band), w_pad=0.35, h_pad=1.1)
    return fig
