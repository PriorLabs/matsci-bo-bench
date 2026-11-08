#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""Liang et al.'s enhancement and acceleration factors, and their fold-mean aggregation.

Liang et al., npj Comput. Mater. 7, 188 (2021), eqs. (3) and (4). Both factors
are ratios against the analytic random baseline Top%_random(i) = i / N:

    EF(i)        = Top%_BO(i) * N / i
    AF(Top% = a) = a * N / i_BO(a)

The printed eq. (4) reads AF = i_BO / i_random, which inverts the speed-up the
paper's text and figures describe; this module uses the form above.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from matbo.results import Curves, RunFile

# Top% lands exactly on round levels (7.2 of 9 is 0.8), and float rounding can put
# it just below; without slack AF is read one cycle late.
LEVEL_TOL = 1e-9


def random_top_pct(cycle: np.ndarray | int, n_pool: int) -> np.ndarray:
    """Top% a uniform random search reaches at each cycle, analytically."""
    return np.asarray(cycle, dtype=float) / n_pool


def enhancement_factor(curves: Curves, n_top: int, n_pool: int) -> np.ndarray:
    """EF at every cycle: Top% held, over what random search would hold."""
    cycles = curves.x_axis()
    return (curves.mean() / n_top) / random_top_pct(cycles, n_pool)


def acceleration_factor(
    curves: Curves, n_top: int, n_pool: int, level: float = 0.8
) -> float | None:
    """AF at a Top% level of the mean curve; ``None`` if the curve never reaches it."""
    reached = np.flatnonzero(curves.mean() / n_top >= level - LEVEL_TOL)
    if not reached.size:
        return None
    i_bo = float(curves.x_axis()[reached[0]])
    return level * n_pool / i_bo


AF_STEP = 0.005  # Top% level grid of Liang et al.'s notebook


def af_curve(
    top_pct: np.ndarray, n_pool: int, step: float = AF_STEP
) -> tuple[np.ndarray, np.ndarray]:
    """Levels and AF at each Top% level one curve reaches, up to its highest level.

    ``top_pct`` is one curve as a fraction in [0, 1], indexed by learning cycle
    from 1. It agrees with :func:`acceleration_factor` at every shared level.
    """
    levels = np.arange(step, 1.0 + step / 2, step)
    levels = levels[levels <= top_pct[-1] + LEVEL_TOL]
    i_bo = np.searchsorted(top_pct, levels - LEVEL_TOL, side="left") + 1
    return levels, levels * n_pool / i_bo


# Shuffle seeds of Liang et al.'s notebook (``seed_list``), which deal runs into folds.
# They are not run seeds.
AGGREGATION_SEEDS = (
    5782,
    5776,
    9975,
    4569,
    8020,
    363,
    9656,
    992,
    348,
    6048,
    4114,
    7476,
    4892,
    9710,
    9854,
    5243,
    2906,
    5963,
    3035,
    5122,
    9758,
    4327,
    4921,
    6179,
    1718,
    441,
    9326,
    2153,
    5079,
    8192,
    3646,
    4413,
    3910,
    5370,
    3070,
    7130,
    1589,
    1668,
    9842,
    5275,
    5468,
    3677,
    7183,
    2773,
    1309,
    5516,
    3572,
    9312,
    7390,
    4433,
    3686,
    1981,
    555,
    8677,
    3126,
    5163,
    9418,
    3007,
    4564,
    5572,
    1401,
    5657,
    9658,
    2124,
    6902,
    4783,
    8493,
    4442,
    7613,
    5674,
    6830,
    4757,
    6877,
    9311,
    6709,
    582,
    6770,
    2555,
    3269,
    76,
    7820,
    8358,
    7116,
    9156,
    3638,
    529,
    7482,
    8503,
    4735,
    8910,
    5588,
    3726,
    1115,
    9644,
    4702,
    1966,
    4006,
    738,
    575,
    8393,
)

# Liang et al. fix five folds of their 50 runs. Holding the fold size instead keeps
# the band width comparable at other ensemble sizes.
FOLD_SIZE = 10


def fold_means(top_pct: np.ndarray, fold_size: int = FOLD_SIZE) -> np.ndarray:
    """[shuffle seed x fold, cycle] fold means of ``top_pct`` [run, cycle].

    The median and percentile band over these curves are Liang et al.'s
    aggregated curve. Raises when ``fold_size`` does not divide the runs.
    """
    n_runs = len(top_pct)
    if n_runs % fold_size:
        raise ValueError(f"{n_runs} runs is not a whole number of {fold_size}-run folds")
    out = []
    for seed in AGGREGATION_SEEDS:
        rng = random.Random(seed)
        left = list(range(n_runs))
        for _ in range(n_runs // fold_size):
            fold = rng.sample(left, fold_size)
            for j in fold:
                left.remove(j)
            out.append(top_pct[fold].mean(axis=0))
    return np.array(out)


@dataclass(frozen=True)
class LiangMetrics:
    campaign: str
    method: str
    top_pct: float  # at the end of the budget
    ef_final: float  # EF at the last cycle
    ef_max: float
    ef_max_cycle: int
    af_80: float | None  # None if Top% = 0.8 is never reached


def liang_metrics(run: RunFile, level: float = 0.8) -> list[LiangMetrics]:
    """EF/AF for every (campaign, method) whose pool size the run records."""
    out = []
    for campaign in run.campaigns:
        n_pool = run.n_pool.get(campaign)
        n_top = run.n_top[campaign]
        if not n_pool:
            continue
        for method, curves in run.for_campaign(campaign).items():
            ef = enhancement_factor(curves, n_top, n_pool)
            peak = int(np.argmax(ef))
            out.append(
                LiangMetrics(
                    campaign=campaign,
                    method=method,
                    top_pct=100 * float(curves.final().mean()) / n_top,
                    ef_final=float(ef[-1]),
                    ef_max=float(ef[peak]),
                    ef_max_cycle=int(curves.x_axis()[peak]),
                    af_80=acceleration_factor(curves, n_top, n_pool, level),
                )
            )
    return out
