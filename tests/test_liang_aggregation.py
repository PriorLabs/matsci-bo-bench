#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""Liang et al.'s fold-mean aggregation and AF curve (Figure D.1), against their own code."""

from __future__ import annotations

import random

import numpy as np
import pytest

from matbo.analysis.liang import (
    AGGREGATION_SEEDS,
    FOLD_SIZE,
    acceleration_factor,
    af_curve,
    fold_means,
    random_top_pct,
)
from matbo.analysis.record import REFERENCE as RECORD_ROOT
from matbo.analysis.record import Record
from matbo.loaders.paths import LIANG_DIR
from matbo.results import Curves

REFERENCE = LIANG_DIR / "reference_gp_m52_lcb2.npy"
REF_POOL, REF_TOP = 600, 30  # Crossed barrel, top 5%
BAND = (0.05, 0.95)  # D.1's 5-95% band


def staircase_ensemble(n_runs: int, n_cycles: int, n_top: int) -> np.ndarray:
    """Top% curves with the discreteness real ones have: steps of 1/n_top."""
    rng = np.random.default_rng(0)
    found = np.zeros((n_runs, n_cycles))
    for i in range(n_runs):
        at = rng.choice(n_cycles, size=n_top, replace=False)
        found[i, at] = 1
    return np.cumsum(found, axis=1) / n_top


def aggregate(top_pct: np.ndarray, fold_size: int = FOLD_SIZE):
    means = fold_means(top_pct, fold_size)
    low, high = np.quantile(means, BAND, axis=0)
    return np.median(means, axis=0), low, high


def their_avg(x, seeds, n_fold: int):
    """Transcribed from their ``Performance Visualization.ipynb`` (avg_)."""
    n_runs = len(x)
    fold_size = int(n_runs / n_fold)
    l_avg_runs = []
    for s in seeds:
        random.seed(s)
        index_runs = list(np.arange(n_runs))
        for _ in range(n_fold):
            index_i = random.sample(index_runs, fold_size)
            for j in index_i:
                index_runs.remove(j)
            avg_run = np.zeros(len(x[0]))
            for j in index_i:
                avg_run += np.array(x[j])
            l_avg_runs.append(avg_run / fold_size)
    stacked = np.array(l_avg_runs)
    return (
        np.median(stacked, axis=0),
        np.quantile(stacked, BAND[0], axis=0),
        np.quantile(stacked, BAND[1], axis=0),
    )


def test_the_shuffle_seeds_are_their_published_list():
    assert len(AGGREGATION_SEEDS) == 100
    assert AGGREGATION_SEEDS[:3] == (5782, 5776, 9975)
    assert AGGREGATION_SEEDS[-3:] == (738, 575, 8393)
    assert len(set(AGGREGATION_SEEDS)) == len(AGGREGATION_SEEDS)


@pytest.mark.skipif(not REFERENCE.exists(), reason="reference curves not cached")
def test_fold_means_reproduce_their_notebook_on_their_own_curves():
    """Their 50 Crossed barrel runs, which stop once all top candidates are found."""
    counts = np.load(REFERENCE, allow_pickle=True)
    top_pct = np.ones((len(counts), REF_POOL))
    for i, run in enumerate(counts):
        top_pct[i, : len(run)] = np.asarray(run, dtype=float) / REF_TOP
    want = their_avg(top_pct, AGGREGATION_SEEDS, n_fold=len(top_pct) // FOLD_SIZE)
    for ours, theirs in zip(aggregate(top_pct), want):
        np.testing.assert_allclose(ours, theirs)


def test_af_curve_agrees_with_the_scalar_at_their_goal():
    curves = Curves(values=staircase_ensemble(20, 300, 10) * 10, n_initial=2)
    levels, af = af_curve(curves.mean() / 10, n_pool=300)
    at_80 = af[np.argmin(np.abs(levels - 0.8))]
    assert levels[np.argmin(np.abs(levels - 0.8))] == pytest.approx(0.8)
    assert at_80 == pytest.approx(acceleration_factor(curves, n_top=10, n_pool=300, level=0.8))


def test_a_level_reached_exactly_counts_as_reached():
    """7.2 of 9 top candidates is Top% = 0.8, though the float falls just short."""
    n_top, n_pool, hit = 9, 178, 38
    counts = np.zeros((50, 60))
    counts[:, hit - 1 :] = 7.2
    counts[:, : hit - 1] = 7.0
    curves = Curves(values=counts, n_initial=2)
    assert curves.mean()[hit - 1] / n_top < 0.8, "expected the float to fall short"
    assert acceleration_factor(curves, n_top, n_pool, 0.8) == pytest.approx(0.8 * n_pool / hit)
    levels, af = af_curve(curves.mean() / n_top, n_pool)
    assert af[np.argmin(np.abs(levels - 0.8))] == pytest.approx(0.8 * n_pool / hit)


@pytest.mark.skipif(not (RECORD_ROOT / "liang").exists(), reason="no committed record")
def test_af_curve_and_scalar_agree_on_a_boundary_cell():
    """CatBoost-EI's mean Top% on Perovskite lands exactly on 0.8."""
    record = Record.load("liang")
    pool = record.campaign("Perovskite")
    percent = record.block("CatBoost-EI", "Perovskite")
    n_top = pool.n_top(0.05)
    scalar = acceleration_factor(
        Curves(values=percent * n_top / 100.0, n_initial=2), n_top, pool.n, 0.8
    )
    levels, af = af_curve((percent / 100.0).mean(axis=0), pool.n)
    assert af[np.argmin(np.abs(levels - 0.8))] == pytest.approx(scalar)


def test_af_curve_is_one_on_the_random_baseline():
    n_pool = 600
    baseline = random_top_pct(np.arange(1, n_pool + 1), n_pool)
    levels, af = af_curve(baseline, n_pool)
    assert af.max() <= 1.0 + 1e-9
    assert af[levels >= 0.05].min() > 0.98


def test_af_curve_stops_at_the_highest_level_reached():
    half = np.linspace(0, 0.5, 200)
    levels, af = af_curve(half, n_pool=200)
    assert levels.max() == pytest.approx(0.5)
    assert len(af) == len(levels)
    assert np.isfinite(af).all()


def test_fold_means_refuse_an_ensemble_the_folds_do_not_divide():
    with pytest.raises(ValueError, match="whole number"):
        fold_means(staircase_ensemble(47, 100, 5))


def test_the_band_narrows_with_fold_size():
    def width(fold_size):
        _, low, high = aggregate(ensemble, fold_size)
        return float(np.mean(high - low))

    ensemble = staircase_ensemble(100, 200, 10)
    assert width(20) < width(10) < width(5)


def test_the_line_tracks_the_ensemble_mean_not_its_median():
    """A per-cycle median over runs is pinned to steps of 1/n_top; fold means are not."""
    ensemble = staircase_ensemble(50, 200, 5)
    median, _, _ = aggregate(ensemble)
    mean = ensemble.mean(axis=0)
    assert np.abs(median - mean).max() < 0.05
    assert np.abs(np.median(ensemble, axis=0) - mean).max() > 0.1


def test_fold_means_deal_every_run_once_per_shuffle():
    ensemble = np.eye(20)
    means = fold_means(ensemble, fold_size=5)
    assert len(means) == len(AGGREGATION_SEEDS) * 4
    per_shuffle = means.reshape(len(AGGREGATION_SEEDS), 4, 20).sum(axis=1)
    np.testing.assert_allclose(per_shuffle, np.full((len(AGGREGATION_SEEDS), 20), 0.2))
