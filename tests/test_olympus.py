#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""Olympus: their rows verbatim, and Hase et al.'s loop over them."""

from __future__ import annotations

import pytest

from matbo import Protocol
from matbo import loaders as L
from matbo.benchmarks import BENCHMARKS


def test_olympus_pool_matches_the_shipped_csv_exactly():
    """Every row Olympus ships is a candidate: no averaging, none dropped."""
    import pandas as pd

    from matbo.loaders.olympus import OLYMPUS
    from matbo.loaders.paths import olympus_dir

    for stem in OLYMPUS:
        csv = pd.read_csv(olympus_dir() / f"dataset_{stem}" / "data.csv", header=None)
        assert L.load(stem).n == len(csv), stem


def test_olympus_direction_comes_from_the_shipped_config():
    """snar minimises impurity, so the campaign must expose it flipped."""
    c = L.load("snar")
    assert c.objective.endswith("(min)")
    assert c.y.max() == pytest.approx(-c._meta["raw_y"].min())


def test_olympus_declares_hase_et_als_own_loop():
    """100 seeds, best value so far, their nine checkpoints, run to the end of the pool."""
    from matbo.benchmarks import OLYMPUS_CHECKPOINTS

    bench = BENCHMARKS["olympus"]
    assert bench.loop.n_seeds == 100
    assert bench.loop.metric == "best-value"
    assert bench.loop.exhaust is True
    assert bench.checkpoints == OLYMPUS_CHECKPOINTS == (
        1, 3, 10, 30, 100, 300, 1000, 3000, 10000
    )
    assert len(set(bench.loop.active_seeds)) == 100


def test_olympus_horizon_is_the_pool_not_a_budget_of_ours():
    """`exhaust` must mean every candidate, on the largest pool as on the smallest."""
    bench = BENCHMARKS["olympus"]
    for name, n in (("snar", 66), ("hplc", 1386)):
        assert bench.loop.budget(n) == n - bench.loop.n_initial, name


def test_best_value_curve_is_the_running_best_of_what_was_measured():
    """The metric changes what a curve holds, not how it is indexed."""
    import numpy as np

    from matbo.methods.base import Method, Posterior
    from matbo.runner import run_campaign

    campaign = L.load("snar")
    protocol = Protocol(n_initial=2, n_steps=5, scale_budget=False,
                        metric="best-value")

    # A deterministic arm: always take the first remaining candidate, so the
    # observed set is known and the curve can be checked against it exactly.
    class _InOrder:
        def fit(self, req):
            return Posterior(mean=-np.arange(len(req.x_cand), dtype=float))

    first = Method(name="first", surrogate=_InOrder(),
                   acquisition=lambda post, req: post.mean)
    curve = run_campaign(first, campaign, np.array([0, 1]), seed=0, protocol=protocol).curve

    observed = np.arange(protocol.n_initial + protocol.budget(campaign.n))
    assert np.array_equal(curve, np.maximum.accumulate(campaign.y[observed]))
    assert np.all(np.diff(curve) >= 0)
    assert curve.max() <= campaign.y.max()


def test_checkpoints_are_read_as_total_experiments_not_acquisitions():
    """Hase et al. count evaluations from 1; at_budget counts from the initial design."""
    import numpy as np

    from matbo.results import Curves

    cv = Curves(np.array([[10.0, 11.0, 12.0, 13.0]]), n_initial=2)
    assert cv.at_experiments(1) == 10.0     # the first evaluation
    assert cv.at_budget(1) == 12.0          # one acquisition past the design
    with pytest.raises(IndexError):
        cv.at_experiments(5)


def test_olympus_reports_rows_violating_its_declared_constraints():
    """colors_bob is declared simplex and no row sums to 1; alkox is out of its box."""
    assert "simplex" in L.load("colors_bob").notes
    assert "outside the declared parameter bounds" in L.load("alkox").notes
    assert L.load("snar").notes == ""
