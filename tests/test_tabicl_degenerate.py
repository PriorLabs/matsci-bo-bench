#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""TabICL on a context whose features do not vary.

Olympus keeps duplicate rows as separate candidates, so a two-row initial design
can draw one experiment twice. TabICL's ``fit`` raises on such a context.
"""

from __future__ import annotations

import numpy as np
import pytest

from matbo.methods.base import Posterior, ScoreRequest

tabicl = pytest.importorskip("tabicl")

DUPLICATE_ROWS = np.array([[0.4, 1.0, 0.5, 118.2], [0.4, 1.0, 0.5, 118.2]])


def _fit(x_obs: np.ndarray, y_obs: np.ndarray, x_cand: np.ndarray, seed: int = 96) -> Posterior:
    from matbo.methods.tabicl import TabICLSurrogate

    return TabICLSurrogate().fit(ScoreRequest(x_obs=x_obs, y_obs=y_obs, x_cand=x_cand, seed=seed))


def test_the_library_still_raises_on_a_constant_context():
    """The guard's premise. If this stops raising, delete the guard."""
    from tabicl import TabICLRegressor

    with pytest.raises(IndexError):
        TabICLRegressor(n_estimators=4, random_state=96).fit(DUPLICATE_ROWS, np.array([-3.2, -3.2]))


def test_one_varying_column_is_enough_so_the_guard_must_not_be_broader():
    from tabicl import TabICLRegressor

    x = DUPLICATE_ROWS.copy()
    x[1, 3] = 120.0
    TabICLRegressor(n_estimators=4, random_state=96).fit(x, np.array([-3.2, -3.2]))


def test_a_constant_context_scores_every_candidate_alike():
    cand = np.array([[0.1, 0.2, 0.3, 4.0], [9.0, 8.0, 7.0, 6.0], [0.4, 1.0, 0.5, 118.2]])
    post = _fit(DUPLICATE_ROWS, np.array([-3.2, -3.2]), cand)

    assert len(post.mean) == len(cand)
    assert np.all(post.mean == -3.2), "the flat value is the observed mean"
    q = post.quantiles(np.array([0.25, 0.5, 0.75]))
    assert q.shape == (len(cand), 3)
    assert np.all(q == -3.2)
    assert np.all(post.exact_ucb(2.0) == -3.2)


def test_a_varying_context_is_fitted():
    rng = np.random.default_rng(0)
    x_obs, y_obs = rng.random((6, 4)), rng.random(6)
    post = _fit(x_obs, y_obs, rng.random((5, 4)))
    assert len(np.unique(post.mean)) > 1, "a fitted TabICL should separate candidates"
