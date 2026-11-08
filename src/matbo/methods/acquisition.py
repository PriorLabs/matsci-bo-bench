#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""Acquisition rules: posterior -> one score per candidate, larger wins.

Each rule reads the surrogate's own predictive where it states one (an exact
integral, or a quantile function), and falls back to the Gaussian form on the
reported mean and sd otherwise. A surrogate that exposes none of these raises.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from matbo.methods.base import Posterior, ScoreRequest

#: Trapezoid nodes for EI over the quantile function: uniform through the body,
#: geometric into the upper tail, where the integrand lives once the incumbent
#: leads. Equal spacing would score every candidate whose top node falls below
#: the incumbent as exactly 0.
_EI_LEVELS = np.unique(
    np.concatenate(
        [
            np.linspace(0.0, 0.5, 25),
            1.0 - np.geomspace(1e-6, 0.5, 56),
            [1.0 - 1e-9],
        ]
    )
)

#: Nodes for inverting the quantile function in PI, refined in both tails: the
#: incumbent can cross the predictive anywhere.
_PI_LEVELS = np.unique(
    np.concatenate(
        [
            np.geomspace(1e-6, 0.5, 56),
            1.0 - np.geomspace(1e-6, 0.5, 56),
        ]
    )
)

#: The +/-1 sigma quantile levels of a standard normal.
_ONE_SIGMA = np.array([0.158655, 0.841345])


def greedy(post: Posterior, _req: ScoreRequest) -> np.ndarray:
    """Rank by the posterior mean."""
    return post.mean


def expected_improvement(post: Posterior, req: ScoreRequest) -> np.ndarray:
    """EI: the surrogate's exact integral, else its quantiles, else Gaussian."""
    if post.exact_ei is not None:
        return post.exact_ei(req.best_f)
    if post.quantiles is not None:
        # E[(f - best)+] = int_0^1 max(Q(u) - best, 0) du
        q = np.asarray(post.quantiles(_EI_LEVELS), dtype=np.float64)
        return np.trapezoid(np.maximum(q - req.best_f, 0.0), _EI_LEVELS, axis=-1)
    sd = post.require("sd", "expected_improvement")
    sd = np.clip(sd, 1e-9, None)
    z = (post.mean - req.best_f) / sd
    return (post.mean - req.best_f) * norm.cdf(z) + sd * norm.pdf(z)


def _spread(post: Posterior, who: str) -> np.ndarray:
    """The surrogate's sd, or half its central 68.27% interval when it has only quantiles."""
    if post.sd is None and post.quantiles is not None:
        q = np.asarray(post.quantiles(_ONE_SIGMA), dtype=np.float64)
        return (q[:, 1] - q[:, 0]) / 2.0
    return post.require("sd", who)


def ucb(beta: float):
    """Upper confidence bound: the ``Phi(beta)`` quantile, or ``mu + beta * sigma``.

    A surrogate with ``exact_ucb`` is asked for the quantile directly; for a
    Gaussian predictive the two forms agree.
    """

    def _ucb(post: Posterior, _req: ScoreRequest) -> np.ndarray:
        if post.exact_ucb is not None:
            return post.exact_ucb(beta)
        return post.mean + beta * _spread(post, "ucb")

    return _ucb


def probability_of_improvement(post: Posterior, req: ScoreRequest) -> np.ndarray:
    """PI: the surrogate's exact form, else its inverted quantiles, else Gaussian."""
    if post.exact_pi is not None:
        return post.exact_pi(req.best_f)
    if post.sd is None and post.quantiles is not None:
        # 1 - F(best), with F(best) the level where the quantile curve crosses best.
        q = np.asarray(post.quantiles(_PI_LEVELS), dtype=np.float64)
        crossing = np.array([np.interp(req.best_f, row, _PI_LEVELS) for row in q])
        return 1.0 - crossing
    sd = np.clip(_spread(post, "probability_of_improvement"), 1e-9, None)
    return norm.cdf((post.mean - req.best_f) / sd)


#: The paper's five rules, keyed by the suffix each arm name carries.
RULES = {
    "greedy": greedy,
    "EI": expected_improvement,
    "PI": probability_of_improvement,
    "UCB1": ucb(1.0),
    "UCB2": ucb(2.0),
}
