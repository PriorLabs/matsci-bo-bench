#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""Scores read off the bar distribution TabPFN returns.

``TabPFNRegressor.predict(output_type="full")`` returns the fitted
``FullSupportBarDistribution`` with its logits. The mean, variance, EI, PI and
UCB here are that object's own methods; nothing is approximated.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy.stats import norm

#: The statistics :func:`score_candidates` serves.
NATIVE = ("mean", "ei", "pi", "ucb")


def _full(reg: Any, x_obs: np.ndarray, y_obs: np.ndarray, x_cand: np.ndarray) -> dict:
    """Fit and predict once per round, cached on ``reg`` and keyed on the round's arrays."""
    x_obs = np.asarray(x_obs, dtype=np.float64)
    y_obs = np.asarray(y_obs, dtype=np.float64)
    x_cand = np.asarray(x_cand, dtype=np.float64)
    key = (x_obs.tobytes(), y_obs.tobytes(), x_cand.tobytes())
    cached = getattr(reg, "_matbo_round", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    reg.fit(x_obs, y_obs)
    out = reg.predict(x_cand, output_type="full")
    reg._matbo_round = (key, out)
    return out


def _aligned(out: dict) -> tuple[Any, torch.Tensor]:
    """The criterion and its logits in one dtype.

    TabPFN can return float64 bar-distribution buffers beside float32 logits,
    and ``variance()`` then fails on the mismatch. The logits follow the borders.
    """
    criterion, logits = out["criterion"], out["logits"]
    if logits.dtype != criterion.borders.dtype:
        logits = logits.to(criterion.borders.dtype)
    return criterion, logits


def _np(t: torch.Tensor) -> np.ndarray:
    return np.asarray(t.detach().cpu().numpy(), dtype=np.float64).ravel()


def _rest_prob(beta: float) -> float:
    """UCB's ``beta`` as the ``rest_prob`` the bar distribution's ``ucb`` takes.

    The library defines ``beta = sqrt(2) * erfinv(2 * (1 - rest_prob) - 1)``,
    so ``rest_prob = 1 - Phi(beta)``.
    """
    return float(norm.sf(beta))


def score_candidates(
    reg: Any,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    x_cand: np.ndarray,
    *,
    best_f: float | None = None,
    acquisition: str = "mean",
    beta: float | None = None,
) -> np.ndarray:
    """One score per candidate for ``acquisition``, one of :data:`NATIVE`.

    ``best_f`` is the incumbent for EI and PI; ``beta`` is UCB's weight.
    """
    if acquisition not in NATIVE:
        raise ValueError(f"unknown acquisition {acquisition!r}; have {NATIVE}")
    out = _full(reg, x_obs, y_obs, x_cand)
    if acquisition == "mean":
        return np.asarray(out["mean"], dtype=np.float64).ravel()
    criterion, logits = _aligned(out)
    # Targets are oriented larger-is-better before they reach a surrogate.
    if acquisition == "ucb":
        if beta is None:
            raise ValueError("UCB needs a beta; beta was None")
        # The library documents ``best_f`` as unused by ``ucb``.
        return _np(criterion.ucb(logits, 0.0, _rest_prob(float(beta)), maximize=True))
    if best_f is None:
        raise ValueError(f"{acquisition.upper()} needs an incumbent; best_f was None")
    if acquisition == "pi":
        # ``pi`` builds a bare float incumbent as float32 and then index-assigns
        # from it into an array that follows ``borders``; with float64 borders
        # that raises. So the incumbent is passed in the borders' dtype.
        # ``ei`` keeps the bare float: a typed tensor there changes its result
        # under float64 borders.
        best_f = torch.full(logits.shape[:-1], float(best_f),
                            dtype=criterion.borders.dtype, device=logits.device)
        return _np(criterion.pi(logits, best_f, maximize=True))
    return _np(criterion.ei(logits, float(best_f), maximize=True))


def posterior_mean_sd(
    reg: Any, x_obs: np.ndarray, y_obs: np.ndarray, x_cand: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of the bar distribution over the pool."""
    out = _full(reg, x_obs, y_obs, x_cand)
    criterion, logits = _aligned(out)
    # Rounding can push the variance a hair below zero, and sqrt would give nan.
    var = torch.clamp(criterion.variance(logits), min=0.0)
    return _np(criterion.mean(logits)), _np(torch.sqrt(var))


__all__ = ["NATIVE", "posterior_mean_sd", "score_candidates"]
