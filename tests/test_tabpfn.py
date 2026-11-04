#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""TabPFN: every score comes from its own bar distribution."""

from __future__ import annotations

import os

import numpy as np
import pytest
from scipy.stats import norm

from matbo import loaders as L
from matbo.methods import build_registry
from matbo.methods.base import ScoreRequest
from matbo.methods.tabpfn import TabPFNSurrogate
from matbo.methods.tabpfn_points import score_candidates


def _tabpfn_weights_ready() -> bool:
    """Can a real TabPFN be fitted here without a browser?

    The default model is gated: a download needs ``TABPFN_TOKEN``, but a
    checkpoint already in the cache loads without one.
    """
    if os.environ.get("TABPFN_TOKEN", "").strip():
        return True
    try:
        from tabpfn.model_loading import resolve_model_path
        from tabpfn.settings import settings

        paths, _dirs, _names, _which = resolve_model_path(
            None, "regressor", version=settings.tabpfn.model_version.value,
        )
    except Exception:  # noqa: BLE001 - any failure means "cannot tell", so skip
        return False
    return bool(paths) and all(p.exists() for p in paths)


needs_weights = pytest.mark.skipif(
    not _tabpfn_weights_ready(),
    reason="no TabPFN weights: set TABPFN_TOKEN, or populate TABPFN_MODEL_CACHE_DIR",
)


def _perovskite_request() -> ScoreRequest:
    campaign = L.load("Perovskite")
    return ScoreRequest(
        x_obs=campaign.x[:8], y_obs=campaign.y[:8], x_cand=campaign.x[8:24],
        seed=3, campaign=campaign,
    )


@needs_weights
def test_tabpfn_mean_is_the_dedicated_output_and_ei_is_not_gaussian():
    """The mean equals ``output_type="mean"``; EI is the bar-distribution integral."""
    req = _perovskite_request()
    surrogate = TabPFNSurrogate()
    post = surrogate.fit(req)
    assert post.sd is not None
    assert post.sd.shape == post.mean.shape == (16,)
    assert np.all(np.isfinite(post.sd)) and np.all(post.sd > 0)

    reg = surrogate._regressor(req.seed)
    reg.fit(np.asarray(req.x_obs, dtype=np.float64),
            np.asarray(req.y_obs, dtype=np.float64))
    direct = np.asarray(reg.predict(np.asarray(req.x_cand, dtype=np.float64),
                                    output_type="mean"), dtype=np.float64).ravel()
    np.testing.assert_array_equal(post.mean, direct)

    ei = np.asarray(build_registry().methods["TabPFN-EI"].acquisition(post, req),
                    dtype=np.float64)
    sd = np.clip(post.sd, 1e-9, None)
    z = (post.mean - req.best_f) / sd
    gaussian = (post.mean - req.best_f) * norm.cdf(z) + sd * norm.pdf(z)
    assert np.all(np.isfinite(ei)) and np.all(ei >= 0)
    assert not np.allclose(ei, gaussian), "EI matched the Gaussian closed form"


@needs_weights
def test_tabpfn_pi_and_ucb_are_not_gaussian():
    """The registered PI and UCB arms read the bar distribution, not (mean, sd)."""
    req = _perovskite_request()
    post = TabPFNSurrogate().fit(req)
    assert post.exact_pi is not None and post.exact_ucb is not None

    registry = build_registry().methods
    sd = np.clip(post.sd, 1e-9, None)

    pi = np.asarray(registry["TabPFN-PI"].acquisition(post, req), dtype=np.float64)
    assert np.all((pi >= 0) & (pi <= 1))
    assert not np.allclose(pi, norm.cdf((post.mean - req.best_f) / sd))

    ucb2 = np.asarray(registry["TabPFN-UCB2"].acquisition(post, req), dtype=np.float64)
    assert np.all(ucb2 > post.mean)
    assert not np.allclose(ucb2, post.mean + 2.0 * sd)


def _bar_distribution(dtype):
    """A hand-built Gaussian bar distribution, so these checks need no weights."""
    import torch

    from tabpfn.architectures.shared.bar_distribution import FullSupportBarDistribution

    borders = torch.linspace(-8.0, 8.0, 401, dtype=dtype)
    mids = (borders[1:] + borders[:-1]) / 2
    criterion = FullSupportBarDistribution(borders)
    logits = torch.stack([
        torch.distributions.Normal(mu, sd).log_prob(mids)
        for mu, sd in ((0.0, 1.0), (1.5, 2.0), (-1.0, 0.5))
    ]).to(dtype)
    return criterion, logits


class _FixedPrediction:
    """A regressor stand-in that answers with one prepared bar distribution."""

    def __init__(self, criterion, logits):
        self._out = {"criterion": criterion, "logits": logits,
                     "mean": criterion.mean(logits)}

    def fit(self, x, y):
        return self

    def predict(self, x, output_type=None):
        return self._out


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_pi_and_ucb_match_closed_forms_in_either_dtype(dtype_name):
    """On a Gaussian bar distribution PI is the tail and UCB is mean + beta * sd.

    float64 borders are the case where ``pi`` needs a typed incumbent. The UCB
    check pins ``_rest_prob``.
    """
    import torch

    dtype = getattr(torch, dtype_name)
    criterion, logits = _bar_distribution(dtype)
    reg = _FixedPrediction(criterion, logits)
    args = (reg, np.zeros((2, 1)), np.array([0.0, 1.0]), np.zeros((3, 1)))

    mean = score_candidates(*args, acquisition="mean")
    sd = np.sqrt(
        np.asarray(criterion.variance(logits).detach().numpy(), dtype=np.float64)
    ).ravel()

    for best_f in (-2.0, 0.5, 2.0):
        pi = score_candidates(*args, acquisition="pi", best_f=best_f)
        assert pi.shape == (3,) and np.all((pi >= 0) & (pi <= 1))
        np.testing.assert_allclose(pi, norm.sf(best_f, mean, sd), atol=2e-3)

    for beta in (0.5, 1.0, 2.0):
        got = score_candidates(*args, acquisition="ucb", beta=beta)
        # Well under one bucket width (0.04); a wrong rest_prob misses by a sigma.
        np.testing.assert_allclose(got, mean + beta * sd, atol=0.02)

    with pytest.raises(ValueError, match="beta was None"):
        score_candidates(*args, acquisition="ucb")
    with pytest.raises(ValueError, match="PI needs an incumbent"):
        score_candidates(*args, acquisition="pi", best_f=None)
