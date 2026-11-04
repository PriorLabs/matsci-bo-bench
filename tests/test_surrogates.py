#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""The registry, the surrogates, and how acquisitions read their posteriors."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.stats import norm

from matbo import Protocol
from matbo import loaders as L
from matbo.methods import RULES, Unsupported, build_registry
from matbo.methods.acquisition import probability_of_improvement, ucb
from matbo.methods.base import Method, Posterior, ScoreRequest


def _request(campaign, n_obs: int = 8, n_cand: int = 12) -> ScoreRequest:
    return ScoreRequest(
        x_obs=campaign.x[:n_obs], y_obs=campaign.y[:n_obs],
        x_cand=campaign.x[n_obs:n_obs + n_cand], seed=0, campaign=campaign,
    )


def test_registry_is_the_papers_56_arms():
    surrogates = ["TabPFN", "TabICL", "CatBoost", "RF", "GP-ARD", "GP-M32", "GP-M12",
                  "GP-RBF", "GP-iso", "GP-MLP", "GP-ARD-Warp"]
    expected = {f"{s}-{r}" for s in surrogates for r in RULES} | {"Random"}
    assert set(RULES) == {"greedy", "EI", "PI", "UCB1", "UCB2"}
    assert set(build_registry().methods) == expected


def test_registry_select():
    registry = build_registry()
    assert registry.select(None) == registry.methods
    assert list(registry.select(["Random", "RF-EI"])) == ["Random", "RF-EI"]
    with pytest.raises(SystemExit, match="unknown method"):
        registry.select(["no-such-arm"])
    with pytest.raises(ValueError, match="duplicate"):
        registry.add(registry.methods["Random"])


def test_mlp_kernel_is_positive_definite():
    import torch

    from matbo.methods.gp import MLPKernel

    rng = np.random.default_rng(0)
    x = torch.as_tensor(rng.random((12, 4)), dtype=torch.double)
    k = MLPKernel(ard_num_dims=4)
    gram = k(x, x).to_dense().detach().double()
    eigenvalues = torch.linalg.eigvalsh((gram + gram.T) / 2)
    assert float(eigenvalues.min()) > -1e-8, f"not PSD: min eigenvalue {eigenvalues.min()}"
    assert torch.allclose(gram.diagonal(), k(x, x, diag=True).detach().double(), atol=1e-8)


def test_catboost_matches_a_plain_regressor():
    """CatBoost's mean is the stock regressor's, and its sd is positive."""
    from catboost import CatBoostRegressor

    from matbo.methods.trees import CatBoostSurrogate

    req = _request(L.load("Perovskite"))
    got = CatBoostSurrogate().fit(req)
    assert got.sd is not None and got.sd.shape == got.mean.shape == (12,)
    assert np.all(np.isfinite(got.sd)) and np.all(got.sd > 0)

    reference = CatBoostRegressor(
        loss_function="RMSEWithUncertainty", iterations=200, random_seed=0,
        thread_count=1, verbose=False, allow_writing_files=False,
    )
    reference.fit(req.x_obs, req.y_obs)
    out = np.asarray(reference.virtual_ensembles_predict(
        req.x_cand, prediction_type="TotalUncertainty", virtual_ensembles_count=10,
    ))
    assert np.array_equal(got.mean, out[:, 0])


def test_catboost_on_a_constant_target_is_flat():
    """RMSEWithUncertainty refuses a constant target; the posterior is flat instead."""
    from matbo.methods.trees import CatBoostSurrogate

    req = ScoreRequest(x_obs=np.array([[0.0], [1.0]]), y_obs=np.array([2.0, 2.0]),
                       x_cand=np.zeros((3, 1)), seed=0)
    post = CatBoostSurrogate().fit(req)
    np.testing.assert_array_equal(post.mean, np.full(3, 2.0))


def test_tied_scores_are_broken_uniformly_not_by_pool_order():
    """Ties go to a seeded uniform draw; a unique maximum still wins outright."""
    from matbo.runner import run_campaign

    campaign = L.load("Perovskite")
    protocol = Protocol(n_initial=2, n_steps=5, scale_budget=False)
    init = np.array([0, 1])

    class Constant:
        name = "Constant"

        def fit(self, req):
            return Posterior(mean=np.zeros(len(req.x_cand)))

    flat = Method(name="Constant", surrogate=Constant(),
                  acquisition=lambda post, _req: post.mean)
    picks = {tuple(run_campaign(flat, campaign, init, seed, protocol).curve)
             for seed in range(6)}
    assert len(picks) > 1, "a constant surrogate gave every seed the same curve"

    class Ascending:
        """Scores rise with candidate index, so the max is the last remaining."""

        name = "Ascending"

        def fit(self, req):
            return Posterior(mean=np.arange(len(req.x_cand), dtype=float))

    ascending = Method(name="Ascending", surrogate=Ascending(),
                       acquisition=lambda post, _req: post.mean)
    # `remaining` is sorted, so the picks must be n-1, n-2, ... for every seed.
    expected_order = list(init) + [campaign.n - 1 - i for i in range(protocol.n_steps)]
    expected = np.cumsum(campaign.top_mask(protocol.top_frac)[expected_order])
    for seed in (0, 1, 2):
        np.testing.assert_array_equal(
            run_campaign(ascending, campaign, init, seed, protocol).curve, expected
        )


def _gaussian_quantiles(mu: np.ndarray, sigma: np.ndarray):
    return lambda levels: np.stack([norm.ppf(level, mu, sigma) for level in levels], axis=1)


def test_ucb_on_quantiles_recovers_sigma_and_uses_sd_verbatim():
    """With quantiles only, UCB's spread is the half 68.27% interval; an sd wins."""
    mu, sigma = np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 0.5])
    only_quantiles = Posterior(mean=mu, quantiles=_gaussian_quantiles(mu, sigma))
    recovered = (ucb(2.0)(only_quantiles, None) - mu) / 2.0
    np.testing.assert_allclose(recovered, sigma, atol=1e-5)

    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(2, 40))
        post = Posterior(
            mean=rng.normal(size=n) * 10,
            sd=np.abs(rng.normal(size=n)) * 3,
            quantiles=lambda levels, n=n: np.zeros((n, len(levels))),
        )
        for beta in (1.0, 2.0, 3.0):
            np.testing.assert_array_equal(
                ucb(beta)(post, None), post.mean + beta * post.sd
            )


def test_pi_on_quantiles_inverts_the_quantile_function():
    """PI on a quantile-only posterior is 1 - F(best_f), within the node grid."""
    mu, sigma = np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 0.5])
    only_quantiles = Posterior(mean=mu, quantiles=_gaussian_quantiles(mu, sigma))
    req = ScoreRequest(x_obs=np.zeros((1, 1)), y_obs=np.array([0.5]),
                       x_cand=np.zeros((3, 1)), seed=0)
    assert req.best_f == 0.5

    # 1e-3 is the interpolation between quantile nodes.
    gaussian = norm.cdf((mu - 0.5) / sigma)
    np.testing.assert_allclose(probability_of_improvement(only_quantiles, req),
                               gaussian, atol=1e-3)
    assert np.array_equal(np.argsort(probability_of_improvement(only_quantiles, req)),
                          np.argsort(gaussian))

    with_sd = Posterior(
        mean=mu, sd=sigma,
        quantiles=lambda levels: np.zeros((3, len(np.asarray(levels)))),
    )
    np.testing.assert_allclose(probability_of_improvement(with_sd, req), gaussian,
                               atol=1e-12)

    with pytest.raises(Unsupported):
        probability_of_improvement(Posterior(mean=mu), req)


def test_exact_ucb_is_preferred_over_the_quantile_spread():
    """``exact_ucb`` is read directly; on a skewed predictive it differs from mu + beta*sigma."""
    mu, sigma = np.array([0.0, 1.0, -2.0]), np.array([1.0, 2.0, 0.5])
    quantiles = lambda levels: mu[:, None] + sigma[:, None] * norm.ppf(
        np.asarray(levels))[None, :]
    post = Posterior(
        mean=mu, quantiles=quantiles,
        exact_ucb=lambda beta: quantiles(np.array([float(norm.cdf(beta))]))[:, 0],
    )
    req = ScoreRequest(x_obs=np.zeros((1, 1)), y_obs=np.array([0.5]),
                       x_cand=np.zeros((3, 1)), seed=0)
    for beta in (1.0, 2.0, 3.0):
        np.testing.assert_allclose(ucb(beta)(post, req), mu + beta * sigma, atol=1e-9)

    lognormal = lambda levels: np.exp(norm.ppf(np.asarray(levels)))[None, :] * np.ones((3, 1))
    skewed_exact = Posterior(
        mean=np.full(3, float(np.exp(0.5))), quantiles=lognormal,
        exact_ucb=lambda beta: lognormal(np.array([float(norm.cdf(beta))]))[:, 0],
    )
    skewed_spread = Posterior(mean=np.full(3, float(np.exp(0.5))), quantiles=lognormal)
    exact_3 = ucb(3.0)(skewed_exact, req)[0]
    np.testing.assert_allclose(exact_3, np.exp(3.0), rtol=1e-9)
    assert ucb(3.0)(skewed_spread, req)[0] < 0.5 * exact_3


def test_pi_and_ucb_prefer_the_exact_forms():
    """``exact_pi`` and ``exact_ucb`` are used when present; otherwise the Gaussian form."""
    mu, sigma = np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 0.5])
    req = ScoreRequest(x_obs=np.zeros((2, 1)), y_obs=np.array([0.0, 0.5]),
                       x_cand=np.zeros((3, 1)), seed=0)

    # Exact forms that happen to be Gaussian must agree with the closed forms;
    # this pins the level `ucb` asks for.
    exact = Posterior(
        mean=mu, sd=sigma,
        exact_pi=lambda b: norm.sf(b, mu, sigma),
        exact_ucb=lambda beta: norm.ppf(norm.cdf(beta), mu, sigma),
    )
    for beta in (0.5, 1.0, 2.0, 3.0):
        np.testing.assert_allclose(ucb(beta)(exact, req), mu + beta * sigma, atol=1e-9)
    np.testing.assert_allclose(probability_of_improvement(exact, req),
                               norm.cdf((mu - req.best_f) / sigma), atol=1e-9)

    skewed = Posterior(
        mean=mu, sd=sigma,
        exact_pi=lambda b: np.array([0.9, 0.1, 0.2]),
        exact_ucb=lambda beta: np.array([7.0, 1.0, 2.0]),
    )
    assert np.argmax(probability_of_improvement(skewed, req)) == 0
    assert np.argmax(ucb(2.0)(skewed, req)) == 0
    assert np.argmax(mu + 2.0 * sigma) == 1

    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(2, 40))
        post = Posterior(mean=rng.normal(size=n) * 10,
                         sd=np.abs(rng.normal(size=n)) * 3 + 1e-3)
        gaussian_req = ScoreRequest(x_obs=np.zeros((2, 1)), y_obs=rng.normal(size=2),
                                    x_cand=np.zeros((n, 1)), seed=0)
        for beta in (1.0, 2.0, 3.0):
            np.testing.assert_array_equal(
                ucb(beta)(post, gaussian_req), post.mean + beta * post.sd
            )
        np.testing.assert_allclose(
            probability_of_improvement(post, gaussian_req),
            norm.cdf((post.mean - gaussian_req.best_f) / np.clip(post.sd, 1e-9, None)),
        )


def test_every_registered_arm_can_actually_score():
    """Every arm's acquisition can read its surrogate's posterior.

    Each surrogate is fitted once and every rule scores that posterior. An
    ImportError fails the test. A surrogate that cannot fit here (TabPFN
    without weights) is reported as a warning.
    """
    req = _request(L.load("Perovskite"))

    posteriors, broken, unfittable = {}, [], []
    for name, method in sorted(build_registry().methods.items()):
        key = id(method.surrogate)
        if key not in posteriors:
            try:
                posteriors[key] = method.surrogate.fit(req)
            except ImportError:
                raise
            except Exception as exc:  # noqa: BLE001 - reported below
                posteriors[key] = f"{type(exc).__name__}: {exc}"
        post = posteriors[key]
        if isinstance(post, str):
            unfittable.append(f"{name} ({post})")
            continue
        try:
            scores = method.acquisition(post, req)
        except Unsupported as exc:
            broken.append(f"{name}: {exc}")
            continue
        if not np.all(np.isfinite(scores)):
            broken.append(f"{name}: produced non-finite scores")

    assert not broken, "registered arms that cannot score:\n  " + "\n  ".join(broken)
    if unfittable:
        warnings.warn(
            f"pairing unchecked for {len(unfittable)} arm(s): " + "; ".join(unfittable),
            stacklevel=2,
        )
