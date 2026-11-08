#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""Surrogates, posteriors, acquisitions, and the arms they compose into.

An arm is one surrogate scored under one acquisition rule. A rule that needs a
capability the posterior does not have raises :class:`Unsupported` instead of
substituting an approximation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from matbo.campaign import Campaign


@dataclass(frozen=True)
class ScoreRequest:
    """The observed rows and the candidate pool at one acquisition round."""

    x_obs: np.ndarray
    y_obs: np.ndarray
    x_cand: np.ndarray
    seed: int
    campaign: Campaign | None = None

    @property
    def best_f(self) -> float:
        """The incumbent: the best objective value observed so far."""
        return float(np.max(self.y_obs))


class Unsupported(NotImplementedError):
    """Raised when an acquisition needs something the posterior lacks."""


@dataclass
class Posterior:
    """A surrogate's predictive belief over the candidate pool.

    Optional fields are capabilities. ``sd is None`` means the surrogate states
    no uncertainty, not that its uncertainty is zero.
    """

    mean: np.ndarray
    sd: np.ndarray | None = None
    #: levels -> (n_cand, n_levels) quantiles of the predictive.
    quantiles: Callable[[np.ndarray], np.ndarray] | None = None
    #: best_f -> EI computed from the surrogate's own predictive.
    exact_ei: Callable[[float], np.ndarray] | None = None
    #: best_f -> P(f > best_f) from the surrogate's own predictive.
    exact_pi: Callable[[float], np.ndarray] | None = None
    #: beta -> the predictive's Phi(beta) quantile; ``mean + beta * sd`` if Gaussian.
    exact_ucb: Callable[[float], np.ndarray] | None = None

    def require(self, attr: str, who: str) -> np.ndarray:
        """Return ``attr``, or raise :class:`Unsupported` naming ``who`` if it is missing."""
        value = getattr(self, attr, None)
        if value is None:
            raise Unsupported(f"{who} needs `{attr}`, which this surrogate does not provide.")
        return value


class Surrogate(Protocol):
    """Fits the observed rows and predicts over the candidate pool."""

    name: str

    def fit(self, request: ScoreRequest) -> Posterior: ...


#: Turns a posterior into one score per candidate; larger wins.
Acquisition = Callable[[Posterior, ScoreRequest], np.ndarray]


@dataclass(frozen=True)
class Method:
    """One benchmark arm: a surrogate scored under an acquisition rule."""

    name: str
    surrogate: Surrogate
    acquisition: Acquisition

    def score(self, request: ScoreRequest) -> np.ndarray:
        scores = self.acquisition(self.surrogate.fit(request), request)
        scores = np.asarray(scores, dtype=np.float64).ravel()
        if len(scores) != len(request.x_cand):
            raise ValueError(
                f"{self.name}: scored {len(scores)} candidates, expected {len(request.x_cand)}"
            )
        return scores


@dataclass
class Registry:
    """The arms a run can draw on, keyed by name."""

    methods: dict[str, Method] = field(default_factory=dict)

    def add(self, method: Method) -> Method:
        if method.name in self.methods:
            raise ValueError(f"duplicate method name {method.name!r}")
        self.methods[method.name] = method
        return method

    def select(self, names: list[str] | None) -> dict[str, Method]:
        """The named arms, or all of them when ``names`` is empty or None."""
        if not names:
            return dict(self.methods)
        unknown = [n for n in names if n not in self.methods]
        if unknown:
            raise SystemExit(
                f"unknown method(s) {unknown}; available: {', '.join(sorted(self.methods))}"
            )
        return {n: self.methods[n] for n in names}
