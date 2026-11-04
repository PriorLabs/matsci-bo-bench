#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""TabICLv2 as a BO surrogate (Qu et al., arXiv:2602.11139).

TabICL states quantiles but no density or sd, so EI and PI read its quantile
function and UCB reads one quantile directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import norm

from matbo.methods.base import Posterior, ScoreRequest

#: One regressor per (seed, n_estimators, device), reused across rounds so the
#: checkpoint loads once. ``fit`` fully resets the estimator, so reuse gives the
#: same predictions as a fresh instance.
_CACHE: dict[tuple[int, int, str], Any] = {}


@dataclass
class TabICLSurrogate:
    """TabICLv2's predictive mean and quantiles over the pool."""

    n_estimators: int = 8
    device: str = "cpu"
    name: str = field(default="TabICL", init=False)

    def _regressor(self, seed: int) -> Any:
        from tabicl import TabICLRegressor

        key = (seed, self.n_estimators, self.device)
        if key not in _CACHE:
            _CACHE[key] = TabICLRegressor(
                n_estimators=self.n_estimators, device=self.device, random_state=seed
            )
        return _CACHE[key]

    def fit(self, req: ScoreRequest) -> Posterior:
        x_obs = np.asarray(req.x_obs, dtype=np.float64)
        y_obs = np.asarray(req.y_obs, dtype=np.float64)
        cand = np.asarray(req.x_cand, dtype=np.float64)

        # TabICL's fit raises IndexError when no feature column varies. Olympus
        # keeps duplicate rows as separate candidates, so a two-row initial
        # design can draw the same experiment twice. Such a context carries no
        # ranking, so the posterior is flat and the runner's tie-break picks.
        if x_obs.size and not (x_obs.max(axis=0) > x_obs.min(axis=0)).any():
            flat = np.full(len(cand), float(y_obs.mean()) if y_obs.size else 0.0)
            return Posterior(
                mean=flat,
                quantiles=lambda levels: np.repeat(
                    flat[:, None], len(np.asarray(levels)), axis=1
                ),
                exact_ucb=lambda _beta: flat.copy(),
            )

        reg = self._regressor(req.seed)
        reg.fit(x_obs, y_obs)
        mean = np.asarray(reg.predict(cand, output_type="mean"), dtype=np.float64).ravel()

        def quantiles(levels: np.ndarray) -> np.ndarray:
            return np.asarray(
                reg.predict(cand, output_type="quantiles", alphas=list(map(float, levels))),
                dtype=np.float64,
            )

        def exact_ucb(beta: float) -> np.ndarray:
            return np.asarray(quantiles(np.array([float(norm.cdf(beta))])),
                              dtype=np.float64)[:, 0]

        return Posterior(mean=mean, quantiles=quantiles, exact_ucb=exact_ucb)
