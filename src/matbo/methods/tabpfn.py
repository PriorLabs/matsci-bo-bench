#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""TabPFN as a BO surrogate.

The mean, sd, EI, PI and UCB all come from the bar distribution that TabPFN
returns with ``predict(output_type="full")``; see :mod:`matbo.methods.tabpfn_points`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from matbo.methods.base import Posterior, ScoreRequest
from matbo.methods.tabpfn_points import posterior_mean_sd, score_candidates


@dataclass
class TabPFNSurrogate:
    """TabPFN-3's predictive distribution over the pool, at the library defaults."""

    n_estimators: int = 8
    device: str = "cpu"
    name: str = field(default="TabPFN", init=False)

    def _regressor(self, seed: int) -> Any:
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion

        # Pinned to TabPFN-3, the model behind the recorded traces; newer
        # tabpfn releases default to a different checkpoint.
        return TabPFNRegressor.create_default_for_version(
            ModelVersion.V3,
            n_estimators=self.n_estimators,
            device=self.device,
            random_state=seed,
            inference_precision=torch.float32,
            # None lets TabPFN infer categorical columns itself.
            categorical_features_indices=None,
        )

    def fit(self, req: ScoreRequest) -> Posterior:
        # A fresh regressor per round. tabpfn_points caches the round's single
        # fit and prediction on it, so every statistic below reads one forward pass.
        reg = self._regressor(req.seed)
        x_obs, y_obs, x_cand = req.x_obs, req.y_obs, req.x_cand

        def _score(acquisition: str, best_f: float | None = None,
                   beta: float | None = None) -> np.ndarray:
            return np.asarray(
                score_candidates(
                    reg, x_obs, y_obs, x_cand,
                    best_f=req.best_f if best_f is None else best_f,
                    acquisition=acquisition,
                    beta=beta,
                ),
                dtype=np.float64,
            )

        mean = _score("mean")
        _, sd = posterior_mean_sd(reg, x_obs, y_obs, x_cand)
        return Posterior(
            mean=mean,
            sd=np.asarray(sd, dtype=np.float64),
            exact_ei=lambda b: _score("ei", b),
            exact_pi=lambda b: _score("pi", b),
            exact_ucb=lambda beta: _score("ucb", beta=beta),
        )
