#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""Tree-ensemble surrogates, and the random baseline."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from matbo.methods.base import Posterior, ScoreRequest


@dataclass
class RandomForestSurrogate:
    """Random forest; the spread across trees is the uncertainty."""

    n_estimators: int = 100
    name: str = field(default="RF", init=False)

    def fit(self, req: ScoreRequest) -> Posterior:
        # One job: the runner already parallelises over (arm, seed).
        rf = RandomForestRegressor(
            n_estimators=self.n_estimators, random_state=req.seed, n_jobs=1
        )
        rf.fit(req.x_obs, req.y_obs)
        per_tree = np.stack([t.predict(req.x_cand) for t in rf.estimators_])
        return Posterior(mean=per_tree.mean(0), sd=per_tree.std(0).clip(1e-9))


@dataclass
class CatBoostSurrogate:
    """CatBoost with ``RMSEWithUncertainty``; sd is the total predictive spread.

    The total is the knowledge (ensemble) variance plus the data variance from
    ``virtual_ensembles_predict``. The knowledge term alone is so small at these
    context sizes that EI over it underflows to zero.
    """

    iterations: int = 200
    virtual_ensembles: int = 10
    name: str = field(default="CatBoost", init=False)

    def fit(self, req: ScoreRequest) -> Posterior:
        from catboost import CatBoostRegressor

        # RMSEWithUncertainty refuses a constant target, which a two-row initial
        # design hits on coarse objectives. Such a context carries no ranking,
        # so the posterior is flat and the runner's tie-break picks.
        if req.y_obs.size == 0 or np.ptp(req.y_obs) == 0:
            flat = np.full(len(req.x_cand), float(req.y_obs[0]) if req.y_obs.size else 0.0)
            return Posterior(mean=flat, sd=np.full(len(req.x_cand), 1e-9))

        cat = CatBoostRegressor(
            loss_function="RMSEWithUncertainty",
            iterations=self.iterations,
            random_seed=req.seed,
            # One thread: the runner already parallelises over (arm, seed).
            thread_count=1,
            verbose=False,
            # Otherwise parallel workers race on catboost_info/.
            allow_writing_files=False,
        )
        cat.fit(req.x_obs, req.y_obs)
        # (n, 3): mean, knowledge variance, data variance.
        out = np.asarray(
            cat.virtual_ensembles_predict(
                req.x_cand,
                prediction_type="TotalUncertainty",
                virtual_ensembles_count=self.virtual_ensembles,
            )
        )
        mean, knowledge, data = out[:, 0], out[:, 1], out[:, 2]
        var = knowledge + data
        return Posterior(mean=mean, sd=np.sqrt(np.clip(var, 0.0, None)).clip(1e-9))


@dataclass
class RandomSurrogate:
    """Uniform random scores: the baseline every arm has to beat."""

    name: str = field(default="Random", init=False)

    def fit(self, req: ScoreRequest) -> Posterior:
        rng = np.random.default_rng(req.seed * 7919 + len(req.x_obs))
        return Posterior(mean=rng.random(len(req.x_cand)))
