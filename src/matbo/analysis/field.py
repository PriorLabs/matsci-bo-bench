# Copyright (c) Prior Labs GmbH 2026. Licensed under the Apache License, Version 2.0.
"""The 56-arm paper field, its normalized improvement (NI) and rankings.

NI is min-max over exactly the FIELD arms, per campaign and budget, after
averaging seeds (paper App. C). Arms outside FIELD are never read.
"""

from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np
import pandas as pd

from matbo.analysis.record import Record

BUDGETS = np.geomspace(0.05, 0.50, 8)
# arange, not linspace: linspace rounds a few half-row budgets to a different evaluation.
DENSE_BUDGETS = np.arange(0.05, 0.501, 0.01)
METRICS = {"liang": "topcount", "olympus": "best-value"}
FAMILIES = ("TabPFN", "TabICL", "GP", "CatBoost", "RF")
RULES = ("greedy", "EI", "PI", "UCB1", "UCB2")
KERNELS = ("GP-ARD", "GP-M32", "GP-M12", "GP-RBF", "GP-iso", "GP-MLP", "GP-ARD-Warp")
FIELD = (
    *(f"{s}-{r}" for s in ("TabPFN", "TabICL", "CatBoost", "RF", *KERNELS) for r in RULES),
    "Random",
)


def family(arm: str) -> str:
    return "GP" if arm.startswith("GP-") else arm.split("-")[0]


def budget_columns(n: int, budgets) -> np.ndarray:
    """0-based curve columns at fractions `budgets` of an `n`-row pool."""
    return np.maximum(1, np.rint(np.asarray(budgets) * n).astype(int)) - 1


def normalize(values: np.ndarray) -> np.ndarray:
    """Min-max across the last (arm) axis; NaN where all arms tie."""
    lo = values.min(axis=-1, keepdims=True)
    span = values.max(axis=-1, keepdims=True) - lo
    return np.divide(values - lo, span, out=np.full_like(values, np.nan), where=span > 0)


@dataclass
class Scored:
    """FIELD curves of one benchmark under one metric."""

    record: Record
    blocks: dict[str, np.ndarray]  # campaign -> [FIELD arm, seed, evaluation]

    def __post_init__(self):
        for campaign, block in self.blocks.items():
            if len(block) != len(FIELD):
                raise ValueError(f"{campaign}: {len(block)} arms, FIELD has {len(FIELD)}")

    def sample(self, budgets=BUDGETS) -> np.ndarray:
        """[campaign, budget, arm, seed] on the metric's own scale."""
        return np.stack(
            [
                block[:, :, budget_columns(block.shape[2], budgets)].transpose(2, 0, 1)
                for block in self.blocks.values()
            ]
        )

    def ni(self, budgets=BUDGETS) -> np.ndarray:
        """[campaign, budget, arm] normalized improvement."""
        return normalize(self.sample(budgets).mean(axis=-1))

    def ranking(self) -> pd.Series:
        """NI-AUC over the linear 5-50% window, campaign-averaged; Random excluded."""
        # [arm, budget], contiguous: summing along a strided axis changes the last bit.
        mean = np.ascontiguousarray(np.nanmean(self.ni(DENSE_BUDGETS), axis=0).T)
        auc = np.trapezoid(mean, DENSE_BUDGETS) / np.ptp(DENSE_BUDGETS)
        return pd.Series(auc, index=FIELD).drop("Random").sort_values(ascending=False)

    def best(self) -> list[str]:
        """The top-ranked arm of each family, then Random."""
        ranked = self.ranking()
        return [next(a for a in ranked.index if family(a) == f) for f in FAMILIES] + ["Random"]


def score(benchmark: str, metric: str | None = None) -> Scored:
    """FIELD curves of `benchmark`, on its own metric unless `metric` is given."""
    return _score(benchmark, metric or METRICS[benchmark])


@lru_cache(maxsize=None)
def _score(benchmark: str, metric: str) -> Scored:
    record = Record.load(benchmark)
    record = replace(record, protocol=replace(record.protocol, metric=metric))
    missing = [
        f"{a} on {c}" for a in FIELD for c in record.campaigns if not record.path(a, c).exists()
    ]
    if missing:
        raise ValueError(f"{benchmark}: incomplete paper field: {missing}")
    blocks = {}
    for campaign in record.campaigns:
        values = [record.block(a, campaign) for a in FIELD]
        if min(v.shape[1] for v in values) != record.campaign(campaign).n:
            raise ValueError(f"{campaign}: paper figures need traces through pool exhaustion")
        blocks[campaign] = np.stack(values)
    return Scored(record, blocks)


def campaign_band(ni: np.ndarray) -> np.ndarray:
    """2.5-97.5% percentile band of the campaign mean, 400 campaign resamples."""
    rng = np.random.default_rng(0)
    draws = [np.nanmean(ni[rng.integers(len(ni), size=len(ni))], axis=0) for _ in range(400)]
    return np.nanpercentile(draws, [2.5, 97.5], axis=0)


def kernel_values(scored: Scored) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """[campaign, arm] raw log-budget AUC, with 95% paired seed-bootstrap bounds."""
    x = np.log10(BUDGETS)
    per = np.trapezoid(scored.sample(), x, axis=1) / np.ptp(x)
    rng = np.random.default_rng(0)
    draws = [
        per[:, :, rng.integers(per.shape[2], size=per.shape[2])].mean(axis=2) for _ in range(400)
    ]
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    return per.mean(axis=2), lo, hi
