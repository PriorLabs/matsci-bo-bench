#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""Seed-level tests (Table D.1) and best acquisition rules (Figure D.9).

Each seed's score is the AUC of the raw metric over the linear 5-50% budget
window, oriented so that larger is better. Table D.1 tests the Figure 1 arms with a Welch t-test per campaign, and ranks
their families across campaigns with a Friedman test and the Nemenyi
critical difference.

    python -m matbo.analysis.posthoc --out figures/paper/posthoc
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from matbo.analysis.field import (
    DENSE_BUDGETS, FAMILIES, FIELD, KERNELS, RULES, Scored, budget_columns, score,
)

SURROGATES = ("TabPFN", "TabICL", "CatBoost", "RF", *KERNELS)
SUBSTITUTED = "-substituted"  # run-tag marker of a seed rerun with Random acquisition
ALPHA = 0.05


def seed_auc(curves: np.ndarray, n: int) -> np.ndarray:
    """[seed] mean of each curve over the linear 5-50% window of an `n`-row pool."""
    if curves.shape[1] != n:
        raise ValueError(f"curves end at {curves.shape[1]} of {n} evaluations")
    dense = curves[:, budget_columns(n, DENSE_BUDGETS)]
    return np.trapezoid(dense, DENSE_BUDGETS, axis=1) / np.ptp(DENSE_BUDGETS)


@dataclass
class CampaignScores:
    """Per-seed AUC of every FIELD arm on one campaign."""

    benchmark: str
    campaign: str
    n: int
    seeds: tuple[int, ...]
    sign: int  # original units = sign * oriented
    units: str
    scores: dict[str, np.ndarray]  # arm -> [seed], oriented
    substituted: dict[str, set[int]] = field(default_factory=dict)


def campaign_scores(scored: Scored, campaign: str) -> CampaignScores:
    """Per-seed scores; raises if the arms do not share each seed's initial design."""
    record = scored.record
    pool = record.campaign(campaign)
    block = scored.blocks[campaign]
    rows = {arm: record.rows(arm, campaign) for arm in FIELD}
    seeds = tuple(record.protocol.seeds)
    k = record.protocol.n_initial
    unpaired = [s for i, s in enumerate(seeds)
                if len({tuple(sorted(r[i]["trace"][:k])) for r in rows.values()}) > 1]
    if unpaired:
        raise ValueError(f"{campaign}: initial designs differ across arms for seeds {unpaired}")
    if record.protocol.metric == "topcount":
        sign, units = 1, "Top% (percentage points)"
    else:
        sign = -1 if pool.objective.endswith("(min)") else 1
        units = f"{pool.objective}, original units"
    substituted = {arm: {int(r["seed"]) for r in rs if SUBSTITUTED in r["run_tag"]}
                   for arm, rs in rows.items()}
    return CampaignScores(
        record.benchmark, campaign, pool.n, seeds, sign, units,
        {arm: seed_auc(block[i], pool.n) for i, arm in enumerate(FIELD)},
        {arm: s for arm, s in substituted.items() if s})


def _base(cs: CampaignScores, arms) -> dict:
    flags = sorted({f"{a}:seed {s}" for a in arms for s in cs.substituted.get(a, ())})
    return {"benchmark": cs.benchmark, "campaign": cs.campaign, "pool": cs.n,
            "units": cs.units, "goal": "maximize" if cs.sign > 0 else "minimize",
            "n_seeds": len(cs.seeds), "substituted": ", ".join(flags)}


def acquisition_cost(cs: CampaignScores) -> pd.DataFrame:
    """Per surrogate: each rule's mean AUC and its gap to the best rule on this campaign.

    `best` marks every rule tied for best.
    """
    rows = []
    for s in SURROGATES:
        arms = [f"{s}-{r}" for r in RULES]
        mean = np.stack([cs.scores[a] for a in arms], axis=1).mean(axis=0)
        top = mean.max()
        for r, rule in enumerate(RULES):
            rows.append({
                **_base(cs, arms), "surrogate": s, "rule": rule,
                "mean": cs.sign * mean[r], "cost": top - mean[r], "best": mean[r] == top})
    return pd.DataFrame(rows)


def seed_tests(scores: list[CampaignScores], arms: dict[str, str]) -> pd.DataFrame:
    """Welch t-test per campaign of the TabPFN arm against each other family's arm.

    `arms` maps each family to its Figure 1 arm. `mean_diff` is oriented:
    positive favors TabPFN.
    """
    rows = []
    ours = arms["TabPFN"]
    for cs in scores:
        for fam, arm in arms.items():
            if fam == "TabPFN":
                continue
            t, o = cs.scores[ours], cs.scores[arm]
            p = stats.ttest_ind(t, o, equal_var=False).pvalue
            diff = t.mean() - o.mean()
            # Two identical columns give p = nan: not separated.
            verdict = "better" if p < ALPHA and diff > 0 else "worse" if p < ALPHA else "not separated"
            rows.append({"benchmark": cs.benchmark, "campaign": cs.campaign, "family": fam,
                         "comparator": arm, "mean_diff": diff, "welch_p": p, "verdict": verdict})
    return pd.DataFrame(rows)


def family_ranks(summary: pd.DataFrame) -> tuple[pd.Series, float, float]:
    """Mean rank (1 = best), Friedman p-value and Nemenyi critical difference.

    `summary` has one row per campaign and one column per family.
    """
    n, k = summary.shape
    ranks = summary.rank(axis=1, ascending=False, method="average").mean()
    p = stats.friedmanchisquare(*summary.to_numpy().T).pvalue
    cd = stats.studentized_range.ppf(1 - ALPHA, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * n))
    return ranks.sort_values(), p, cd


@dataclass(frozen=True)
class Tables:
    cost: pd.DataFrame
    seed_tests: pd.DataFrame
    summaries: dict[str, pd.DataFrame]  # "mean"/"median" -> [campaign, family] seed-AUC summary
    campaigns: list[str]  # by pool size


@lru_cache(maxsize=None)
def tables(benchmark: str) -> Tables:
    """Table D.1 and Figure D.9 inputs of one benchmark, on its own metric."""
    scored = score(benchmark)
    record = scored.record
    scores = [campaign_scores(scored, c) for c in record.campaigns]
    cost = pd.concat(acquisition_cost(cs) for cs in scores)
    arms = dict(zip((*FAMILIES, "Random"), scored.best()))
    summaries = {stat: pd.DataFrame({cs.campaign: {f: getattr(np, stat)(cs.scores[a]) for f, a in arms.items()}
                                     for cs in scores}).T
                 for stat in ("mean", "median")}
    campaigns = sorted(record.campaigns, key=lambda c: record.campaign(c).n)
    return Tables(cost, seed_tests(scores, arms), summaries, campaigns)


def main():
    parser = argparse.ArgumentParser(description="Write Table D.1 and the table behind Figure D.9.")
    parser.add_argument("--out", type=Path, default=Path("figures/paper/posthoc"))
    args = parser.parse_args()
    ranks = []
    for benchmark in ("liang", "olympus"):
        t = tables(benchmark)
        folder = args.out / benchmark
        folder.mkdir(parents=True, exist_ok=True)
        t.seed_tests.to_csv(folder / "seed_tests.csv", index=False)
        t.cost.to_csv(folder / "acquisition_cost.csv", index=False)
        print(f"wrote {folder}", flush=True)
    for stat in ("mean", "median"):
        pooled = pd.concat(tables(b).summaries[stat] for b in ("liang", "olympus"))
        mean_rank, p, cd = family_ranks(pooled)
        ranks.append(mean_rank.rename("mean_rank").rename_axis("family").reset_index()
                     .assign(summary=stat, friedman_p=p, critical_difference=cd))
    pd.concat(ranks).to_csv(args.out / "family_ranks.csv", index=False)
    print(f"wrote {args.out / 'family_ranks.csv'}", flush=True)


if __name__ == "__main__":
    main()
