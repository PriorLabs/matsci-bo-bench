#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""Seed-level tests (Table D.1), paired seed comparisons (Figure D.9) and best acquisition rules (Figure D.10).

Each seed's score is the AUC of the raw metric over the linear 5-50% budget
window, oriented so that larger is better. Paired differences are bootstrapped
by resampling whole seeds jointly across every contrast of one campaign.
Table D.1 tests the Figure 1 arms with a Welch t-test per campaign, and ranks
their families across campaigns with a Friedman test and the Nemenyi
critical difference.

    python -m matbo.analysis.posthoc --out figures/paper/posthoc
"""

from __future__ import annotations

import argparse
import zlib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from matbo.analysis.field import (
    DENSE_BUDGETS, FAMILIES, FIELD, KERNELS, RULES, Scored, budget_columns, score,
)

COMPARATORS = (*KERNELS, "RF", "CatBoost", "TabICL")
SURROGATES = ("TabPFN", "TabICL", "CatBoost", "RF", *KERNELS)
N_BOOT = 10_000
ANALYSIS_SEED = 20261018
# Part of each bootstrap's RNG key; changing it changes every interval.
PAIRED_KEY = "A"
SUBSTITUTED = "-substituted"  # run-tag marker of a seed rerun with Random acquisition
RECORDED, DROPPED = "recorded", "substituted-seeds-dropped"
ALPHA = 0.05


def seed_auc(curves: np.ndarray, n: int) -> np.ndarray:
    """[seed] mean of each curve over the linear 5-50% window of an `n`-row pool."""
    if curves.shape[1] != n:
        raise ValueError(f"curves end at {curves.shape[1]} of {n} evaluations")
    dense = curves[:, budget_columns(n, DENSE_BUDGETS)]
    return np.trapezoid(dense, DENSE_BUDGETS, axis=1) / np.ptp(DENSE_BUDGETS)


def bootstrap(deltas: np.ndarray, rng: np.random.Generator, n_boot: int = N_BOOT):
    """95% percentile interval of the column means, resampling rows (seeds) jointly."""
    n = len(deltas)
    flat = deltas.reshape(n, -1)
    centre = flat.mean(axis=0)
    weights = np.zeros((n_boot, n))
    np.add.at(weights, (np.arange(n_boot)[:, None], rng.integers(n, size=(n_boot, n))), 1)
    means = centre + weights @ (flat - centre) / n
    lo, hi = np.percentile(means, [2.5, 97.5], axis=0)
    return lo.reshape(deltas.shape[1:]), hi.reshape(deltas.shape[1:])


def rng_for(*key: str) -> np.random.Generator:
    return np.random.default_rng([ANALYSIS_SEED, zlib.crc32("/".join(key).encode())])


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


def paired_pairs() -> list[tuple[str, str, str]]:
    """(rule, TabPFN arm, other arm) under one rule, including Random."""
    return [(rule, f"TabPFN-{rule}", other) for rule in RULES
            for other in (*(f"{c}-{rule}" for c in COMPARATORS), "Random")]


def _variants(cs: CampaignScores, pairs):
    """(variant, seeds kept, pairs): all seeds, then each pair set without its substituted seeds."""
    yield RECORDED, list(cs.seeds), pairs
    by_drop: dict[frozenset, list] = {}
    for p in pairs:
        drop = cs.substituted.get(p[1], set()) | cs.substituted.get(p[2], set())
        if drop:
            by_drop.setdefault(frozenset(drop), []).append(p)
    for drop, ps in by_drop.items():
        yield DROPPED, [s for s in cs.seeds if s not in drop], ps


def _base(cs: CampaignScores, variant: str, n: int, arms) -> dict:
    flags = sorted({f"{a}:seed {s}" for a in arms for s in cs.substituted.get(a, ())})
    return {"benchmark": cs.benchmark, "campaign": cs.campaign, "pool": cs.n,
            "units": cs.units, "goal": "maximize" if cs.sign > 0 else "minimize",
            "analysis_variant": variant, "n_seeds": n,
            "substituted": ", ".join(flags) if variant == RECORDED else ""}


def paired_effects(cs: CampaignScores, pairs) -> pd.DataFrame:
    """Mean paired AUC difference (first - second, larger is better) with a seed bootstrap."""
    rows = []
    for variant, seeds, ps in _variants(cs, pairs):
        keep = [cs.seeds.index(s) for s in seeds]
        deltas = np.stack([cs.scores[a][keep] - cs.scores[b][keep] for _, a, b in ps], axis=1)
        lo, hi = bootstrap(deltas, rng_for(PAIRED_KEY, cs.benchmark, cs.campaign, variant))
        mean, sd = deltas.mean(axis=0), deltas.std(axis=0, ddof=1)
        for k, (rule, a, b) in enumerate(ps):
            d = deltas[:, k]
            rows.append({
                **_base(cs, variant, len(keep), (a, b)), "rule": rule, "first": a, "second": b,
                "mean_diff": mean[k], "ci_low": lo[k], "ci_high": hi[k], "sd_diff": sd[k],
                "win": (d > 0).mean(), "tie": (d == 0).mean(), "loss": (d < 0).mean()})
    return pd.DataFrame(rows)


def acquisition_cost(cs: CampaignScores, scale: float = 1.0) -> pd.DataFrame:
    """Per surrogate: each rule's mean AUC and its gap to the best rule on this campaign.

    `cost_scaled` divides the gap by `scale`; `best` marks every rule tied for best.
    """
    rows = []
    for s in SURROGATES:
        arms = [f"{s}-{r}" for r in RULES]
        mean = np.stack([cs.scores[a] for a in arms], axis=1).mean(axis=0)
        top = mean.max()
        for r, rule in enumerate(RULES):
            rows.append({
                **_base(cs, RECORDED, len(cs.seeds), arms), "surrogate": s, "rule": rule,
                "mean": cs.sign * mean[r], "cost": top - mean[r],
                "cost_scaled": (top - mean[r]) / scale, "best": mean[r] == top})
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
    paired: pd.DataFrame
    cost: pd.DataFrame
    seed_tests: pd.DataFrame
    summaries: dict[str, pd.DataFrame]  # "mean"/"median" -> [campaign, family] seed-AUC summary
    campaigns: list[str]  # by pool size
    scale: dict[str, float] | None  # olympus: pool best minus Random's mean AUC


@lru_cache(maxsize=None)
def tables(benchmark: str) -> Tables:
    """D.1, D.9 and D.10 inputs of one benchmark, on its own metric."""
    scored = score(benchmark)
    record = scored.record
    scores = [campaign_scores(scored, c) for c in record.campaigns]
    scale = None
    if record.protocol.metric == "best-value":
        scale = {cs.campaign: record.campaign(cs.campaign).y.max() - cs.scores["Random"].mean()
                 for cs in scores}
    paired = pd.concat(paired_effects(cs, paired_pairs()) for cs in scores)
    cost = pd.concat(acquisition_cost(cs, scale[cs.campaign] if scale else 1.0)
                     for cs in scores)
    arms = dict(zip((*FAMILIES, "Random"), scored.best()))
    summaries = {stat: pd.DataFrame({cs.campaign: {f: getattr(np, stat)(cs.scores[a]) for f, a in arms.items()}
                                     for cs in scores}).T
                 for stat in ("mean", "median")}
    campaigns = sorted(record.campaigns, key=lambda c: record.campaign(c).n)
    return Tables(paired, cost, seed_tests(scores, arms), summaries, campaigns, scale)


def main():
    parser = argparse.ArgumentParser(description="Write Table D.1 and the tables behind Figures D.9 and D.10.")
    parser.add_argument("--out", type=Path, default=Path("figures/paper/posthoc"))
    args = parser.parse_args()
    ranks = []
    for benchmark in ("liang", "olympus"):
        t = tables(benchmark)
        folder = args.out / benchmark
        folder.mkdir(parents=True, exist_ok=True)
        t.seed_tests.to_csv(folder / "seed_tests.csv", index=False)
        t.paired.to_csv(folder / "paired_effects.csv", index=False)
        t.cost.to_csv(folder / "acquisition_cost.csv", index=False)
        if t.scale:
            pd.Series(t.scale, name="random_to_pool_best_gap").rename_axis("campaign").to_csv(
                folder / "figure_scale.csv")
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
