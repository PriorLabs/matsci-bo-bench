#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""Contracts of the paired and best-rule tables behind Figures D.9 and D.10."""

from __future__ import annotations

import csv
import shutil

import numpy as np
import pandas as pd
import pytest

from matbo.analysis import posthoc as P
from matbo.analysis.record import REFERENCE, Record
from matbo.analysis.field import DENSE_BUDGETS, FIELD, RULES, Scored

SEEDS = (7, 3, 11, 5)


def scores(values: dict[str, np.ndarray], sign=1, substituted=None) -> P.CampaignScores:
    full = {a: np.asarray(v, float) for a, v in values.items()}
    return P.CampaignScores("synthetic", "c", 100, SEEDS, sign, "units", full, substituted or {})


def recorded(table: pd.DataFrame, variant=P.RECORDED) -> pd.DataFrame:
    return table[table.analysis_variant == variant]


def test_identical_arms_give_zero_difference_and_zero_width():
    cs = scores({"a": [1, 5, 2, 8], "b": [1, 5, 2, 8]})
    row = recorded(P.paired_effects(cs, [("r", "a", "b")])).iloc[0]
    assert row.mean_diff == row.ci_low == row.ci_high == 0
    assert row.tie == 1


def test_constant_improvement_is_exact_and_reverses_with_the_arms():
    cs = scores({"a": [3.5, 7.5, 4.5, 10.5], "b": [1, 5, 2, 8]})
    t = recorded(P.paired_effects(cs, [("r", "a", "b"), ("r", "b", "a")]))
    forward, backward = t.iloc[0], t.iloc[1]
    assert forward.mean_diff == forward.ci_low == forward.ci_high == 2.5
    assert backward.mean_diff == backward.ci_low == backward.ci_high == -2.5
    assert forward.win == backward.loss == 1


def test_paired_difference_reproduces_a_hand_computed_example():
    cs = scores({"S-greedy": [4, 4, 4, 4], "S-EI": [5, 3, 6, 6]})
    row = recorded(P.paired_effects(cs, [("EI", "S-EI", "S-greedy")])).iloc[0]
    assert row.mean_diff == 1.0
    assert (row.win, row.tie, row.loss) == (0.75, 0, 0.25)
    assert row.sd_diff == pytest.approx(np.std([1, -1, 2, 2], ddof=1))


def test_bootstrap_is_repeatable_and_resamples_seeds_jointly():
    deltas = np.random.default_rng(1).normal(size=(30, 3, 2))
    deltas[:, 1] = deltas[:, 0]
    lo1, hi1 = P.bootstrap(deltas, P.rng_for("x"))
    lo2, hi2 = P.bootstrap(deltas, P.rng_for("x"))
    np.testing.assert_array_equal(lo1, lo2)
    np.testing.assert_array_equal(hi1, hi2)
    np.testing.assert_allclose(lo1[0], lo1[1])
    assert not np.allclose(lo1, P.bootstrap(deltas, P.rng_for("y"))[0])
    lo, hi = P.bootstrap(np.full((50, 1), 0.1), P.rng_for("x"))
    assert lo == hi == 0.1


def test_paired_pairs_set_tabpfn_against_every_surrogate_and_random():
    pairs = P.paired_pairs()
    assert len(pairs) == len(RULES) * 11
    assert {a for _, a, _ in pairs} == {f"TabPFN-{r}" for r in RULES}
    assert {b for _, _, b in pairs} <= set(FIELD)


def test_substituted_seed_is_dropped_from_both_arms_of_its_contrast():
    arms = {f"GP-M12-{r}": np.arange(4.0) + i for i, r in enumerate(RULES)}
    arms.update({f"TabPFN-{r}": np.arange(4.0) for r in RULES})
    cs = scores(arms, substituted={"GP-M12-UCB1": {5}})
    pairs = [(r, f"TabPFN-{r}", f"GP-M12-{r}") for r in RULES]
    t = P.paired_effects(cs, pairs)
    dropped = recorded(t, P.DROPPED)
    assert set(zip(dropped["first"], dropped.second)) == {("TabPFN-UCB1", "GP-M12-UCB1")}
    assert (dropped.n_seeds == 3).all()
    assert recorded(t).query("second == 'GP-M12-UCB1'").substituted.str.contains("seed 5").all()


def test_minimisation_keeps_improvement_positive_and_reports_source_units():
    values = {f"{s}-{r}": -np.array([10.0, 12, 14, 16]) + i
              for s in P.SURROGATES for i, r in enumerate(RULES)}
    cs = scores(values, sign=-1)
    diff = recorded(P.paired_effects(cs, [("EI", "TabPFN-EI", "TabPFN-greedy")])).iloc[0]
    assert diff.mean_diff == 1
    cost = P.acquisition_cost(cs).query("surrogate == 'TabPFN'").set_index("rule")
    assert cost.loc["greedy", "mean"] == 13
    assert cost.loc["UCB2", "mean"] == 9
    assert cost.best[cost.best].index.tolist() == ["UCB2"]
    assert (cost.goal == "minimize").all()


def test_acquisition_cost_is_the_gap_to_the_best_rule_and_marks_every_tie():
    by_rule = {"greedy": 10.0, "EI": 12.0, "PI": 12.0, "UCB1": 11.0, "UCB2": 6.0}
    cs = scores({f"{s}-{r}": np.full(4, v) for s in P.SURROGATES for r, v in by_rule.items()})
    cost = P.acquisition_cost(cs, 4.0).query("surrogate == 'TabPFN'").set_index("rule")
    assert cost.cost.to_dict() == {"greedy": 2, "EI": 0, "PI": 0, "UCB1": 1, "UCB2": 6}
    assert cost.best[cost.best].index.tolist() == ["EI", "PI"]
    assert cost.loc["UCB2", "cost_scaled"] == 1.5


@pytest.mark.parametrize("n", [66, 90, 94, 1386])
def test_seed_auc_reads_the_columns_the_ranking_reads(n):
    curve = np.arange(1.0, n + 1)[None]
    block = np.broadcast_to(curve, (len(FIELD), 1, n)).copy()
    dense = Scored(None, {"c": block}).sample(DENSE_BUDGETS)[0, :, 0, 0]
    want = np.trapezoid(dense, DENSE_BUDGETS) / np.ptp(DENSE_BUDGETS)
    assert P.seed_auc(curve, n)[0] == pytest.approx(want)
    with pytest.raises(ValueError, match="end at"):
        P.seed_auc(curve[:, :-1], n)


@pytest.fixture
def perovskite(tmp_path):
    """A liang folder holding every FIELD arm on Perovskite."""
    folder = tmp_path / "liang"
    folder.mkdir()
    shutil.copy(REFERENCE / "liang" / "protocol.json", folder)
    for arm in FIELD:
        shutil.copy(REFERENCE / "liang" / f"{arm}__Perovskite.trace.csv", folder)
    return folder


def rewrite(path, edit):
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
        names = list(rows[0])
    rows = edit(rows)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, names)
        writer.writeheader()
        writer.writerows(rows)


def perovskite_scores(folder) -> P.CampaignScores:
    record = Record.load("liang", folder.parent)
    block = np.stack([record.block(a, "Perovskite") for a in FIELD])
    return P.campaign_scores(Scored(record, {"Perovskite": block}), "Perovskite")


def test_liang_scores_are_top_percent_and_row_order_is_irrelevant(perovskite):
    before = perovskite_scores(perovskite)
    assert before.units.startswith("Top%")
    assert (Record.load("liang", perovskite.parent).block("Random", "Perovskite")[:, -1]
            == 100).all()
    rewrite(perovskite / "Random__Perovskite.trace.csv", lambda rows: rows[::-1])
    after = perovskite_scores(perovskite)
    for arm in FIELD:
        np.testing.assert_array_equal(before.scores[arm], after.scores[arm])


def test_duplicate_seed_is_refused(perovskite):
    rewrite(perovskite / "Random__Perovskite.trace.csv", lambda rows: rows + rows[:1])
    with pytest.raises(ValueError, match="duplicate seed"):
        Record.load("liang", perovskite.parent).rows("Random", "Perovskite")


def test_repeated_candidate_is_refused(perovskite):
    def repeat(rows):
        rows[0]["s5"] = rows[0]["s4"]
        return rows

    rewrite(perovskite / "TabPFN-greedy__Perovskite.trace.csv", repeat)
    with pytest.raises(ValueError, match="repeats a candidate"):
        Record.load("liang", perovskite.parent).rows("TabPFN-greedy", "Perovskite")


def test_differing_initial_designs_are_refused(perovskite):
    def swap(rows):
        rows[0]["s1"], rows[0]["s3"] = rows[0]["s3"], rows[0]["s1"]
        return rows

    rewrite(perovskite / "Random__Perovskite.trace.csv", swap)
    with pytest.raises(ValueError, match="initial designs differ"):
        perovskite_scores(perovskite)


def test_seed_tests_separate_a_clear_shift_and_not_identical_arms():
    arms = {"TabPFN": "t", "GP": "g", "RF": "r"}
    cs = scores({"t": [5.0, 6, 7, 8], "g": [1.0, 2, 3, 4], "r": [5.0, 6, 7, 8]})
    t = P.seed_tests([cs], arms).set_index("family")
    assert t.loc["GP", "verdict"] == "better" and t.loc["GP", "mean_diff"] == 4
    assert t.loc["RF", "verdict"] == "not separated"


def test_family_ranks_follow_the_summary_and_the_nemenyi_cd():
    summary = pd.DataFrame([[3.0, 2, 1]] * 4 + [[1.0, 2, 3]], columns=["a", "b", "c"])
    ranks, p, cd = P.family_ranks(summary)
    assert ranks.to_dict() == {"a": 1.4, "b": 2.0, "c": 2.6}
    # Demsar (2006), Table 5: q_0.05 = 2.343 for three classifiers.
    assert cd == pytest.approx(2.343 * np.sqrt(3 * 4 / (6 * 5)), abs=1e-3)
