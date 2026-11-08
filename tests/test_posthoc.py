#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""Contracts of Table D.1 and the best-rule table behind Figure D.9."""

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


def test_minimisation_keeps_improvement_positive_and_reports_source_units():
    values = {f"{s}-{r}": -np.array([10.0, 12, 14, 16]) + i
              for s in P.SURROGATES for i, r in enumerate(RULES)}
    cs = scores(values, sign=-1)
    cost = P.acquisition_cost(cs).query("surrogate == 'TabPFN'").set_index("rule")
    assert cost.loc["greedy", "mean"] == 13
    assert cost.loc["UCB2", "mean"] == 9
    assert cost.best[cost.best].index.tolist() == ["UCB2"]
    assert (cost.goal == "minimize").all()


def test_acquisition_cost_is_the_gap_to_the_best_rule_and_marks_every_tie():
    by_rule = {"greedy": 10.0, "EI": 12.0, "PI": 12.0, "UCB1": 11.0, "UCB2": 6.0}
    cs = scores({f"{s}-{r}": np.full(4, v) for s in P.SURROGATES for r, v in by_rule.items()})
    cost = P.acquisition_cost(cs).query("surrogate == 'TabPFN'").set_index("rule")
    assert cost.cost.to_dict() == {"greedy": 2, "EI": 0, "PI": 0, "UCB1": 1, "UCB2": 6}
    assert cost.best[cost.best].index.tolist() == ["EI", "PI"]


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
