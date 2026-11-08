# Copyright (c) Prior Labs GmbH 2026. Licensed under the Apache License, Version 2.0.
"""Contracts needed to reproduce the paper's figures from the committed record."""

from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
from matbo.analysis.record import Record  # noqa: E402
from matbo.visualisation import FIGURES  # noqa: E402
from matbo.analysis import field as F  # noqa: E402
from matbo.analysis.field import FIELD, Scored, kernel_values, normalize, score  # noqa: E402


def test_scored_holds_exactly_the_field():
    with pytest.raises(ValueError, match="FIELD has 56"):
        Scored(None, {"synthetic": np.ones((len(FIELD) + 1, 2, 4))})


def test_ni_spans_the_field():
    block = np.ones((len(FIELD), 2, 4))
    block[FIELD.index("Random")] = 0
    block[FIELD.index("TabPFN-EI")] = 2
    ni = Scored(None, {"synthetic": block}).ni()
    assert np.all(ni[..., FIELD.index("Random")] == 0)
    assert np.all(ni[..., FIELD.index("TabPFN-EI")] == 1)


def test_undefined_ni_does_not_destroy_raw_kernel_measurements():
    block = np.full((len(FIELD), 2, 20), 6.0)
    scored = Scored(None, {"synthetic": block})
    assert np.isnan(scored.ni()).all()
    assert np.isnan(normalize(np.array([[5., 5.]]))).all()
    for values in kernel_values(scored):
        np.testing.assert_allclose(values, 6)


def test_incomplete_paper_field_is_refused(monkeypatch):
    path = Record.path
    monkeypatch.setattr(Record, "path", lambda self, arm, campaign: (
        Path("/nonexistent") if arm == "Random" else path(self, arm, campaign)))
    with pytest.raises(ValueError, match="incomplete paper field.*Random"):
        F._score.__wrapped__("liang", "topcount")


@pytest.mark.parametrize("benchmark,expected", [
    ("liang", ["TabPFN-UCB1", "TabICL-PI", "GP-ARD-Warp-UCB1", "CatBoost-UCB1", "RF-greedy",
               "Random"]),
    ("olympus", ["TabPFN-EI", "TabICL-EI", "GP-MLP-PI", "CatBoost-PI", "RF-PI", "Random"]),
])
def test_representatives_match_the_paper_linear_budget_ranking(benchmark, expected):
    scored = score(benchmark)
    assert scored.best() == expected
    ranked = scored.ranking()
    assert len(ranked) == 55
    assert set(ranked.index) == set(FIELD) - {"Random"}
    assert ranked.index[0] == expected[0]


@pytest.mark.parametrize("number,axes", [
    ("1", 2), ("2", 1), ("D.1", 15), ("D.2", 10), ("D.3", 2), ("D.4", 1), ("D.5", 1),
    ("D.6", 2), ("D.7", 2), ("D.8", 10), ("D.9", 1),
])
def test_paper_figures_build(number, axes):
    fig = FIGURES[number]()
    try:
        fig.canvas.draw()
        assert len(fig.axes) == axes
        if number in ("D.4", "D.5"):
            labels = {line.get_label() for line in fig.axes[0].lines}
            assert labels & set(FIELD) == set(FIELD)
        if number == "D.1":
            assert [ax.get_ylabel().split('\n')[0] for ax in fig.axes[::3]] == [
                "Perovskite", "AutoAM", "AgNP", "P3HT", "Crossed barrel"]
        if number == "D.8":
            inverted = {ax.get_xlabel() for ax in fig.axes if ax.yaxis_inverted()}
            assert inverted == {"benzylation", "snar", "colors_bob", "colors_n9", "photo_pce10",
                                "photo_wf3"}
    finally:
        matplotlib.pyplot.close(fig)
