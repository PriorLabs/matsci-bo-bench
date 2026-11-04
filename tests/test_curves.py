#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""Curve layout: which column is which experiment."""

from __future__ import annotations

import numpy as np
import pytest

from matbo import Curves, Protocol


def test_curve_width_is_initial_plus_budget():
    p = Protocol(n_initial=2, n_steps=20, scale_budget=False)
    assert p.curve_width(100) == 2 + p.budget(100)


def test_at_budget_counts_past_the_initial_design():
    values = np.tile(np.arange(22, dtype=float), (4, 1))
    cv = Curves(values, n_initial=2)
    assert cv.budget == 20
    assert np.array_equal(cv.at_budget(0), values[:, 1])
    assert np.array_equal(cv.at_budget(cv.budget), cv.final())


def test_at_budget_rejects_out_of_range():
    cv = Curves(np.zeros((3, 22)), n_initial=2)
    with pytest.raises(IndexError):
        cv.at_budget(21)


def test_x_axis_counts_total_experiments():
    """A 2+20 campaign ran 22 experiments, not 24."""
    cv = Curves(np.zeros((3, 22)), n_initial=2)
    assert cv.x_axis()[-1] == 22
