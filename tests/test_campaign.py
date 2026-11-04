#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""Campaign invariants: what a loaded pool must be before anything reads it."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import SYNTHETIC
from matbo import Campaign
from matbo import loaders as L


@pytest.mark.parametrize("name", ["Perovskite", "AgNP", "snar"])
def test_campaign_loads_finite(name):
    c = L.load(name)
    assert c.n > 1 and c.dim > 0
    assert np.isfinite(c.x).all() and np.isfinite(c.y).all()


def test_minimisation_campaign_is_flipped():
    """Perovskite minimises instability, so its oriented y is the negated pooled value."""
    from matbo.benchmarks import LIANG
    from matbo.loaders import exact_csv
    from matbo.loaders.paths import LIANG_DIR
    from matbo.pool import build_pool

    raw = exact_csv.read_csv(LIANG_DIR / "Perovskite_dataset.csv")
    features, objective = list(raw.columns)[:-1], raw.columns[-1]
    # Repeats collapse before orientation, so compare against the pooled values.
    worst = build_pool(raw, features, objective, LIANG.pool)[objective].max()
    assert L.load("Perovskite").y.min() == pytest.approx(-worst)


def test_top_mask_size_matches_n_top():
    c = L.load("Perovskite")
    assert c.top_mask(0.05).sum() == c.n_top(0.05)


def test_campaign_rejects_nonfinite():
    x = np.array([[0.0], [np.nan]])
    with pytest.raises(ValueError, match="finite"):
        Campaign(name="bad", family="t", x=x, y=np.array([1.0, 2.0]),
                 objective="o", pool=SYNTHETIC)
