#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""Provenance: our pools against the sizes the sources report."""

from __future__ import annotations

import pytest

from matbo import loaders as L
from matbo import provenance


@pytest.mark.parametrize("name", L.names())
def test_pool_matches_the_published_size(name):
    """Liang et al. duplicate-average as we do and Olympus pools keep every row, so all match."""
    campaign = L.load(name)
    assert not provenance.unexplained(campaign)
    assert campaign.n == provenance.PAPER_POOL[name].rows
