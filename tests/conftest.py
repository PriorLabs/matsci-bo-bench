#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""Fixtures shared by the tests."""

from __future__ import annotations

from matbo.pool import PoolRules

# Pool rules for synthetic campaigns built inside tests.
SYNTHETIC = PoolRules(
    follows="synthetic test fixture, no source",
    replicates="keep", order="source", direction="per-campaign",
    constraints="unconstrained", not_reproduced=(),
)
