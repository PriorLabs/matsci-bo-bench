#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""How raw source rows become a candidate pool.

Every field of :class:`PoolRules` is required, so a benchmark must state how it
treats repeated measurements, what order its candidates are in, where the
optimisation direction comes from, and which parts of its source's protocol it
does not reproduce. :func:`build_pool` is the only path from a source frame to
a Campaign's ``x``/``y``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# How repeated measurements of one design point collapse to one candidate.
REPLICATES = {
    # Liang et al.: the mean of the *distinct* repeat values, so 5, 5, 9 -> 7.
    "unique-mean",
    # Every measured row is its own candidate.
    "keep",
}
# The order candidates sit in. Trace indices refer to this order.
ORDER = {
    # Sorted by feature value, as a groupby leaves them.
    "grouped",
    # As the source file ships them.
    "source",
}
# Where "larger is better" comes from.
DIRECTION = {
    # A per-campaign flag in the loader.
    "per-campaign",
    # The dataset's own config, read at load time.
    "config-default-goal",
}
# Whether the source declares a feasible region for its parameters.
CONSTRAINTS = {
    "unconstrained",
    # The loader checks rows against the declared region and reports
    # violations. It never drops rows.
    "declared",
}


@dataclass(frozen=True)
class PoolRules:
    """One benchmark's pool construction. No field has a default."""

    # The protocol this reproduces, as a citation.
    follows: str
    replicates: str
    order: str
    direction: str
    constraints: str
    # Parts of the source's own protocol this pool does not reproduce, each
    # with the reason.
    not_reproduced: tuple[str, ...]

    def __post_init__(self):
        for field_name, allowed in (
            ("replicates", REPLICATES), ("order", ORDER),
            ("direction", DIRECTION), ("constraints", CONSTRAINTS),
        ):
            value = getattr(self, field_name)
            if value not in allowed:
                raise ValueError(
                    f"{field_name}={value!r} is not one of {sorted(allowed)}"
                )
        if self.replicates == "keep" and self.order == "grouped":
            raise ValueError(
                "order='grouped' is the residue of averaging; with "
                "replicates='keep' there is nothing to group, so the order is "
                "the source's -- declare order='source'"
            )
        if not self.follows:
            raise ValueError("follows must name the protocol this reproduces")

    def summary(self) -> str:
        """One line for the run log."""
        return (
            f"pool rules: follows {self.follows}; replicates={self.replicates}, "
            f"order={self.order}, direction={self.direction}, "
            f"constraints={self.constraints}"
        )


def build_pool(
    frame: pd.DataFrame,
    features: list[str],
    objective: str,
    rules: PoolRules,
) -> pd.DataFrame:
    """Collapse ``frame`` to one row per candidate, with ``features + [objective]`` columns."""
    if rules.replicates == "keep":
        return frame[[*features, objective]].reset_index(drop=True)

    grouped = (
        frame.groupby(features, sort=rules.order == "grouped")[objective]
        .agg(lambda v: v.unique().mean())
        .to_frame()
        .reset_index()
    )
    return grouped[[*features, objective]]


def oriented(y: np.ndarray, maximize: bool) -> np.ndarray:
    """Negate a minimisation objective so larger is always better."""
    y = np.asarray(y, dtype=np.float64)
    return y if maximize else -y
