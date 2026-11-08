#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""Pool sizes the sources report, and what a run prints about each pool.

Every liang and olympus pool is expected to match its source exactly, so any
difference is reported as a loader bug.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from matbo.campaign import Campaign


@dataclass(frozen=True)
class Published:
    """Rows and features the source reports for one campaign."""

    rows: int
    features: int
    source: str


# Liang et al. duplicate-average as we do, so these match after averaging.
_LIANG = "Liang et al., npj Comput. Mater. 7, 188 (2021), Table 1"
# Rows of the package's bundled data.csv. Where the dataset description
# disagrees (snar's says 67 samples; the csv ships 66), the csv is what a run
# can reproduce.
_OLYMPUS = "Hase et al., MLST 2, 035021 -- bundled olympus data.csv row count"

PAPER_POOL: dict[str, Published] = {
    "Crossed barrel": Published(600, 4, _LIANG),
    "P3HT": Published(178, 5, _LIANG),
    "AgNP": Published(164, 5, _LIANG),
    "Perovskite": Published(94, 3, _LIANG),
    "AutoAM": Published(100, 4, _LIANG),
    "snar": Published(66, 4, _OLYMPUS),
    "benzylation": Published(73, 4, _OLYMPUS),
    "colors_n9": Published(102, 3, _OLYMPUS),
    "alkox": Published(208, 4, _OLYMPUS),
    "colors_bob": Published(241, 5, _OLYMPUS),
    "suzuki": Published(247, 4, _OLYMPUS),
    "fullerenes": Published(246, 3, _OLYMPUS),
    "photo_pce10": Published(1040, 4, _OLYMPUS),
    "photo_wf3": Published(1040, 4, _OLYMPUS),
    "hplc": Published(1386, 6, _OLYMPUS),
}


def describe(campaign: Campaign, top_frac: float = 0.05) -> list[str]:
    """Lines a run prints about this pool. Reports only; never filters."""
    lines = []
    published = PAPER_POOL.get(campaign.name)
    raw = campaign._meta.get("raw_rows")
    if published:
        bits = []
        if published.rows != campaign.n:
            bits.append(f"paper reports {published.rows} (UNEXPLAINED -- the loader may be wrong)")
            if raw is not None and raw != campaign.n:
                bits.append(f"{raw} rows read, {raw - campaign.n} collapsed or dropped")
        if published.features != campaign.dim:
            bits.append(f"paper reports {published.features} features, we build {campaign.dim}")
        if bits:
            lines.append("pool differs from the source: " + "; ".join(bits))
        elif raw is not None and raw != campaign.n:
            lines.append(
                f"pool matches the source ({published.rows}); {raw} rows "
                f"duplicate-averaged to {campaign.n}"
            )

    if campaign.notes:
        lines.append(f"declared constraints: {campaign.notes}")

    values, counts = np.unique(campaign.y, return_counts=True)
    frac = counts.max() / campaign.n
    if frac > 0.05:
        lines.append(
            f"degenerate target: {frac:.0%} of the pool is y={values[counts.argmax()]:.4g} "
            "(reported, not excluded)"
        )
    if not campaign.is_discriminative(top_frac):
        lines.append(
            "target is too flat for the top set to be well defined (reported, not excluded)"
        )
    return lines


def unexplained(campaign: Campaign) -> bool:
    """True when the pool's size or width differs from what the source reports."""
    published = PAPER_POOL.get(campaign.name)
    if published is None:
        return False
    return published.rows != campaign.n or published.features != campaign.dim
