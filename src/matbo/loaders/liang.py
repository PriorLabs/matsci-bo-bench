#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""The five experimental materials campaigns from Liang et al.

    Liang, Gongora et al., "Benchmarking the performance of Bayesian
    optimization across multiple experimental materials science domains",
    npj Computational Materials 7, 188 (2021).
    code + data: https://github.com/PV-Lab/Benchmarking

Each csv is (features..., objective) with the objective last. Perovskite and
AgNP minimise; the other three maximise.
"""

from __future__ import annotations

from matbo.benchmarks import LIANG as _BENCH
from matbo.campaign import Campaign
from matbo.loaders import exact_csv
from matbo.loaders.paths import LIANG_DIR
from matbo.pool import build_pool, oriented

# (stem, higher-is-better)
LIANG = [
    ("Crossed barrel", True),
    ("P3HT", True),
    ("AgNP", False),
    ("Perovskite", False),
    ("AutoAM", True),
]


def load(stem: str, maximize: bool) -> Campaign:
    raw = exact_csv.read_csv(LIANG_DIR / f"{stem}_dataset.csv")
    features, objective = list(raw.columns)[:-1], raw.columns[-1]
    grouped = build_pool(raw, features, objective, _BENCH.pool)
    return Campaign(
        name=stem,
        family="liang",
        x=grouped[features].to_numpy(dtype=float),
        y=oriented(grouped[objective].to_numpy(dtype=float), maximize),
        objective=f"{objective} ({'max' if maximize else 'min'})",
        pool=_BENCH.pool,
        _meta={"raw_rows": len(raw)},
    )
