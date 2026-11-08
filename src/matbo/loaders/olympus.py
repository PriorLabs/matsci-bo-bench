#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""Ten experimentally derived campaigns from Olympus.

    Hase et al., Mach. Learn.: Sci. Technol. 2, 035021.

Every shipped row is a candidate, in shipped order, with the direction taken
from the config's ``default_goal``. Rows are read from the ``olymp`` package's
bundled csv and json; the package itself is never imported.

Quirks of the shipped data:

* Configs state parameter bounds as ``low``/``high`` in six datasets and
  ``lower``/``upper`` in four. :func:`_bounds` reads both.
* ``colors_bob`` is declared ``simplex`` but its rows sum to 0.64-4.41, and
  every ``alkox`` row falls outside its declared box. Violations are reported
  in ``Campaign.notes``, never dropped.
* ``dataset_alkox/config.json`` is a copy of snar's, so alkox's parameter names
  and bounds are snar's. Columns are positional, so no number depends on the
  names.
"""

from __future__ import annotations

import json

import numpy as np

from matbo.benchmarks import OLYMPUS as _BENCH
from matbo.campaign import Campaign
from matbo.loaders import exact_csv
from matbo.loaders.paths import olympus_dir
from matbo.pool import build_pool, oriented

OLYMPUS = [
    "snar",
    "benzylation",
    "colors_n9",
    "alkox",
    "colors_bob",
    "suzuki",
    "fullerenes",
    "photo_pce10",
    "photo_wf3",
    "hplc",
]

# Defects in the shipped metadata, appended to the constraint report so a
# reader knows what a violation count means.
CONFIG_DEFECTS = {
    "alkox": (
        "config.json is dataset_snar's, copied -- same parameter names and "
        "bounds, only the measurement renamed. The real space is Catalase "
        "[0.05, 1.0] / HRP [0.5, 10.0] / AlcOx [2.0, 8.0] / pH [6.0, 8.0] "
        "(Hase et al., table V), which the csv columns match exactly, so the "
        "violations above are snar's box, not alkox's measurements"
    ),
}


def _bounds(param: dict) -> tuple[float, float] | None:
    """A parameter's declared box, under either of the two config schemas."""
    if "low" in param and "high" in param:
        return float(param["low"]), float(param["high"])
    if "lower" in param and "upper" in param:
        return float(param["lower"]), float(param["upper"])
    return None


def _constraint_violations(x: np.ndarray, cfg: dict) -> list[str]:
    """Rows failing the config's declared feasible region, as report lines."""
    declared = cfg.get("constraints", {}).get("parameters", "none")
    lines = []
    if declared == "simplex":
        off = int((np.abs(x.sum(axis=1) - 1.0) > 1e-6).sum())
        if off:
            sums = x.sum(axis=1)
            lines.append(
                f"declared simplex, but {off}/{len(x)} rows do not sum to 1 "
                f"(range {sums.min():.3f}-{sums.max():.3f})"
            )
    boxes = [_bounds(p) for p in cfg["parameters"]]
    if declared in ("cube", "none") and all(b is not None for b in boxes):
        lo = np.array([b[0] for b in boxes])
        hi = np.array([b[1] for b in boxes])
        off = int((~((x >= lo - 1e-9) & (x <= hi + 1e-9)).all(axis=1)).sum())
        if off:
            lines.append(f"{off}/{len(x)} rows fall outside the declared parameter bounds")
    return lines


def load(stem: str) -> Campaign:
    """Load one Olympus campaign under the benchmark's declared pool rules."""
    root = olympus_dir() / f"dataset_{stem}"
    if not root.is_dir():
        raise FileNotFoundError(
            f"Olympus campaign {stem!r} needs the package's bundled datasets, and "
            f"{root} does not exist.\n"
            "  Obtain it: `uv pip install olymp --no-deps` (the package fails to "
            "*import* on matplotlib >= 3.9, but we only read its csv+json), or point "
            "OLYMPUS_DIR at a checkout's olympus/datasets directory."
        )
    cfg = json.loads((root / "config.json").read_text())
    features = [p["name"] for p in cfg["parameters"]]
    objectives = [m["name"] for m in cfg["measurements"]]
    if len(objectives) != 1:
        raise ValueError(f"{stem}: expected one measurement, config declares {objectives}")
    # The csvs have no header row; the names come from the config.
    df = exact_csv.read_csv(root / "data.csv", header=None, names=features + objectives)

    objective = objectives[0]
    maximize = cfg["default_goal"] == "maximize"
    pooled = build_pool(df, features, objective, _BENCH.pool)
    x = pooled[features].to_numpy(dtype=float)
    y_raw = pooled[objective].to_numpy(dtype=float)
    ok = np.isfinite(x).all(1) & np.isfinite(y_raw)
    x, y_raw = x[ok], y_raw[ok]

    notes = _constraint_violations(x, cfg)
    if stem in CONFIG_DEFECTS:
        notes.append(CONFIG_DEFECTS[stem])

    return Campaign(
        name=stem,
        family=_BENCH.name,
        x=x,
        y=oriented(y_raw, maximize),
        objective=f"{objective} ({'max' if maximize else 'min'})",
        pool=_BENCH.pool,
        notes="; ".join(notes),
        _meta={
            "raw_rows": len(df),
            "stem": stem,
            # The source's own units and direction, for reporting.
            "goal": cfg["default_goal"],
            "raw_y": y_raw,
            "declared_constraints": cfg.get("constraints", {}),
        },
    )
