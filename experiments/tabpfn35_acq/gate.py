#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""Offline gate: rules on held-out synthetic pools, paired per (pool, seed).

Pools come from a seed stream gen.py never draws. The score is Top-5% found,
averaged over the benchmark's 5-50% budgets, or (``METRIC=best-value``) the
best value found as a percentage of the pool's range.

    POOLS=0:40 RULES=PTop,Head OUT=results/gate python experiments/tabpfn35_acq/gate.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lockstep import _regressor, run_lockstep, stop_length  # noqa: E402
from synthetic import pool, pool_mixed  # noqa: E402

from matbo.analysis.field import DENSE_BUDGETS, budget_columns  # noqa: E402

SEEDS = 4


def main() -> int:
    start, stop = (int(v) for v in os.environ["POOLS"].split(":"))
    rules = os.environ["RULES"].split(",")
    out_dir = Path(os.environ["OUT"])
    out_dir.mkdir(parents=True, exist_ok=True)
    reg = _regressor("cuda" if torch.cuda.is_available() else "cpu", 8)
    metric = os.environ.get("METRIC", "topcount")
    # "mixed" pools come from a seed stream rl.py never draws ([13, it] there).
    mixed = os.environ.get("PRIOR", "gp") == "mixed"
    for p in range(start, stop):
        path = out_dir / f"pool{p:05d}.json"
        if path.exists():
            continue
        if mixed:
            x, y = pool_mixed(np.random.default_rng([17, p]), 50, 300)
        else:
            x, y = pool(np.random.default_rng([11, p]))
        n = len(y)
        top = np.zeros(n, dtype=bool)
        top[np.argsort(-y)[: int(np.ceil(n * 0.05))]] = True
        runs = [(r, s) for r in rules for s in range(SEEDS)]
        inits = {
            s: np.random.default_rng([11, p, s]).choice(n, 2, replace=False) for s in range(SEEDS)
        }
        traces = [list(map(int, inits[s])) for _, s in runs]
        traces = run_lockstep(
            reg,
            x,
            y,
            runs,
            traces,
            stop_length(n, 0.5),
            0.05,
            32,
            log=lambda m: None,
            metric=metric,
        )
        cols = budget_columns(n, DENSE_BUDGETS)
        res = {}
        for (r, s), tr in zip(runs, traces, strict=True):
            if metric == "topcount":
                curve = 100.0 * np.cumsum(top[tr]) / top.sum()
            else:
                curve = 100.0 * (np.maximum.accumulate(y[tr]) - y.min()) / (np.ptp(y) or 1.0)
            auc = np.trapezoid(curve[cols], DENSE_BUDGETS) / np.ptp(DENSE_BUDGETS)
            res.setdefault(r, []).append(float(auc))
        path.write_text(json.dumps({"n": n, "auc": res}))
        print(
            f"pool {p}: n={n} " + " ".join(f"{r}={np.mean(v):.1f}" for r, v in res.items()),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
