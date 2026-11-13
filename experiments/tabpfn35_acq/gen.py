#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""Training data for the acquisition head: synthetic states with lookahead targets.

For a state (queried rows of a synthetic pool) and a candidate ``c``, the
target is the realized Top-5% gain of "measure c, then one PTop pick":
``1[c in top] + 1[c2 in top]``, where ``c2`` is PTop's pick after ``c``'s value
is revealed. A rule that maximizes its expectation is a two-step lookahead with
a PTop rollout, which no one-step rule is.

    POOLS=0:200 OUT=results/head_data python experiments/tabpfn35_acq/gen.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lockstep import _regressor  # noqa: E402
from rules import Round, features, p_top  # noqa: E402
from synthetic import pool, sample_state  # noqa: E402

TOP_FRAC = 0.05
STATES_PER_POOL = 6
N_CAND = 16  # 12 by PTop, 4 at random


def _round(out, y_obs, n, budget):
    return Round(
        out=out,
        y_obs=y_obs,
        n_pool=n,
        top_frac=TOP_FRAC,
        rng=np.random.default_rng(0),
        budget_frac=budget,
    )


def main() -> int:
    start, stop = (int(v) for v in os.environ["POOLS"].split(":"))
    out_dir = Path(os.environ["OUT"])
    out_dir.mkdir(parents=True, exist_ok=True)
    reg = _regressor("cuda" if torch.cuda.is_available() else "cpu", 8)
    for p in range(start, stop):
        path = out_dir / f"pool{p:05d}.npz"
        if path.exists():
            continue
        t0 = time.time()
        rng = np.random.default_rng([7, p])
        x, y = pool(rng)
        n = len(y)
        top = np.zeros(n, dtype=bool)
        top[np.argsort(-y)[: int(np.ceil(n * TOP_FRAC))]] = True
        feats, targets, groups = [], [], []
        for s in range(STATES_PER_POOL):
            obs = sample_state(y, rng, max_obs=n // 2)
            rest = np.setdiff1d(np.arange(n), obs)
            out = reg.predict_batched([x[obs]], [y[obs]], [x[rest]], output_type="full")[0]
            if out["logits"].shape[-1] == 1:
                continue
            r = _round(out, y[obs], n, len(obs) / n)
            f = features(r)
            order = np.argsort(-p_top(r))
            pick = np.concatenate(
                [order[:12], rng.choice(order[12:], min(4, len(order) - 12), replace=False)]
            )
            cands = rest[pick]
            x_tr = [x[np.append(obs, c)] for c in cands]
            y_tr = [y[np.append(obs, c)] for c in cands]
            x_te = [x[rest[rest != c]] for c in cands]
            outs = reg.predict_batched(x_tr, y_tr, x_te, output_type="full")
            v = []
            for c, o, yt in zip(cands, outs, y_tr, strict=True):
                left = rest[rest != c]
                if o["logits"].shape[-1] == 1:
                    c2 = rng.choice(left)
                else:
                    c2 = left[int(np.argmax(p_top(_round(o, yt, n, (len(obs) + 1) / n))))]
                v.append(float(top[c]) + float(top[c2]))
            feats.append(f[pick])
            targets.append(np.column_stack([v, top[cands]]))
            groups.append(np.full(len(pick), s))
        np.savez(
            path, x=np.concatenate(feats), t=np.concatenate(targets), g=np.concatenate(groups), n=n
        )
        print(f"pool {p}: n={n} {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
