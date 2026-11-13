#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""Fit the acquisition head on gen.py's states; hold out a fifth of the pools.

python experiments/tabpfn35_acq/train.py results/head_data results/head.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

WIDTH = 64


def load(folder: Path):
    xs, ts, gs, ps = [], [], [], []
    for i, path in enumerate(sorted(folder.glob("pool*.npz"))):
        d = np.load(path)
        xs.append(d["x"])
        ts.append(d["t"])
        gs.append(d["g"] + 100 * i)
        ps.append(np.full(len(d["x"]), i))
    x = np.nan_to_num(np.concatenate(xs), posinf=1e6, neginf=-1e6)
    return x, np.concatenate(ts), np.concatenate(gs), np.concatenate(ps)


def pick_value(score, v, g):
    """Mean target of each state's top-scored candidate."""
    out = []
    for k in np.unique(g):
        m = g == k
        out.append(v[m][np.argmax(score[m])])
    return float(np.mean(out))


def main() -> int:
    x, t, g, p = load(Path(sys.argv[1]))
    v = t[:, 0]
    test = p % 5 == 0
    mu, sd = x[~test].mean(0), x[~test].std(0) + 1e-6
    z = torch.as_tensor((x - mu) / sd, dtype=torch.float32)
    tv = torch.as_tensor(v, dtype=torch.float32)
    torch.manual_seed(0)
    net = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], WIDTH),
        torch.nn.GELU(),
        torch.nn.Linear(WIDTH, WIDTH),
        torch.nn.GELU(),
        torch.nn.Linear(WIDTH, 1),
    )
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    tr = np.flatnonzero(~test)
    rng = np.random.default_rng(0)
    for epoch in range(60):
        for idx in np.array_split(rng.permutation(tr), max(1, len(tr) // 256)):
            loss = torch.mean((net(z[idx]).squeeze(-1) - tv[idx]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 10 == 9:
            with torch.no_grad():
                pred = net(z).squeeze(-1).numpy()
            print(
                f"epoch {epoch + 1}: train mse {np.mean((pred[~test] - v[~test]) ** 2):.4f} "
                f"test mse {np.mean((pred[test] - v[test]) ** 2):.4f}"
            )
    with torch.no_grad():
        pred = net(z).squeeze(-1).numpy()
    gt, vt = g[test], v[test]
    print(f"held-out states: {len(np.unique(gt))}, candidates {test.sum()}")
    print(
        f"  mean 2-step value of pick: head {pick_value(pred[test], vt, gt):.4f}  "
        f"PTop {pick_value(x[test, 0], vt, gt):.4f}  "
        f"oracle {pick_value(vt, vt, gt):.4f}"
    )
    torch.save(
        {
            "n_in": x.shape[1],
            "width": WIDTH,
            "state": net.state_dict(),
            "mu": torch.as_tensor(mu),
            "sd": torch.as_tensor(sd),
        },
        sys.argv[2],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
