#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""Synthetic pools: GP draws (random Fourier features) on experiment-like designs.

Grids, uniform boxes, categorical products and mixes, with warped outputs and
optional replicates, sized like the benchmark's smaller campaigns.
"""

import itertools

import numpy as np

MIN_N, MAX_N = 60, 300
MAX_DIM = 16


def pool(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One pool (X [n, d], y [n]); larger y is better."""
    n = int(np.exp(rng.uniform(np.log(MIN_N), np.log(MAX_N))))
    x, y = design_gp(n, rng)
    return x.astype(np.float64), (-y if rng.random() < 0.5 else y).astype(np.float64)


def _design(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """(encoded X, latent Z) for one of: a grid, a uniform box, a categorical product, or a mix."""
    kind = rng.choice(["grid", "uniform", "categorical", "mixed"])
    n_cont = {
        "grid": rng.integers(2, 7),
        "uniform": rng.integers(2, 13),
        "categorical": 0,
        "mixed": rng.integers(1, 4),
    }[kind]
    n_cat = {
        "grid": 0,
        "uniform": 0,
        "categorical": rng.integers(2, 6),
        "mixed": rng.integers(1, 4),
    }[kind]
    levels = [rng.integers(3, 13) for _ in range(n_cont)] if kind != "uniform" else []
    options = [int(rng.integers(2, 13)) for _ in range(n_cat)]
    sizes = levels + options
    if sizes and np.prod(sizes, dtype=float) <= 50_000:
        grid = np.array(list(itertools.product(*[range(s) for s in sizes])))
        rows = grid[rng.choice(len(grid), min(n, len(grid)), replace=False)]
    else:
        rows = (
            np.stack([rng.integers(0, s, n) for s in sizes], 1) if sizes else np.zeros((n, 0), int)
        )
    m = len(rows)
    cont = (
        rng.random((m, n_cont)) if kind == "uniform" else rows[:, :n_cont] / (np.array(levels) - 1)
    )
    # Each option of a categorical input sits at a random point of a 2-d latent space.
    lat = [rng.normal(size=(o, 2))[rows[:, n_cont + j]] for j, o in enumerate(options)]
    onehot = [np.eye(o)[rows[:, n_cont + j]] for j, o in enumerate(options)]
    x = np.concatenate([cont] + onehot, 1)
    z = np.concatenate([cont] + lat, 1)
    return x, z


def design_gp(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A GP draw (random Fourier features, ARD lengthscales) on an experiment-like design."""
    x, z = _design(n, rng)
    while x.shape[1] > MAX_DIM or len(x) < MIN_N:
        x, z = _design(n, rng)
    ls = np.exp(rng.normal(np.log(0.4), 0.7, z.shape[1]))
    r = 256
    w = rng.normal(size=(z.shape[1], r)) / ls[:, None]
    f = np.sqrt(2 / r) * np.cos(z @ w + rng.uniform(0, 2 * np.pi, r)) @ rng.normal(size=r)
    f = f + z @ rng.normal(0, rng.uniform(0, 0.5), z.shape[1])
    f = (f - f.mean()) / (f.std() + 1e-9)
    warp = rng.choice(["none", "exp", "sigmoid", "abs"])
    if warp == "exp":
        f = np.exp(rng.uniform(0.5, 2.0) * f)
    elif warp == "sigmoid":
        f = 100 / (1 + np.exp(-rng.uniform(1, 3) * (f - rng.uniform(0, 1.5))))
    elif warp == "abs":
        f = -np.abs(f - rng.normal())
    noise = np.exp(rng.uniform(np.log(1e-3), np.log(0.3))) * (f.std() + 1e-9)
    y = f + rng.normal(size=len(f)) * noise
    if rng.random() < 0.2:  # replicated settings with fresh noise
        dup = rng.choice(len(x), max(1, int(rng.uniform(0.1, 0.3) * len(x))), replace=True)
        x = np.concatenate([x, x[dup]])
        y = np.concatenate([y, f[dup] + rng.normal(size=len(dup)) * noise])
    return x, y


def sample_state(y: np.ndarray, rng: np.random.Generator, max_obs: int) -> np.ndarray:
    """Indices of the queried set: 2 random rows plus picks that a BO run might have made.

    A quarter of states continue at random; the rest pick with weight exp(beta * rank percentile),
    which enriches the queried set in good rows the way an exploiting policy does.
    """
    n = len(y)
    hi = max(4, min(n // 2, max_obs))
    n_obs = int(round(np.exp(rng.uniform(np.log(3), np.log(hi)))))
    init = rng.choice(n, 2, replace=False)
    w = np.ones(n)
    if rng.random() >= 0.25:
        u = np.argsort(np.argsort(y)) / (n - 1)
        w = np.exp(rng.uniform(1, 12) * (u - 1))
    w[init] = 0
    picks = rng.choice(n, n_obs - 2, replace=False, p=w / w.sum())
    return np.concatenate([init, picks])
