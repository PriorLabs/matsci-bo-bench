#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""Acquisition rules read off TabPFN's bar distribution over the pool.

Every rule takes one round's ``predict(output_type="full")`` output and returns
one score per candidate; larger wins. Targets are oriented larger-is-better.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import norm


@dataclass(frozen=True)
class Round:
    """One dataset's predictive over its candidates, plus what the rules need."""

    out: dict
    y_obs: np.ndarray
    n_pool: int
    top_frac: float
    rng: np.random.Generator
    #: Picks made so far (initial design included) over the pool size.
    budget_frac: float = 0.0
    #: The benchmark's metric: "topcount" (Top-5% found) or "best-value".
    metric: str = "topcount"

    @property
    def criterion(self):
        return self.out["criterion"]

    @property
    def logits(self) -> torch.Tensor:
        logits = self.out["logits"]
        borders = self.criterion.borders
        return logits.to(borders.dtype) if logits.dtype != borders.dtype else logits

    @property
    def probs(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=-1)


def _np(t: torch.Tensor) -> np.ndarray:
    return np.asarray(t.detach().cpu().numpy(), dtype=np.float64).ravel()


def greedy(r: Round) -> np.ndarray:
    return np.asarray(r.out["mean"], dtype=np.float64).ravel()


def ei(r: Round) -> np.ndarray:
    return _np(r.criterion.ei(r.logits, float(np.max(r.y_obs)), maximize=True))


def pi(r: Round) -> np.ndarray:
    c, logits = r.criterion, r.logits
    best = torch.full(
        logits.shape[:-1], float(np.max(r.y_obs)), dtype=c.borders.dtype, device=logits.device
    )
    return _np(c.pi(logits, best, maximize=True))


def ucb(beta: float):
    def _ucb(r: Round) -> np.ndarray:
        return _np(r.criterion.ucb(r.logits, 0.0, float(norm.sf(beta)), maximize=True))

    return _ucb


def thompson(r: Round) -> np.ndarray:
    """One draw from each candidate's marginal predictive, uniform within its bar."""
    borders = r.criterion.borders
    cum = torch.cumsum(r.probs, dim=-1)
    u = torch.as_tensor(r.rng.random(cum.shape[0]), dtype=cum.dtype, device=cum.device)
    idx = torch.searchsorted(cum, u[:, None]).squeeze(-1).clamp(0, len(borders) - 2)
    w = torch.as_tensor(r.rng.random(cum.shape[0]), dtype=cum.dtype, device=cum.device)
    return _np(borders[idx] + w * (borders[idx + 1] - borders[idx]))


def top_threshold(r: Round, k: int) -> float:
    """The plug-in top-``k`` cut of the pool.

    The largest ``t`` with ``#{observed >= t} + sum_c P(y_c >= t) >= k``: the
    expected number of pool rows at or above ``t`` is ``k``. Candidate survival
    is read at the bar borders and interpolated linearly inside a bar.
    """
    borders = r.criterion.borders
    # P(y_c >= b_j) summed over candidates, at every border b_j.
    surv = 1.0 - torch.cumsum(r.probs, dim=-1)
    surv = torch.cat([torch.ones_like(surv[:, :1]), surv], dim=-1).clamp(0.0, 1.0)
    s_tot = _np(surv.sum(0))
    b = _np(borders)
    y = np.sort(np.asarray(r.y_obs, dtype=np.float64))
    grid = np.unique(np.concatenate([b, y]))
    g = np.interp(grid, b, s_tot) + (len(y) - np.searchsorted(y, grid, side="left"))
    ok = np.flatnonzero(g >= k)
    i = ok[-1]
    if i + 1 >= len(grid):
        return float(grid[i])
    # No observation lies strictly inside (lo, hi), so g is linear there; it
    # steps down at lo when an observation sits exactly on it.
    lo, hi = grid[i], grid[i + 1]
    above = len(y) - np.searchsorted(y, lo, side="right")
    h_lo = np.interp(lo, b, s_tot) + above
    h_hi = np.interp(hi, b, s_tot) + above
    if h_lo < k or h_lo <= h_hi:
        return float(lo)
    t = lo + (h_lo - k) / (h_lo - h_hi) * (hi - lo)
    return float(min(max(t, lo), hi))


def p_top(r: Round) -> np.ndarray:
    """P(y_c is in the pool's top ``top_frac``): Liang et al.'s metric, one step ahead."""
    k = int(np.ceil(r.n_pool * r.top_frac))
    t = top_threshold(r, k)
    c, logits = r.criterion, r.logits
    ys = torch.full((logits.shape[0], 1), t, dtype=c.borders.dtype, device=logits.device)
    return 1.0 - _np(c.cdf(logits, ys))


def features(r: Round) -> np.ndarray:
    """[candidate, feature] summary of each candidate's predictive, free of the target's scale.

    Values are standardized by the observed targets' spread and centred on the
    plug-in top-5% cut, so a head trained on one pool family reads any pool.
    """
    c, logits = r.criterion, r.logits
    y = np.asarray(r.y_obs, dtype=np.float64)
    scale = float(np.std(y)) or 1.0
    k = int(np.ceil(r.n_pool * r.top_frac))
    tau = top_threshold(r, k)
    inc = float(y.max())
    n_cand = logits.shape[0]

    def surv(t: float) -> np.ndarray:
        ys = torch.full((n_cand, 1), t, dtype=c.borders.dtype, device=logits.device)
        return 1.0 - _np(c.cdf(logits, ys))

    mean = greedy(r)
    q = np.stack([_np(c.icdf(logits, p)) for p in (0.1, 0.5, 0.9)], axis=1)
    p_top = surv(tau)
    ei_tau = _np(c.ei(logits, tau, maximize=True)) / scale
    rank = np.argsort(np.argsort(-p_top)) / max(n_cand - 1, 1)
    found = float(np.sum(y >= tau)) / k
    cols = [
        p_top,
        surv(inc),
        np.log(p_top + 1e-6),
        ei_tau,
        _np(c.ei(logits, inc, maximize=True)) / scale,
        (mean - tau) / scale,
        (q[:, 2] - q[:, 0]) / scale,
        (q - tau) / scale,
        rank,
        np.full(n_cand, r.budget_frac),
        np.full(n_cand, found),
        np.full(n_cand, (inc - tau) / scale),
        np.full(n_cand, float(np.sum(p_top))),
    ]
    return np.column_stack([np.asarray(v, dtype=np.float64).reshape(n_cand, -1) for v in cols])


_HEAD = None


def head(r: Round) -> np.ndarray:
    """A small MLP over :func:`features`, trained on synthetic pools (``HEAD_PATH``)."""
    global _HEAD
    if _HEAD is None:
        import os

        blob = torch.load(os.environ["HEAD_PATH"], map_location="cpu", weights_only=True)
        net = torch.nn.Sequential(
            torch.nn.Linear(blob["n_in"], blob["width"]),
            torch.nn.GELU(),
            torch.nn.Linear(blob["width"], blob["width"]),
            torch.nn.GELU(),
            torch.nn.Linear(blob["width"], 1),
        )
        net.load_state_dict(blob["state"])
        _HEAD = (net.double().eval(), blob["mu"].double(), blob["sd"].double())
    net, mu, sd = _HEAD
    f = torch.as_tensor(np.nan_to_num(features(r), posinf=1e6, neginf=-1e6))
    with torch.no_grad():
        return net((f - mu) / sd).squeeze(-1).numpy()


def pool_head(r: Round) -> np.ndarray:
    """The pool-level set-attention head (``POOL_HEAD_PATH``); see pool_head.py."""
    from pool_head import score_round

    return score_round(r)


RULES = {
    "greedy": greedy,
    "EI": ei,
    "PI": pi,
    "UCB1": ucb(1.0),
    "UCB2": ucb(2.0),
    "TS": thompson,
    "PTop": p_top,
    "Head": head,
    "PoolHead": pool_head,
}
