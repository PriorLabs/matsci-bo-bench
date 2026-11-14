#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""Pool-level acquisition head: set attention over every candidate's predictive.

Each candidate is a token: the scale-free summary of :func:`rules.features`
plus its survival curve on a fixed grid around the plug-in top-5% cut. A small
transformer encoder lets candidates see each other, and a linear readout adds a
residual to the metric's one-step rule (log PTop for Top-5% found, log EI for
best value). The readout starts at zero, so an untrained head is that rule.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from rules import Round, _np, ei, features, p_top, top_threshold
from torch import nn

#: Survival thresholds, in units of the observed targets' sd around the top-5% cut.
GRID = np.linspace(-3.0, 3.0, 13)


def base_logit(r: Round) -> np.ndarray:
    """The metric's one-step rule, as a log score."""
    if r.metric == "best-value":
        scale = float(np.std(r.y_obs)) or 1.0
        return np.log(ei(r) / scale + 1e-9)
    return np.log(p_top(r) + 1e-9)


def tokens(r: Round) -> tuple[np.ndarray, np.ndarray]:
    """([candidate, feature] tokens, [candidate] base logit)."""
    c, logits = r.criterion, r.logits
    y = np.asarray(r.y_obs, dtype=np.float64)
    scale = float(np.std(y)) or 1.0
    tau = top_threshold(r, int(np.ceil(r.n_pool * r.top_frac)))
    ys = torch.as_tensor(tau + GRID * scale, dtype=c.borders.dtype, device=logits.device)
    surv = 1.0 - c.cdf(logits, ys.repeat(logits.shape[0], 1))
    base = base_logit(r)
    tok = np.column_stack(
        [
            features(r),
            _np(surv).reshape(logits.shape[0], -1),
            base,
            np.full(len(base), float(r.metric == "best-value")),
        ]
    )
    return np.nan_to_num(tok, posinf=1e6, neginf=-1e6), base


class PoolHead(nn.Module):
    def __init__(self, n_in: int, width: int = 64, depth: int = 2, heads: int = 4):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_in, width), nn.GELU(), nn.Linear(width, width))
        layer = nn.TransformerEncoderLayer(
            width, heads, 2 * width, dropout=0.0, batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(layer, depth)
        self.out = nn.Linear(width, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.register_buffer("mu", torch.zeros(n_in))
        self.register_buffer("sd", torch.ones(n_in))

    def forward(self, tok: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        """[batch, candidate] policy logits; ``tok`` [batch, candidate, n_in]."""
        h = self.enc(self.inp((tok - self.mu) / self.sd))
        return base + self.out(h).squeeze(-1)


_LOADED: PoolHead | None = None


def load(path: str) -> PoolHead:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    net = PoolHead(blob["n_in"], **blob["arch"])
    net.load_state_dict(blob["state"])
    return net.eval()


def score_round(r: Round) -> np.ndarray:
    global _LOADED
    if _LOADED is None:
        _LOADED = load(os.environ["POOL_HEAD_PATH"])
    tok, base = tokens(r)
    with torch.no_grad():
        out = _LOADED(
            torch.as_tensor(tok, dtype=torch.float32)[None],
            torch.as_tensor(base, dtype=torch.float32)[None],
        )
    return out[0].double().numpy()
