#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""The pool-based campaign: the one representation every loader returns."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import numpy as np

from matbo.pool import PoolRules


@dataclass(frozen=True)
class Campaign:
    """A finite library of enumerated candidates with measured objectives.

    ``y`` is oriented so that larger is better; ``objective`` names the raw
    measurement and its direction.
    """

    name: str
    family: str
    x: np.ndarray  # (n, d) float64
    y: np.ndarray  # (n,) float64, larger is better
    objective: str
    pool: PoolRules
    notes: str = ""
    _meta: dict = field(default_factory=dict, compare=False)

    def __post_init__(self):
        if self.x.ndim != 2:
            raise ValueError(f"{self.name}: x must be 2-D, got shape {self.x.shape}")
        if self.y.ndim != 1 or len(self.y) != len(self.x):
            raise ValueError(
                f"{self.name}: y must be 1-D of length {len(self.x)}, got {self.y.shape}"
            )
        if not np.isfinite(self.x).all() or not np.isfinite(self.y).all():
            raise ValueError(f"{self.name}: x and y must be finite; filter before constructing")
        if len(self.y) < 2:
            raise ValueError(f"{self.name}: needs at least 2 candidates, got {len(self.y)}")

    @property
    def n(self) -> int:
        return len(self.y)

    @property
    def dim(self) -> int:
        return self.x.shape[1]

    def n_top(self, frac: float) -> int:
        return int(math.ceil(self.n * frac))

    def fingerprint(self) -> str:
        """Twelve hex characters identifying this pool's ``x`` and ``y``, in row order.

        A trace is a list of row indices, so it decodes only against the pool it
        was run on. Every saved trace carries this, and every reader checks it.
        """
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.x, dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(self.y, dtype=np.float64).tobytes())
        return h.hexdigest()[:12]

    def top_mask(self, frac: float) -> np.ndarray:
        """1 for the best ``ceil(n * frac)`` rows, 0 elsewhere.

        Ties at the cut are broken by argsort order, as in Liang et al.'s code.
        """
        mask = np.zeros(self.n, dtype=int)
        mask[np.argsort(-self.y)[: self.n_top(frac)]] = 1
        return mask

    def is_discriminative(self, frac: float) -> bool:
        """False when the objective is too flat for a top set to mean anything."""
        return float(np.std(self.y)) > 1e-6 and len(np.unique(self.y)) > self.n_top(frac)
