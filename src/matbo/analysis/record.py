#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""Decode the traces in `reference_results/<benchmark>/` into metric curves.

A trace holds pool row indices, so it decodes only against the pool whose
fingerprint it carries. Rows come back in the protocol's seed order, not in
sorted seed order: paired and resampled statistics depend on it.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path

import numpy as np

from matbo import loaders as L
from matbo.campaign import Campaign
from matbo.protocol import Protocol
from matbo.runner import curve_from_trace

REFERENCE = Path(__file__).resolve().parents[3] / "reference_results"


@lru_cache(maxsize=None)
def _load_campaign(name: str) -> Campaign:
    return L.load(name)


@dataclass(frozen=True)
class Record:
    """One benchmark folder: its protocol, campaigns and traces."""

    benchmark: str
    folder: Path
    protocol: Protocol

    @classmethod
    def load(cls, benchmark: str, root: Path = REFERENCE) -> Record:
        folder = root / benchmark
        saved = json.loads((folder / "protocol.json").read_text())["protocol"]
        return cls(benchmark, folder, Protocol.from_json(json.dumps(saved)))

    @cached_property
    def campaigns(self) -> list[str]:
        """The benchmark's campaigns as the loader defines them."""
        return L.names(families=[self.benchmark])

    def campaign(self, name: str) -> Campaign:
        return _load_campaign(name)

    def path(self, arm: str, campaign: str) -> Path:
        return self.folder / f"{arm}__{campaign.replace(' ', '_')}.trace.csv"

    def rows(self, arm: str, campaign: str) -> list[dict]:
        """Validated trace rows in protocol seed order; each row's `trace` is an index array."""
        path = self.path(arm, campaign)
        pool = self.campaign(campaign)
        with path.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        by_seed: dict[int, dict] = {}
        for row in rows:
            seed = int(row["seed"])
            if seed in by_seed:
                raise ValueError(f"{path.name}: duplicate seed {seed}")
            if row["pool_fingerprint"] != pool.fingerprint():
                raise ValueError(f"{path.name}: trace on pool {row['pool_fingerprint']}, "
                                 f"loader builds {pool.fingerprint()}")
            keys = sorted((k for k in row
                           if k.startswith("s") and k[1:].isdigit() and row[k] not in ("", None)),
                          key=lambda k: int(k[1:]))
            trace = np.array([int(row[k]) for k in keys])
            if trace.min() < 0 or trace.max() >= pool.n:
                raise ValueError(f"{path.name}: seed {seed} indexes outside the pool")
            if len(np.unique(trace)) != len(trace):
                raise ValueError(f"{path.name}: seed {seed} repeats a candidate")
            row["trace"] = trace
            by_seed[seed] = row
        order = list(self.protocol.seeds)
        if set(by_seed) != set(order):
            raise ValueError(f"{path.name}: missing seeds {sorted(set(order) - set(by_seed))[:5]}, "
                             f"undeclared seeds {sorted(set(by_seed) - set(order))[:5]}")
        return [by_seed[s] for s in order]

    def block(self, arm: str, campaign: str) -> np.ndarray:
        """[seed, evaluation] curves of the protocol's metric.

        Top% in percent for `topcount`; the oriented incumbent for `best-value`.
        """
        pool = self.campaign(campaign)
        scale = (100.0 / int(pool.top_mask(self.protocol.top_frac).sum())
                 if self.protocol.metric == "topcount" else 1.0)
        curves = [scale * curve_from_trace(r["trace"], pool, self.protocol)
                  for r in self.rows(arm, campaign)]
        width = min(len(c) for c in curves)
        return np.vstack([c[:width] for c in curves])
