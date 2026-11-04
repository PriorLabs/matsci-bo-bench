#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""Saved runs: curves, traces and the protocol that produced them.

A curve's first ``n_initial`` columns are the initial design. Callers index
curves through :class:`Curves`, never by raw column.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from matbo.protocol import Protocol


@dataclass(frozen=True)
class Curves:
    """Curves for one method on one campaign: (n_seeds, n_columns).

    The values are the run's ``protocol.metric``. The column layout is the
    same for every metric.
    """

    values: np.ndarray
    n_initial: int

    def __post_init__(self):
        if self.values.ndim != 2:
            raise ValueError(f"curves must be 2-D (seeds, rounds), got {self.values.shape}")
        if not 1 <= self.n_initial <= self.values.shape[1]:
            raise ValueError(
                f"n_initial must be between 1 and the {self.values.shape[1]} curve columns, "
                f"got {self.n_initial}"
            )

    @property
    def n_seeds(self) -> int:
        return self.values.shape[0]

    @property
    def n_columns(self) -> int:
        return self.values.shape[1]

    @property
    def budget(self) -> int:
        """Acquisitions represented, excluding the initial design."""
        return self.n_columns - self.n_initial

    def at_budget(self, b: int) -> np.ndarray:
        """Per-seed value after ``b`` acquisitions past the initial design."""
        if not 0 <= b <= self.budget:
            raise IndexError(
                f"budget {b} outside 0..{self.budget} for a curve of "
                f"{self.n_columns} columns with n_initial={self.n_initial}"
            )
        return self.values[:, self.n_initial + b - 1]

    def at_experiments(self, k: int) -> np.ndarray:
        """Per-seed value after ``k`` experiments, initial design included.

        Hase et al.'s checkpoints count evaluations from 1, so this is the
        accessor for them, not :meth:`at_budget`.
        """
        if not 1 <= k <= self.n_columns:
            raise IndexError(
                f"{k} experiments outside 1..{self.n_columns} for this curve"
            )
        return self.values[:, k - 1]

    def final(self) -> np.ndarray:
        return self.values[:, -1]

    def x_axis(self) -> np.ndarray:
        """Experiments performed at each column, 1-based, initial design included."""
        return np.arange(1, self.n_columns + 1)

    def mean(self) -> np.ndarray:
        return self.values.mean(axis=0)

    def stderr(self) -> np.ndarray:
        return self.values.std(axis=0) / np.sqrt(self.n_seeds)


class IncompatibleProtocol(RuntimeError):
    """A saved run cannot be merged with this one without changing meaning."""


class IncompatiblePool(IncompatibleProtocol):
    """Runs refer to different candidate pools under the same campaign name."""


class RunFile:
    """A run's curves and traces, plus the protocol that produced them.

    Saved as one ``.npz``. Keys are ``<campaign>|<method>`` for curves,
    ``<campaign>|<method>|__trace`` for traces, and ``<campaign>|__<field>``
    for per-campaign metadata.
    """

    KEY_PROTOCOL = "__protocol"
    # Pool indices in measurement order, one row per seed.
    KEY_TRACE = "__trace"
    # The pool the traces index; see Campaign.fingerprint.
    KEY_FINGERPRINT = "__fingerprint"
    KEY_NTOP = "__n_top"
    KEY_NPOOL = "__n_pool"

    def __init__(self, protocol: Protocol):
        self.protocol = protocol
        self.curves: dict[tuple[str, str], Curves] = {}
        self.traces: dict[tuple[str, str], np.ndarray] = {}
        self.fingerprint: dict[str, str] = {}
        self.n_top: dict[str, int] = {}
        self.n_pool: dict[str, int] = {}

    def add(self, campaign: str, method: str, curves: Curves, n_top: int,
            n_pool: int, traces: np.ndarray, fingerprint: str):
        self.validate_campaign(campaign, n_top, n_pool, fingerprint)
        self.curves[(campaign, method)] = curves
        self.traces[(campaign, method)] = np.asarray(traces, dtype=np.int32)
        self.n_top[campaign] = n_top
        self.n_pool[campaign] = n_pool
        self.fingerprint[campaign] = fingerprint

    def validate_campaign(self, campaign: str, n_top: int, n_pool: int,
                          fingerprint: str) -> None:
        """Refuse to attach one campaign name to two different pools."""
        conflicts = []
        current_fp = self.fingerprint.get(campaign)
        if current_fp is not None and current_fp != fingerprint:
            conflicts.append(f"fingerprint {current_fp} != {fingerprint}")
        current_n = self.n_pool.get(campaign)
        if current_n is not None and current_n != n_pool:
            conflicts.append(f"pool size {current_n} != {n_pool}")
        current_top = self.n_top.get(campaign)
        if current_top is not None and current_top != n_top:
            conflicts.append(f"n_top {current_top} != {n_top}")
        if conflicts:
            raise IncompatiblePool(
                f"campaign {campaign!r} refers to incompatible pools: "
                + "; ".join(conflicts)
                + ". Use a different TAG or discard the superseded run; its trace "
                  "indices cannot be merged safely."
            )

    @property
    def campaigns(self) -> list[str]:
        return sorted({c for c, _ in self.curves})

    @property
    def methods(self) -> list[str]:
        return sorted({m for _, m in self.curves})

    def for_campaign(self, campaign: str) -> dict[str, Curves]:
        return {m: c for (ds, m), c in self.curves.items() if ds == campaign}

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {f"{ds}|{m}": c.values for (ds, m), c in self.curves.items()}
        payload |= {f"{ds}|{m}|{self.KEY_TRACE}": t
                    for (ds, m), t in self.traces.items()}
        payload |= {f"{ds}|{self.KEY_FINGERPRINT}": np.array(fp)
                    for ds, fp in self.fingerprint.items()}
        payload |= {f"{ds}|{self.KEY_NTOP}": np.array(n) for ds, n in self.n_top.items()}
        payload |= {f"{ds}|{self.KEY_NPOOL}": np.array(n) for ds, n in self.n_pool.items()}
        payload[self.KEY_PROTOCOL] = np.array(self.protocol.to_json())
        np.savez(path, **payload)

    @classmethod
    def load(cls, path: str) -> RunFile:
        raw = np.load(path, allow_pickle=False)
        out = cls(Protocol.from_json(str(raw[cls.KEY_PROTOCOL])))
        for key in raw.files:
            if key == cls.KEY_PROTOCOL:
                continue
            ds, _, method = key.partition("|")
            if method == cls.KEY_NTOP:
                out.n_top[ds] = int(raw[key])
            elif method == cls.KEY_NPOOL:
                out.n_pool[ds] = int(raw[key])
            elif method == cls.KEY_FINGERPRINT:
                out.fingerprint[ds] = str(raw[key])
            elif method.endswith(f"|{cls.KEY_TRACE}"):
                out.traces[(ds, method.removesuffix(f"|{cls.KEY_TRACE}"))] = raw[key]
            else:
                out.curves[(ds, method)] = Curves(raw[key], out.protocol.n_initial)
        return out

    def merge_from(self, path: str, log: Callable[[str], None] = print) -> list[str]:
        """Fold a saved run's arms in; arms already in this run win.

        Returns the keys carried over. Raises :class:`IncompatibleProtocol`
        before changing anything if the saved run's protocol or pools differ.
        """
        if not os.path.exists(path):
            return []
        prior = RunFile.load(path)
        clashes = self.protocol.incompatible_with(prior.protocol)
        if clashes:
            raise IncompatibleProtocol(
                f"{os.path.basename(path)} was recorded with a different "
                f"{', '.join(clashes)}; its curves are not comparable to this run's. "
                f"Use a different TAG, or re-run with the saved settings: "
                f"{prior.protocol.to_json()}"
            )
        for (ds, method) in prior.curves:
            if (ds, method) not in self.curves:
                self.validate_campaign(
                    ds, prior.n_top[ds], prior.n_pool[ds], prior.fingerprint[ds]
                )
        kept = []
        for (ds, method), curves in prior.curves.items():
            if (ds, method) in self.curves:
                continue
            self.add(ds, method, curves, prior.n_top[ds], prior.n_pool[ds],
                     traces=prior.traces[(ds, method)],
                     fingerprint=prior.fingerprint[ds])
            kept.append(f"{ds}|{method}")
        if kept:
            log(f"  merged {len(kept)} curve set(s) from {os.path.basename(path)}")
        return kept

    def manifest(self) -> dict:
        """What this run was, for the config.json beside curves.npz."""
        return {
            "protocol": json.loads(self.protocol.to_json()),
            "campaigns": self.campaigns,
            "methods": self.methods,
            "n_top": self.n_top,
            "n_pool": self.n_pool,
            "pool_fingerprint": self.fingerprint,
        }
