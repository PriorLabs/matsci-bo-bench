#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""The experimental protocol: resolved once per run, saved with its results, compared on merge."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, replace

# What a saved curve holds. Two runs that disagree on it cannot be merged.
METRICS = {
    # Cumulative count of top-`top_frac` candidates held (Liang et al.).
    "topcount",
    # Best objective value observed so far, in oriented units (Hase et al.).
    "best-value",
}


@dataclass(frozen=True)
class Protocol:
    """Everything that changes what a number means.

    The defaults serve tests and hand-built protocols. A benchmark run takes
    its loop from :mod:`matbo.benchmarks`, which states every field.
    """

    n_initial: int = 2
    top_frac: float = 0.05
    n_seeds: int = 16
    n_steps: int = 20
    metric: str = "topcount"

    # Scale the budget to the pool size, capped at n_steps. Off gives a flat
    # `min(n_steps, n - n_initial)`. Ignored when `exhaust` is on.
    scale_budget: bool = True

    # Run until every candidate is measured. Overrides n_steps and
    # scale_budget.
    exhaust: bool = False

    # Recorded in every saved protocol. This code runs only a uniform random
    # initial design of `n_initial` rows, so these are the only accepted values.
    init_regime: str = "random"
    init_frac: float | None = None

    # Changes speed and float arithmetic, not the experiment; see
    # `incompatible_with`.
    device: str = "cpu"
    # Liang et al.'s published seed list (the first 50 of the 300 their
    # notebook ships).
    seeds: tuple[int, ...] = field(
        default_factory=lambda: (
            4295,
            8508,
            326,
            3135,
            1549,
            2528,
            1274,
            6545,
            5971,
            6269,
            2422,
            4287,
            9320,
            4932,
            951,
            4304,
            1745,
            5956,
            7620,
            4545,
            6003,
            9885,
            5548,
            9477,
            30,
            8992,
            7559,
            5034,
            9071,
            6437,
            3389,
            9816,
            8617,
            3712,
            3626,
            1660,
            3309,
            2427,
            9872,
            938,
            5156,
            7409,
            7672,
            3411,
            3559,
            9966,
            7331,
            8273,
            8484,
            5127,
        )
    )
    # A half-open [start, stop) window into the active seeds, so one arm's
    # ensemble can be split across processes. `n_seeds` stays the full count,
    # so a shard reads as partial until merge_runs.py stacks the shards and
    # writes the protocol back with `seed_slice=None`.
    seed_slice: tuple[int, int] | None = None

    def __post_init__(self):
        if self.init_regime != "random":
            raise ValueError(f"init_regime must be 'random', got {self.init_regime!r}")
        if self.init_frac is not None:
            raise ValueError(f"init_frac must be None, got {self.init_frac}")
        if not 0 < self.top_frac < 1:
            raise ValueError(f"top_frac must be in (0, 1), got {self.top_frac}")
        if self.n_initial < 1:
            raise ValueError(f"n_initial must be at least 1, got {self.n_initial}")
        if self.n_seeds > len(self.seeds):
            raise ValueError(f"n_seeds={self.n_seeds} exceeds {len(self.seeds)} available seeds")
        if self.seed_slice is not None:
            start, stop = self.seed_slice
            if not 0 <= start < stop <= self.n_seeds:
                raise ValueError(
                    f"seed_slice must be 0 <= start < stop <= n_seeds={self.n_seeds}, "
                    f"got {self.seed_slice}"
                )
        if self.metric not in METRICS:
            raise ValueError(f"metric must be one of {sorted(METRICS)}, got {self.metric!r}")

    @property
    def active_seeds(self) -> tuple[int, ...]:
        seeds = self.seeds[: self.n_seeds]
        if self.seed_slice is None:
            return seeds
        start, stop = self.seed_slice
        return seeds[start:stop]

    def budget(self, n: int) -> int:
        """Acquisitions to run on a pool of ``n`` candidates."""
        if self.exhaust:
            return n - self.n_initial
        if self.scale_budget:
            return min(self.n_steps, max(15, n // 4), n - self.n_initial)
        return min(self.n_steps, n - self.n_initial)

    def curve_width(self, n: int) -> int:
        """Columns in a saved curve: the initial design plus the budget."""
        return self.n_initial + self.budget(n)

    @classmethod
    def from_env(cls, base: Protocol) -> Protocol:
        """Apply the environment variables that are set on top of a benchmark's loop.

        ``SMOKE`` defaults to 2 seeds and no exhaustion. ``DEVICE`` is read
        from the environment, never from ``base``.
        """

        def _flag(name: str, default: bool) -> bool:
            return os.environ.get(name, "1" if default else "0") != "0"

        smoke = bool(os.environ.get("SMOKE"))
        return replace(
            base,
            n_initial=int(os.environ.get("N_INITIAL", base.n_initial)),
            top_frac=float(os.environ.get("TOP_FRAC", base.top_frac)),
            n_seeds=int(os.environ.get("N_SEEDS", 2 if smoke else base.n_seeds)),
            n_steps=int(os.environ.get("N_STEPS", base.n_steps)),
            metric=os.environ.get("METRIC", base.metric),
            scale_budget=_flag("SCALE_BUDGET", base.scale_budget),
            exhaust=_flag("EXHAUST", False if smoke else base.exhaust),
            device=os.environ.get("DEVICE") or _default_device(),
            seed_slice=_seed_slice(os.environ.get("SEED_SLICE"), base.seed_slice),
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, blob: str) -> Protocol:
        raw = json.loads(blob)
        # JSON has no tuples, and a list would compare unequal to the tuple.
        raw["seeds"] = tuple(raw["seeds"])
        if raw.get("seed_slice") is not None:
            raw["seed_slice"] = tuple(raw["seed_slice"])
        return cls(**raw)

    def incompatible_with(self, other: Protocol) -> list[str]:
        """Fields that make two runs' curves not comparable.

        ``device`` is excluded. A different device changes float arithmetic and
        can send a campaign down a different path, but the runs remain draws of
        the same experiment. It stays recorded in every saved protocol.
        """
        mine, theirs = asdict(self), asdict(other)
        return sorted(k for k in mine if k != "device" and mine[k] != theirs[k])


def _seed_slice(raw: str | None, base: tuple[int, int] | None) -> tuple[int, int] | None:
    """``SEED_SLICE="10:20"`` -> ``(10, 20)``. Empty or unset keeps ``base``."""
    if not raw:
        return base
    start, sep, stop = raw.partition(":")
    if not sep or not start.strip().isdigit() or not stop.strip().isdigit():
        raise ValueError(f"SEED_SLICE must be 'start:stop' with both bounds, got {raw!r}")
    return int(start), int(stop)


def _default_device() -> str:
    """CUDA when present, else CPU. MPS is never chosen: it is slower at these context sizes."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
