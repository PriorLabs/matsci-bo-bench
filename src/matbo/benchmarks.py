#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""The two benchmarks: how each builds its pools and runs its loop.

Each benchmark states its :class:`~matbo.pool.PoolRules` and every
comparability-relevant field of its :class:`~matbo.protocol.Protocol`, so no
result depends on a dataclass default. :func:`resolve_loop` refuses a run over
benchmarks whose loops disagree.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from matbo.pool import PoolRules
from matbo.protocol import Protocol

# Fields a benchmark must state. `seeds` defaults to Liang et al.'s list,
# `device` does not change the experiment, and `seed_slice` describes how one
# process divides the work, not the benchmark.
_REQUIRED_LOOP = {f.name for f in fields(Protocol)} - {"seeds", "device", "seed_slice"}


def _loop(**kwargs) -> Protocol:
    """A Protocol with every comparability-relevant field stated."""
    missing = _REQUIRED_LOOP - set(kwargs)
    if missing:
        raise ValueError(f"benchmark loop must state every field; missing {sorted(missing)}")
    return Protocol(**kwargs)


# Shared by both benchmarks. The horizon and the ensemble size come from each
# source, so every entry states those itself.
_SAMPLING = dict(
    n_initial=2,
    top_frac=0.05,
    init_regime="random",
    init_frac=None,
    metric="topcount",
)

# Hase et al. run 100 independent campaigns but publish no seeds, so the count
# is theirs and the seed values are ours.
_OLYMPUS_SEEDS = tuple(range(100))

# The evaluation counts Hase et al. compare planners at. All are read off one
# trace; those beyond a pool's size are unreachable.
OLYMPUS_CHECKPOINTS = (1, 3, 10, 30, 100, 300, 1000, 3000, 10000)


@dataclass(frozen=True)
class Benchmark:
    """One benchmark: where its rows come from, how they are pooled, how it runs."""

    name: str
    pool: PoolRules
    loop: Protocol
    # What the benchmark is, and any caveat against the source it is named after.
    note: str
    # Evaluation counts the source reports results at. Reporting positions
    # only, so not part of the Protocol.
    checkpoints: tuple[int, ...] = ()


BENCHMARKS: dict[str, Benchmark] = {}


def _register(bench: Benchmark) -> Benchmark:
    if bench.name in BENCHMARKS:
        raise ValueError(f"duplicate benchmark {bench.name!r}")
    BENCHMARKS[bench.name] = bench
    return bench


# Liang et al.'s pools and loop: 50 ensembles from their seed list, run until
# the pool is exhausted (`for i in np.arange(len(index_learn))` in their code).
# n_steps and scale_budget are inert under exhaust but apply with EXHAUST=0.
LIANG = _register(
    Benchmark(
        name="liang",
        pool=PoolRules(
            follows="Liang et al., npj Comput. Mater. 7, 188 (2021)",
            replicates="unique-mean",
            order="grouped",
            direction="per-campaign",
            constraints="unconstrained",
            not_reproduced=(),
        ),
        loop=_loop(**_SAMPLING, n_seeds=50, n_steps=20, scale_budget=True, exhaust=True),
        note="the reference protocol, reproduced end to end",
    )
)

# Every row Olympus ships, in shipped order, with the config's direction and
# declared constraints. The loop is Hase et al.'s as far as a finite pool
# allows: 100 seeds, best value so far, run to the pool's end in place of their
# 10,000 emulator evaluations.
OLYMPUS = _register(
    Benchmark(
        name="olympus",
        pool=PoolRules(
            follows="Hase et al., Mach. Learn.: Sci. Technol. 2, 035021 -- bundled datasets",
            replicates="keep",
            order="source",
            direction="config-default-goal",
            constraints="declared",
            not_reproduced=(
                "Olympus's emulator protocol. The shipped emulators are TensorFlow 1 "
                "checkpoints, and `olymp` pins tensorflow==1.15, which has no wheel "
                "for Python 3.11 or macOS arm64. Running them would also require "
                "every surrogate to optimise an acquisition over a continuous space "
                "instead of ranking a finite pool, which is a different benchmark.",
                "Olympus's train/validate/test split and cross-validation folds, "
                "which exist to fit that emulator. The whole dataset is the pool here.",
                "Olympus's first two evaluations. Their planners choose from "
                "evaluation 1; here evaluations 1-2 are the shared random initial "
                "design. At checkpoint 1 every arm is therefore identical, and at 3 "
                "only one acquisition separates them.",
                "Olympus's budget of 10,000 evaluations, which exceeds every pool "
                "(66-1386 rows). Checkpoints 3000 and 10000 are unreachable on all "
                "ten campaigns, and 1000 on all but hplc, photo_pce10 and photo_wf3. "
                "Each run measures every candidate, the deepest horizon a pool allows.",
            ),
        ),
        loop=_loop(
            **(_SAMPLING | dict(n_seeds=100, metric="best-value")),
            seeds=_OLYMPUS_SEEDS,
            n_steps=20,
            scale_budget=True,
            exhaust=True,
        ),
        checkpoints=OLYMPUS_CHECKPOINTS,
        note=(
            "Hase et al.'s rows verbatim, under their loop as far as a finite pool "
            "allows: 100 seeds, best value so far at their nine checkpoints, run to "
            "the end of the pool in place of their 10,000 emulator evaluations. "
            "They optimise a continuous space through an emulator; we rank a "
            "finite pool"
        ),
    )
)


def get(name: str) -> Benchmark:
    if name not in BENCHMARKS:
        raise KeyError(f"unknown benchmark {name!r}; known: {', '.join(sorted(BENCHMARKS))}")
    return BENCHMARKS[name]


def resolve_loop(names: list[str]) -> Protocol:
    """The shared loop of ``names``. Raises if any two declare different loops."""
    if not names:
        raise ValueError("no benchmark selected")
    loops = {n: get(n).loop for n in names}
    first, base = next(iter(loops.items()))
    for other, loop in loops.items():
        differing = base.incompatible_with(loop)
        if differing:
            raise ValueError(
                f"benchmarks {first!r} and {other!r} declare different loops "
                f"({', '.join(differing)}); run them separately"
            )
    return base
