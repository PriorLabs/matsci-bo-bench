#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""Pool-based Bayesian optimisation of experimental materials campaigns.

The pieces, in the order data flows through them:

    Benchmark / PoolRules       what a benchmark is, declared in full
    Campaign                    one candidate pool
    Protocol                    the loop, saved with every result
    Surrogate x Acquisition     the arms
    Curves / RunFile            saved results and their column layout
"""

from matbo.benchmarks import BENCHMARKS, Benchmark
from matbo.campaign import Campaign
from matbo.methods.base import (
    Method,
    Posterior,
    Registry,
    ScoreRequest,
    Surrogate,
    Unsupported,
)
from matbo.pool import PoolRules, build_pool, oriented
from matbo.protocol import Protocol
from matbo.results import Curves, RunFile

__all__ = [
    "BENCHMARKS",
    "Benchmark",
    "Campaign",
    "Curves",
    "Method",
    "PoolRules",
    "Posterior",
    "Protocol",
    "Registry",
    "RunFile",
    "ScoreRequest",
    "Surrogate",
    "Unsupported",
    "build_pool",
    "oriented",
]
