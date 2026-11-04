#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""The paper's 56 arms: eleven surrogates x five acquisition rules, plus Random."""

from __future__ import annotations

from matbo.methods.acquisition import (
    RULES,
    expected_improvement,
    greedy,
    probability_of_improvement,
    ucb,
)
from matbo.methods.base import (
    Method,
    Posterior,
    Registry,
    ScoreRequest,
    Surrogate,
    Unsupported,
)
from matbo.methods.gp import KERNELS, GPSurrogate, MLPKernel, fit_failure_report
from matbo.methods.tabicl import TabICLSurrogate
from matbo.methods.tabpfn import TabPFNSurrogate
from matbo.methods.trees import CatBoostSurrogate, RandomForestSurrogate, RandomSurrogate


def surrogates(device: str = "cpu") -> dict[str, Surrogate]:
    """The eleven surrogates, keyed by the prefix each arm name carries."""
    out: dict[str, Surrogate] = {
        "TabPFN": TabPFNSurrogate(device=device),
        "TabICL": TabICLSurrogate(device=device),
        "CatBoost": CatBoostSurrogate(),
        "RF": RandomForestSurrogate(),
    }
    for kernel in KERNELS:
        out[f"GP-{kernel}"] = GPSurrogate(kernel)
    out["GP-ARD-Warp"] = GPSurrogate("ARD", warp=True)
    return out


def build_registry(device: str = "cpu") -> Registry:
    reg = Registry()
    for prefix, surrogate in surrogates(device).items():
        for rule, acquisition in RULES.items():
            reg.add(Method(name=f"{prefix}-{rule}", surrogate=surrogate, acquisition=acquisition))
    reg.add(Method(name="Random", surrogate=RandomSurrogate(), acquisition=greedy))
    return reg


__all__ = [
    "RULES",
    "CatBoostSurrogate",
    "GPSurrogate",
    "MLPKernel",
    "Method",
    "Posterior",
    "RandomForestSurrogate",
    "RandomSurrogate",
    "Registry",
    "ScoreRequest",
    "Surrogate",
    "TabICLSurrogate",
    "TabPFNSurrogate",
    "Unsupported",
    "build_registry",
    "expected_improvement",
    "fit_failure_report",
    "greedy",
    "probability_of_improvement",
    "surrogates",
    "ucb",
]
