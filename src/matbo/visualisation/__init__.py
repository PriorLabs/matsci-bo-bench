# Copyright (c) Prior Labs GmbH 2026. Licensed under the Apache License, Version 2.0.
"""Render the paper's twelve data figures: ``python -m matbo.visualisation``."""

from functools import partial

from .campaigns import liang, olympus
from .figures import all_arms, budget, ranking
from .kernels import kernel_variation
from .posthoc import best_rule, paired

FIGURES = {
    "1": budget,
    "2": partial(kernel_variation, "liang"),
    "D.1": liang,
    "D.2": olympus,
    "D.3": ranking,
    "D.4": partial(all_arms, "liang"),
    "D.5": partial(all_arms, "olympus"),
    "D.6": partial(budget, "topcount"),
    "D.7": partial(budget, "best-value"),
    "D.8": partial(kernel_variation, "olympus"),
    "D.9": paired,
    "D.10": best_rule,
}

__all__ = ["FIGURES"]
