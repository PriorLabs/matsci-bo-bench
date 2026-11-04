#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""Where the datasets and run artefacts live."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

# src/matbo/loaders/paths.py -> repo root
_REPO = Path(__file__).resolve().parents[3]

DATA_ROOT = Path(os.environ.get("MATBO_DATA", _REPO / "data"))
LIANG_DIR = Path(os.environ.get("MATBENCH_DIR", DATA_ROOT / "liang_2021"))
RESULTS_DIR = Path(os.environ.get("MATBO_RESULTS", _REPO / "results"))
REFERENCE_DIR = _REPO / "reference_results"


def olympus_dir() -> Path:
    """Olympus's bundled datasets, found without importing the package.

    The ``olymp`` distribution fails to import on matplotlib >= 3.9, but its
    datasets are plain csv and json. ``OLYMPUS_DIR`` overrides the lookup.
    """
    override = os.environ.get("OLYMPUS_DIR")
    if override:
        return Path(override)
    spec = importlib.util.find_spec("olympus")
    if not spec or not spec.origin:
        return Path("")
    return Path(spec.origin).parent / "datasets"
