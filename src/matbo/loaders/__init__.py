#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""The campaign registry: every campaign by name, and the benchmark family it belongs to."""

from __future__ import annotations

from collections.abc import Callable

from matbo.campaign import Campaign
from matbo.loaders import liang as _liang
from matbo.loaders import olympus as _olympus

# name -> zero-argument loader. Each entry reads disk only when called.
CAMPAIGNS: dict[str, Callable[[], Campaign]] = {}
FAMILY_OF: dict[str, str] = {}


def _register(name: str, family: str, loader: Callable[[], Campaign]):
    if name in CAMPAIGNS:
        raise ValueError(f"duplicate campaign name {name!r}")
    CAMPAIGNS[name] = loader
    FAMILY_OF[name] = family


for _stem, _max in _liang.LIANG:
    _register(_stem, "liang", lambda s=_stem, m=_max: _liang.load(s, m))
for _stem in _olympus.OLYMPUS:
    _register(_stem, "olympus", lambda s=_stem: _olympus.load(s))


def load(name: str) -> Campaign:
    if name not in CAMPAIGNS:
        raise KeyError(f"unknown campaign {name!r}; known: {', '.join(sorted(CAMPAIGNS))}")
    return CAMPAIGNS[name]()


def names(families: list[str] | None = None, only: list[str] | None = None) -> list[str]:
    """Campaign names for the requested families, optionally filtered by ``only``.

    Each key in ``only`` matches case-insensitively. A key that names a campaign
    exactly selects only that campaign; any other key is a substring match, so
    ``ONLY=photo`` selects both photo campaigns.
    """
    out = [n for n in CAMPAIGNS if not families or FAMILY_OF[n] in families]
    if only:
        keys = [k.strip().lower() for k in only]
        exact = {n for n in out if n.lower() in keys}
        loose = {
            n for n in out if any(k in n.lower() for k in keys if k not in {m.lower() for m in out})
        }
        out = [n for n in out if n in exact | loose]
    return out


__all__ = ["CAMPAIGNS", "FAMILY_OF", "Campaign", "load", "names"]
