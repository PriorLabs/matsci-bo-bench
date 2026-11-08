#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""Merge safety: a refused merge changes nothing, in memory or on disk."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import SYNTHETIC

from matbo import Campaign, Curves, Protocol


def _run_file(scale_budget=True, methods=("A",), n_cols=22):
    from matbo.results import RunFile

    run = RunFile(Protocol(n_seeds=3, scale_budget=scale_budget))
    for m in methods:
        run.add(
            "camp",
            m,
            Curves(np.ones((3, n_cols)), 2),
            n_top=5,
            n_pool=100,
            traces=np.zeros((3, n_cols), dtype=int),
            fingerprint="pool",
        )
    return run


def test_merge_carries_prior_methods(tmp_path):
    path = str(tmp_path / "curves.npz")
    _run_file(methods=("A", "B")).save(path)
    later = _run_file(methods=("C",))
    later.merge_from(path)
    assert sorted(later.methods) == ["A", "B", "C"]


def test_merge_does_not_overwrite_recomputed_methods(tmp_path):
    path = str(tmp_path / "curves.npz")
    _run_file(methods=("A",)).save(path)
    later = _run_file(methods=("A",))
    later.curves[("camp", "A")] = Curves(np.full((3, 22), 9.0), 2)
    later.merge_from(path)
    assert later.curves[("camp", "A")].values[0, 0] == 9.0


def test_incompatible_merge_raises_and_leaves_file_intact(tmp_path):
    from matbo.results import IncompatibleProtocol, RunFile

    path = str(tmp_path / "curves.npz")
    _run_file(scale_budget=False, methods=("A", "B")).save(path)
    later = _run_file(scale_budget=True, methods=("C",))
    with pytest.raises(IncompatibleProtocol, match="scale_budget"):
        later.merge_from(path)
    assert sorted(RunFile.load(path).methods) == ["A", "B"]


def test_merge_refuses_a_different_pool_before_mutating(tmp_path):
    """A trace cannot move under the fingerprint and metadata of another pool."""
    from matbo.results import IncompatiblePool

    path = str(tmp_path / "curves.npz")
    prior = _run_file(methods=("old",))
    prior.n_pool["camp"] = 100
    prior.fingerprint["camp"] = "old-pool"
    prior.save(path)

    current = _run_file(methods=("new",))
    current.n_pool["camp"] = 120
    current.fingerprint["camp"] = "new-pool"
    with pytest.raises(IncompatiblePool, match="incompatible pools"):
        current.merge_from(path, log=lambda _m: None)

    assert current.methods == ["new"]
    assert current.fingerprint["camp"] == "new-pool"
    assert ("camp", "old") not in current.curves


def test_adding_an_arm_refuses_a_pool_loaded_by_an_earlier_merge(tmp_path):
    """The normal MERGE=1 order is load the old file, then compute a new arm."""
    from matbo.results import IncompatiblePool, RunFile

    path = str(tmp_path / "curves.npz")
    prior = _run_file(methods=("old",))
    prior.n_pool["camp"] = 100
    prior.fingerprint["camp"] = "old-pool"
    prior.save(path)

    current = RunFile(prior.protocol)
    current.merge_from(path, log=lambda _m: None)
    with pytest.raises(IncompatiblePool, match="fingerprint"):
        current.add(
            "camp",
            "new",
            Curves(np.ones((3, 22)), 2),
            n_top=5,
            n_pool=100,
            traces=np.zeros((3, 22), dtype=int),
            fingerprint="new-pool",
        )
    assert current.methods == ["old"]


def test_run_suite_preflights_a_merged_pool_before_fitting():
    """run_suite must reject a moved pool before fitting anything."""
    from matbo.methods import build_registry
    from matbo.results import IncompatiblePool, RunFile
    from matbo.runner import run_suite

    x = np.arange(40, dtype=float).reshape(20, 2)
    campaign = Campaign(
        name="camp", family="f", x=x, y=np.arange(20.0), objective="o", pool=SYNTHETIC
    )
    protocol = Protocol(n_seeds=1, n_steps=1, scale_budget=False)
    prior = RunFile(protocol)
    prior.add(
        "camp",
        "old",
        Curves(np.zeros((1, 3)), 2),
        n_top=1,
        n_pool=20,
        traces=np.zeros((1, 3), dtype=int),
        fingerprint="superseded-pool",
    )

    with pytest.raises(IncompatiblePool, match="fingerprint"):
        run_suite(
            [campaign],
            build_registry().select(["Random"]),
            protocol,
            into=prior,
            log=lambda _m: None,
        )
    assert prior.methods == ["old"]


def test_saved_protocol_survives_a_roundtrip(tmp_path):
    from matbo.results import RunFile

    path = str(tmp_path / "curves.npz")
    run = _run_file(scale_budget=False)
    run.save(path)
    assert RunFile.load(path).protocol == run.protocol
