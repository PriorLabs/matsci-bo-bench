#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""Acquisition traces: the record every curve is recomputed from."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from matbo import Campaign, Protocol
from matbo import loaders as L
from conftest import SYNTHETIC


@pytest.mark.parametrize("metric", ["topcount", "best-value"])
def test_trace_reproduces_the_curve_the_loop_returned(metric):
    """Every metric must be recomputable from the trace."""
    from matbo.methods import build_registry
    from matbo.runner import curve_from_trace, run_campaign

    campaign = L.load("snar")
    protocol = Protocol(n_seeds=1, n_steps=8, scale_budget=False, metric=metric)
    method = build_registry().methods["Random"]
    init = np.array([0, 1])

    result = run_campaign(method, campaign, init, seed=3, protocol=protocol)
    np.testing.assert_array_equal(
        curve_from_trace(result.trace, campaign, protocol), result.curve
    )
    # The trace is what was measured: no repeats, initial design first.
    assert len(set(result.trace.tolist())) == len(result.trace)
    np.testing.assert_array_equal(result.trace[: protocol.n_initial], init)
    assert len(result.trace) == protocol.n_initial + protocol.budget(campaign.n)


def test_a_trace_scores_a_metric_its_run_never_computed():
    """A best-value run's trace gives the TopCount curve a TopCount run would."""
    from matbo.methods import build_registry
    from matbo.runner import curve_from_trace, run_campaign

    campaign = L.load("snar")
    as_run = Protocol(n_seeds=1, n_steps=8, scale_budget=False, metric="best-value")
    result = run_campaign(build_registry().methods["Random"], campaign,
                          np.array([0, 1]), seed=5, protocol=as_run)

    rescored = dataclasses.replace(as_run, metric="topcount")
    recomputed = curve_from_trace(result.trace, campaign, rescored)
    direct = run_campaign(build_registry().methods["Random"], campaign,
                          np.array([0, 1]), seed=5, protocol=rescored).curve
    np.testing.assert_array_equal(recomputed, direct)


def test_pool_fingerprint_changes_when_the_pool_does():
    """The fingerprint must move for any change that renumbers or edits rows."""
    x = np.arange(12, dtype=float).reshape(6, 2)
    y = np.array([1.0, 5.0, 2.0, 4.0, 3.0, 0.0])
    base = Campaign(name="c", family="f", x=x, y=y, objective="o", pool=SYNTHETIC)

    assert base.fingerprint() == Campaign(
        name="other-name", family="f", x=x.copy(), y=y.copy(), objective="o",
        pool=SYNTHETIC).fingerprint(), "the name is not the pool"

    order = np.array([1, 0, 2, 3, 4, 5])
    reordered = Campaign(name="c", family="f", x=x[order], y=y[order],
                         objective="o", pool=SYNTHETIC)
    assert reordered.fingerprint() != base.fingerprint()

    dropped = Campaign(name="c", family="f", x=x[:5], y=y[:5], objective="o",
                       pool=SYNTHETIC)
    assert dropped.fingerprint() != base.fingerprint()

    nudged_y = y.copy()
    nudged_y[3] += 1e-9
    assert Campaign(name="c", family="f", x=x, y=nudged_y, objective="o",
                    pool=SYNTHETIC).fingerprint() != base.fingerprint()


def test_traces_survive_save_load_and_merge(tmp_path):
    """Traces and fingerprints survive a save, a load and a merge."""
    from matbo.methods import build_registry
    from matbo.results import RunFile
    from matbo.runner import run_suite

    path = str(tmp_path / "curves.npz")
    protocol = Protocol(n_seeds=2, n_steps=3, device="cpu")
    campaign = L.load("snar")
    run = run_suite([campaign], build_registry().select(["Random"]), protocol,
                    log=lambda _m: None)
    run.save(path)

    loaded = RunFile.load(path)
    key = ("snar", "Random")
    np.testing.assert_array_equal(loaded.traces[key], run.traces[key])
    assert loaded.traces[key].shape == (2, protocol.n_initial + 3)
    assert loaded.fingerprint["snar"] == campaign.fingerprint()

    # A second pass merges the first one's arm, trace and all.
    second = RunFile(protocol)
    second.merge_from(path, log=lambda _m: None)
    np.testing.assert_array_equal(second.traces[key], run.traces[key])
    assert second.fingerprint["snar"] == campaign.fingerprint()
