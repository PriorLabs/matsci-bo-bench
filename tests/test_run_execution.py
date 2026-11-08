#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""Running the loop: workers, checkpoints, seed shards, and one end-to-end pass.

WORKERS and SEED_SLICE divide the work and must change nothing else.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from matbo import Protocol
from matbo import loaders as L


def test_parallel_and_sequential_agree(tmp_path):
    """WORKERS must return exactly the sequential curves and traces."""
    from matbo.methods import build_registry
    from matbo.runner import run_suite

    campaign = L.load("Perovskite")
    protocol = Protocol(n_seeds=4, n_steps=4, device="cpu")
    methods = build_registry().select(["GP-iso-EI", "Random"])

    serial = run_suite([campaign], methods, protocol, log=lambda _m: None)
    parallel = run_suite([campaign], methods, protocol, log=lambda _m: None, workers=2)
    for key, curves in serial.curves.items():
        np.testing.assert_array_equal(curves.values, parallel.curves[key].values)
        np.testing.assert_array_equal(serial.traces[key], parallel.traces[key])


def test_checkpoint_during_a_second_pass_keeps_the_first(tmp_path):
    """A pass's checkpoints must contain the arms an earlier pass wrote."""
    from matbo.methods import build_registry
    from matbo.results import RunFile
    from matbo.runner import run_suite

    path = str(tmp_path / "curves.npz")
    campaign = L.load("Perovskite")
    protocol = Protocol(n_seeds=2, n_steps=3, device="cpu")

    first = run_suite(
        [campaign], build_registry().select(["Random"]), protocol, log=lambda _m: None
    )
    first.save(path)

    prior = RunFile(protocol)
    prior.merge_from(path, log=lambda _m: None)
    seen = []
    run_suite(
        [campaign],
        build_registry().select(["GP-iso-EI"]),
        protocol,
        log=lambda _m: None,
        into=prior,
        checkpoint=lambda r: seen.append(sorted(r.methods)),
    )
    assert seen, "checkpoint never fired"
    assert seen[0] == ["GP-iso-EI", "Random"], f"checkpoint dropped the first pass: {seen[0]}"


def test_partial_checkpoints_keep_arms_row_aligned(tmp_path):
    """Mid-campaign saves hold the same seed prefix for every arm."""
    from matbo.methods import build_registry
    from matbo.results import RunFile
    from matbo.runner import run_suite

    path = str(tmp_path / "curves.npz")
    seen = []

    def checkpoint(run):
        run.save(path)
        saved = RunFile.load(path)
        seen.append(tuple(saved.curves[("Perovskite", m)].n_seeds for m in ("GP-iso-EI", "Random")))

    run_suite(
        [L.load("Perovskite")],
        build_registry().select(["GP-iso-EI", "Random"]),
        Protocol(n_seeds=12, n_steps=3, device="cpu"),
        log=lambda _m: None,
        checkpoint=checkpoint,
        workers=2,
        checkpoint_seeds=4,
    )

    assert len(seen) > 1, f"expected mid-campaign saves, got {seen}"
    assert all(a == b for a, b in seen), f"arms fell out of alignment: {seen}"
    assert seen[-1] == (12, 12)


def test_seed_shards_are_the_same_experiment_as_the_whole_ensemble():
    """A shard's rows are the matching rows of the undivided run."""
    from matbo.methods import build_registry
    from matbo.runner import run_suite

    campaign = L.load("Perovskite")
    whole = Protocol(n_seeds=4, n_steps=4, device="cpu")
    methods = build_registry().select(["Random"])

    full = run_suite([campaign], methods, whole, log=lambda _m: None)
    rows = []
    for window in ((0, 2), (2, 4)):
        shard = dataclasses.replace(whole, seed_slice=window)
        assert shard.active_seeds == whole.active_seeds[window[0] : window[1]]
        part = run_suite([campaign], methods, shard, log=lambda _m: None)
        rows.append(part.curves[("Perovskite", "Random")].values)

    np.testing.assert_array_equal(np.vstack(rows), full.curves[("Perovskite", "Random")].values)


def test_a_seed_slice_outside_the_ensemble_is_refused():
    """A window past n_seeds would run fewer seeds than it names."""
    with pytest.raises(ValueError, match="seed_slice"):
        Protocol(n_seeds=4, seed_slice=(2, 9))
    with pytest.raises(ValueError, match="seed_slice"):
        Protocol(n_seeds=4, seed_slice=(3, 3))


def test_smoke_thirty_acquisitions_end_to_end(tmp_path):
    """The loop, a worker pool, the save, the reload and the trace, together."""
    from matbo.methods import build_registry
    from matbo.results import RunFile
    from matbo.runner import curve_from_trace, run_suite

    campaign = L.load("Perovskite")
    protocol = Protocol(n_seeds=3, n_steps=30, scale_budget=False, device="cpu")
    path = str(tmp_path / "curves.npz")

    run_suite(
        [campaign],
        build_registry().select(["Random", "GP-iso-EI"]),
        protocol,
        log=lambda _m: None,
        checkpoint=lambda r: r.save(path),
        workers=2,
    )

    saved = RunFile.load(path)
    for arm in ("Random", "GP-iso-EI"):
        key = ("Perovskite", arm)
        curves = saved.curves[key]
        assert curves.values.shape == (3, protocol.n_initial + 30)
        assert np.all(np.diff(curves.values, axis=1) >= 0), "TopCount fell"
        for row, trace in zip(curves.values, saved.traces[key], strict=True):
            np.testing.assert_array_equal(curve_from_trace(trace, campaign, protocol), row)
    # A surrogate that silently scored as Random would pass everything above.
    assert not np.array_equal(
        saved.traces[("Perovskite", "Random")], saved.traces[("Perovskite", "GP-iso-EI")]
    )
