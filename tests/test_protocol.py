#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""Protocols: comparability between runs, and each benchmark declaring its own loop."""

from __future__ import annotations

import dataclasses

import pytest

from conftest import SYNTHETIC
from matbo import Protocol
from matbo import loaders as L
from matbo.benchmarks import BENCHMARKS, Benchmark, _loop, resolve_loop
from matbo.pool import PoolRules


def test_incompatible_protocols_are_detected():
    a = Protocol(scale_budget=True)
    b = Protocol(scale_budget=False)
    assert "scale_budget" in a.incompatible_with(b)
    assert a.incompatible_with(a) == []


def test_device_is_recorded_but_does_not_refuse_a_merge():
    assert Protocol(device="cpu").incompatible_with(Protocol(device="cuda")) == []
    assert "device" in Protocol().to_json()
    assert Protocol(device="cpu", n_initial=2).incompatible_with(
        Protocol(device="cuda", n_initial=5)) == ["n_initial"]


def test_exhaust_overrides_the_step_horizon():
    for n_steps, scale in ((4, True), (20, True), (50, False), (1000, False)):
        p = Protocol(n_initial=2, n_steps=n_steps, scale_budget=scale, exhaust=True)
        assert p.budget(94) == 92
    assert Protocol(n_initial=2, n_steps=4, scale_budget=False).budget(94) == 4


def test_liang_declares_their_loop():
    loop = BENCHMARKS["liang"].loop
    assert loop.exhaust and loop.n_seeds == 50 and loop.metric == "topcount"
    assert loop.active_seeds == Protocol().seeds[:50]


def test_protocol_roundtrips():
    p = Protocol(n_seeds=8, scale_budget=False, seed_slice=(2, 4))
    assert Protocol.from_json(p.to_json()) == p


def test_every_family_is_a_declared_benchmark():
    assert set(L.FAMILY_OF.values()) == set(BENCHMARKS)


def test_campaign_carries_the_rules_its_benchmark_declares():
    for name, family in L.FAMILY_OF.items():
        assert L.load(name).pool == BENCHMARKS[family].pool, name


def test_liang_and_olympus_loops_are_refused_together():
    with pytest.raises(ValueError, match="metric|n_seeds"):
        resolve_loop(["liang", "olympus"])


def test_benchmarks_with_different_loops_are_refused_together():
    assert resolve_loop(["liang"]) == BENCHMARKS["liang"].loop
    odd = Benchmark(
        name="odd", pool=SYNTHETIC,
        loop=dataclasses.replace(BENCHMARKS["liang"].loop, n_steps=99),
        note="test fixture",
    )
    BENCHMARKS["odd"] = odd
    try:
        with pytest.raises(ValueError, match="n_steps"):
            resolve_loop(["liang", "odd"])
    finally:
        del BENCHMARKS["odd"]


def test_benchmark_loop_must_be_stated_in_full():
    with pytest.raises(ValueError, match="must state every field"):
        _loop(n_initial=2, top_frac=0.05)


def test_pool_rules_reject_a_meaningless_combination():
    """`keep` leaves nothing to group, so `grouped` would be a false claim."""
    with pytest.raises(ValueError, match="nothing to group"):
        PoolRules(follows="x", replicates="keep", order="grouped",
                  direction="per-campaign", constraints="unconstrained",
                  not_reproduced=())


def test_env_overrides_the_benchmark_loop_but_does_not_replace_it(monkeypatch):
    base = BENCHMARKS["liang"].loop
    monkeypatch.setenv("N_SEEDS", "4")
    resolved = Protocol.from_env(base=base)
    assert resolved.n_seeds == 4
    assert resolved.n_steps == base.n_steps and resolved.top_frac == base.top_frac
    assert resolved.seeds == base.seeds and resolved.exhaust == base.exhaust


def test_env_leaves_an_unset_loop_untouched(monkeypatch):
    for var in ("SMOKE", "N_INITIAL", "TOP_FRAC", "N_SEEDS", "N_STEPS", "METRIC",
                "SCALE_BUDGET", "EXHAUST", "SEED_SLICE"):
        monkeypatch.delenv(var, raising=False)
    for bench in BENCHMARKS.values():
        assert Protocol.from_env(base=bench.loop).incompatible_with(bench.loop) == []


def test_only_random_initial_designs_are_accepted():
    with pytest.raises(ValueError, match="init_regime"):
        Protocol(init_regime="local")
    with pytest.raises(ValueError, match="init_frac"):
        Protocol(init_frac=0.2)
