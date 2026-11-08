#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""The committed traces still decode against their pools and declared loops.

A trace holds pool row indices, so a loader change can silently remap it; each
trace carries `Campaign.fingerprint()`, which is re-derived here.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from matbo import loaders as L
from matbo.protocol import Protocol
from matbo.runner import curve_from_trace

REFERENCE = Path(__file__).resolve().parent.parent / "reference_results"
FOLDERS = sorted(p.name for p in REFERENCE.iterdir() if p.is_dir()) if REFERENCE.exists() else []


def _columns(row: dict, prefix: str) -> list[str]:
    """`s1..sN` in numeric order, stopping at the row's own end."""
    keys = [
        k
        for k in row
        if k.startswith(prefix) and k[len(prefix) :].isdigit() and row[k] not in ("", None)
    ]
    return sorted(keys, key=lambda k: int(k[len(prefix) :]))


def _arms(folder: Path) -> list[Path]:
    """Every `<arm>__<campaign>.trace.csv` in the folder."""
    return sorted(folder.glob("*.trace.csv"))


@pytest.mark.skipif(not FOLDERS, reason="no reference_results/ in this checkout")
@pytest.mark.parametrize("folder_name", FOLDERS)
def test_reference_traces_decode_against_the_pool_they_index(folder_name):
    """Each trace matches its pool fingerprint and decodes to the protocol's curve width."""
    folder = REFERENCE / folder_name
    meta = folder / "protocol.json"
    assert meta.exists(), f"{folder_name} has no protocol.json"
    protocol = Protocol.from_json(json.dumps(json.loads(meta.read_text())["protocol"]))

    arms = _arms(folder)
    if not arms:
        pytest.skip(f"{folder_name} holds no traces")

    checked = 0
    campaigns: dict[str, object] = {}
    for trace_path in arms:
        with trace_path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                name = row["campaign"]
                if name not in campaigns:
                    campaigns[name] = L.load(name)
                assert row["pool_fingerprint"] == campaigns[name].fingerprint(), (
                    f"{folder_name}/{trace_path.stem} {name}: trace on pool "
                    f"{row['pool_fingerprint']}, loader builds {campaigns[name].fingerprint()}"
                )
                trace = np.array([int(row[k]) for k in _columns(row, "s")])
                curve = curve_from_trace(trace, campaigns[name], protocol)
                assert len(curve) == protocol.curve_width(campaigns[name].n), (
                    f"{folder_name}/{trace_path.stem} {name} seed {row['seed']}: "
                    f"decodes to {len(curve)} evaluations, protocol says "
                    f"{protocol.curve_width(campaigns[name].n)}"
                )
                checked += 1
    assert checked, f"{folder_name}: traced arms present but no rows checked"


@pytest.mark.skipif(not FOLDERS, reason="no reference_results/ in this checkout")
def test_every_result_file_has_a_trace():
    """Every result CSV in the record has a trace beside it."""
    found = set()
    for folder_name in FOLDERS:
        folder = REFERENCE / folder_name
        for path in sorted(folder.glob("*.csv")):
            if path.name.endswith(".trace.csv"):
                continue
            if not (folder / f"{path.stem}.trace.csv").exists():
                found.add(f"{folder_name}/{path.stem}")
    assert not found, f"result files without a trace: {sorted(found)}"


def test_declared_loop_reproduces_the_record_it_is_named_after():
    """Each benchmark declares the loop its committed record was run at."""
    from matbo.benchmarks import BENCHMARKS

    checked = 0
    for name, bench in BENCHMARKS.items():
        blob = REFERENCE / name / "protocol.json"
        if not blob.exists():
            continue
        recorded = Protocol.from_json(json.dumps(json.loads(blob.read_text())["protocol"]))
        clashes = bench.loop.incompatible_with(recorded)
        assert not clashes, (
            f"benchmark {name!r} declares a loop its own record was not run at: "
            f"{clashes}. Declared {bench.loop.to_json()}"
        )
        checked += 1
    assert checked, "no benchmark had a checked-in protocol.json to compare against"


def test_every_arm_in_the_record_resolves_in_the_registry():
    """Every arm name in the record resolves in the method registry."""
    from matbo.methods import build_registry

    registered = set(build_registry().methods)
    found = {path.name.split("__")[0] for path in REFERENCE.glob("*/*.trace.csv")}
    assert found, "no traces found"
    orphaned = sorted(found - registered)
    assert not orphaned, f"arms in the record with no registry entry: {orphaned}"
