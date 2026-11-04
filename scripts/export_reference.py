#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""Write a run's traces into ``reference_results/<benchmark>/``.

    BENCHMARK=liang TAG=liang python scripts/export_reference.py

Writes one ``<arm>__<campaign>.trace.csv`` per (arm, campaign): one row per
seed, columns ``s1..sN`` holding the pool row measured at each evaluation.
Every metric is recomputed from these (``runner.curve_from_trace``), so the
curves are not exported. A campaign already in a file is replaced; other
campaigns in it are kept.

The folder's ``protocol.json`` pins one protocol, and a run under a different
one is refused. Arms with fewer seeds than the protocol are skipped unless
``ALLOW_PARTIAL=1``.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

from matbo.benchmarks import BENCHMARKS
from matbo.loaders.paths import REFERENCE_DIR, RESULTS_DIR
from matbo.protocol import Protocol
from matbo.results import RunFile

# The leading columns; the rest are s1..sN. A decoder must check
# `pool_fingerprint` before reading the indices.
TRACE_KEYS = ["run_tag", "benchmark", "campaign", "method", "seed", "pool",
              "pool_fingerprint"]


def _merged(path: Path, rows: list[dict], width: int) -> tuple[list[dict], int]:
    """New rows over the file's existing rows, replacing whole campaigns.

    The file takes its widest row's width; shorter rows are blank past their end.
    """
    if not path.exists():
        return rows, width
    with path.open(newline="") as fh:
        existing = list(csv.DictReader(fh))
    replacing = {r["campaign"] for r in rows}
    kept = [r for r in existing if r["campaign"] not in replacing]
    for row in kept:
        width = max(width, sum(1 for k, v in row.items()
                               if k.startswith("s") and k[1:].isdigit() and v != ""))
    return kept + rows, width


def _write_protocol(folder: Path, run: RunFile, tag: str) -> None:
    """Pin the folder to one protocol; exit if the run's differs."""
    path = folder / "protocol.json"
    if path.exists():
        existing = json.loads(path.read_text())
        differing = run.protocol.incompatible_with(
            Protocol.from_json(json.dumps(existing["protocol"]))
        )
        if differing:
            sys.exit(
                f"REFUSED: {folder} was written under a different protocol "
                f"({', '.join(differing)}); use a separate folder."
            )
        payload = {"protocol": existing["protocol"],
                   "run_tags": sorted(set(existing["run_tags"]) | {tag})}
    else:
        payload = {"protocol": json.loads(run.protocol.to_json()), "run_tags": [tag]}
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> int:
    tag = os.environ.get("TAG")
    benchmark = os.environ.get("BENCHMARK")
    if not tag or benchmark not in BENCHMARKS:
        sys.exit("usage: BENCHMARK=<liang|olympus> TAG=<results subdirectory> "
                 "python scripts/export_reference.py")
    curves_path = RESULTS_DIR / tag / "curves.npz"
    if not curves_path.exists():
        sys.exit(f"no curves at {curves_path}")

    run = RunFile.load(str(curves_path))
    folder = REFERENCE_DIR / benchmark
    folder.mkdir(parents=True, exist_ok=True)
    _write_protocol(folder, run, tag)

    seeds = run.protocol.active_seeds
    allow_partial = bool(os.environ.get("ALLOW_PARTIAL"))
    written, skipped = [], []
    for (campaign, method), traces in sorted(run.traces.items()):
        # The runner checkpoints a seed prefix mid-campaign, so a live tag can
        # hold fewer seeds than the protocol.
        if len(traces) < run.protocol.n_seeds and not allow_partial:
            skipped.append(f"{campaign}|{method} ({len(traces)}/{run.protocol.n_seeds})")
            continue
        rows = []
        for seed, trace in zip(seeds, traces):
            row = {
                "run_tag": tag, "benchmark": benchmark, "campaign": campaign,
                "method": method, "seed": seed, "pool": run.n_pool[campaign],
                "pool_fingerprint": run.fingerprint[campaign],
            }
            row |= {f"s{k + 1}": int(v) for k, v in enumerate(trace)}
            rows.append(row)
        path = folder / f"{method}__{campaign.replace(' ', '_')}.trace.csv"
        rows, width = _merged(path, rows, traces.shape[1])
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=TRACE_KEYS + [f"s{k + 1}" for k in range(width)],
                restval="",
            )
            writer.writeheader()
            writer.writerows(rows)
        written.append(path.name)

    for cell in skipped:
        print(f"  SKIPPED (incomplete): {cell}")
    print(f"wrote {len(written)} trace file(s) to {folder}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
