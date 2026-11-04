#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""Which (arm, campaign) cells ``reference_results/`` holds at the full seed count.

    python scripts/coverage.py

Prints one grid per benchmark, the 56 registered arms down and its campaigns
across: ``done`` at the folder protocol's seed count, a number for fewer seeds,
``.`` for none.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from matbo import loaders as L
from matbo.benchmarks import BENCHMARKS
from matbo.loaders.paths import REFERENCE_DIR
from matbo.methods import build_registry


def _held(folder: Path) -> dict[tuple[str, str], int]:
    """(arm, campaign) -> distinct seeds in the folder's trace files."""
    seen: dict[tuple[str, str], set[str]] = {}
    for path in sorted(folder.glob("*.trace.csv")):
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                seen.setdefault((row["method"], row["campaign"]), set()).add(row["seed"])
    return {key: len(seeds) for key, seeds in seen.items()}


def main() -> int:
    arms = list(build_registry().methods)
    missing = 0
    for bench_name, bench in BENCHMARKS.items():
        campaigns = L.names(families=[bench_name])
        folder = REFERENCE_DIR / bench_name
        held = _held(folder) if folder.exists() else {}
        want = bench.loop.n_seeds
        if (folder / "protocol.json").exists():
            want = json.loads((folder / "protocol.json").read_text())["protocol"]["n_seeds"]

        done = sum(1 for a in arms for c in campaigns if held.get((a, c), 0) >= want)
        total = len(arms) * len(campaigns)
        missing += total - done
        print(f"\n{bench_name.upper()}  {len(campaigns)} campaigns x {len(arms)} arms "
              f"= {total} cells, {done} done at {want} seeds")
        width = max(len(a) for a in arms) + 1
        print(f"  {'':{width}s}" + "".join(f"{c[:9]:>10s}" for c in campaigns))
        for arm in arms:
            cells = ""
            for c in campaigns:
                n = held.get((arm, c), 0)
                cells += f"{'.' if not n else (str(n) if n < want else 'done'):>10s}"
            print(f"  {arm:{width}s}{cells}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
