#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""Print one arm's mean curve on one campaign, decoded from its committed traces.

    python scripts/read_reference.py liang GP-ARD-EI Perovskite
    python scripts/read_reference.py olympus RF-EI snar --metric topcount

Exits if the loader no longer builds the pool the traces were recorded on.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace

import numpy as np

from matbo import loaders as L
from matbo.loaders.paths import REFERENCE_DIR
from matbo.protocol import Protocol
from matbo.runner import curve_from_trace


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("benchmark")
    ap.add_argument("arm")
    ap.add_argument("campaign")
    ap.add_argument(
        "--metric",
        default=None,
        help="override the folder protocol's metric (topcount, best-value)",
    )
    ap.add_argument("--seed", type=int, default=None, help="one seed instead of the mean")
    args = ap.parse_args()

    folder = REFERENCE_DIR / args.benchmark
    path = folder / f"{args.arm}__{args.campaign.replace(' ', '_')}.trace.csv"
    if not path.exists():
        sys.exit(
            f"no such record: {path}\n"
            f"  available arms on {args.campaign}: "
            + ", ".join(
                sorted(
                    p.name.split("__")[0]
                    for p in folder.glob(f"*__{args.campaign.replace(' ', '_')}.trace.csv")
                )
            )
        )

    protocol = Protocol.from_json(
        json.dumps(json.loads((folder / "protocol.json").read_text())["protocol"])
    )
    if args.metric:
        protocol = replace(protocol, metric=args.metric)

    campaign = L.load(args.campaign)
    curves, seeds = [], []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row["pool_fingerprint"] != campaign.fingerprint():
                sys.exit(
                    f"REFUSED: trace recorded against pool {row['pool_fingerprint']}, "
                    f"loader now builds {campaign.fingerprint()}"
                )
            if args.seed is not None and int(row["seed"]) != args.seed:
                continue
            idx = sorted(
                (
                    k
                    for k in row
                    if k.startswith("s") and k[1:].isdigit() and row[k] not in ("", None)
                ),
                key=lambda k: int(k[1:]),
            )
            trace = np.array([int(row[k]) for k in idx])
            curves.append(curve_from_trace(trace, campaign, protocol))
            seeds.append(int(row["seed"]))

    if not curves:
        sys.exit(f"no rows matched (seed={args.seed})")
    arr = np.array(curves)
    print(
        f"{args.benchmark}/{args.arm} on {args.campaign}: "
        f"{len(seeds)} seed(s) x {arr.shape[1]} evaluations, metric={protocol.metric}"
    )
    mean = arr.mean(axis=0)
    for n in (8, 16, 50, 100, arr.shape[1] - 1):
        if 0 < n < arr.shape[1]:
            print(f"  @{n:<5d} {mean[n]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
