#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""Check every campaign against its source, and a run's Random arm against chance.

    python scripts/check_provenance.py [results/<tag>/curves.npz]

1. pools    -- each loaded pool's size and feature count against
               :data:`matbo.provenance.PAPER_POOL`. Exits non-zero on any
               difference.
2. baseline -- for a TopCount run, the Random arm against Liang et al.'s
               analytic random baseline, E[TopCount after n draws] = n * n_top / N.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from matbo import loaders as L
from matbo import provenance
from matbo.results import RunFile


def check_pools() -> int:
    """Loaded pool against the size the source paper reports."""
    print("=" * 78)
    print("POOLS  (loaded vs published)\n")
    print(f"{'campaign':18s}{'ours':>7s}{'paper':>8s}{'raw':>7s}{'dim':>5s}  note")
    unexplained = 0
    for family in ("liang", "olympus"):
        print(f"-- {family} --")
        for name in L.names(families=[family]):
            try:
                campaign = L.load(name)
            except FileNotFoundError as exc:
                print(f"{name:18s}{'-':>7s}{'-':>8s}{'-':>7s}{'-':>5s}  "
                      f"unavailable: {str(exc).splitlines()[0][:52]}")
                continue
            published = provenance.PAPER_POOL.get(name)
            raw = campaign._meta.get("raw_rows")
            notes = provenance.describe(campaign)
            if provenance.unexplained(campaign):
                unexplained += 1
            print(
                f"{name:18s}{campaign.n:7d}"
                f"{published.rows if published else '-':>8}"
                f"{raw if raw else '-':>7}{campaign.dim:5d}  "
                + ("; ".join(notes)[:60] if notes else "")
            )
    return unexplained


def check_random_baseline(run: RunFile) -> list[float]:
    """The Random arm against the hypergeometric mean n * n_top / N.

    Checkpoints within a campaign are correlated, so only the mean z is judged.
    """
    print("\n" + "=" * 78)
    print("RANDOM BASELINE  (simulated vs analytic n * n_top / N)\n")
    print(f"{'campaign':18s}{'n_meas':>7s}{'ours':>8s}{'analytic':>10s}{'z':>7s}")
    zs = []
    for name in run.campaigns:
        curves = run.for_campaign(name).get("Random")
        if curves is None:
            continue
        n_pool, n_top = run.n_pool[name], run.n_top[name]
        width = curves.n_columns
        for column in sorted({width // 4, width // 2, width - 1}):
            n = column + 1  # column j holds the count after j+1 measurements
            if n >= n_pool:
                continue  # an exhausted pool holds every top row; variance is 0
            p = n_top / n_pool
            mu = n * p
            var = n * p * (1 - p) * (n_pool - n) / (n_pool - 1)
            observed = curves.values[:, column].mean()
            z = (observed - mu) / np.sqrt(var / curves.n_seeds)
            zs.append(z)
            print(f"{name:18s}{n:7d}{observed:8.2f}{mu:10.2f}{z:7.2f}")
    if zs:
        verdict = "ok" if abs(np.mean(zs)) < 1.5 else "INVESTIGATE"
        print(f"\nmean z over {len(zs)} checkpoints: {np.mean(zs):+.2f} ({verdict}; "
              "|mean z| < 1.5 is consistent with the analytic baseline)")
    return zs


def main() -> int:
    unexplained = check_pools()
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if path and path.exists():
        print(f"\nsaved run: {path}")
        run = RunFile.load(str(path))
        if run.protocol.metric == "topcount":
            check_random_baseline(run)
        else:
            print(f"baseline check skipped: it needs TopCount curves, "
                  f"this run is {run.protocol.metric}")
    elif path:
        print(f"\nno run at {path}")
    else:
        print("\npass a curves.npz to also run the baseline check")
    if unexplained:
        print(f"\nFAILED: {unexplained} pool(s) differ from the source")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
