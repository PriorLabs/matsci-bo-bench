#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""Join several runs' ``curves.npz`` into one, across arms and across seed shards.

    OUT=results/liang/curves.npz python scripts/merge_runs.py results/liang_*/curves.npz

Disjoint (campaign, arm) pairs are unioned. Shards of one arm's ensemble
(``SEED_SLICE``) are stacked back in seed order, so row *i* is
``protocol.seeds[i]`` again; shards must start at 0 and chain without gaps.
The merged file is written with ``seed_slice=None``.

Use this when parallel processes wrote separate tags. Passes run one after
another into one tag need only ``MERGE=1``.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from dataclasses import replace

import numpy as np

from matbo.results import Curves, RunFile


def _load(paths: list[str]) -> list[tuple[str, RunFile]]:
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        sys.exit(f"REFUSED: {len(missing)} source(s) do not exist, first: {missing[0]}")
    return [(p, RunFile.load(p)) for p in paths]


def _check_protocols(runs) -> None:
    """Every source must agree on the loop, seed window aside."""
    base_path, base = runs[0]
    for path, run in runs[1:]:
        clashes = [k for k in base.protocol.incompatible_with(run.protocol)
                   if k != "seed_slice"]
        if clashes:
            sys.exit(
                f"REFUSED, nothing written -- {path} differs from {base_path} in "
                f"{', '.join(clashes)}; its curves are not comparable to theirs."
            )


def _check_pools(runs) -> None:
    """One pool per campaign across the sources."""
    seen: dict[str, tuple[str, str]] = {}
    for path, run in runs:
        for campaign, fp in run.fingerprint.items():
            if campaign in seen and seen[campaign][0] != fp:
                sys.exit(
                    f"REFUSED, nothing written -- {campaign} has fingerprint {fp} in "
                    f"{path} and {seen[campaign][0]} in {seen[campaign][1]}: "
                    "these are different pools."
                )
            seen.setdefault(campaign, (fp, path))


def _stack(key, shards, n_seeds: int):
    """Shards of one (campaign, arm) -> one curve set and trace, or exit.

    ``shards`` is a list of ``(seed_slice, Curves, trace, path)``.
    """
    campaign, method = key
    if len(shards) == 1 and shards[0][0] is None:
        _, curves, trace, _ = shards[0]
        return curves, trace, None

    unsharded = [p for sl, _, _, p in shards if sl is None]
    if unsharded:
        sys.exit(f"REFUSED -- {campaign}|{method} has both a whole-ensemble run "
                 f"({unsharded[0]}) and {len(shards) - len(unsharded)} shard(s); "
                 "one of them is the run you want, not both.")

    ordered = sorted(shards, key=lambda s: s[0][0])
    cursor = 0
    for (start, stop), curves, _, path in ordered:
        if start != cursor:
            sys.exit(
                f"REFUSED -- {campaign}|{method}: seeds [{cursor}, {start}) are "
                f"missing before {path}. Stacking across a gap would put row i on "
                "the wrong seed; re-run the missing shard."
            )
        if curves.n_seeds != stop - start:
            sys.exit(
                f"REFUSED -- {campaign}|{method}: {path} holds {curves.n_seeds} row(s) "
                f"for window [{start}, {stop}); it is still running."
            )
        cursor = stop

    widths = {c.n_columns for _, c, _, _ in ordered}
    if len(widths) > 1:
        sys.exit(f"REFUSED -- {campaign}|{method}: shards have {sorted(widths)} columns; "
                 "they were run to different budgets.")

    values = np.vstack([c.values for _, c, _, _ in ordered])
    stacked_trace = np.vstack([t for _, _, t, _ in ordered])
    short = f"{cursor}/{n_seeds} seeds" if cursor < n_seeds else None
    return Curves(values, ordered[0][1].n_initial), stacked_trace, short


def main() -> int:
    paths = sys.argv[1:]
    out = os.environ.get("OUT")
    if not out or not paths:
        sys.exit("usage: OUT=<dest curves.npz> python scripts/merge_runs.py "
                 "<curves.npz> [<curves.npz> ...]")

    runs = _load(paths)
    _check_protocols(runs)
    _check_pools(runs)

    protocol = runs[0][1].protocol
    merged = RunFile(replace(protocol, seed_slice=None))

    groups: dict[tuple[str, str], list] = defaultdict(list)
    n_top: dict[str, int] = {}
    n_pool: dict[str, int] = {}
    fingerprint: dict[str, str] = {}
    for path, run in runs:
        for key, curves in run.curves.items():
            groups[key].append((run.protocol.seed_slice, curves,
                                run.traces[key], path))
        n_top |= run.n_top
        n_pool |= run.n_pool
        fingerprint |= run.fingerprint

    sharded, partial = 0, []
    for key in sorted(groups):
        campaign, method = key
        curves, trace, short = _stack(key, groups[key], protocol.n_seeds)
        if len(groups[key]) > 1:
            sharded += 1
        if short:
            partial.append(f"{campaign}|{method} ({short})")
        merged.add(campaign, method, curves, n_top[campaign], n_pool[campaign],
                   traces=trace, fingerprint=fingerprint[campaign])

    merged.save(out)
    print(f"{out}: {len(merged.curves)} curve set(s) from {len(paths)} file(s), "
          f"{len(merged.campaigns)} campaign(s), {len(merged.methods)} arm(s)")
    if sharded:
        print(f"  {sharded} curve set(s) stitched from seed shards")
    if partial:
        # export_reference.py refuses these without ALLOW_PARTIAL.
        print(f"  PARTIAL ({len(partial)}): " + ", ".join(partial[:6])
              + (" ..." if len(partial) > 6 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
