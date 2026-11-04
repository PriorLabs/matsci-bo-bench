#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""Run arms on one benchmark and save their curves and traces.

Each run writes ``$MATBO_RESULTS/<TAG>/``: ``curves.npz`` (curves, traces and
protocol), ``config.json`` and ``run.log``.

    FAMILIES=liang python scripts/run_benchmark.py                  # all 56 arms
    FAMILIES=olympus ONLY=snar METHODS=RF-EI,Random python scripts/run_benchmark.py
    FAMILIES=liang SMOKE=1 python scripts/run_benchmark.py          # 2 seeds, 2 campaigns
    FAMILIES=liang METHODS=GP-ARD-EI MERGE=1 TAG=liang python scripts/run_benchmark.py

Environment:
    FAMILIES   benchmark (required): liang or olympus
    METHODS    comma-separated arm names; unset runs all 56
    ONLY       comma-separated campaign names or substrings
    TAG        results subdirectory (default: run, or smoke under SMOKE)
    MERGE      fold this run into the tag's existing curves.npz
    WORKERS    processes for (arm, seed) pairs; results do not depend on it
    SEED_SLICE start:stop window into the seeds, for splitting one arm's ensemble
    N_SEEDS, N_STEPS, EXHAUST, ...  override the benchmark's loop (see Protocol.from_env)
"""

from __future__ import annotations

import json
import os
import sys

from matbo import loaders as L
from matbo.benchmarks import BENCHMARKS, get as get_benchmark, resolve_loop
from matbo.loaders.paths import RESULTS_DIR
from matbo.methods import build_registry
from matbo.protocol import Protocol
from matbo.results import IncompatiblePool, IncompatibleProtocol, RunFile
from matbo.runner import run_suite


def _csv(name: str) -> list[str] | None:
    raw = os.environ.get(name)
    return [v.strip() for v in raw.split(",") if v.strip()] if raw else None


def main() -> int:
    smoke = bool(os.environ.get("SMOKE"))
    tag = os.environ.get("TAG", "smoke" if smoke else "run")

    families = _csv("FAMILIES")
    if not families:
        sys.exit(
            "FAMILIES is required -- there is no default benchmark.\n"
            "  available: " + ", ".join(sorted(BENCHMARKS)) + "\n"
            + "\n".join(f"    {n:16s} {b.note}" for n, b in sorted(BENCHMARKS.items()))
        )
    unknown = [f for f in families if f not in BENCHMARKS]
    if unknown:
        sys.exit(f"unknown benchmark(s) {', '.join(unknown)}; "
                 f"known: {', '.join(sorted(BENCHMARKS))}")

    try:
        protocol = Protocol.from_env(base=resolve_loop(families))
    except ValueError as exc:
        sys.exit(str(exc))

    names = L.names(families=families, only=_csv("ONLY"))
    if smoke:
        names = names[:2]
    if not names:
        sys.exit(f"no campaigns matched families={families} only={_csv('ONLY')}")

    methods = build_registry(device=protocol.device).select(_csv("METHODS"))

    out_dir = RESULTS_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    # Append: MERGE runs several passes into one tag.
    log_file = log_path.open("a")

    def log(msg: str):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    for family in families:
        bench = get_benchmark(family)
        log(f"\nBENCHMARK {bench.name} -- {bench.note}")
        log(f"  {bench.pool.summary()}")
        for skipped in bench.pool.not_reproduced:
            log(f"  NOT reproduced: {skipped}")

    # A campaign without data on disk (Olympus not installed) is skipped and
    # fails the exit code at the end.
    campaigns, missing = [], []
    for name in names:
        try:
            campaigns.append(L.load(name))
        except FileNotFoundError as exc:
            missing.append(name)
            log(f"{name}: SKIPPED -- {exc}")
    if not campaigns:
        log("FAILED: no campaign could be loaded")
        return 2

    curves_path = out_dir / "curves.npz"
    # Merge before the loop, so every checkpoint keeps the earlier passes and an
    # incompatible file is refused before any compute is spent.
    prior = RunFile(protocol)
    if os.environ.get("MERGE"):
        try:
            prior.merge_from(str(curves_path), log=log)
        except IncompatibleProtocol as exc:
            log(f"REFUSED merge, nothing written -- {exc}")
            return 2

        # Check every loaded pool against the merged file before the first
        # campaign starts.
        try:
            for campaign in campaigns:
                prior.validate_campaign(
                    campaign.name,
                    campaign.n_top(protocol.top_frac),
                    campaign.n,
                    campaign.fingerprint(),
                )
        except IncompatiblePool as exc:
            log(f"REFUSED merge, nothing written -- {exc}")
            return 2

    run = run_suite(
        campaigns, methods, protocol, log=log,
        checkpoint=lambda r: r.save(str(curves_path)),
        workers=int(os.environ.get("WORKERS", "1")),
        into=prior,
    )
    run.save(str(curves_path))
    (out_dir / "config.json").write_text(json.dumps(run.manifest(), indent=2))

    _report(run, log)

    log(f"wrote {out_dir}/  (curves.npz, config.json, run.log)")

    if missing:
        log(f"FAILED: {len(missing)} campaign(s) had no data: {', '.join(missing)}")
    log_file.close()
    return 1 if missing else 0


def _report(run: RunFile, log) -> None:
    """One line per (campaign, arm): the mean of the curve's final value over seeds."""
    topcount = run.protocol.metric == "topcount"
    label = "Top% at end" if topcount else "best value at end (oriented)"
    log(f"\n{'campaign':16s} {'method':22s} {label:>28s} {'seeds':>6s}")
    for campaign in run.campaigns:
        for method, cv in sorted(run.for_campaign(campaign).items()):
            final = cv.final().mean()
            value = f"{100 * final / run.n_top[campaign]:.1f}%" if topcount else f"{final:.4g}"
            log(f"{campaign:16s} {method:22s} {value:>28s} {cv.n_seeds:6d}")


if __name__ == "__main__":
    raise SystemExit(main())
