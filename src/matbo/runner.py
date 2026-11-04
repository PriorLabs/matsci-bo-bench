#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""The acquisition loop: pool-based active learning over a finite library.

Each round scores every unmeasured row and measures the best one. Ties are
broken uniformly on a separate random stream, never by pool order (see
:func:`run_campaign`).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np

from matbo import provenance
from matbo.campaign import Campaign
from matbo.methods.base import Method, ScoreRequest
from matbo.methods.gp import fit_failure_report
from matbo.protocol import Protocol
from matbo.results import Curves, RunFile


def choose_init(campaign: Campaign, rng: np.random.Generator, protocol: Protocol) -> np.ndarray:
    """The initial design: ``n_initial`` rows drawn uniformly without replacement."""
    return rng.choice(campaign.n, size=protocol.n_initial, replace=False)


@dataclass(frozen=True)
class CampaignRun:
    """One seed's campaign.

    ``trace`` is the pool indices in acquisition order, initial design first.
    ``curve`` is the protocol's metric read off it; :func:`curve_from_trace`
    recomputes it.
    """

    curve: np.ndarray
    trace: np.ndarray


def run_campaign(
    method: Method,
    campaign: Campaign,
    init_idx: np.ndarray,
    seed: int,
    protocol: Protocol,
) -> CampaignRun:
    """Run one campaign for one seed. Returns its trace and the protocol's curve.

    The curve includes the initial design in its first ``n_initial`` columns.
    """
    budget = protocol.budget(campaign.n)
    observed = list(init_idx)
    remaining = np.setdiff1d(np.arange(campaign.n), init_idx)
    # `np.argmax` would take the first maximum, i.e. the next row in pool
    # order, which is a systematic traversal on a grouped pool. The stream is
    # separate from the initial design's, and a no-op wherever the maximum is
    # unique.
    tie_rng = np.random.default_rng([seed, 0])

    for _ in range(budget):
        if remaining.size == 0:
            break
        scores = method.score(
            ScoreRequest(
                x_obs=campaign.x[observed],
                y_obs=campaign.y[observed],
                x_cand=campaign.x[remaining],
                seed=seed,
                campaign=campaign,
            )
        )
        best = np.flatnonzero(scores == np.max(scores))
        pick = remaining[int(best[0] if best.size == 1 else tie_rng.choice(best))]
        observed.append(pick)
        remaining = remaining[remaining != pick]

    trace = np.asarray(observed, dtype=np.int32)
    return CampaignRun(curve_from_trace(trace, campaign, protocol), trace)


def curve_from_trace(
    trace: np.ndarray, campaign: Campaign, protocol: Protocol,
) -> np.ndarray:
    """The protocol's metric along a trace.

    ``topcount`` is the running count of top-``top_frac`` rows measured.
    ``best-value`` is the running maximum of the oriented objective.
    """
    if protocol.metric == "best-value":
        return np.maximum.accumulate(campaign.y[trace])
    return np.cumsum(campaign.top_mask(protocol.top_frac)[trace])


_WORKER_REGISTRY: dict[str, Method] = {}


def _run_one(job):
    """One (method, seed) :class:`CampaignRun`, in a worker process.

    Takes the method's name, not the method: acquisition rules are closures,
    which do not pickle. Each worker builds the registry once.
    """
    import torch

    from matbo.methods import build_registry

    campaign, method_name, seed, init_idx, protocol, threads = job
    torch.set_num_threads(threads)
    if not _WORKER_REGISTRY:
        _WORKER_REGISTRY.update(build_registry(device=protocol.device).methods)
    try:
        result = run_campaign(_WORKER_REGISTRY[method_name], campaign, init_idx, seed, protocol)
    except Exception as exc:
        # Surrogate libraries are imported only in workers, so the parent may
        # be unable to unpickle their exception classes.
        raise RuntimeError(
            f"{method_name} seed {seed} on {campaign.name}: "
            f"{type(exc).__name__}: {exc}"
        ) from None
    return method_name, seed, result


def _complete_prefix(collected, methods, seeds):
    """The seeds every method has finished, as a prefix of the protocol's order.

    Curve rows are positional and paired across arms, so a partial save must
    hold the same seeds for every arm.
    """
    prefix = []
    for seed in seeds:
        if not all(seed in collected[name] for name in methods):
            break
        prefix.append(seed)
    return prefix


def _run_parallel(campaign, methods, protocol, inits, workers, log,
                  partial=None, every_seeds=5):
    """Every (method, seed) of one campaign across a process pool.

    Returns the same runs as the sequential loop, in the protocol's seed order.
    """
    jobs = [
        (campaign, name, seed, init_idx, protocol, 1)
        for name in methods
        for seed, init_idx in inits.items()
    ]
    started = time.time()
    saved = 0
    collected: dict[str, dict[int, CampaignRun]] = {name: {} for name in methods}
    every = 1 if len(jobs) <= 60 else max(1, len(jobs) // 50)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_one, job) for job in jobs]
        done = 0
        for fut in as_completed(futures):
            name, seed, result = fut.result()
            collected[name][seed] = result
            done += 1
            if done == 1 or done % every == 0 or done == len(jobs):
                elapsed = time.time() - started
                eta = elapsed / done * (len(jobs) - done)
                log(f"    {campaign.name}: {done}/{len(jobs)} runs "
                    f"({100 * done // len(jobs)}%), {elapsed / 60:.0f}m elapsed, "
                    f"~{eta / 60:.0f}m left")
            if partial is not None:
                prefix = _complete_prefix(collected, methods, list(inits))
                if len(prefix) >= saved + every_seeds and len(prefix) < len(inits):
                    saved = len(prefix)
                    partial({n: [collected[n][s] for s in prefix] for n in methods},
                            len(prefix))
    runs = {name: [collected[name][s] for s in inits] for name in methods}
    log(f"  {len(jobs)} runs on {workers} workers in {time.time() - started:.0f}s")
    return runs


def run_suite(
    campaigns: list[Campaign],
    methods: dict[str, Method],
    protocol: Protocol,
    log: Callable[[str], None] = print,
    checkpoint: Callable[[RunFile], None] | None = None,
    workers: int = 1,
    into: RunFile | None = None,
    checkpoint_seeds: int = 5,
) -> RunFile:
    """Every method on every campaign, over the protocol's active seeds.

    ``checkpoint`` is called after each campaign and every
    ``checkpoint_seeds`` seeds within one, with the finished seed prefix.
    ``into`` is a RunFile holding an earlier pass, so checkpoints keep it.
    """
    run = into if into is not None else RunFile(protocol)
    log(f"protocol: {protocol.to_json()}")
    log(f"methods: {', '.join(methods)}")

    for campaign in campaigns:
        budget = protocol.budget(campaign.n)
        n_top = campaign.n_top(protocol.top_frac)
        fingerprint = campaign.fingerprint()
        # Refuse a moved pool before fitting anything.
        run.validate_campaign(campaign.name, n_top, campaign.n, fingerprint)
        log(
            f"{campaign.name}:\n  pool={campaign.n} dim={campaign.dim} n_top={n_top} "
            f"budget={budget} metric={protocol.metric} "
            f"objective={campaign.objective!r}"
        )
        for line in provenance.describe(campaign, protocol.top_frac):
            log(f"  {line}")

        # The initial design depends on the seed only, so every arm starts
        # from the same rows and the comparison is paired.
        inits = {
            seed: choose_init(campaign, np.random.default_rng(seed), protocol)
            for seed in protocol.active_seeds
        }
        n_planned = len(inits)

        def _store(name, results, _c=campaign, _t=n_top, _fp=fingerprint):
            """One arm's finished rows into the run; row i is seed i of the prefix."""
            run.add(
                _c.name, name,
                Curves(np.array([r.curve for r in results], dtype=float),
                       protocol.n_initial),
                _t, _c.n,
                traces=np.array([r.trace for r in results], dtype=np.int32),
                fingerprint=_fp,
            )

        def save_partial(done_runs, n_done, _c=campaign, _n=n_planned):
            """Write the finished seed prefix mid-campaign."""
            if checkpoint is None:
                return
            for nm, rows in done_runs.items():
                _store(nm, rows)
            checkpoint(run)
            log(f"    checkpointed {n_done}/{_n} seeds of {_c.name}")

        if workers > 1:
            runs = _run_parallel(campaign, methods, protocol, inits, workers, log,
                                 partial=save_partial, every_seeds=checkpoint_seeds)
        else:
            runs = {m: [] for m in methods}
            for i, (seed, init_idx) in enumerate(inits.items(), start=1):
                line = [f"  {campaign.name} seed {seed}"]
                for name, method in methods.items():
                    t0 = time.time()
                    result = run_campaign(method, campaign, init_idx, seed, protocol)
                    runs[name].append(result)
                    curve = result.curve
                    reached = (f"{curve[-1]:.4g}" if protocol.metric == "best-value"
                               else f"{curve[-1]:.0f}/{n_top}")
                    line.append(f"{name} {reached} ({time.time() - t0:.0f}s)")
                log("  ".join(line))
                if i % checkpoint_seeds == 0 and i < len(inits):
                    save_partial(runs, i)

        for name, results in runs.items():
            _store(name, results)
        width = max((len(r.trace) for results in runs.values() for r in results),
                    default=0)
        log(f"  traces: {len(runs)} arm(s) x {n_planned} seeds x {width} "
            f"selections recorded, pool fingerprint {fingerprint}")
        for line in fit_failure_report():
            log(f"  {line}")
        if checkpoint:
            checkpoint(run)
    return run
