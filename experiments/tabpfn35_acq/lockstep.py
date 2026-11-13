#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""Run TabPFN-3.5 arms on one campaign, every (rule, seed) in lockstep.

At a given step every (rule, seed) run has the same number of observed rows,
so one ``predict_batched`` call fits them all. The loop is the benchmark's:
the same initial design per seed as ``matbo.runner.choose_init``, one pick per
round, ties broken uniformly on a per-seed stream. Runs stop at ``STOP_FRAC``
of the pool: NI and the seed AUCs read no budget beyond 50%.

    BENCH=liang CAMPAIGN=AgNP RULES=EI,PTop SEEDS=0:50 OUT=results/tabpfn35 \
        python experiments/tabpfn35_acq/lockstep.py

Each step's state is saved to ``OUT/<campaign>/state.npz``, so a restarted job
resumes where it stopped. Every random draw is keyed on (seed, step), so a
resumed run is identical to an uninterrupted one.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rules import RULES, Round  # noqa: E402

from matbo import loaders as L  # noqa: E402
from matbo.benchmarks import get as get_benchmark  # noqa: E402
from matbo.runner import choose_init  # noqa: E402

TRACE_KEYS = ["run_tag", "benchmark", "campaign", "method", "seed", "pool", "pool_fingerprint"]


def _regressor(device: str, n_estimators: int):
    """TabPFN-3.5 holding its loaded model, so a refit neither reloads nor re-places it.

    ``predict_batched`` fits a clone per dataset, and a clone given a checkpoint
    path reads the checkpoint and copies it to the device on every fit.
    """
    from tabpfn import TabPFNRegressor
    from tabpfn.base import ModelSpecs
    from tabpfn.constants import ModelVersion

    kwargs = dict(
        n_estimators=n_estimators, device=device, random_state=0, categorical_features_indices=None
    )
    base = TabPFNRegressor.create_default_for_version(ModelVersion.V3_5, **kwargs)
    rng = np.random.default_rng(0)
    base.fit(rng.random((8, 2)), rng.random(8))
    spec = ModelSpecs(
        model=base.models_[0],
        architecture_config=base.configs_[0],
        inference_config=base.inference_config_,
        norm_criterion=base.znorm_space_bardist_,
    )
    return TabPFNRegressor(model_path=spec, **kwargs)


def stop_length(n: int, frac: float) -> int:
    """Trace length that covers every budget column up to ``frac`` of the pool."""
    return min(n, int(np.rint(frac * n)) + 1)


def run_lockstep(
    reg,
    x: np.ndarray,
    y: np.ndarray,
    runs: list[tuple[str, int]],
    traces: list[list[int]],
    length: int,
    top_frac: float,
    chunk: int,
    state_path: Path | None = None,
    log=print,
) -> list[list[int]]:
    """Extend every (rule, seed) trace to ``length`` picks; ``traces`` start as initial designs."""
    n = len(y)
    started = time.time()
    while len(traces[0]) < length:
        step = len(traces[0])
        t0 = time.time()
        masks = []
        for tr in traces:
            m = np.ones(n, dtype=bool)
            m[tr] = False
            masks.append(m)
        x_tr = [x[tr] for tr in traces]
        y_tr = [y[tr] for tr in traces]
        x_te = [x[m] for m in masks]
        outs = []
        for i in range(0, len(runs), chunk):
            outs += reg.predict_batched(
                x_tr[i : i + chunk], y_tr[i : i + chunk], x_te[i : i + chunk], output_type="full"
            )
        t_fit = time.time() - t0
        for j, ((rule, seed), out) in enumerate(zip(runs, outs, strict=True)):
            rnd = Round(
                out=out,
                y_obs=y_tr[j],
                n_pool=n,
                top_frac=top_frac,
                rng=np.random.default_rng([seed, step, 1]),
                budget_frac=step / n,
            )
            remaining = np.flatnonzero(masks[j])
            if out["logits"].shape[-1] == 1:
                # TabPFN answers a constant target with a point mass: every candidate ties.
                scores = np.zeros(len(remaining))
            else:
                scores = np.asarray(RULES[rule](rnd), dtype=np.float64)
            if len(scores) != len(remaining) or not np.isfinite(scores).all():
                raise RuntimeError(f"{rule} seed {seed} step {step}: bad scores")
            best = np.flatnonzero(scores == scores.max())
            if best.size > 1:
                best = [np.random.default_rng([seed, step, 0]).choice(best)]
            traces[j].append(int(remaining[best[0]]))
        if state_path is not None:
            np.savez(
                state_path,
                rules=np.array([r for r, _ in runs]),
                seeds=np.array([s for _, s in runs]),
                traces=np.array(traces),
            )
        if step % 10 == 0 or len(traces[0]) == length:
            log(
                f"  step {step + 1}/{length}: fit {t_fit:.1f}s, total {time.time() - t0:.1f}s, "
                f"elapsed {(time.time() - started) / 60:.1f}m"
            )
    return traces


def main() -> int:
    bench = get_benchmark(os.environ["BENCH"])
    protocol = bench.loop
    name = os.environ["CAMPAIGN"]
    # Specs are whitespace-separated, so "Crossed barrel" arrives as "Crossed_barrel".
    campaign = L.load(name if name in L.names() else name.replace("_", " "))
    rules = os.environ["RULES"].split(",")
    start, stop = (int(v) for v in os.environ.get("SEEDS", f"0:{protocol.n_seeds}").split(":"))
    seeds = list(protocol.seeds[: protocol.n_seeds])[start:stop]
    out_dir = Path(os.environ["OUT"]) / campaign.name.replace(" ", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = os.environ.get("TAG", "tabpfn35")
    prefix = os.environ.get("PREFIX", "TabPFN35")
    chunk = int(os.environ.get("CHUNK", "32"))
    device = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    reg = _regressor(device, int(os.environ.get("N_ESTIMATORS", "8")))

    n = campaign.n
    length = stop_length(n, float(os.environ.get("STOP_FRAC", "0.5")))
    runs = [(rule, seed) for rule in rules for seed in seeds]
    state_path = out_dir / "state.npz"
    if state_path.exists():
        saved = np.load(state_path, allow_pickle=False)
        if len(saved["rules"]) == len(runs):
            assert list(zip(saved["rules"], saved["seeds"])) == runs
        else:  # states written before per-run rule labels
            assert list(saved["rules"]) == rules and list(saved["seeds"]) == seeds
        traces = saved["traces"].tolist()
    else:
        inits = {s: choose_init(campaign, np.random.default_rng(s), protocol) for s in seeds}
        traces = [list(map(int, inits[s])) for _, s in runs]
    print(
        f"{campaign.name}: n={n} dim={campaign.dim} length={length} runs={len(runs)} "
        f"from step {len(traces[0])}",
        flush=True,
    )

    started = time.time()
    traces = run_lockstep(
        reg,
        campaign.x,
        campaign.y,
        runs,
        traces,
        length,
        protocol.top_frac,
        chunk,
        state_path,
        log=lambda m: print(m, flush=True),
    )

    fp = campaign.fingerprint()
    for rule in rules:
        path = out_dir / f"{prefix}-{rule}__{campaign.name.replace(' ', '_')}.trace.csv"
        with path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(TRACE_KEYS + [f"s{i + 1}" for i in range(length)])
            for (r, seed), tr in zip(runs, traces, strict=True):
                if r == rule:
                    w.writerow(
                        [tag, bench.name, campaign.name, f"{prefix}-{rule}", seed, n, fp, *tr]
                    )
    (out_dir / "done.json").write_text(
        json.dumps(
            {
                "rules": rules,
                "seeds": seeds,
                "length": length,
                "minutes": (time.time() - started) / 60,
            }
        )
    )
    print(f"done in {(time.time() - started) / 60:.1f}m", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
