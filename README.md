# Out-of-the-Box Tabular Foundation Models for Bayesian Optimization in Materials Discovery

Code and results for the paper by Klemens Flöge, Simon Bing, Leo Grinsztajn,
Eliott Kalfon, Lennart Purucker, Noah Hollmann and Frank Hutter (AI4Mat
workshop, NeurIPS 2026).

We compare tabular foundation models (TabPFN, TabICL) with Gaussian processes,
random forests and CatBoost as surrogates for pool-based Bayesian optimization
on real experimental campaigns. Each round, the optimizer picks the next
candidate to measure from a finite pool of settings that were measured in the
source study.

## What is in this repository

| path | contents |
|---|---|
| `src/matbo/` | data loaders, surrogates, acquisition rules, the BO loop, analysis and figures |
| `reference_results/` | the traces behind every number in the paper: which candidate each arm picked at each step, for every campaign and seed |
| `scripts/` | run arms, merge runs, export traces, check coverage and pool provenance |
| `data/` | the Liang et al. datasets ([licenses and sources](data/README.md)) |
| `docs/PROVENANCE.md` | how each candidate pool is built from its source |

**Benchmarks.** `liang`: 5 materials campaigns from
[Liang et al. (2021)](https://www.nature.com/articles/s41524-021-00656-9),
50 seeds, scored on Top-5% found. `olympus`: 10 chemistry and colorimetry
campaigns from [Häse et al. (2021)](https://iopscience.iop.org/article/10.1088/2632-2153/abedc8),
100 seeds, scored on best value found. Every run starts from 2 random points,
picks one candidate per round, and continues until the pool is exhausted.

**Arms.** 11 surrogates × 5 acquisition rules, plus Random: 56 arms.

- Surrogates: `TabPFN`, `TabICL`, `CatBoost`, `RF`, and seven GP kernels
  (`GP-ARD`, `GP-M32`, `GP-M12`, `GP-RBF`, `GP-iso`, `GP-MLP`, `GP-ARD-Warp`).
- Rules: `greedy`, `EI`, `PI`, `UCB1`, `UCB2`.
- An arm is named `<surrogate>-<rule>`, for example `TabPFN-UCB1`. The registry
  is [`src/matbo/methods/__init__.py`](src/matbo/methods/__init__.py).

## Install

```bash
uv sync --locked --extra dev
uv pip install olymp --no-deps   # the Olympus datasets; see data/README.md
```

`uv.lock` pins the library versions the paper's runs used. The TabPFN weights
are gated: accept the license once when TabPFN first asks, or set `TABPFN_TOKEN`.

## Reproduce the figures

All figures are computed from `reference_results/`. No model is fitted.

```bash
uv run python -m matbo.visualisation               # Figures 1, 2, D.1-D.10 -> figures/paper/
uv run python -m matbo.visualisation 1 D.3         # selected figures only
uv run python -m matbo.analysis.posthoc            # tables behind D.9 and D.10 -> figures/paper/posthoc/
```

Normalized improvement (NI) rescales each campaign and budget so that the worst
of the 56 arms scores 0 and the best scores 1 (App. C).

## Re-run the benchmark

The full grid is expensive. For example, one GP arm on `olympus` runs 100 seeds
through pools of up to 1,386 candidates. To re-run some or all arms and
compare them with the record:

```bash
# one arm on one campaign, 2 processes
FAMILIES=olympus ONLY=snar METHODS=RF-EI,Random WORKERS=2 TAG=check \
  uv run python scripts/run_benchmark.py

# every arm on a benchmark, one arm at a time
FAMILIES=liang TAG=liang ./scripts/run_arms.sh

# write a finished run into reference_results/, then check that no cell is missing
BENCHMARK=liang TAG=liang uv run python scripts/export_reference.py
uv run python scripts/coverage.py
```

Runs write to `results/<TAG>/`. `scripts/merge_runs.py` joins runs that were
split by arm or by seed (`SEED_SLICE=start:stop`).

A re-run picks the same candidates as the recorded trace when the numerics
match. Most cells we re-ran on a different machine matched exactly. Some
didn't: GP fits that fail and fall back to the prior, and some random-forest
ties, depend on the platform's floating-point behaviour.

`reference_results/README.md` describes the trace format and one documented
substituted seed (`GP-M12-UCB1` on `photo_pce10`, seed 5).

## Tests

```bash
uv run pytest tests -q
```

The tests check, among other things, that every committed trace decodes
against the current pool (`pool_fingerprint`) and that the figures use exactly
the 56 arms.

## License

Apache 2.0 (see `LICENSE`). The datasets keep their own licenses
(see [`data/README.md`](data/README.md)).
