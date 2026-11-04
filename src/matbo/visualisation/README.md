# Reproduce the paper figures

```sh
python -m matbo.visualisation                  # all twelve figures, PDF and PNG
python -m matbo.visualisation 1 2 D.10          # selected figure numbers
python -m matbo.visualisation --out /tmp/paper  # another output directory
python -m matbo.analysis.posthoc                # tables behind D.9 and D.10
```

Figures go to `figures/paper/`, tables to `figures/paper/posthoc/<benchmark>/`.
Inputs are the traces in `reference_results/` and their source pools; no model
is fitted. Install the project and the Olympus pool data first
(`uv sync --extra dev`, then `uv pip install olymp --no-deps`).

| Paper figure | Output stem | Content | Implementation |
|---|---|---|---|
| 1 | `figure_1` | Best arm per family, each benchmark's own metric | `figures.budget` |
| 2 | `figure_2` | `liang` kernel comparison, UCB1, raw Top-5% | `kernels.kernel_variation` |
| D.1 | `figure_D_1` | `liang` Top%, enhancement and acceleration factors | `campaigns.liang` |
| D.2 | `figure_D_2` | `olympus` per-campaign best-value curves | `campaigns.olympus` |
| D.3 | `figure_D_3` | NI-AUC ranking of the 55 factorial arms | `figures.ranking` |
| D.4 | `figure_D_4` | All 56 arms on `liang` | `figures.all_arms` |
| D.5 | `figure_D_5` | All 56 arms on `olympus` | `figures.all_arms` |
| D.6 | `figure_D_6` | Both benchmarks scored on Top-5% | `figures.budget` |
| D.7 | `figure_D_7` | Both benchmarks scored on best value | `figures.budget` |
| D.8 | `figure_D_8` | `olympus` kernel comparison, PI, raw best value | `kernels.kernel_variation` |
| D.9 | `figure_D_9` | Paired differences, TabPFN minus each comparator | `posthoc.paired` |
| D.10 | `figure_D_10` | Best acquisition rule per surrogate and campaign | `posthoc.best_rule` |

## Tables

- `paired_effects.csv`: D.9. Mean paired AUC difference (TabPFN minus each
  other surrogate under the same rule, and minus Random), 95% seed-bootstrap
  interval (10,000 resamples), SD and win/tie/loss fractions. Rows with
  `analysis_variant = substituted-seeds-dropped` repeat a contrast without the
  seeds that a run recorded as substituted.
- `acquisition_cost.csv`: D.10. Mean AUC of every rule per surrogate and
  campaign, its gap to the best rule (`cost`) and whether it is best.
- `figure_scale.csv` (`olympus` only): the D.9 denominator per campaign, pool
  best minus Random's mean AUC.

AUC here is each seed's raw metric integrated over the linear 5-50% budget
window and divided by its width. A positive paired difference favors the
first arm, also on minimization campaigns; `mean` is in source units.

## Numerical conventions

- NI is min-max over exactly the 56 `FIELD` arms, per campaign and budget,
  after averaging seeds; then NI is averaged across campaigns. Rows where all
  arms tie are undefined.
- Rankings integrate NI over linear budget from 5% to 50% on a 1% grid. Each
  family's representative is the top-ranked arm of that family.
- Figures 1, D.6 and D.7 show eight log-spaced budgets with 400 campaign
  bootstrap resamples. Figures 2 and D.8 integrate each seed's raw values over
  the log-budget window and use 400 paired seed bootstrap resamples.
- D.1 shows the median and 5-95% band of Liang et al.'s fold means. D.2 shows
  mean curves and 25-75% intervals over the 100 runs.
