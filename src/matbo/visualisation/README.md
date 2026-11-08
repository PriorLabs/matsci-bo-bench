# Reproduce the paper figures

```sh
python -m matbo.visualisation                  # all eleven figures, PDF and PNG
python -m matbo.visualisation 1 2 D.9           # selected figure numbers
python -m matbo.visualisation --out /tmp/paper  # another output directory
python -m matbo.analysis.posthoc                # Table D.1 and the table behind D.9
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
| D.9 | `figure_D_9` | Best acquisition rule per surrogate and campaign | `posthoc.best_rule` |

## Tables

- `seed_tests.csv`: Table D.1. Welch t-test per campaign of the TabPFN arm of
  Figure 1 against each other family's arm (`mean_diff`, `welch_p`, `verdict`).
- `family_ranks.csv` (one file for both benchmarks): mean rank of each family
  over all 15 campaigns, by mean and by median seed AUC, with the Friedman
  p-value and the Nemenyi critical difference.
- `acquisition_cost.csv`: D.9. Mean AUC of every rule per surrogate and
  campaign, its gap to the best rule (`cost`) and whether it is best.

AUC here is each seed's raw metric integrated over the linear 5-50% budget
window and divided by its width. A positive `mean_diff` favors TabPFN, also on
minimization campaigns; `mean` is in source units.

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
