# Reference results

The traces behind every number in the paper. Each benchmark folder has one file
per (arm, campaign):

```
reference_results/
  liang/      protocol.json, <arm>__<campaign>.trace.csv   56 arms x 5 campaigns, 50 seeds
  olympus/    protocol.json, <arm>__<campaign>.trace.csv   56 arms x 10 campaigns, 100 seeds
```

`protocol.json` holds the loop that produced every file in its folder. The
exporter refuses a run whose protocol differs.

## Trace files

There is one row per seed. The leading columns are `run_tag`, `benchmark`,
`campaign`, `method`, `seed`, `pool` (pool size) and `pool_fingerprint`. After
them, `s1 ... sN` are the pool indices measured at evaluations 1 to N. The
first two are the shared random initial design. Every run continues until the
pool is exhausted.

Every metric is a function of the trace: Top%, best value, and Liang et al.'s
enhancement and acceleration factors. So a run can be scored on any of them
without being re-run:

```python
import numpy as np
import pandas as pd

from matbo import loaders as L

t = pd.read_csv("reference_results/olympus/RF-PI__photo_pce10.trace.csv")
pool = L.load("photo_pce10")
assert (t.pool_fingerprint == pool.fingerprint()).all()  # the indices refer to this pool
picks = t[[c for c in t.columns if c[:1] == "s" and c[1:].isdigit()]].to_numpy()
top_found = np.cumsum(pool.top_mask(0.05)[picks], axis=1)
```

A trace only makes sense with the pool it indexes. If the loader changes how
it handles repeats, which rows it drops, or their order, index 37 refers to a
different experiment. The fingerprint check catches this.

For a given seed, all arms start from the same initial design. Sort on `seed`
to pair arms.

## One substituted seed

In `olympus/GP-M12-UCB1__photo_pce10.trace.csv`, the row with `seed=5` is the
`Random` arm's seed-5 trace, not a `GP-M12-UCB1` run. Its `run_tag` ends in
`:Random-substituted`. The other 99 rows are the arm's own. The paper reports
this (App. D).

`GP-M12-UCB1` cannot finish that seed. Its lengthscale for a near-constant
input dimension fits to exactly 0, the Matérn-1/2 distance divides by zero, and
some candidates score NaN at acquisition 6. This happens on every hardware
configuration tried. A lower bound on the lengthscale avoids the NaN, but it
also changes the GP fits on other campaigns, so the reported traces do not use
it.
