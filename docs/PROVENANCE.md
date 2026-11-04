# Pool provenance

How each candidate pool relates to the dataset it comes from.
`src/matbo/provenance.py` holds the published counts, and every run reports
where its pool differs from them.

## `liang`: Liang et al. (2021)

Repeated measurements of one setting are averaged into one candidate, as in
Liang et al. The pools then match their Table 1:

| campaign | paper size / dims | ours | raw csv rows |
|---|---|---|---|
| P3HT/CNT | 178 / 5 | 178 / 5 | 233 |
| AgNP | 164 / 5 | 164 / 5 | 3295 |
| Perovskite | 94 / 3 | 94 / 3 | 139 |
| Crossed barrel | 600 / 4 | 600 / 4 | 1800 |
| AutoAM | 100 / 4 | 100 / 4 | 100 |

The loop follows their Methods and notebook: two initial points drawn without
replacement, batch size 1, `n_top = ceil(0.05 n)`, 50 seeds from their
published seed list, and every run continues until the pool is exhausted.

## `olympus`: Häse et al. (2021)

Every row in the `olymp` package's bundled `data.csv` is one candidate, in the
order the file lists them. Nothing is averaged or dropped:

| campaign | rows | n_top |
|---|---|---|
| snar | 66 | 4 |
| benzylation | 73 | 4 |
| colors_n9 | 102 | 6 |
| alkox | 208 | 11 |
| colors_bob | 241 | 13 |
| fullerenes | 246 | 13 |
| suzuki | 247 | 13 |
| photo_pce10 | 1040 | 52 |
| photo_wf3 | 1040 | 52 |
| hplc | 1386 | 70 |

All ten campaigns have one objective and only continuous parameters. The
optimization direction comes from each config's `default_goal`.

Häse et al. optimize over a continuous space through a Bayesian neural network
emulator fitted to these rows. Here, the measured rows are the candidate pool,
so no method can propose an unmeasured setting. The emulators are TensorFlow 1
checkpoints and cannot be loaded with current Python. There are 100 seeds
(0-99), as in their 100 independent runs.

Two campaigns do not satisfy the constraints their own configs declare:

- `colors_bob` declares a simplex, but its 241 rows sum to between 0.638 and
  4.405.
- All 208 rows of `alkox` fall outside the `low`/`high` box in its config.

The loader reports both and drops no rows. Also, the bundled
`dataset_snar/description.txt` lists 67 samples and the target `e_factor`.
The data file has 66 rows, and the config names the target `impurity`.
