# Datasets

| benchmark | source | rows read from | license |
|---|---|---|---|
| `liang` | Liang et al., *npj Comput. Mater.* **7**, 188 (2021). [Paper](https://www.nature.com/articles/s41524-021-00656-9), [code](https://github.com/PV-Lab/Benchmarking) | `liang_2021/`, copied from PV-Lab/Benchmarking | MIT |
| `olympus` | Häse et al., *Mach. Learn.: Sci. Technol.* **2**, 035021 (2021). [Paper](https://iopscience.iop.org/article/10.1088/2632-2153/abedc8), [code](https://github.com/aspuru-guzik-group/olympus) | the `data.csv` and `config.json` files bundled with the `olymp` package (not copied here) | MIT |

Install the Olympus datasets with `uv pip install olymp --no-deps`. The package
pins TensorFlow 1.15 and fails to import on current matplotlib. Only its
bundled files are read, so its dependencies are not needed. To read the files
from a checkout instead, set `OLYMPUS_DIR` to its `olympus/datasets` directory.

[`docs/PROVENANCE.md`](../docs/PROVENANCE.md) describes how each pool is built
from these rows.

`liang_2021/reference_gp_m52_lcb2.npy` is `TopCount_collection_GP_Matern52_LCB2.npy`
from PV-Lab/Benchmarking: Liang et al.'s own GP results on Crossed barrel. A
test uses it to check that our fold aggregation matches theirs.
