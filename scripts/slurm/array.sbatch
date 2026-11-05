#!/usr/bin/env bash
#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0
#
#SBATCH --job-name=tfm-bo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --requeue
#
# One array task: one arm on every campaign of the benchmark, for one seed range.
# Submitted by submit.sh, which sets FAMILIES, TAG, CELLS, REPO, PY and RESULTS.
set -euo pipefail

line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CELLS")
[[ -n "$line" ]] || { echo "no cell at line $SLURM_ARRAY_TASK_ID of $CELLS" >&2; exit 1; }
IFS=$'\t' read -r arm window <<< "$line"

# One process per seed with one thread each; more threads per process oversubscribe.
export WORKERS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# A separate tag per task, so no two tasks write the same curves.npz.
export TAG="$TAG/cells/${arm}__s${window/:/-}"
export METHODS="$arm" SEED_SLICE="$window" FAMILIES MATBO_RESULTS="$RESULTS"
export DEVICE=cpu
export PYTHONPATH="$REPO/src"

echo "host=$(hostname) arm=$arm seeds=$window workers=$WORKERS tag=$TAG"
cd "$REPO"
srun "$PY" scripts/run_benchmark.py
