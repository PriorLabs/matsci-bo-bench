#!/usr/bin/env bash
#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0
#
# Run a benchmark on a Slurm cluster as a job array: one task per (arm, seed range).
#
#   FAMILIES=liang TAG=liang ./scripts/slurm/submit.sh                 # print the plan
#   FAMILIES=liang TAG=liang CONFIRM=1 ./scripts/slurm/submit.sh       # submit it
#   FAMILIES=olympus TAG=olympus ARMS="RF-EI,Random" SEEDS_PER_JOB=25 CONFIRM=1 ./scripts/slurm/submit.sh
#
# Cluster settings (partition, account, time, cpus) come from sbatch itself:
# set SBATCH_PARTITION, SBATCH_ACCOUNT, SBATCH_TIMELIMIT, or pass SBATCH_ARGS.
# When every task has finished, run ./scripts/slurm/collect.sh.
set -euo pipefail
cd "$(dirname "$0")/../.."

FAMILIES="${FAMILIES:?set FAMILIES to liang or olympus}"
TAG="${TAG:?set TAG, the results directory under MATBO_RESULTS}"
PY="${PY:-$PWD/.venv/bin/python}"
SEEDS_PER_JOB="${SEEDS_PER_JOB:-10}"
RESULTS="${MATBO_RESULTS:-$PWD/results}"
export PYTHONPATH="$PWD/src"

if [[ -z "${ARMS:-}" ]]; then
  ARMS=$("$PY" -c "from matbo.methods import build_registry; print(' '.join(build_registry().methods))")
fi
N_SEEDS=$("$PY" -c "from matbo.benchmarks import BENCHMARKS; print(BENCHMARKS['$FAMILIES'].loop.n_seeds)")

mkdir -p "$RESULTS/$TAG"
CELLS="$RESULTS/$TAG/cells.tsv"
: > "$CELLS"
read -r -a arms <<< "${ARMS//,/ }"
for arm in "${arms[@]}"; do
  for ((start = 0; start < N_SEEDS; start += SEEDS_PER_JOB)); do
    stop=$((start + SEEDS_PER_JOB < N_SEEDS ? start + SEEDS_PER_JOB : N_SEEDS))
    printf '%s\t%s:%s\n' "$arm" "$start" "$stop" >> "$CELLS"
  done
done
n=$(wc -l < "$CELLS" | tr -d ' ')

echo "benchmark=$FAMILIES tag=$TAG arms=${#arms[@]} seeds=$N_SEEDS per_job=$SEEDS_PER_JOB tasks=$n"
echo "cells: $CELLS"
if [[ -z "${CONFIRM:-}" ]]; then
  echo "plan only; set CONFIRM=1 to submit"
  exit 0
fi

mkdir -p "$RESULTS/$TAG/logs"
# shellcheck disable=SC2086
sbatch --array="1-$n" --output="$RESULTS/$TAG/logs/%A_%a.log" ${SBATCH_ARGS:-} \
  --export=ALL,FAMILIES="$FAMILIES",TAG="$TAG",CELLS="$CELLS",REPO="$PWD",PY="$PY",RESULTS="$RESULTS" \
  scripts/slurm/array.sbatch
