#!/usr/bin/env bash
#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0
#
# Run arms one at a time into one results tag, merging each into its curves.npz.
#
#   FAMILIES=liang TAG=liang ./scripts/run_arms.sh                  # all 56 arms
#   FAMILIES=olympus TAG=olympus ARMS="RF-EI GP-ARD-EI Random" ./scripts/run_arms.sh
#
# One arm per pass, so a crash loses at most the arm in progress. ARMS
# defaults to every registered arm. Everything else (ONLY, WORKERS, N_SEEDS,
# SEED_SLICE, DEVICE, MATBO_RESULTS) passes through to run_benchmark.py.
set -euo pipefail
cd "$(dirname "$0")/.."

FAMILIES="${FAMILIES:?set FAMILIES to liang or olympus}"
TAG="${TAG:?set TAG, the results directory under MATBO_RESULTS}"
PY="${PY:-.venv/bin/python}"
export PYTHONPATH="${PYTHONPATH:-src}"
export WORKERS="${WORKERS:-1}"

if [[ -z "${ARMS:-}" ]]; then
  ARMS=$("$PY" -c "from matbo.methods import build_registry; print(' '.join(build_registry().methods))")
fi

# Two runs writing one TAG overwrite each other's curves.npz.
if pgrep -f "[r]un_benchmark.py" >/dev/null 2>&1; then
  echo "REFUSED: run_benchmark.py is already running here." >&2
  exit 1
fi

read -r -a _arms <<< "${ARMS//,/ }"
echo "arms=${#_arms[@]} tag=$TAG families=$FAMILIES workers=$WORKERS"
for _arm in "${_arms[@]}"; do
  echo
  echo "=== $_arm ($(date -u +%H:%M:%SZ)) ==="
  FAMILIES="$FAMILIES" TAG="$TAG" METHODS="$_arm" MERGE=1 "$PY" scripts/run_benchmark.py
done
echo
echo "=== done: ${MATBO_RESULTS:-results}/$TAG ==="
echo "export with: BENCHMARK=$FAMILIES TAG=$TAG $PY scripts/export_reference.py"
