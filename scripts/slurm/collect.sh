#!/usr/bin/env bash
#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0
#
# Merge the finished array tasks of one run into a single curves.npz.
#
#   TAG=liang ./scripts/slurm/collect.sh
#   BENCHMARK=liang TAG=liang .venv/bin/python scripts/export_reference.py   # then export
set -euo pipefail
cd "$(dirname "$0")/../.."

TAG="${TAG:?set TAG, the tag given to submit.sh}"
PY="${PY:-.venv/bin/python}"
RESULTS="${MATBO_RESULTS:-results}"

shopt -s nullglob
parts=("$RESULTS/$TAG"/cells/*/curves.npz)
expected=$(wc -l < "$RESULTS/$TAG/cells.tsv" | tr -d ' ')
echo "found ${#parts[@]} of $expected task results"
(( ${#parts[@]} == expected )) || { echo "REFUSED: some tasks have not finished" >&2; exit 1; }
OUT="$RESULTS/$TAG/curves.npz" PYTHONPATH=src "$PY" scripts/merge_runs.py "${parts[@]}"
