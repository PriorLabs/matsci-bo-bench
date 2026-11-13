#!/usr/bin/env bash
# Runs the JOBS specs PAR at a time and mirrors $HOME/results/$RUN to $BUCKET/$RUN.
set -u
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OUT=$HOME/results/$RUN
mkdir -p "$OUT/logs"
# Resume from what an earlier (preempted) node uploaded.
gcloud storage rsync -r "$BUCKET/$RUN" "$OUT" >/dev/null 2>&1
( while true; do sleep 600; gcloud storage rsync -r "$OUT" "$BUCKET/$RUN" >/dev/null 2>&1; done ) &
sync_pid=$!
one() {
  read -r bench campaign rules seeds <<<"$1"
  local name="${bench}_${campaign}_${seeds/:/-}_${rules//,/+}"
  BENCH=$bench CAMPAIGN=$campaign RULES=$rules SEEDS=$seeds OUT="$OUT/$bench/${seeds/:/-}" \
    TAG=$RUN .venv/bin/python experiments/tabpfn35_acq/lockstep.py >"$OUT/logs/$name.log" 2>&1
  echo "$name rc=$? $(tail -1 "$OUT/logs/$name.log")"
}
export -f one
IFS=';' read -ra specs <<<"$JOBS"
printf '%s\n' "${specs[@]}" | xargs -P "$PAR" -I{} bash -c 'one "{}"'
kill $sync_pid
gcloud storage rsync -r "$OUT" "$BUCKET/$RUN" >/dev/null 2>&1
echo "synced to $BUCKET/$RUN"
