#!/usr/bin/env bash
# Runs the JOBS specs PAR at a time and mirrors $HOME/results/$RUN to $BUCKET/$RUN.
set -u
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OUT=$HOME/results/$RUN
mkdir -p "$OUT/logs"
# Mirror only this node's own campaign folders: nodes share $BUCKET/$RUN.
mirror() {
  local spec bench campaign seeds dir
  IFS=';' read -ra all <<<"$JOBS"
  for spec in "${all[@]}"; do
    read -r bench campaign _ seeds <<<"$spec"
    dir="$bench/${seeds/:/-}/$campaign"
    mkdir -p "$OUT/$dir"
    if [ "$1" = pull ]; then
      gcloud storage rsync -r "$BUCKET/$RUN/$dir" "$OUT/$dir" >/dev/null 2>&1
    else
      gcloud storage rsync -r "$OUT/$dir" "$BUCKET/$RUN/$dir" >/dev/null 2>&1
    fi
  done
  [ "$1" = push ] && gcloud storage rsync -r "$OUT/logs" "$BUCKET/$RUN/logs/$(hostname)" >/dev/null 2>&1
}
mirror pull  # resume what a preempted node uploaded
( while true; do sleep 600; mirror push; done ) &
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
mirror push
echo "synced to $BUCKET/$RUN"
