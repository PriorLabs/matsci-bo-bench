#!/usr/bin/env bash
# Step 1 on one node: generate head data, train the head, run the offline gate.
# Everything lands in $BUCKET/$RUN; a restarted node resumes from there.
set -u
export OMP_NUM_THREADS=4
OUT=$HOME/results/$RUN
mkdir -p "$OUT"
gcloud storage rsync -r "$BUCKET/$RUN" "$OUT" >/dev/null 2>&1
( while true; do sleep 300; gcloud storage rsync -r "$OUT" "$BUCKET/$RUN" >/dev/null 2>&1; done ) &
sync_pid=$!
py=.venv/bin/python
for r in 0:60 60:120 120:180; do
  POOLS=$r OUT=$OUT/data $py experiments/tabpfn35_acq/gen.py >"$OUT/gen_${r/:/-}.log" 2>&1 &
done
wait $(jobs -p | grep -v "^$sync_pid$")
$py experiments/tabpfn35_acq/train.py "$OUT/data" "$OUT/head.pt" 2>&1 | tee "$OUT/train.log"
for r in 0:14 14:28 28:40; do
  HEAD_PATH=$OUT/head.pt POOLS=$r RULES=PTop,Head,UCB1 OUT=$OUT/gate \
    $py experiments/tabpfn35_acq/gate.py >"$OUT/gate_${r/:/-}.log" 2>&1 &
done
wait $(jobs -p | grep -v "^$sync_pid$")
kill $sync_pid
gcloud storage rsync -r "$OUT" "$BUCKET/$RUN" >/dev/null 2>&1
echo "head run done"
