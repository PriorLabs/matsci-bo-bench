#!/usr/bin/env bash
# Pool-head evaluation on one node: synthetic gate (both metrics), then the
# preregistered campaigns. $HEAD_RUN holds the trained head; $BASE_RULES are
# step-0 rules to rerun on this backbone (empty for v3.5, which has them).
set -u
export OMP_NUM_THREADS=4
OUT=$HOME/results/$RUN
mkdir -p "$OUT"
gcloud storage cp "$BUCKET/$HEAD_RUN/pool_head.pt" "$OUT/pool_head.pt" >/dev/null 2>&1
export POOL_HEAD_PATH=$OUT/pool_head.pt
gcloud storage rsync -r "$BUCKET/$RUN" "$OUT" >/dev/null 2>&1
( while true; do sleep 300; gcloud storage rsync -r "$OUT" "$BUCKET/$RUN" >/dev/null 2>&1; done ) &
sync_pid=$!
py=.venv/bin/python
PRIOR=mixed POOLS=0:20 METRIC=topcount RULES=PTop,PoolHead OUT=$OUT/gate_top \
  $py experiments/tabpfn35_acq/gate.py >"$OUT/gate_top.log" 2>&1 &
PRIOR=mixed POOLS=0:20 METRIC=best-value RULES=EI,PoolHead OUT=$OUT/gate_best \
  $py experiments/tabpfn35_acq/gate.py >"$OUT/gate_best.log" 2>&1 &
wait $(jobs -p | grep -v "^$sync_pid$")
for spec in "liang AutoAM PoolHead${BASE_TOP:+,$BASE_TOP} 0:50" \
            "olympus benzylation PoolHead${BASE_BEST:+,$BASE_BEST} 0:100" \
            "olympus colors_bob PoolHead${BASE_BEST:+,$BASE_BEST} 0:100" \
            "olympus fullerenes PoolHead${BASE_BEST:+,$BASE_BEST} 0:100"; do
  read -r bench campaign rules seeds <<<"$spec"
  BENCH=$bench CAMPAIGN=$campaign RULES=$rules SEEDS=$seeds OUT=$OUT/$bench TAG=$RUN \
    $py experiments/tabpfn35_acq/lockstep.py >"$OUT/${bench}_$campaign.log" 2>&1 &
done
wait $(jobs -p | grep -v "^$sync_pid$")
kill $sync_pid
gcloud storage rsync -r "$OUT" "$BUCKET/$RUN" >/dev/null 2>&1
echo "eval done"
