#!/bin/bash
# Recycle wave: concurrent batches of 3 (distinct FL_SUBNET_OCTET per slot).
# Batch A: recycle seeds 41-43; B: 44-46; C (time-guarded): skip+aging smoke + bundle seeds.
set +e
cd ~/fl-framework && source .venv/bin/activate
OUT=campaigns/p1/exp10_recycle
run_one() { # $1=config $2=outdir $3=octet
  [ -f "$2/results/report.json" ] && { echo "skip $2 (done)"; return; }
  mkdir -p "$2"
  echo "$(date '+%H:%M:%S') START octet=$3 -> $2"
  FL_SUBNET_OCTET=$3 timeout 2900 python -m scripts.run -c "$1" -o "$2" >> "$2/run.log" 2>&1
  echo "$(date '+%H:%M:%S') END   rc=$? $( [ -f "$2/results/report.json" ] && echo OK || echo NO_REPORT ) $2"
}
echo "=== batch A (recycle 41-43) $(date '+%H:%M:%S') ==="
run_one configs/experiments/phase1b/exp10_recycle/eps03_recycle/seed41.yaml $OUT/eps03_recycle/seed41 9 &
run_one configs/experiments/phase1b/exp10_recycle/eps03_recycle/seed42.yaml $OUT/eps03_recycle/seed42 10 &
run_one configs/experiments/phase1b/exp10_recycle/eps03_recycle/seed43.yaml $OUT/eps03_recycle/seed43 11 &
wait
echo "=== batch B (recycle 44-46) $(date '+%H:%M:%S') ==="
run_one configs/experiments/phase1b/exp10_recycle/eps03_recycle/seed44.yaml $OUT/eps03_recycle/seed44 9 &
run_one configs/experiments/phase1b/exp10_recycle/eps03_recycle/seed45.yaml $OUT/eps03_recycle/seed45 10 &
run_one configs/experiments/phase1b/exp10_recycle/eps03_recycle/seed46.yaml $OUT/eps03_recycle/seed46 11 &
wait
echo "=== batch C (time-guarded extras) $(date '+%H:%M:%S') ==="
NOW=$(TZ=Europe/Zurich date +%H%M)
if [ "$NOW" -lt 1650 ]; then
  # skip+aging smoke (ratchet fix demo) + bundle seeds
  python - <<'PYC'
import yaml
c=yaml.safe_load(open("configs/experiments/phase1b/exp1_grid/coverage_eft_eps03/seed41.yaml"))
c["training"]["total_rounds"]=6
c["training"]["late_layer_policy"]="recycle_last_delta"
c["training"]["skip_feedback"]="shed"
c["training"]["aging_mode"]="additive_capped"; c["training"]["aging_lambda"]=0.5; c["training"]["aging_tau_max"]=3
yaml.dump(c,open("/tmp/smoke_skip_aging.yaml","w"),default_flow_style=False,sort_keys=False)
PYC
  run_one /tmp/smoke_skip_aging.yaml $OUT/skip_aging_smoke 9 &
  run_one configs/experiments/phase1b/exp10_recycle/eps03_recycle_aging/seed41.yaml $OUT/eps03_recycle_aging/seed41 10 &
  run_one configs/experiments/phase1b/exp10_recycle/eps03_recycle_aging/seed42.yaml $OUT/eps03_recycle_aging/seed42 11 &
  wait
else
  echo "batch C skipped (past 16:50 guard)"
fi
echo "=== RECYCLE WAVE DONE $(date '+%H:%M:%S') ==="
