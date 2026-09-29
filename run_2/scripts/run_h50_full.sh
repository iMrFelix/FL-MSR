#!/bin/bash
# 50r extension → full 4-arm n=3 triptych (2026-07-23, meeting pushed to 17:00).
# Restartable: skips any seed with a report.json (mono 41-43 + recycle 42/43 done).
# Runs the remainder: recycle41 (CIFAR-race re-run), drop 41-43, aging 41-43.
# 3-WAY ONLY (octets 9/10/11). NEVER run while another campaign is live (6-way deadlocks
# the box — see writeup/moltres memory). Output shares campaigns/h50min so the analyzer
# sees all four arms at once.
set +e
cd ~/fl-framework && source .venv/bin/activate
OUT=campaigns/h50min
DONE="$OUT/H50FULL.DONE"
C=configs/experiments/horizon50
mkdir -p "$OUT"; : > "$DONE"

# Warm the CIFAR cache single-process FIRST, so concurrent runs never race a re-download
# (the recycle/seed41 "pickle data was truncated" failure).
echo "$(date '+%H:%M:%S') warming CIFAR cache..." | tee -a "$DONE"
python -c "import tensorflow as tf; tf.keras.datasets.cifar10.load_data(); print('cifar ok')" >> "$DONE" 2>&1

run_one() { local cfg="$1" d="$2" oct="$3"
  [ -f "$d/results/report.json" ] && { echo "$(date '+%H:%M:%S') skip $d (done)" | tee -a "$DONE"; return; }
  mkdir -p "$d"; echo "$(date '+%H:%M:%S') START oct=$oct $d" | tee -a "$DONE"
  FL_SUBNET_OCTET="$oct" timeout 3600 python -m scripts.run -c "$cfg" -o "$d" >> "$d/run.log" 2>&1
  echo "$(date '+%H:%M:%S') END rc=$? $([ -f "$d/results/report.json" ] && echo OK || echo NO_REPORT) $d" | tee -a "$DONE"
}
batch() { run_one "$1" "$2" 9 & run_one "$3" "$4" 10 & run_one "$5" "$6" 11 & wait
  echo "$(date '+%H:%M:%S') --- batch done ---" | tee -a "$DONE"; }

echo "=== H50FULL START $(date '+%F %H:%M:%S') · 3-way ===" | tee -a "$DONE"
# priority order: finish recycle, then drop (worst-case anchor), then aging
batch $C/recycle_eps03/seed41.yaml       $OUT/recycle_eps03/seed41 \
      $C/drop_eps03/seed41.yaml          $OUT/drop_eps03/seed41 \
      $C/drop_eps03/seed42.yaml          $OUT/drop_eps03/seed42
batch $C/drop_eps03/seed43.yaml          $OUT/drop_eps03/seed43 \
      $C/recycle_aging_eps03/seed41.yaml $OUT/recycle_aging_eps03/seed41 \
      $C/recycle_aging_eps03/seed42.yaml $OUT/recycle_aging_eps03/seed42
run_one $C/recycle_aging_eps03/seed43.yaml $OUT/recycle_aging_eps03/seed43 9
echo "=== H50FULL DONE $(date '+%F %H:%M:%S') ===" | tee -a "$DONE"
