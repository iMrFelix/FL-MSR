#!/bin/bash
# Minimal 50r RECOVERY runner after the 6-way campaign deadlocked (2026-07-23).
# Root cause: 6 concurrent docker stacks (30 containers) deadlocked inter-container
# networking → every run hung at round ~4. Fix: 3-WAY (validated in wave-3, 45/45),
# octets 9/10/11. Meeting-critical arms first: mono + recycle, seeds 41-43.
# Output to campaigns/h50min (separate from the aborted campaigns/h50).
set +e
cd ~/fl-framework && source .venv/bin/activate
OUT=campaigns/h50min
DONE="$OUT/H50MIN.DONE"
C=configs/experiments/horizon50
mkdir -p "$OUT"; : > "$DONE"

run_one() { # cfg outdir octet
  local cfg="$1" d="$2" oct="$3"
  [ -f "$d/results/report.json" ] && { echo "$(date '+%H:%M:%S') skip $d" | tee -a "$DONE"; return; }
  mkdir -p "$d"
  echo "$(date '+%H:%M:%S') START oct=$oct $d" | tee -a "$DONE"
  FL_SUBNET_OCTET="$oct" timeout 3600 python -m scripts.run -c "$cfg" -o "$d" >> "$d/run.log" 2>&1
  echo "$(date '+%H:%M:%S') END rc=$? $([ -f "$d/results/report.json" ] && echo OK || echo NO_REPORT) $d" | tee -a "$DONE"
}

echo "=== H50MIN START $(date '+%F %H:%M:%S') · 3-way · octets 9/10/11 ===" | tee -a "$DONE"
# batch 1: mono 41-43
run_one $C/mono/seed41.yaml $OUT/mono/seed41 9  &
run_one $C/mono/seed42.yaml $OUT/mono/seed42 10 &
run_one $C/mono/seed43.yaml $OUT/mono/seed43 11 &
wait
echo "$(date '+%H:%M:%S') --- batch 1 (mono) done ---" | tee -a "$DONE"
# batch 2: recycle 41-43
run_one $C/recycle_eps03/seed41.yaml $OUT/recycle_eps03/seed41 9  &
run_one $C/recycle_eps03/seed42.yaml $OUT/recycle_eps03/seed42 10 &
run_one $C/recycle_eps03/seed43.yaml $OUT/recycle_eps03/seed43 11 &
wait
echo "=== H50MIN DONE $(date '+%F %H:%M:%S') ===" | tee -a "$DONE"
