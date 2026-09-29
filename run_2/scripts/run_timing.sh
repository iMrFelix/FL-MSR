#!/bin/bash
# Timing re-measurement runner (2026-08-05, audit NT-01/02/03/05). Restartable.
# SERIAL by design: one stack at a time on octet 0, so no co-tenancy compute
# contention leaks into the quantity being measured.
set +e
cd ~/fl-framework && source .venv/bin/activate
OUT=campaigns/timing
SPECS=configs/experiments/timing/specs.txt
mkdir -p "$OUT"
DONE="$OUT/TIMING.DONE"; : > "$DONE"

echo "=== TIMING START $(date '+%F %H:%M:%S') ===" | tee -a "$DONE"
while IFS=$'\t' read -r mode tier arm seed cfg; do
  d="$OUT/$arm/seed$seed"
  if [ -f "$d/results/report.json" ]; then echo "$(date '+%H:%M:%S') skip $d"; continue; fi
  mkdir -p "$d"
  docker rm -f node-0 node-1 node-2 node-3 monitor >/dev/null 2>&1
  echo "$(date '+%H:%M:%S') START $d"
  FL_SUBNET_OCTET=0 timeout 3000 python -m scripts.run -c "$cfg" -o "$d" >> "$d/run.log" 2>&1
  rc=$?
  ok=$([ -f "$d/results/report.json" ] && echo OK || echo NO_REPORT)
  echo "$(date '+%H:%M:%S') END   rc=$rc $ok $d" | tee -a "$DONE"
  [ "$ok" = "OK" ] && rm -rf "$d/data"
done < "$SPECS"
echo "=== TIMING DONE $(date '+%F %H:%M:%S') ===" | tee -a "$DONE"
