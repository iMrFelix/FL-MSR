#!/bin/bash
# 50-round horizon campaign runner (2026-07-23 wave 2). Reads
# configs/experiments/horizon50/specs.txt and runs WIDTH runs concurrently
# (distinct FL_SUBNET_OCTET per slot), skipping any with a report.json.
# Accuracy-only (timing certified serially in wave-3), so concurrency is safe.
# Restartable: re-running skips completed seeds.
set +e
cd ~/fl-framework && source .venv/bin/activate
OUT=campaigns/h50
SPECS=configs/experiments/horizon50/specs.txt
DONE="$OUT/HORIZON50.DONE"
WIDTH=6
mkdir -p "$OUT"; : > "$DONE"

run_one() { # $1=cfg $2=outdir $3=octet
  local cfg="$1" d="$2" oct="$3"
  if [ -f "$d/results/report.json" ]; then echo "$(date '+%H:%M:%S') skip $d (done)" | tee -a "$DONE"; return; fi
  mkdir -p "$d"
  echo "$(date '+%H:%M:%S') START oct=$oct $d" | tee -a "$DONE"
  FL_SUBNET_OCTET="$oct" timeout 5400 python -m scripts.run -c "$cfg" -o "$d" >> "$d/run.log" 2>&1
  local rc=$?; local ok; ok=$([ -f "$d/results/report.json" ] && echo OK || echo NO_REPORT)
  echo "$(date '+%H:%M:%S') END   rc=$rc $ok $d" | tee -a "$DONE"
}

echo "=== HORIZON50 START $(date '+%F %H:%M:%S') (width=$WIDTH) ===" | tee -a "$DONE"
mapfile -t LINES < "$SPECS"
i=0; n=${#LINES[@]}
while [ "$i" -lt "$n" ]; do
  oct=1
  for _ in $(seq 1 "$WIDTH"); do
    [ "$i" -lt "$n" ] || break
    IFS=$'\t' read -r mode tier arm seed cfg <<< "${LINES[$i]}"
    run_one "$cfg" "$OUT/$arm/seed$seed" "$oct" &
    oct=$((oct + 1)); i=$((i + 1))
  done
  wait
  echo "$(date '+%H:%M:%S') --- batch done ($i/$n) ---" | tee -a "$DONE"
done
echo "=== HORIZON50 DONE $(date '+%F %H:%M:%S') ===" | tee -a "$DONE"
