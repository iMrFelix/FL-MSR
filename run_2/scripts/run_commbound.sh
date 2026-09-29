#!/bin/bash
# h150 late-stage campaign runner (2026-08-05). Restartable. 3-way concurrent.
# 150r/run: timeout 9000s (~2.8h guard; measured ~20s/round + overhead).
set +e
cd ~/fl-framework && source .venv/bin/activate
OUT=campaigns/commbound
SPECS=configs/experiments/commbound/specs.txt
mkdir -p "$OUT"
DONE="$OUT/COMMBOUND.DONE"; : > "$DONE"

run_one() {
  local cfg="$1" d="$2" oct="$3"
  if [ -f "$d/results/report.json" ]; then echo "$(date '+%H:%M:%S') skip $d (done)"; return; fi
  mkdir -p "$d"
  docker rm -f "node-0-o$oct" "node-1-o$oct" "node-2-o$oct" "node-3-o$oct" "monitor-o$oct" >/dev/null 2>&1
  echo "$(date '+%H:%M:%S') START oct=$oct $d"
  FL_SUBNET_OCTET="$oct" timeout 9000 python -m scripts.run -c "$cfg" -o "$d" >> "$d/run.log" 2>&1
  local rc=$?
  local ok; ok=$([ -f "$d/results/report.json" ] && echo OK || echo NO_REPORT)
  echo "$(date '+%H:%M:%S') END   rc=$rc $ok $d" | tee -a "$DONE"
  [ "$ok" = "OK" ] && rm -rf "$d/data"
}

echo "=== COMMBOUND START $(date '+%F %H:%M:%S') ===" | tee -a "$DONE"
mapfile -t ROWS < <(awk -F'\t' '$1=="concurrent"' "$SPECS")
i=0; n=${#ROWS[@]}
while [ $i -lt $n ]; do
  for oct in 9 10 11; do
    [ $i -lt $n ] || break
    IFS=$'\t' read -r m t arm seed cfg <<< "${ROWS[$i]}"
    run_one "$cfg" "$OUT/$arm/seed$seed" "$oct" &
    i=$((i+1))
  done
  wait
done
echo "=== COMMBOUND DONE $(date '+%F %H:%M:%S') ===" | tee -a "$DONE"
