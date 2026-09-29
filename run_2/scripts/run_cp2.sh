#!/bin/bash
# CP2 attribution-control campaign runner (2026-08-05). Restartable (skips any
# run with a report.json). Reads configs/experiments/cp2/specs.txt:
#   mode<TAB>tier<TAB>arm<TAB>seed<TAB>cfgpath
# All arms are accuracy-only -> 3-way concurrent (FL_SUBNET_OCTET 9/10/11),
# tier order (tier 0 = determinism sentinels first).
# Output tree: campaigns/cp2/<arm>/seed<seed>/.
set +e
cd ~/fl-framework && source .venv/bin/activate
OUT=campaigns/cp2
SPECS=configs/experiments/cp2/specs.txt
mkdir -p "$OUT"
DONE="$OUT/CP2.DONE"; : > "$DONE"

run_one() { # $1=cfgpath $2=outdir $3=octet
  local cfg="$1" d="$2" oct="$3"
  if [ -f "$d/results/report.json" ]; then echo "$(date '+%H:%M:%S') skip $d (done)"; return; fi
  mkdir -p "$d"
  docker rm -f "node-0-o$oct" "node-1-o$oct" "node-2-o$oct" "node-3-o$oct" "monitor-o$oct" >/dev/null 2>&1
  echo "$(date '+%H:%M:%S') START oct=$oct $d"
  FL_SUBNET_OCTET="$oct" timeout 3000 python -m scripts.run -c "$cfg" -o "$d" >> "$d/run.log" 2>&1
  local rc=$?
  local ok; ok=$([ -f "$d/results/report.json" ] && echo OK || echo NO_REPORT)
  echo "$(date '+%H:%M:%S') END   rc=$rc $ok $d" | tee -a "$DONE"
  [ "$ok" = "OK" ] && rm -rf "$d/data"
}

echo "=== CP2 START $(date '+%F %H:%M:%S') ===" | tee -a "$DONE"

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

echo "=== CP2 DONE $(date '+%F %H:%M:%S') ===" | tee -a "$DONE"
