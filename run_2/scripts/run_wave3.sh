#!/bin/bash
# Wave-3 campaign runner (2026-07-23 pre-meeting sprint). Restartable
# (skips any run with a report.json). Reads configs/experiments/wave3/specs.txt:
#   mode<TAB>tier<TAB>arm<TAB>seed<TAB>cfgpath
# Phase 1: SERIAL timing arms get the whole box (mono first, then byte_balanced)
#          — no co-tenancy, so t_eps / wire timing is clean.
# Phase 2: CONCURRENT accuracy arms run 3-way (FL_SUBNET_OCTET 9/10/11), tier order.
# Output tree: campaigns/w3/<arm>/seed<seed>/.  Per-run data dir is deleted after
# the report is captured to reclaim disk (each carries a duplicated 10k test set).
set +e
cd ~/fl-framework && source .venv/bin/activate
OUT=campaigns/w3
SPECS=configs/experiments/wave3/specs.txt
mkdir -p "$OUT"
DONE="$OUT/WAVE3.DONE"; : > "$DONE"

run_one() { # $1=cfgpath $2=outdir $3=octet
  local cfg="$1" d="$2" oct="$3"
  if [ -f "$d/results/report.json" ]; then echo "$(date '+%H:%M:%S') skip $d (done)"; return; fi
  mkdir -p "$d"
  # Clear any stale containers for this octet (timed-out prior run) before launch.
  if [ "$oct" = "0" ]; then
    docker rm -f node-0 node-1 node-2 node-3 monitor >/dev/null 2>&1
  else
    docker rm -f "node-0-o$oct" "node-1-o$oct" "node-2-o$oct" "node-3-o$oct" "monitor-o$oct" >/dev/null 2>&1
  fi
  echo "$(date '+%H:%M:%S') START oct=$oct $d"
  FL_SUBNET_OCTET="$oct" timeout 3000 python -m scripts.run -c "$cfg" -o "$d" >> "$d/run.log" 2>&1
  local rc=$?
  local ok; ok=$([ -f "$d/results/report.json" ] && echo OK || echo NO_REPORT)
  echo "$(date '+%H:%M:%S') END   rc=$rc $ok $d" | tee -a "$DONE"
  # Reclaim the big per-run data dir (4x duplicated test set) once captured.
  [ "$ok" = "OK" ] && rm -rf "$d/data"
}

echo "=== WAVE3 START $(date '+%F %H:%M:%S') ===" | tee -a "$DONE"

# ---- Phase 1: serial timing arms (mono first, then byte_balanced) ----
awk -F'\t' '$1=="serial" && $3=="mono"'   "$SPECS" | while IFS=$'\t' read -r m t arm seed cfg; do
  run_one "$cfg" "$OUT/$arm/seed$seed" 0
done
awk -F'\t' '$1=="serial" && $3!="mono"'   "$SPECS" | while IFS=$'\t' read -r m t arm seed cfg; do
  run_one "$cfg" "$OUT/$arm/seed$seed" 0
done

# ---- Phase 2: concurrent accuracy arms, 3-way by octet, in tier order ----
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

echo "=== WAVE3 DONE $(date '+%F %H:%M:%S') ===" | tee -a "$DONE"
