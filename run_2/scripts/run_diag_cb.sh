#!/bin/bash
# Combined diagnostic + competent-baseline runner (2026-08-05). Restartable.
# Phase 1: campaigns/diag  — the 4 collapse-diagnostic 50r runs (specs tier order)
# Phase 2: campaigns/cb    — competent-baseline triptych (20r n=6 first, then 50r n=3)
# All concurrent 3-way (FL_SUBNET_OCTET 9/10/11). 50r runs need a longer timeout.
set +e
cd ~/fl-framework && source .venv/bin/activate

run_one() { # $1=cfgpath $2=outdir $3=octet $4=timeout $5=done-file
  local cfg="$1" d="$2" oct="$3" to="$4" done_f="$5"
  if [ -f "$d/results/report.json" ]; then echo "$(date '+%H:%M:%S') skip $d (done)"; return; fi
  mkdir -p "$d"
  docker rm -f "node-0-o$oct" "node-1-o$oct" "node-2-o$oct" "node-3-o$oct" "monitor-o$oct" >/dev/null 2>&1
  echo "$(date '+%H:%M:%S') START oct=$oct $d"
  FL_SUBNET_OCTET="$oct" timeout "$to" python -m scripts.run -c "$cfg" -o "$d" >> "$d/run.log" 2>&1
  local rc=$?
  local ok; ok=$([ -f "$d/results/report.json" ] && echo OK || echo NO_REPORT)
  echo "$(date '+%H:%M:%S') END   rc=$rc $ok $d" | tee -a "$done_f"
  [ "$ok" = "OK" ] && rm -rf "$d/data"
}

run_specs() { # $1=specs $2=outbase $3=timeout $4=done-file
  local specs="$1" outbase="$2" to="$3" done_f="$4"
  mapfile -t ROWS < <(awk -F'\t' '$1=="concurrent"' "$specs")
  local i=0 n=${#ROWS[@]}
  while [ $i -lt $n ]; do
    for oct in 9 10 11; do
      [ $i -lt $n ] || break
      IFS=$'\t' read -r m t arm seed cfg <<< "${ROWS[$i]}"
      run_one "$cfg" "$outbase/$arm/seed$seed" "$oct" "$to" "$done_f" &
      i=$((i+1))
    done
    wait
  done
}

DIAG_DONE=campaigns/diag/DIAG.DONE
CB_DONE=campaigns/cb/CB.DONE
mkdir -p campaigns/diag campaigns/cb
: > "$DIAG_DONE"; : > "$CB_DONE"

echo "=== DIAG START $(date '+%F %H:%M:%S') ===" | tee -a "$DIAG_DONE"
run_specs configs/experiments/diag/specs.txt campaigns/diag 6000 "$DIAG_DONE"
echo "=== DIAG DONE $(date '+%F %H:%M:%S') ===" | tee -a "$DIAG_DONE"

echo "=== CB START $(date '+%F %H:%M:%S') ===" | tee -a "$CB_DONE"
run_specs configs/experiments/cb/specs.txt campaigns/cb 6000 "$CB_DONE"
echo "=== CB DONE $(date '+%F %H:%M:%S') ===" | tee -a "$CB_DONE"
