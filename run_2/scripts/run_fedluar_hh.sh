#!/bin/bash
# fedluar_hh runner (2026-08-05): FedLUAR-native vs ImpRoute head-to-head at
# matched communication budgets.  Restartable, 3-way concurrent.
#
# Runs from an ISOLATED tree (~/fl-framework-fedluar) so the src/ patch that
# adds skip_feedback={fedluar_random,fedluar_cyclic} can never reach the
# campaigns running out of ~/fl-framework.  The venv and the keras CIFAR
# cache are user-level and shared; `src` is not pip-installed, so `import src`
# resolves from cwd — this tree.  Docker images are rebuilt from THIS tree by
# scripts.run on every run (fl-node:latest is a shared tag; every campaign
# runner rebuilds it from its own tree, so the tag is self-correcting).
#
# 81 runs x 20 rounds; measured ~11 min per 3-way batch -> ~5 h.
# specs.txt is tier-ordered, so an interrupted night still lands tier 0
# (the powered n=6 headline four-way at Comm ~= 0.500) first.
set +e
TREE="${FL_HH_TREE:-$HOME/fl-framework-fedluar}"
cd "$TREE" || { echo "FATAL: no tree at $TREE"; exit 1; }
source "${FL_HH_VENV:-$HOME/fl-framework/.venv}/bin/activate" || {
  echo "FATAL: no venv"; exit 1; }

OUT=campaigns/fedluar_hh
SPECS=configs/experiments/fedluar/specs.txt
mkdir -p "$OUT"
DONE="$OUT/FEDLUAR_HH.DONE"; : > "$DONE"

run_one() {
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

echo "=== FEDLUAR_HH START $(date '+%F %H:%M:%S') tree=$TREE ===" | tee -a "$DONE"
echo "=== specs: $(wc -l < "$SPECS") rows ===" | tee -a "$DONE"
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
echo "=== FEDLUAR_HH DONE $(date '+%F %H:%M:%S') ===" | tee -a "$DONE"
