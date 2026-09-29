#!/bin/bash
# Serial phase-1b backbone runner (restartable). Reads specs.txt, runs each config
# sequentially to campaigns/p1/<exp>/<arm>/seed<seed>/, skipping any with a report.json.
set +e
cd ~/fl-framework && source .venv/bin/activate
OUT=campaigns/p1
SPECS=configs/experiments/phase1b/specs.txt
mkdir -p "$OUT"
echo "=== phase1b backbone start $(date '+%F %H:%M:%S') ==="
total=$(wc -l < "$SPECS"); i=0
while IFS=$'\t' read -r exp arm seed cfg; do
  i=$((i+1))
  d="$OUT/$exp/$arm/seed$seed"
  if [ -f "$d/results/report.json" ]; then echo "[$i/$total] skip $exp/$arm/seed$seed (done)"; continue; fi
  mkdir -p "$d"
  echo "[$i/$total] $(date '+%H:%M:%S') RUN $exp/$arm/seed$seed"
  timeout 2400 python -m scripts.run -c "$cfg" -o "$d" >> "$d/run.log" 2>&1
  rc=$?
  ok=$([ -f "$d/results/report.json" ] && echo OK || echo NO_REPORT)
  echo "[$i/$total] $(date '+%H:%M:%S') done rc=$rc $ok"
done < "$SPECS"
echo "=== phase1b backbone ALL DONE $(date '+%F %H:%M:%S') ==="
