#!/bin/bash
# Stability-sweep runner (2026-08-05). Restartable. 3-way concurrent, tier order.
set +e
cd ~/fl-framework && source .venv/bin/activate
OUT=campaigns/w3verify
SPECS=configs/experiments/w3verify/specs.txt
mkdir -p "$OUT"
DONE="$OUT/W3VERIFY.DONE"; : > "$DONE"

# CIFAR cache warm-up — MANDATORY before any concurrent batch.
# Concurrent stacks calling tf.keras.datasets.cifar10.load_data() race on the
# keras cache: the loud failure is `pickle data was truncated` (observed 3x in
# this campaign), and the silent failure is a run training on partially
# materialised data — which is how campaigns/w3verify_corrupt_20260805 acquired a
# run whose output could not be reproduced. Warming single-process first makes
# the extraction happen exactly once. Pattern taken from run_h50_full.sh:17-18.
echo "$(date '+%H:%M:%S') warming CIFAR cache (single process)..." | tee -a "$DONE"
python -c "import tensorflow as tf; tf.keras.datasets.cifar10.load_data(); print('cifar ok')" >> "$DONE" 2>&1
echo "$(date '+%H:%M:%S') cache warm complete" | tee -a "$DONE"

run_one() {
  local cfg="$1" d="$2" oct="$3"
  if [ -f "$d/results/report.json" ]; then echo "$(date '+%H:%M:%S') skip $d (done)"; return; fi
  mkdir -p "$d"
  docker rm -f "node-0-o$oct" "node-1-o$oct" "node-2-o$oct" "node-3-o$oct" "monitor-o$oct" >/dev/null 2>&1
  # Remove this run's compose network if a previous generation left it behind.
  # Symptom otherwise: rc=1 within ~7s because the project's network already
  # exists with an incompatible subnet (observed 2026-08-06 01:11 on two cells).
  # Project name is <arm>-<seed> (src/launcher.py:_project_name).
  local proj; proj="$(basename "$(dirname "$d")")-$(basename "$d")"
  docker network rm "${proj}_fl-net" >/dev/null 2>&1
  echo "$(date '+%H:%M:%S') START oct=$oct $d"
  FL_SUBNET_OCTET="$oct" timeout 3000 python -m scripts.run -c "$cfg" -o "$d" >> "$d/run.log" 2>&1
  local rc=$?
  local ok; ok=$([ -f "$d/results/report.json" ] && echo OK || echo NO_REPORT)
  echo "$(date '+%H:%M:%S') END   rc=$rc $ok $d" | tee -a "$DONE"
  # Record the materialised partition's fingerprint BEFORE reclaiming the data
  # dir. Silent shard corruption (duplicate images with identical labels) has
  # been observed under concurrent launches and is otherwise undetectable after
  # the fact — the KB's proposed instrument, now mandatory. Cheap: 4 hashes.
  if [ "$ok" = "OK" ] && [ -d "$d/data" ]; then
    python - "$d" <<'PY' >> "$d/results/data_digests.json" 2>/dev/null
import hashlib, json, sys, glob
import numpy as np
run = sys.argv[1]
out = {}
for p in sorted(glob.glob(f"{run}/data/node-*.npz")):
    d = np.load(p)
    x, y = d["x_train"], d["y_train"]
    flat = x.reshape(len(x), -1)
    out[p.split("/")[-1]] = {
        "x_sha256": hashlib.sha256(np.ascontiguousarray(x)).hexdigest(),
        "y_sha256": hashlib.sha256(np.ascontiguousarray(y)).hexdigest(),
        "n": int(len(x)),
        "duplicate_images": int(len(x) - len(np.unique(flat, axis=0))),
    }
print(json.dumps(out, indent=1))
PY
  fi
  [ "$ok" = "OK" ] && rm -rf "$d/data"
}

echo "=== W3VERIFY START $(date '+%F %H:%M:%S') ===" | tee -a "$DONE"
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
echo "=== W3VERIFY DONE $(date '+%F %H:%M:%S') ===" | tee -a "$DONE"
