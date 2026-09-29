#!/bin/bash
# fedluar_late runner (2026-08-06). Restartable. 3-way concurrent.
# 9 runs: {fedluar,luarand,luarcyc}_d7 x seeds {41,42,43} x 100 rounds.
# Answers 09 Q3b: does FedLUAR's importance metric gain signal late in training?
set +e
# Run from THIS script's own tree, not a hardcoded ~/fl-framework.
# src/launcher.py:build_images rebuilds fl-node:latest from the launching tree
# on every run, so the tree a campaign is launched from IS the code it runs —
# hardcoding the path would silently run someone else's src.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" && source .venv/bin/activate
OUT=campaigns/fedluar_late
SPECS=configs/experiments/fedluar_late/specs.txt
mkdir -p "$OUT"
DONE="$OUT/FEDLUAR_LATE.DONE"; : > "$DONE"

# CIFAR cache warm-up — MANDATORY before any concurrent batch (11 Sec 11.6).
# Concurrent stacks racing on the keras cache is the prime suspect for the
# SILENT corruption: a partially materialised shard gets several percent of its
# images replaced by ALL-ZERO rows while the labels stay byte-identical, so
# every sanity check anybody normally runs still passes. Warm single-process
# first so the extraction happens exactly once.
echo "$(date '+%H:%M:%S') warming CIFAR cache (single process)..." | tee -a "$DONE"
python -c "import tensorflow as tf; tf.keras.datasets.cifar10.load_data(); print('cifar ok')" >> "$DONE" 2>&1
echo "$(date '+%H:%M:%S') cache warm complete" | tee -a "$DONE"

# HARD GATE — materialise all 9 partitions SERIALLY before any concurrency.
# The cache warm-up alone is NOT sufficient: the corruption race re-occurred
# with the cache already staged (kb/11 §11.6). Doing the materialisation itself
# in one process removes the race instead of detecting it afterwards. The gate
# zero-row scans every shard and records its SHA256; if ANY shard is corrupt we
# do not launch at all. At the observed rate a 9-run campaign expects one
# silent casualty, and this is the campaign the FedLUAR verdict rests on.
echo "$(date '+%H:%M:%S') pre-materialising partitions serially..." | tee -a "$DONE"
python -m scripts.prematerialize "$SPECS" "$OUT" 2>&1 | tee -a "$DONE"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo "$(date '+%H:%M:%S') *** PARTITION GATE FAILED — NOT LAUNCHING ***" | tee -a "$DONE"
  exit 1
fi
echo "$(date '+%H:%M:%S') partition gate PASSED (manifest in $OUT/PARTITION_MANIFEST.json)" | tee -a "$DONE"

# From here on the partitions are materialised and verified. If any run finds
# its shards missing/corrupt anyway, it must FAIL LOUDLY rather than quietly
# re-materialising — inside a concurrent batch that re-materialisation IS the
# keras-cache race, so the back door would undo the gate above.
export FL_REQUIRE_PREMATERIALIZED=1

run_one() {
  local cfg="$1" d="$2" oct="$3"
  if [ -f "$d/results/report.json" ]; then echo "$(date '+%H:%M:%S') skip $d (done)"; return; fi
  mkdir -p "$d"
  docker rm -f "node-0-o$oct" "node-1-o$oct" "node-2-o$oct" "node-3-o$oct" "monitor-o$oct" >/dev/null 2>&1
  # Remove this run's compose network if a previous generation left it behind
  # (rc=1 within ~7s otherwise: the project's network exists with an
  # incompatible subnet). Project name is <arm>-<seed>, src/launcher.py.
  local proj; proj="$(basename "$(dirname "$d")")-$(basename "$d")"
  docker network rm "${proj}_fl-net" >/dev/null 2>&1
  echo "$(date '+%H:%M:%S') START oct=$oct $d"
  FL_SUBNET_OCTET="$oct" timeout 30000 python -m scripts.run -c "$cfg" -o "$d" >> "$d/run.log" 2>&1
  local rc=$?
  local ok; ok=$([ -f "$d/results/report.json" ] && echo OK || echo NO_REPORT)
  echo "$(date '+%H:%M:%S') END   rc=$rc $ok $d" | tee -a "$DONE"
  # Fingerprint the materialised partition BEFORE reclaiming data/.
  # zero_rows is the AUTHORITATIVE corruption detector (kb/13 Sec 13.5): it is
  # arm-independent and works at n=1, unlike comparing a hash against sibling
  # runs, which needs a majority of clean siblings and mis-called cells in both
  # directions on fedluar_hh where that support was thin.
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
    zero_rows = int((flat == 0).all(1).sum())
    out[p.split("/")[-1]] = {
        "x_sha256": hashlib.sha256(np.ascontiguousarray(x)).hexdigest(),
        "y_sha256": hashlib.sha256(np.ascontiguousarray(y)).hexdigest(),
        "n": int(len(x)),
        "zero_rows": zero_rows,
        "zero_pct": round(100.0 * zero_rows / len(x), 3),
        "duplicate_images": int(len(x) - len(np.unique(flat, axis=0))),
        "corrupt": bool(zero_rows > 0),
    }
print(json.dumps(out, indent=1))
PY
    if grep -q '"corrupt": true' "$d/results/data_digests.json" 2>/dev/null; then
      echo "$(date '+%H:%M:%S') *** CORRUPT PARTITION $d — EXCLUDE FROM ANALYSIS" | tee -a "$DONE"
    fi
  fi
  [ "$ok" = "OK" ] && rm -rf "$d/data"
}

echo "=== FEDLUAR_LATE START $(date '+%F %H:%M:%S') ===" | tee -a "$DONE"
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
echo "=== FEDLUAR_LATE DONE $(date '+%F %H:%M:%S') ===" | tee -a "$DONE"
