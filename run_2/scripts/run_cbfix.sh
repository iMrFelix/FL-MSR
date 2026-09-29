#!/bin/bash
# cbfix runner (2026-08-06). Restartable. 3-way concurrent.
# 18 runs = 3 arms (mono/drop/recycle) x 6 seeds x 20 rounds.
# Replaces campaigns/cb/recycle_eps03_r20 (quarantine table, kb/10 §10.4).
set +e
# Run from THIS script's own tree, not a hardcoded ~/fl-framework.
# src/launcher.py:build_images rebuilds fl-node:latest from the launching tree
# on every run, so the tree a campaign is launched from IS the code it runs —
# hardcoding the path would silently run someone else's src.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" && source .venv/bin/activate

# ---------------------------------------------------------------------------
# DEPLOY-TREE PRECONDITION — assert, do not assume.
# ---------------------------------------------------------------------------
# moltres's tree did NOT carry the partition fix while the stab census was
# running (syncing src mid-campaign would rebuild fl-node:latest under the live
# census). So this runner can be dropped onto a tree that predates the fix.
#
# The dangerous case is a PARTIAL sync: if scripts/prematerialize.py arrives but
# src/launcher.py does not, the serial gate still runs, FL_REQUIRE_PREMATERIALIZED
# becomes a silent no-op, and a mid-campaign shard rejection re-materialises
# concurrently — the exact back door the fix closed, reopened by a deployment
# accident and invisible in the logs. A missing prematerialize.py fails loudly on
# its own; a missing launcher fix does NOT. Hence this check.
#
# Prevention over detection, applied to our own deployment.
for _req in scripts/prematerialize.py src/launcher.py; do
  [ -f "$_req" ] || { echo "ABORT: $_req missing — tree predates the partition fix. Full sync + rebuild first."; exit 1; }
done
for _sym in _existing_partitions FL_REQUIRE_PREMATERIALIZED; do
  grep -q "$_sym" src/launcher.py || {
    echo "ABORT: src/launcher.py lacks '$_sym'."
    echo "  This tree predates the partition fix, so the serial gate below would"
    echo "  run but its enforcement would be a NO-OP. Full src/tests/scripts/configs"
    echo "  sync + image rebuild, then relaunch. Refusing to start."
    exit 1
  }
done
# D9 — prefer a BEHAVIOURAL check over the textual one above: the greps prove
# the symbols are present, the tests prove the guard actually refuses. Run them
# when pytest exists; fall back to the textual result otherwise, because a
# missing pytest on a deploy box must not block a launch whose precondition is
# already satisfied. Report which check was used — an unstated fallback is how
# a weaker guarantee gets mistaken for a stronger one.
if python -c "import pytest" >/dev/null 2>&1; then
  if python -m pytest tests/test_partition_reuse.py -q >/tmp/cbfix_guard_tests.log 2>&1; then
    echo "$(date '+%H:%M:%S') deploy-tree precondition OK (BEHAVIOURAL: partition-reuse guard tests pass)"
  else
    echo "ABORT: tests/test_partition_reuse.py FAILED on this tree — the partition"
    echo "  guard does not behave correctly here even though its symbols are present."
    tail -15 /tmp/cbfix_guard_tests.log
    exit 1
  fi
else
  echo "$(date '+%H:%M:%S') deploy-tree precondition OK (TEXTUAL only — pytest absent, guard behaviour unverified)"
fi
OUT=campaigns/cbfix
SPECS=configs/experiments/cbfix/specs.txt
mkdir -p "$OUT"
DONE="$OUT/CBFIX.DONE"; : > "$DONE"

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
# do not launch at all. At the observed corruption rate an 18-run campaign
# expects ~2 silent casualties, and this campaign exists to REPLACE runs that
# a different silent defect already cost us — re-running it dirty would be the
# kb/11 11.6 lesson for the third time in one night.
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
  FL_SUBNET_OCTET="$oct" timeout 9000 python -m scripts.run -c "$cfg" -o "$d" >> "$d/run.log" 2>&1
  local rc=$?
  local ok; ok=$([ -f "$d/results/report.json" ] && echo OK || echo NO_REPORT)
  echo "$(date '+%H:%M:%S') END   rc=$rc $ok $d" | tee -a "$DONE"
  # Fingerprint the materialised partition BEFORE reclaiming data/.
  # zero_rows is the AUTHORITATIVE corruption detector (kb/13 Sec 13.5): it is
  # arm-independent and works at n=1, unlike comparing a hash against sibling
  # runs, which needs a majority of clean siblings and mis-called cells in both
  # directions on fedluar_hh where that support was thin.
  if [ "$ok" = "OK" ] && [ -d "$d/data" ]; then
    # D4 — stderr goes to a FILE, never /dev/null. Silencing the corruption
    # detector's own failures means a crashed digest step is indistinguishable
    # from a clean run: data_digests.json ends up absent or truncated, the
    # analyzer's generation filter then treats the cell as prior-generation,
    # and it is silently dropped from the census instead of being flagged.
    # A detector that can fail quietly is not a detector.
    python - "$d" <<'PY' >> "$d/results/data_digests.json" 2>>"$d/results/data_digests.err"
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
    if [ -s "$d/results/data_digests.err" ]; then
      echo "$(date '+%H:%M:%S') *** DIGEST STEP FAILED for $d — integrity UNVERIFIED, treat as suspect" | tee -a "$DONE"
      head -3 "$d/results/data_digests.err" | tee -a "$DONE"
    fi
    # R2 — an EMPTY .json is the failure D4 cannot see. A signal-killed digest
    # (OOM, rc=137) leaves BOTH .err and .json empty: the shell's "Killed" goes
    # to the runner's stderr, not to .err, so the loud line above never fires
    # and the cell looks like a clean run with no digest. The analyzer's
    # generation filter then drops it as prior-generation — silently.
    if [ ! -s "$d/results/data_digests.json" ]; then
      echo "$(date '+%H:%M:%S') *** DIGEST STEP PRODUCED NOTHING for $d (killed?) — integrity UNVERIFIED, treat as suspect" | tee -a "$DONE"
    fi
    if grep -q '"corrupt": true' "$d/results/data_digests.json" 2>/dev/null; then
      echo "$(date '+%H:%M:%S') *** CORRUPT PARTITION $d — EXCLUDE FROM ANALYSIS" | tee -a "$DONE"
    fi
  fi
  [ "$ok" = "OK" ] && rm -rf "$d/data"
}

echo "=== CBFIX START $(date '+%F %H:%M:%S') ===" | tee -a "$DONE"
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
echo "=== CBFIX DONE $(date '+%F %H:%M:%S') ===" | tee -a "$DONE"
