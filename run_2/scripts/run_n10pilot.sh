#!/bin/bash
# 10-node pilot runner (2026-08-06, pre-meeting). SERIAL, seed-blocked.
# Question: does the accuracy cost of layer-shedding shrink with client count?
# 4 arms x 5 seeds, 20 rounds. Accuracy-only (no timing claims at 10 nodes:
# egress-only shaping cannot emulate aggregator fan-in contention).
#   - SERIAL: every confirmed silent-corruption instance came from concurrent
#     launches; serial has never corrupted. One stack, octet 9.
#   - SEED-BLOCKED: all 4 arms at seed s before moving to s+1 — each block is
#     one temporal occasion (commbound2 lesson) and yields a full paired
#     quartet early, including the eps0-vs-mono equivalence/integrity canary.
#   - Partition digests recorded to <run>/results/PARTITION_SHA256.txt, then
#     data/ is DELETED (charizard disk is tight after fedluar_hh). The digests
#     are the cross-arm integrity check: same seed => same hashes required.
set +e
cd ~/fl-framework && source .venv/bin/activate
OUT=campaigns/n10pilot
mkdir -p "$OUT"
DONE="$OUT/N10PILOT.DONE"; : > "$DONE"
echo "laptop_sha=8f262f1 serial=1 seed_blocked=1 hash_then_delete=1" | tee -a "$DONE"

echo "$(date '+%H:%M:%S') warming CIFAR cache (single process)..." | tee -a "$DONE"
python -c "import tensorflow as tf; tf.keras.datasets.cifar10.load_data(); print('cifar ok')" >> "$DONE" 2>&1
echo "$(date '+%H:%M:%S') cache warm complete" | tee -a "$DONE"

NODES="node-0-o9 node-1-o9 node-2-o9 node-3-o9 node-4-o9 node-5-o9 node-6-o9 node-7-o9 node-8-o9 node-9-o9 monitor-o9"

run_one() {
  local cfg="$1" d="$2"
  if [ -f "$d/results/report.json" ]; then echo "$(date '+%H:%M:%S') skip $d (done)" | tee -a "$DONE"; return; fi
  mkdir -p "$d"
  docker rm -f $NODES >/dev/null 2>&1
  echo "$(date '+%H:%M:%S') START $d" | tee -a "$DONE"
  FL_SUBNET_OCTET=9 timeout 6000 python -m scripts.run -c "$cfg" -o "$d" >> "$d/run.log" 2>&1
  local rc=$?
  if [ -d "$d/data" ]; then
    mkdir -p "$d/results"
    (cd "$d/data" && sha256sum node-*.npz > "../results/PARTITION_SHA256.txt" 2>/dev/null)
  fi
  local ok; ok=$([ -f "$d/results/report.json" ] && echo OK || echo NO_REPORT)
  echo "$(date '+%H:%M:%S') END   rc=$rc $ok $d" | tee -a "$DONE"
  [ "$ok" = "OK" ] && rm -rf "$d/data"
}

echo "=== N10PILOT START $(date '+%F %H:%M:%S') ===" | tee -a "$DONE"
for seed in 41 42 43 44 45; do
  for arm in mono eps0 drop_eps03 cyclic_k7; do
    run_one "configs/experiments/n10pilot/$arm/seed$seed.yaml" "$OUT/$arm/seed$seed"
  done
done
echo "=== N10PILOT DONE $(date '+%F %H:%M:%S') ===" | tee -a "$DONE"
