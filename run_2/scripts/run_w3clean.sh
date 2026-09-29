#!/bin/bash
# Clean re-run of the wave-3 core accuracy table + pooled renormalize (2026-08-06).
# Purpose: replace the corruption-clouded July reference with verifiable data.
# Design decisions (pre-registered):
#   - STRICTLY SERIAL: one run at a time. Every confirmed silent-corruption
#     instance came from 3-way-concurrent launches; serial runs have never
#     corrupted. This also removes the serial-mono / concurrent-treatment
#     launch asymmetry of the original wave-3 (audit note S21).
#   - CIFAR cache warmed single-process before any run (run_stab.sh pattern).
#   - Partition digests: sha256 of every node npz recorded to
#     <run>/data/PARTITION_SHA256.txt IMMEDIATELY after the run, and data/
#     dirs are KEPT (disk on snorlax: 94G; ~30 runs x ~1G fits) so partitions
#     stay independently verifiable.
#   - Same configs as the original wave-3 / renorm6 (untouched).
# Launched from laptop tree sha 8f262f1 (box tree is an rsync, not a git repo).
set +e
cd ~/fl-framework && source .venv/bin/activate
OUT=campaigns/w3clean
mkdir -p "$OUT"
DONE="$OUT/W3CLEAN.DONE"; : > "$DONE"
echo "laptop_sha=8f262f1 serial=1 keep_data=1" | tee -a "$DONE"

echo "$(date '+%H:%M:%S') warming CIFAR cache (single process)..." | tee -a "$DONE"
python -c "import tensorflow as tf; tf.keras.datasets.cifar10.load_data(); print('cifar ok')" >> "$DONE" 2>&1
echo "$(date '+%H:%M:%S') cache warm complete" | tee -a "$DONE"

run_one() {
  local cfg="$1" d="$2"
  if [ -f "$d/results/report.json" ]; then echo "$(date '+%H:%M:%S') skip $d (done)" | tee -a "$DONE"; return; fi
  mkdir -p "$d"
  docker rm -f node-0-o9 node-1-o9 node-2-o9 node-3-o9 monitor-o9 >/dev/null 2>&1
  echo "$(date '+%H:%M:%S') START $d" | tee -a "$DONE"
  FL_SUBNET_OCTET=9 timeout 6000 python -m scripts.run -c "$cfg" -o "$d" >> "$d/run.log" 2>&1
  local rc=$?
  if [ -d "$d/data" ]; then (cd "$d/data" && sha256sum node-*.npz > PARTITION_SHA256.txt 2>/dev/null); fi
  local ok; ok=$([ -f "$d/results/report.json" ] && echo OK || echo NO_REPORT)
  echo "$(date '+%H:%M:%S') END   rc=$rc $ok $d" | tee -a "$DONE"
}

echo "=== W3CLEAN START $(date '+%F %H:%M:%S') ===" | tee -a "$DONE"
for seed in 41 42 43 44 45 46; do
  run_one "configs/experiments/wave3/mono/seed$seed.yaml"                "$OUT/mono/seed$seed"
done
for seed in 41 42 43 44 45 46; do
  run_one "configs/experiments/wave3/drop_eps03/seed$seed.yaml"          "$OUT/drop_eps03/seed$seed"
done
for seed in 41 42 43 44 45 46; do
  run_one "configs/experiments/wave3/recycle_eps03/seed$seed.yaml"       "$OUT/recycle_eps03/seed$seed"
done
for seed in 41 42 43 44 45 46; do
  run_one "configs/experiments/wave3/recycle_aging_eps03/seed$seed.yaml" "$OUT/recycle_aging_eps03/seed$seed"
done
for seed in 41 42 43; do
  run_one "configs/experiments/wave3/renormalize_eps03/seed$seed.yaml"   "$OUT/renormalize_eps03/seed$seed"
done
for seed in 44 45 46; do
  run_one "configs/experiments/renorm6/renormalize_eps03/seed$seed.yaml" "$OUT/renormalize_eps03/seed$seed"
done
echo "=== W3CLEAN DONE $(date '+%F %H:%M:%S') ===" | tee -a "$DONE"
