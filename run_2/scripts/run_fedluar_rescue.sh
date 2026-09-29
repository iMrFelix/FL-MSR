#!/bin/bash
# fedluar_hh rescue re-run (staged 2026-08-06, executor session).
#
# *** DO NOT LAUNCH without: Felix's go + verifier-session go + a box
# *** assignment. Staged only, per the executor contract (T6).
#
# Re-runs the 15 cells that campaigns/fedluar_hh_partition_gate.json marks
# CORRUPT (11) or UNVERIFIED (4). The originals ran 3-way CONCURRENT — the
# only launch mode that has ever produced the silent all-zero-row partition
# corruption (writeup/23 s2). This runner is strictly SERIAL (one stack,
# octet 9), matching run_n10pilot.sh, and adds the integrity instrumentation
# the original lacked: CIFAR cache warm-up, partition hashing, and a
# zero-row scan BEFORE data/ is reclaimed. A corrupt materialisation keeps
# its data/ as evidence (never delete — evidence-preservation rule).
#
# Must run FROM the isolated tree (the fedluar_*/luarand_*/luarcyc_* arms
# need the skip_feedback src patch that exists only there):
#   TREE default ~/fl-framework-fedluar; venv is the shared user-level one.
# Launch procedure (after the gate opens) — FULL-TREE, never partial
# (standing rule; a partial sync killed 14 runs once, and the pre-fd23891
# launcher would silently re-materialise every run, voiding the manifest):
#   1. replicate the FULL isolated tree (src/ tests/ scripts/ configs/)
#      from the sha-verified launching configuration to the assigned box;
#      verify: sha256(src/launcher.py) matches the round-2-verified tree
#      AND grep FL_REQUIRE_PREMATERIALIZED src/launcher.py is non-empty
#      AND scripts/prematerialize.py exists
#   2. overlay configs/experiments/fedluar_rescue/ + this script; bash -n
#   3. rebuild images (or rely on scripts.run's per-run rebuild from the
#      launching tree), then: nohup bash run_fedluar_rescue.sh &
#   4. laptop side: incremental_pull.sh + watchdog per standing rule.
# [2026-08-06 ~04:15 state: steps 1-2 already executed on snorlax;
#  launcher sha 112d52a4… verified identical to charizard's launching tree]
# Output: $TREE/campaigns/fedluar_rescue/<arm>/seed<N> — deliberately a NEW
# campaign dir; the quarantined fedluar_hh originals stay untouched.
set +e
TREE="${FL_HH_TREE:-$HOME/fl-framework-fedluar}"
cd "$TREE" || { echo "FATAL: no tree at $TREE"; exit 1; }
source "${FL_HH_VENV:-$HOME/fl-framework/.venv}/bin/activate" || {
  echo "FATAL: no venv"; exit 1; }

OUT=campaigns/fedluar_rescue
CFG=configs/experiments/fedluar_rescue
mkdir -p "$OUT"
DONE="$OUT/FEDLUAR_RESCUE.DONE"; : > "$DONE"
echo "tree_sha=$(git rev-parse --short HEAD 2>/dev/null) serial=1 hash_and_zeroscan_before_delete=1" | tee -a "$DONE"

# 15 rescue cells: the CORRUPT + UNVERIFIED set from
# campaigns/fedluar_hh_partition_gate.json (2026-08-06).
CELLS="
cyclic_k7/seed41
fedluar_d6/seed41
fedluar_d7/seed41
fedluar_d7/seed43
fedluar_d7/seed45
fedluar_d7/seed46
fedluar_d9/seed41
improute_tau2/seed41
improute_tau5/seed41
improute_tau5/seed43
improute_tau8/seed41
luarand_d11/seed41
luarand_d9/seed41
luarand_d9/seed43
luarcyc_d7/seed46
"

echo "$(date '+%H:%M:%S') warming CIFAR cache (single process)..." | tee -a "$DONE"
python -c "import tensorflow as tf; tf.keras.datasets.cifar10.load_data(); print('cifar ok')" >> "$DONE" 2>&1
echo "$(date '+%H:%M:%S') cache warm complete" | tee -a "$DONE"

# VIRGIN CHECK (fd23891 verifier condition): partition adoption carries no
# partition-params fingerprint, so a stale data/ dir of unknown origin would
# be silently adopted. Abort if the output root already holds data dirs
# without a manifest (unknown provenance). A root WITH a manifest is a
# legitimate restart — the gate below re-validates it shard by shard.
if [ -d "$OUT" ] && [ ! -f "$OUT/PARTITION_MANIFEST.json" ] \
   && ls "$OUT"/*/seed*/data/node-*.npz >/dev/null 2>&1; then
  echo "FATAL: $OUT contains data dirs of unknown provenance (no manifest) — refusing to launch" | tee -a "$DONE"
  exit 1
fi

# HARD GATE (fd23891 pattern): materialise all 15 partitions serially in ONE
# process, zero-scan + hash each shard, write PARTITION_MANIFEST.json. Runs
# reuse the verified partitions (launcher _existing_partitions). Nonzero exit
# means a corrupt shard was produced even serially — do not launch, escalate.
# NB: never run this while another campaign is materialising on the same box
# (that concurrency is the corruption mechanism itself).
echo "$(date '+%H:%M:%S') prematerialize gate (serial, verified)..." | tee -a "$DONE"
python -m scripts.prematerialize "$CFG/specs.txt" "$OUT" >> "$DONE" 2>&1
gaterc=$?
if [ "$gaterc" -ne 0 ]; then
  echo "$(date '+%H:%M:%S') FATAL: prematerialize gate failed rc=$gaterc — NOT LAUNCHING" | tee -a "$DONE"
  exit 1
fi
echo "$(date '+%H:%M:%S') gate passed: all partitions verified clean" | tee -a "$DONE"

# Whole-file sha snapshot of every gate-materialised shard. Per-run hashes
# are diffed against this to detect the verifier's back door: silent
# regeneration after adoption-time rejection. (Deterministic clean regen
# would be byte-identical and pass — corruption-bearing regen won't.)
(cd "$OUT" && sha256sum */seed*/data/node-*.npz > PREMATERIALIZED_SHA256.txt)
echo "$(date '+%H:%M:%S') sha snapshot: $(wc -l < "$OUT/PREMATERIALIZED_SHA256.txt") shards" | tee -a "$DONE"

# Hard-fail mode (round-2-verified fix; same pattern as run_fedluar_late.sh):
# if a run's gated shards go missing/corrupt after this point, scripts.run
# must FAIL LOUDLY rather than quietly re-materialise — inside any concurrent
# context that re-materialisation IS the keras-cache race. Requires the
# synced launcher (sha-verified identical to the round-2-verified tree).
export FL_REQUIRE_PREMATERIALIZED=1
echo "hard-fail mode: FL_REQUIRE_PREMATERIALIZED=1" | tee -a "$DONE"

zeroscan() {  # $1 = run dir; writes results/ZERO_SCAN.txt + results/data_digests.json
              # (standard schema so scripts/verify_campaign_integrity.py can
              # gate rescue cells like any other campaign); echoes CLEAN|CORRUPT
  python - "$1" <<'EOF'
import sys, os, glob, json, hashlib
import numpy as np
d = sys.argv[1]
lines, bad, digests = [], 0, {}
for p in sorted(glob.glob(os.path.join(d, "data", "node-*.npz"))):
    with np.load(p) as z:
        x, y = z["x_train"], z["y_train"]
        zt = int(np.count_nonzero(~x.reshape(len(x), -1).any(axis=1))) if len(x) else 0
        nt = len(x)
        zv = nv = 0
        if "x_val" in z.files:
            v = z["x_val"]
            zv = int(np.count_nonzero(~v.reshape(len(v), -1).any(axis=1))) if len(v) else 0
            nv = len(v)
        # hashes identical in construction to scripts/prematerialize.py:scan()
        digests[os.path.basename(p)] = {
            "x_sha256": hashlib.sha256(np.ascontiguousarray(x)).hexdigest(),
            "y_sha256": hashlib.sha256(np.ascontiguousarray(y)).hexdigest(),
            "n": nt, "zero_rows": zt + zv,
        }
    bad += zt + zv
    lines.append(f"{os.path.basename(p)} zero_train={zt}/{nt} zero_val={zv}/{nv}")
os.makedirs(os.path.join(d, "results"), exist_ok=True)
verdict = "CORRUPT" if bad else ("CLEAN" if lines else "NO_DATA")
with open(os.path.join(d, "results", "ZERO_SCAN.txt"), "w") as f:
    f.write("\n".join(lines + [f"verdict={verdict}"]) + "\n")
with open(os.path.join(d, "results", "data_digests.json"), "w") as f:
    json.dump(digests, f, indent=1, sort_keys=True)
print(verdict)
EOF
}

run_one() {
  local cfg="$1" d="$2"
  if [ -f "$d/results/report.json" ]; then echo "$(date '+%H:%M:%S') skip $d (done)" | tee -a "$DONE"; return; fi
  mkdir -p "$d"
  docker rm -f node-0-o9 node-1-o9 node-2-o9 node-3-o9 monitor-o9 >/dev/null 2>&1
  echo "$(date '+%H:%M:%S') START $d" | tee -a "$DONE"
  FL_SUBNET_OCTET=9 timeout 3000 python -m scripts.run -c "$cfg" -o "$d" >> "$d/run.log" 2>&1
  local rc=$?
  local integ="NO_DATA"
  if [ -d "$d/data" ]; then
    mkdir -p "$d/results"
    (cd "$d/data" && sha256sum node-*.npz > "../results/PARTITION_SHA256.txt" 2>/dev/null)
    integ=$(zeroscan "$d" | tail -1)
  fi
  local ok; ok=$([ -f "$d/results/report.json" ] && echo OK || echo NO_REPORT)
  # regen detection (fd23891 verifier condition): per-shard sha must match
  # the gate-time snapshot, and run.log must not mention regeneration.
  local regen="REGEN_NONE"
  if [ -f "$d/results/PARTITION_SHA256.txt" ]; then
    while read -r sha f; do
      grep -q "$sha  ${d#"$OUT"/}/data/$f" "$OUT/PREMATERIALIZED_SHA256.txt" || regen="REGEN_MISMATCH"
    done < "$d/results/PARTITION_SHA256.txt"
  fi
  grep -qi "regenerat" "$d/run.log" 2>/dev/null && regen="${regen}+REGEN_LOGGED"
  echo "$(date '+%H:%M:%S') END   rc=$rc $ok integrity=$integ $regen $d" | tee -a "$DONE"
  # Default: KEEP data/ (target box snorlax has headroom; these 15 cells are
  # exactly where physical evidence matters). Reclaim only if explicitly
  # asked (FL_RESCUE_RECLAIM=1), and even then never for corrupt runs nor
  # for any run whose shards moved or whose log mentions regeneration.
  if [ "${FL_RESCUE_RECLAIM:-0}" = "1" ] && [ "$ok" = "OK" ] \
     && [ "$integ" = "CLEAN" ] && [ "$regen" = "REGEN_NONE" ]; then
    rm -rf "$d/data"
  fi
}

echo "=== FEDLUAR_RESCUE START $(date '+%F %H:%M:%S') tree=$TREE ===" | tee -a "$DONE"
for cell in $CELLS; do
  run_one "$CFG/${cell%/*}/${cell#*/}.yaml" "$OUT/$cell"
done
echo "=== FEDLUAR_RESCUE DONE $(date '+%F %H:%M:%S') ===" | tee -a "$DONE"
