#!/bin/bash
# =============================================================================
# collab_stage2_run.sh v2 — CPU smoke run + first GPU runs on the A100 server.
# v2 after adversarial review (writeup/drafts/review-collab-stage2.md):
# host deps fixed for Python 3.13 (B1); no keras-cache mount — data exported
# once, serially, mounted read-only (M1+M2); results bundle written on EVERY
# exit incl. aborts, all docker runs time-limited, all output logged (M3);
# determinism compared ACROSS two containers (M4); canary check tightened (M5).
#
# USAGE (from the root of the shipped tree; ~3 h — run under nohup or tmux):
#     nohup bash collab_stage2_run.sh > /dev/null 2>&1 &
#     tail -f stage2_console.log
# Skip switches: SKIP_CPU=1 / SKIP_GPU=1. Whatever happens, send us
# collab_stage2_results.tar.gz — it is written even on failure.
# =============================================================================
set -uo pipefail

# Single-instance guard (review N2): two concurrent copies would race on
# the same partitions and kill each other's containers. Loud, early.
if command -v flock >/dev/null 2>&1; then
  exec 9>/tmp/collab_stage2.lock
  flock -n 9 || { echo "another copy of this script is already running"; exit 1; }
fi

OCT=9
OUT=campaigns/collab_smoke
GOUT=gpu_probe_results
NPZ="$PWD/probe_data.npz"
TIMEOUT_RUN=5400          # one 20-round CPU federated run
TIMEOUT_TRAIN=3600        # GPU det/free runs
TIMEOUT_BENCH=2700
TIMEOUT_QUICK=1200
TIMEOUT_BUILD=3600
CPUSETS=("0-23" "24-47" "48-71" "72-95")   # GPU i <-> NUMA node i (survey)
PROBE="$(cd "$(dirname "$0")" && pwd)/collab_gpu_probe.py"
LOG="stage2_console.log"
exec > >(tee -a "$LOG") 2>&1

say()  { echo "[$(date '+%H:%M:%S')] $*"; }
pass() { say "PASS: $*"; }

BUNDLED=0
finalize() {   # runs on EVERY exit — the bundle must survive any failure
  [ "$BUNDLED" = "1" ] && return
  BUNDLED=1
  mkdir -p "$GOUT"
  { uname -a; docker --version 2>&1; date -u; } > "$GOUT/host_env.txt" 2>&1
  command -v nvidia-smi >/dev/null && timeout 30 nvidia-smi \
    --query-gpu=name,driver_version --format=csv,noheader \
    > "$GOUT/driver.txt" 2>&1
  [ -x .venv/bin/pip ] && timeout 60 .venv/bin/pip freeze \
    > "$GOUT/host-freeze.txt" 2>&1
  for img in fl-node fl-node-gpu; do
    timeout 30 docker image inspect "$img:latest" >/dev/null 2>&1 && \
      timeout 60 docker run --rm --entrypoint pip "$img:latest" freeze \
        > "$GOUT/$img-freeze.txt" 2>/dev/null
  done
  local files=("$GOUT" "$LOG")
  for f in "$OUT/PARTITION_MANIFEST.json" "$OUT"/*/seed42/results/report.json \
           "$OUT"/*/seed42/run.log; do
    [ -e "$f" ] && files+=("$f")
  done
  tar -czf collab_stage2_results.tar.gz "${files[@]}" 2>/dev/null
  say "bundle written: collab_stage2_results.tar.gz — SEND US THIS FILE"
}
trap finalize EXIT

abort() { say "ABORT: $*"; exit 1; }

run_probe() {  # run_probe <gpu-idx> <mode> <timeout> <logfile> [extra args...]
  local idx="$1" mode="$2" tmo="$3" log="$4"; shift 4
  local name="probe-$mode-g$idx-$$-$RANDOM"
  timeout -k 30 "$tmo" docker run --rm --init --name "$name" \
    --gpus "device=$idx" --cpuset-cpus "${CPUSETS[$idx]}" --cpuset-mems "$idx" \
    -v "$PROBE":/probe.py:ro -v "$NPZ":/probe_data.npz:ro \
    "$@" fl-node-gpu:latest python /probe.py "$mode" > "$log" 2>&1
  local rc=$?
  [ "$rc" -ge 124 ] && docker rm -f "$name" >/dev/null 2>&1   # timeout leftovers
  return "$rc"
}

collect() { grep "^RESULT_JSON:" "$1" | tail -1 | sed 's/^RESULT_JSON: //'; }

# ============================ P0: preflight + venv ===========================
[ -f src/launcher.py ] || abort "run me from the root of the shipped tree"
[ -f "$PROBE" ] || abort "collab_gpu_probe.py not found next to this script"
command -v docker >/dev/null || abort "docker not found"
timeout 30 docker info >/dev/null 2>&1 || abort "docker daemon not reachable"
docker compose version >/dev/null 2>&1 || abort "compose v2 missing"
FREE_GB=$(df -Pk . | awk 'NR==2 {print int($4/1048576)}')
[ "$FREE_GB" -ge 50 ] || abort "need >=50 GB free, have ${FREE_GB}"
mkdir -p "$GOUT"
pass "P0 preflight (${FREE_GB} GB free)"

# Host venv. pyproject pins protobuf<5.0, which is UNRESOLVABLE with any
# TF that has Python-3.13 wheels (review B1) — so: install runtime deps
# explicitly (letting TF pick its protobuf), then the package WITHOUT deps.
# Safe because the host-side import chain (launcher/prematerialize) never
# loads the generated protobuf modules (verified in review).
if [ ! -x .venv/bin/python ]; then
  say "P0: creating venv + installing deps (TensorFlow ~600 MB)..."
  python3 -m venv .venv || abort "venv creation failed"
  . .venv/bin/activate
  pip install --upgrade pip >/dev/null 2>&1
  pip install "tensorflow>=2.20,<3.0" "pyyaml>=6.0" "pydantic>=2.5" \
              "numpy>=1.24" "docker>=7.0" \
    || abort "host dependency install failed (see above)"
  pip install -e . --no-deps || abort "pip install -e . --no-deps failed"
else
  . .venv/bin/activate
fi
python - <<'PY' || abort "host import chain broken — rm -rf .venv, re-run, send stage2_console.log"
import tensorflow, yaml, pydantic, numpy, docker  # noqa: F401
import scripts.prematerialize                     # pulls src.launcher/schema
print("host import chain OK; TF", tensorflow.__version__)
PY
pass "P0 host venv"

# ============================ CPU STAGE ======================================
if [ "${SKIP_CPU:-0}" != "1" ]; then
  [ -e "$OUT" ] && abort "$OUT exists — must be a virgin root; move it aside"
  say "C2: building CPU images..."
  timeout -k 60 "$TIMEOUT_BUILD" docker build -t fl-node:latest \
    -f docker/Dockerfile.node . || abort "image build failed: fl-node"
  timeout -k 60 "$TIMEOUT_BUILD" docker build -t fl-monitor:latest \
    -f docker/Dockerfile.monitor . || abort "image build failed: fl-monitor"
  NETLOG=$(mktemp)
  if ! timeout 120 docker run --rm --cap-add NET_ADMIN --entrypoint true \
       --pull=never fl-node:latest > "$NETLOG" 2>&1; then
    cat "$NETLOG"
    abort "cannot grant NET_ADMIN — tell us (shaping fallback is our change)"
  fi
  rm -f "$NETLOG"
  pass "C2 images built; NET_ADMIN grantable"

  SPECS=$(mktemp); trap 'rm -f "$SPECS"; finalize' EXIT
  awk -F'\t' '$3=="mono" || $3=="eps0"' configs/experiments/cb2fix/specs.txt \
    > "$SPECS"
  [ "$(wc -l < "$SPECS")" -eq 2 ] || abort "expected 2 smoke rows"
  say "C4: CIFAR warm-up + SERIAL materialisation + integrity gate..."
  timeout 900 python -c \
    "import tensorflow as tf; tf.keras.datasets.cifar10.load_data()" \
    || abort "CIFAR warm-up failed"
  python -m scripts.prematerialize "$SPECS" "$OUT" \
    || abort "partition integrity gate FAILED — send the bundle"
  pass "C4 partitions materialised, all shards scanned clean"

  export FL_REQUIRE_PREMATERIALIZED=1
  run_one() {
    local cfg="$1" d="$2"
    [ -f "$d/results/report.json" ] && { say "skip $d (done)"; return 0; }
    mkdir -p "$d"
    docker rm -f "node-0-o$OCT" "node-1-o$OCT" "node-2-o$OCT" "node-3-o$OCT" \
                 "monitor-o$OCT" >/dev/null 2>&1
    local proj; proj="$(basename "$(dirname "$d")")-$(basename "$d")"
    docker network rm "${proj}_fl-net" >/dev/null 2>&1
    say "C5: START $d (20 rounds, ~25-30 min)"
    FL_SUBNET_OCTET="$OCT" timeout -k 60 "$TIMEOUT_RUN" \
      python -m scripts.run -c "$cfg" -o "$d" >> "$d/run.log" 2>&1
    local rc=$?
    [ -f "$d/results/report.json" ] || abort "no report (rc=$rc) — send bundle"
    [ "$rc" -eq 0 ] || abort "run rc=$rc — send bundle"
    say "C5: END rc=0 $d"
  }
  while IFS=$'\t' read -r _ _ arm seed cfg; do
    run_one "$cfg" "$OUT/$arm/seed$seed"
  done < "$SPECS"

  python - "$OUT" <<'PY' || abort "CPU verification failed (see above)"
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
def series(arm, key):
    rep = json.loads((out / arm / "seed42/results/report.json").read_text())
    return [[n[key] for _, n in sorted(r["nodes"].items())]
            for r in rep["per_round"]]
diffs = 0
lens = {}
for key in ("val_accuracy", "val_loss", "train_loss"):
    m, e = series("mono", key), series("eps0", key)
    lens[key] = (len(m), len(e))
    diffs += sum(1 for a, b in zip(m, e) if a != b)
assert all(v == (20, 20) for v in lens.values()), f"round counts {lens}"
assert diffs == 0, f"{diffs} differing round-metrics"
man = json.loads((out / "PARTITION_MANIFEST.json").read_text())
zeros = sum(s[sp]["zero_rows"] for c in man.values() for s in c.values()
            for sp in ("x_train", "x_val"))
assert zeros == 0, f"{zeros} zero rows"
print("CPU VERDICT: PASS — 20+20 rounds, 3 metrics bit-identical, 0 zero-rows")
PY
  pass "C6 CPU smoke: stack runs, bit-deterministic, partitions clean"
fi

# ============================ GPU STAGE ======================================
if [ "${SKIP_GPU:-0}" != "1" ]; then
  # Stale results from a previous run must never enter this run's verdicts
  # (review N1: a failed det run would silently pair with an old JSON).
  rm -f "$GOUT"/*.json
  # G0: export probe data ONCE, serially, on the host — then mount it ro.
  say "G0: probe dataset (export once, serially; verify if present)..."
  timeout 900 python - "$NPZ" <<'PY' || abort "probe data export/verify FAILED — delete probe_data.npz and re-run"
import hashlib, os, sys
import numpy as np
path = sys.argv[1]
if not os.path.exists(path):
    import tensorflow as tf
    (x, y), _ = tf.keras.datasets.cifar10.load_data()
    x = x.astype("float32") / 255.0
    np.savez(path + ".tmp.npz", x=x, y=y)
    os.replace(path + ".tmp.npz", path)      # atomic: never a torn file
with np.load(path) as d:                     # verify EVERY run (review N4)
    x = d["x"]
    zeros = int((x.reshape(len(x), -1) == 0).all(1).sum())
assert zeros == 0 and len(x) == 50000, f"zeros={zeros} n={len(x)}"
print("probe_data sha256:",
      hashlib.sha256(open(path, "rb").read()).hexdigest()[:16],
      f"n={len(x)} zero_rows=0")
PY
  pass "G0 probe dataset present, scanned clean"

  say "G1: building GPU image (first build downloads several GB)..."
  timeout -k 60 "$TIMEOUT_BUILD" docker build -t fl-node-gpu:latest \
    -f docker/Dockerfile.node.gpu . || abort "GPU image build failed"
  pass "G1 GPU image built"

  say "G2: GPU visibility probe..."
  NAME="probe-gpus-$$"
  timeout -k 30 600 docker run --rm --init --name "$NAME" --gpus all \
    -v "$PROBE":/probe.py:ro -e EXPECT_GPUS=4 \
    fl-node-gpu:latest python /probe.py gpus > "$GOUT/gpus.log" 2>&1 \
    || { docker rm -f "$NAME" >/dev/null 2>&1; abort "GPU visibility failed"; }
  collect "$GOUT/gpus.log" | tee "$GOUT/gpus.json" | grep -q '"ok": true' \
    || abort "did not see 4 GPUs — send the bundle"
  pass "G2 all 4 GPUs visible in-container"

  # G3: timing baseline, then determinism ACROSS TWO SEPARATE CONTAINERS.
  # A G3 failure is a RESULT, not an abort — G4/G5 still run.
  say "G3: free-running timing baseline on GPU 0..."
  run_probe 0 free-once "$TIMEOUT_TRAIN" "$GOUT/free-once.log" \
    && collect "$GOUT/free-once.log" | tee "$GOUT/free-once.json" \
    || say "G3 free-once failed — see bundle"
  say "G3: determinism run 1 of 2 (fresh container)..."
  run_probe 0 det-once "$TIMEOUT_TRAIN" "$GOUT/det-once-a.log" \
    && collect "$GOUT/det-once-a.log" | tee "$GOUT/det-once-a.json" \
    || say "G3 det run 1 failed — see bundle"
  say "G3: determinism run 2 of 2 (fresh container)..."
  run_probe 0 det-once "$TIMEOUT_TRAIN" "$GOUT/det-once-b.log" \
    && collect "$GOUT/det-once-b.log" | tee "$GOUT/det-once-b.json" \
    || say "G3 det run 2 failed — see bundle"
  if [ -s "$GOUT/det-once-a.json" ] && [ -s "$GOUT/det-once-b.json" ]; then
    python - "$GOUT" <<'PY'
import json, sys, os
g = sys.argv[1]
a = json.load(open(os.path.join(g, "det-once-a.json")))
b = json.load(open(os.path.join(g, "det-once-b.json")))
free = None
p = os.path.join(g, "free-once.json")
if os.path.exists(p) and os.path.getsize(p):
    free = json.load(open(p))
same = a["weights_sha256"] == b["weights_sha256"] and a["losses"] == b["losses"]
env_ok = all(r["gpu"] == 1 and r["zero_rows"] == 0
             and r["data_src"] == "/probe_data.npz" for r in (a, b))
verdict = {"cross_container_deterministic": same, "probe_env_ok": env_ok,
           "weights_sha256": [a["weights_sha256"][:16], b["weights_sha256"][:16]],
           "sec_det": [a["seconds"], b["seconds"]],
           "sec_free": free["seconds"] if free else None,
           "gpu_in_container": [a["gpu"], b["gpu"]]}
json.dump(verdict, open(os.path.join(g, "determinism-verdict.json"), "w"))
print("G3 VERDICT:", json.dumps(verdict))
if not env_ok:
    print("G3 *** probe environment suspect (gpu/data/zero-rows) — "
          "read the JSONs before trusting the verdict ***")
if not same:
    print("G3 *** NOT cross-container deterministic — an ANSWER, not a "
          "failure: our canary methodology adapts before step 3 ***")
PY
  fi

  say "G4: throughput benches on GPU 0..."
  run_probe 0 bench-resnet "$TIMEOUT_BENCH" "$GOUT/bench-resnet.log" \
    && collect "$GOUT/bench-resnet.log" | tee "$GOUT/bench-resnet.json" \
    || say "G4 resnet bench failed — see bundle"
  run_probe 0 bench-transformer "$TIMEOUT_BENCH" "$GOUT/bench-transformer.log" \
    && collect "$GOUT/bench-transformer.log" | tee "$GOUT/bench-transformer.json" \
    || say "G4 transformer bench failed — see bundle"

  say "G5: concurrent 4-GPU probe (data from the read-only export — no shared
  cache, no extraction race)..."
  pids=()
  for i in 0 1 2 3; do
    run_probe "$i" quick "$TIMEOUT_QUICK" "$GOUT/quick-gpu$i.log" & pids+=($!)
  done
  G5_OK=1
  for i in 0 1 2 3; do
    wait "${pids[$i]}" || G5_OK=0
    collect "$GOUT/quick-gpu$i.log" > "$GOUT/quick-gpu$i.json" || G5_OK=0
    grep -q '"ok": true' "$GOUT/quick-gpu$i.json" 2>/dev/null || G5_OK=0
  done
  [ "$G5_OK" = "1" ] && pass "G5 all four GPUs trained concurrently, clean" \
    || say "G5 *** at least one GPU probe not clean — see bundle ***"
fi

say "STAGE 2 COMPLETE"
exit 0
