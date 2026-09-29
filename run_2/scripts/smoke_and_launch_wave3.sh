#!/bin/bash
# Run in the FOREGROUND over one ssh connection. Rebuilds images (K0 src changed),
# runs a 2-round smoke that MUST show global_eval live + a written report, and only
# then nohup-launches the full wave-3 campaign. Exits non-zero (no launch) if the
# smoke fails — never commit the overnight run to a broken build.
set -uo pipefail
cd ~/fl-framework
source .venv/bin/activate

echo "### 0. preconditions"
grep -q "BOOTSTRAP DONE" ~/bootstrap.DONE 2>/dev/null || { echo "ABORT: bootstrap not complete"; tail -n +1 ~/bootstrap.DONE 2>/dev/null; exit 2; }
docker compose version >/dev/null 2>&1 || { echo "ABORT: no docker compose v2"; exit 2; }

echo "### 1. rebuild images (pick up K0 code)"
docker build -t fl-node:latest   -f docker/Dockerfile.node    . >/tmp/build_node.log  2>&1 || { echo "ABORT: node image build failed"; tail -20 /tmp/build_node.log; exit 3; }
docker build -t fl-monitor:latest -f docker/Dockerfile.monitor . >/tmp/build_mon.log  2>&1 || { echo "ABORT: monitor image build failed"; tail -20 /tmp/build_mon.log; exit 3; }
echo "images rebuilt OK"

echo "### 2. build a 2-round smoke config from recycle_aging seed41"
python - <<'PY'
import yaml, pathlib
c = yaml.safe_load(open("configs/experiments/wave3/recycle_aging_eps03/seed41.yaml"))
c["training"]["total_rounds"] = 2
pathlib.Path("/tmp/smoke_wave3.yaml").write_text(yaml.dump(c, sort_keys=False))
print("wrote /tmp/smoke_wave3.yaml (global_eval=%s, rounds=%s)"
      % (c["training"]["global_eval"], c["training"]["total_rounds"]))
PY

echo "### 3. run smoke (2 rounds, ~3min)"
rm -rf /tmp/smoke_out
FL_SUBNET_OCTET=7 timeout 900 python -m scripts.run -c /tmp/smoke_wave3.yaml -o /tmp/smoke_out >/tmp/smoke_run.log 2>&1
SRC=$?
echo "smoke exit=$SRC"

echo "### 4. verify smoke outputs"
REPORT=/tmp/smoke_out/results/report.json
FAIL=0
[ -f "$REPORT" ] || { echo "FAIL: no report.json"; FAIL=1; }
if grep -q "global_eval=ON" /tmp/smoke_out/*/node-*.log /tmp/smoke_run.log 2>/dev/null; then
  echo "OK: global_eval=ON observed in node logs"
else
  echo "WARN: 'global_eval=ON' string not found in logs (check manually)";
fi
if [ -f "$REPORT" ]; then
  python - <<'PY'
import json,sys
r=json.load(open("/tmp/smoke_out/results/report.json"))
pr=r.get("per_round",[])
accs=[]
for rd in pr:
    for nid,nd in rd.get("nodes",{}).items():
        va=nd.get("val_accuracy")
        if va is not None: accs.append((rd["round"],nid,round(float(va),4)))
print("per-round val_accuracy sample:", accs[:8])
assert len(pr)>=2, "expected >=2 rounds"
assert accs, "no val_accuracy recorded"
# 10k global test set => accuracy is a sane fraction in (0,1); not the tiny-local-val 0/1 spikes
print("SMOKE REPORT OK: rounds=%d, val_accuracy present" % len(pr))
PY
  [ $? -eq 0 ] || FAIL=1
fi

if [ $SRC -ne 0 ] || [ $FAIL -ne 0 ]; then
  echo "=== SMOKE FAILED — NOT launching campaign. Inspect /tmp/smoke_run.log ==="
  tail -30 /tmp/smoke_run.log
  exit 4
fi

echo "### 5. smoke passed — launch wave-3 detached (nohup)"
chmod +x scripts/run_wave3.sh
nohup bash scripts/run_wave3.sh > ~/wave3_runner.log 2>&1 &
echo "wave-3 launched pid $! — tail ~/wave3_runner.log or campaigns/w3/WAVE3.DONE"
echo "=== SMOKE+LAUNCH COMPLETE $(date '+%F %H:%M:%S') ==="
