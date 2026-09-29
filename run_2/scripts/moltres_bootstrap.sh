#!/usr/bin/env bash
# Fresh-moltres bootstrap: docker(+compose v2), venv, pip -e ., build images, keras check.
# Detached (nohup); progress markers appended to ~/bootstrap.DONE, full log in ~/bootstrap.log.
set -uo pipefail
DONE=~/bootstrap.DONE
: > "$DONE"
step(){ echo "=== [$(date '+%H:%M:%S')] $* ==="; }
ok(){   echo "OK:   $*" | tee -a "$DONE"; }
fail(){ echo "FAIL: $*" | tee -a "$DONE"; }

cd ~/fl-framework || { fail "no repo at ~/fl-framework"; exit 1; }
export DEBIAN_FRONTEND=noninteractive

step "apt base tooling"
apt-get update -y >/dev/null 2>&1 \
  && apt-get install -y python3-venv python3-pip protobuf-compiler rsync curl ca-certificates >/dev/null 2>&1 \
  && ok "apt-base" || fail "apt-base"

step "docker + compose v2 (get.docker.com)"
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sh >/dev/null 2>&1 || fail "docker-install-script"
fi
systemctl enable --now docker >/dev/null 2>&1 || service docker start >/dev/null 2>&1 || true
sleep 3
docker version   >/dev/null 2>&1 && ok "docker-daemon"  || fail "docker-daemon"
docker compose version >/dev/null 2>&1 && ok "compose-v2" || fail "compose-v2"

step "python venv + editable install (tensorflow, this pulls ~500MB)"
python3 -m venv .venv >/dev/null 2>&1 && ok "venv-create" || fail "venv-create"
# shellcheck disable=SC1091
. .venv/bin/activate
pip install --upgrade pip >/dev/null 2>&1
if pip install -e . >/dev/null 2>&1; then ok "pip-install"; else fail "pip-install"; fi
python -c "import tensorflow as tf,numpy,yaml,pydantic; print(tf.__version__)" >/dev/null 2>&1 \
  && ok "py-imports" || fail "py-imports"

step "build docker images (regenerates proto inside)"
if docker build -t fl-node:latest   -f docker/Dockerfile.node    . >/dev/null 2>&1; then ok "img-node";   else fail "img-node";   fi
if docker build -t fl-monitor:latest -f docker/Dockerfile.monitor . >/dev/null 2>&1; then ok "img-monitor"; else fail "img-monitor"; fi

step "keras CIFAR cache check (best-effort, 300s timeout)"
if timeout 300 python -c "import tensorflow as tf; a=tf.keras.datasets.cifar10.load_data(); print('cifar', a[0][0].shape, a[1][0].shape)" >/dev/null 2>&1; then
  ok "keras-cifar"
else
  fail "keras-cifar (stage cache in smoke step)"
fi

echo "=== BOOTSTRAP DONE $(date '+%F %H:%M:%S') ===" | tee -a "$DONE"
