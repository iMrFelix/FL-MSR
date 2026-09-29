#!/bin/bash
# Detached chain guard for the fedluar_hh campaign (2026-08-05).
#
# Waits for the commbound campaign's DONE marker in ~/fl-framework, then
# rebuilds the docker images FROM THE ISOLATED TREE and hands over to
# scripts/run_fedluar_hh.sh.  Nothing here touches ~/fl-framework.
#
# ---------------------------------------------------------------------------
# WHY THE PATTERNS ARE BUILT FROM VARIABLES AND SPLIT STRINGS
# ---------------------------------------------------------------------------
# A guard that greps or pkills a literal script name SELF-MATCHES: the literal
# is sitting in the guard's own /proc/<pid>/cmdline, so `pgrep -f <name>`
# returns the guard, and a `pkill -f <name>` kills the guard before it can do
# anything.  That bit us today.  The discipline here, applied to every pattern
# without exception:
#   1. build the pattern in a shell VARIABLE, from concatenated fragments, so
#      the whole literal never appears in any command line;
#   2. exclude the guard's own PID and its process group explicitly;
#   3. ECHO A MARKER immediately after every pgrep/pkill recording the pattern
#      and the hits, so the log proves what was matched and what was spared.
# This file is also deliberately NOT named after the runner, so the runner
# pattern cannot appear in the guard's argv at all.
# ---------------------------------------------------------------------------
set +e

LOG="${FL_HH_LOG:-$HOME/fl_hh_guard.log}"
exec >>"$LOG" 2>&1

TREE="${FL_HH_TREE:-$HOME/fl-framework-fedluar}"
VENV="${FL_HH_VENV:-$HOME/fl-framework/.venv}"
LOCKDIR="$HOME/.fl_hh_guard.lock"

# --- what we wait for -----------------------------------------------------
# MUST be the TAIL of the box's chain, not merely the campaign named in the
# brief.  On moltres the live chain is `run_commbound.sh; run_stab.sh` (one
# parent shell, 2026-08-05), so waiting on COMMBOUND's marker would start this
# campaign ON TOP of stab and both would crawl.  We wait on STAB instead.
# Override with FL_HH_WAIT_FILE / FL_HH_WAIT_TAG if the chain changes.
WAIT_TAG="${FL_HH_WAIT_TAG:-STAB}"
WAIT_FILE="${FL_HH_WAIT_FILE:-$HOME/fl-framework/campaigns/stab/STAB.DONE}"
# Note the runners TRUNCATE their .DONE at start, so file existence means
# "started", never "finished" — only the appended marker line means finished.
WAIT_PAT="$WAIT_TAG"" DONE"          # assembled: never whole in any argv
RUN_TOKEN="run_fedluar""_hh.sh"      # the runner, as it appears in ITS argv

mark() { echo "[$(date '+%F %H:%M:%S')] $*"; }

mark "=== GUARD ARMED pid=$$ ppid=$PPID tree=$TREE ==="
mark "GUARD wait_file=$WAIT_FILE  (tail of this box's chain)"
mark "GUARD MARKER pattern-built wait_pat=[$WAIT_PAT] run_token=[$RUN_TOKEN]"

# --- single instance (atomic mkdir; no pattern matching involved) ----------
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  mark "GUARD abort: lock $LOCKDIR already held (another guard is armed)"
  exit 0
fi
trap 'rmdir "$LOCKDIR" 2>/dev/null' EXIT
mark "GUARD lock acquired: $LOCKDIR"

# --- is the runner already live?  pattern from a variable, own PID excluded -
HITS="$(pgrep -f -- "$RUN_TOKEN" 2>/dev/null | grep -v -x -- "$$")"
mark "GUARD MARKER pgrep run_token=[$RUN_TOKEN] hits=[$(echo $HITS)] self=$$ (no kill issued)"
if [ -n "$HITS" ]; then
  mark "GUARD abort: runner already running (pids $(echo $HITS))"
  exit 0
fi

# --- 1. wait for the commbound DONE marker --------------------------------
DEADLINE=$(( $(date +%s) + 60*3600 ))   # 60 h cap: never start blind
i=0
while ! grep -q -- "$WAIT_PAT" "$WAIT_FILE" 2>/dev/null; do
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    mark "GUARD ABORT: 60h deadline reached without [$WAIT_PAT]; NOT starting"
    exit 1
  fi
  i=$((i+1))
  [ $((i % 15)) -eq 1 ] && mark "GUARD heartbeat: still waiting for [$WAIT_PAT] in $WAIT_FILE"
  sleep 120
done
mark "GUARD MARKER wait satisfied: [$WAIT_PAT] found in $WAIT_FILE"

# --- 2. settle: let the previous campaign's containers drain ---------------
SETTLE_DEADLINE=$(( $(date +%s) + 45*60 ))
clear_streak=0
while [ "$clear_streak" -lt 3 ]; do
  live="$(docker ps -q 2>/dev/null | wc -l | tr -d ' ')"
  if [ "${live:-1}" -eq 0 ]; then
    clear_streak=$((clear_streak+1))
  else
    clear_streak=0
    mark "GUARD settle: $live container(s) still up, waiting"
  fi
  if [ "$(date +%s)" -ge "$SETTLE_DEADLINE" ]; then
    mark "GUARD ABORT: containers never drained within 45 min; NOT starting"
    exit 1
  fi
  sleep 60
done
mark "GUARD MARKER settled: 0 containers for 3 consecutive checks"

# --- 3. pre-flight: the whole config set must validate on THIS box ---------
# The documented failure mode on these boxes is a PARTIAL sync (cp2 lost 14
# runs to a launcher.py without a matching schema.py).  Validating all 81
# configs against this tree's schema catches exactly that, in seconds,
# before three hours of GPU-free CPU time are committed.
cd "$TREE" || { mark "GUARD ABORT: no tree at $TREE"; exit 1; }
source "$VENV/bin/activate" || { mark "GUARD ABORT: no venv at $VENV"; exit 1; }
python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, ".")
from src.config.schema import load_config
specs = Path("configs/experiments/fedluar/specs.txt")
rows = [ln.split("\t") for ln in specs.read_text().splitlines() if ln.strip()]
bad = []
for row in rows:
    try:
        load_config(row[4])
    except Exception as e:
        bad.append(f"{row[4]}: {e}")
print(f"PREFLIGHT: {len(rows)} configs, {len(bad)} invalid")
for b in bad[:5]:
    print("  INVALID", b)
sys.exit(1 if bad or not rows else 0)
PY
if [ $? -ne 0 ]; then
  mark "GUARD ABORT: config pre-flight failed (partial sync?); NOT starting"
  exit 1
fi
mark "GUARD MARKER pre-flight OK: all configs validate against this tree"

# --- 4. rebuild images from THIS tree -------------------------------------
# Deliberately here and not at sync time: fl-node:latest is a shared tag and
# a rebuild during a live, timing-sensitive campaign would both steal CPU and
# change the code state under the next run.  Post-marker is the only safe
# moment.  scripts.run rebuilds anyway; doing it explicitly puts a build
# failure in the guard log instead of burying it in run-1's log.
mark "GUARD building fl-node:latest from $TREE"
docker build -t fl-node:latest -f docker/Dockerfile.node . \
  && docker build -t fl-monitor:latest -f docker/Dockerfile.monitor .
if [ $? -ne 0 ]; then
  mark "GUARD ABORT: docker build failed; NOT starting"
  exit 1
fi
mark "GUARD MARKER images rebuilt from the isolated tree"

# --- 5. hand over -----------------------------------------------------------
mark "GUARD launching runner ($RUN_TOKEN) inline"
bash "scripts/$RUN_TOKEN"
mark "GUARD runner exited rc=$?"
