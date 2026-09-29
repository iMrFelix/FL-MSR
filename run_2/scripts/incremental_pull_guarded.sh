#!/usr/bin/env bash
# Incremental campaign pull, HARDENED (2026-08-06).
#
# Supersedes scripts/incremental_pull.sh.  Written as a NEW FILE on purpose:
# bash re-reads a running script from its current byte offset, so editing the
# original in place while three loops are executing it risks corrupting their
# control flow mid-cycle.  Migrate loop-by-loop, then retire the old script.
#
# The boxes boot an EPHEMERAL live-image overlay: anything not on the laptop
# dies with the box.  Pull every INTERVAL seconds so the maximum possible loss
# is one interval, not one campaign.
#
# THREE CHANGES over the original, all responses to measured problems:
#
#  (a) FAILURES ARE VISIBLE.  The original was `rsync … >/dev/null 2>&1`.  A
#      preservation tool whose failures are silent is the D4 defect wearing a
#      different hat: when the volume filled, pulls would simply stop
#      preserving and nothing anywhere would say so.  stderr now goes to a log
#      and a nonzero rc prints a loud line.  rc=24 (source files vanished) is
#      EXPECTED against a live campaign tree — the runner deletes each run's
#      data/ the moment it finishes — so it is logged, not screamed about.
#
#  (b) FREE-SPACE GUARD.  Below GUARD_GIB the loop stops transferring data/
#      directories entirely and says so every cycle.  Prevention over
#      detection: a pull that refuses to fill the disk beats discovering a
#      TRUNCATED npz afterwards — and a truncated shard is the worst possible
#      artifact for us, because it reads as corruption evidence in the very
#      tree we treat as ground truth for the corruption story.
#
#  (c) SCOPED EXCLUDES.  data/ is pulled for KEEP_DATA campaigns only.
#      Rationale, per lane: w3clean keeps data/ box-side by design; cbfix and
#      cb2fix digests carry zero_rows computed box-side on the actual training
#      data; the rescue's gate stack includes a box-side ZERO_SCAN;
#      fedluar_hh and the rest are complete, so nothing is inbound.  `stab` is
#      the ONE lane whose runner digests lack zero_rows AND whose box deletes
#      data/ after each run — its shard residue is the only copy that can ever
#      answer the zero-row question for those cells, so it keeps flowing.
#
# Usage: incremental_pull_guarded.sh [INTERVAL] [HOST] [DEST] [REMOTES]
INTERVAL="${1:-600}"
HOST="${2:-moltres-claude}"
DEST="${3:-$HOME/Documents/Uni/01_PhD/Projects/04_Clouds/AI/Custom_Framework/campaigns}"
# Campaigns that need a patched src (e.g. the FedLUAR head-to-head) run from an
# ISOLATED tree, so pulling only fl-framework would silently strand them.
REMOTES="${4:-fl-framework/campaigns fl-framework-fedluar/campaigns}"

KEEP_DATA="${KEEP_DATA:-stab}"   # space-separated campaign names
GUARD_GIB="${GUARD_GIB:-10}"
LOGDIR="${LOGDIR:-$DEST/../.pull-logs}"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/pull_${HOST%-claude}.log"

echo "[$(date '+%F %H:%M:%S')] START host=$HOST interval=${INTERVAL}s keep_data='$KEEP_DATA' guard=${GUARD_GIB}GiB log=$LOG" | tee -a "$LOG"

while true; do
  # Available GiB on the volume holding DEST.  Recomputed EVERY cycle: the
  # guard has to react to a campaign that is filling the disk right now, not
  # to whatever was true when the loop started.
  avail=$(df -g "$DEST" 2>/dev/null | awk 'NR==2{print $4}')
  avail="${avail:-0}"

  filters=()
  if [ "$avail" -lt "$GUARD_GIB" ]; then
    # GUARD TRIPPED — no data/ from anywhere, including KEEP_DATA campaigns.
    # Reports, digests and logs are small and still come through, so analysis
    # is never blocked; only the multi-GB shard residue is held back.
    filters+=(--exclude='data/')
    echo "[$(date '+%H:%M')] *** DISK GUARD: ${avail} GiB free < ${GUARD_GIB} GiB — data/ transfer PAUSED for $HOST (reports/digests still pulling) ***" | tee -a "$LOG"
  else
    # Include rules must precede the exclude: rsync takes the FIRST matching
    # rule, so an anchored include for each KEEP_DATA campaign wins over the
    # generic data/ exclude, and every other campaign falls through to it.
    for c in $KEEP_DATA; do filters+=(--include="/$c/**"); done
    filters+=(--exclude='data/')
  fi

  rc_worst=0
  err=$(mktemp)
  for r in $REMOTES; do
    # stderr is captured PER INVOCATION so the classification below reads only
    # this cycle's message — grepping the shared log would let a stale line
    # from hours ago decide how today's failure is reported.
    rsync -az --timeout=120 "${filters[@]}" -e ssh "$HOST:$r/" "$DEST/" 2>"$err"
    rc=$?
    [ -s "$err" ] && { echo "[$(date '+%F %H:%M:%S')] rsync stderr ($HOST:$r):" >> "$LOG"; cat "$err" >> "$LOG"; }
    if [ "$rc" -eq 0 ]; then
      :
    elif [ "$rc" -eq 24 ]; then
      # "Source files vanished during transfer" — routine here, since a
      # finishing run removes its own data/ while we are reading it.
      echo "[$(date '+%F %H:%M:%S')] note: rc=24 (vanished source files) host=$HOST remote=$r — expected against a live tree" >> "$LOG"
    elif grep -q 'change_dir.*failed: No such file or directory' "$err"; then
      # The remote tree simply does not exist on this host — snorlax has no
      # fl-framework-fedluar/campaigns, for instance.  Absent is not broken.
      # Kept in REMOTES on purpose rather than narrowing the list per host: a
      # campaign that later materialises there (the d7 rescue is slated for
      # snorlax) must be picked up automatically, and a narrowed list would
      # strand it silently — the failure mode this whole rewrite exists to end.
      echo "[$(date '+%F %H:%M:%S')] note: $r absent on $HOST — nothing to pull (will be picked up if it appears)" >> "$LOG"
    else
      echo "[$(date '+%H:%M')] *** PULL FAILED rc=$rc host=$HOST remote=$r — see $LOG ***" | tee -a "$LOG"
      [ "$rc" -gt "$rc_worst" ] && rc_worst=$rc
    fi
  done
  rm -f "$err"

  echo "[$(date '+%H:%M')] pulled ${HOST%-claude} [$REMOTES] free=${avail}GiB rc=$rc_worst -> local reports=$(ls "$DEST"/*/*/seed*/results/report.json 2>/dev/null | wc -l | tr -d ' ')" | tee -a "$LOG"
  sleep "$INTERVAL"
done
