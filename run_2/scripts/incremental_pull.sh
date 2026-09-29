#!/usr/bin/env bash
# Incremental campaign pull (2026-08-05, after charizard was lost with ~4.5h of
# unpulled results). The boxes boot an EPHEMERAL live-image overlay: anything not
# on the laptop dies with the box. Pull every INTERVAL seconds so the maximum
# possible loss is one interval, not one campaign.
INTERVAL="${1:-600}"
HOST="${2:-moltres-claude}"
DEST="${3:-$HOME/Documents/Uni/01_PhD/Projects/04_Clouds/AI/Custom_Framework/campaigns}"
# Remote campaign roots. Campaigns that need a patched src (e.g. the FedLUAR
# head-to-head) run from an ISOLATED tree, so pulling only fl-framework would
# silently leave their results stranded on an ephemeral box.
REMOTES="${4:-fl-framework/campaigns fl-framework-fedluar/campaigns}"
while true; do
  for r in $REMOTES; do
    rsync -az --timeout=120 -e ssh "$HOST:$r/" "$DEST/" >/dev/null 2>&1
  done
  echo "[$(date '+%H:%M')] pulled ${HOST%-claude} [$REMOTES] -> local reports=$(ls "$DEST"/*/*/seed*/results/report.json 2>/dev/null | wc -l)"
  sleep "$INTERVAL"
done
