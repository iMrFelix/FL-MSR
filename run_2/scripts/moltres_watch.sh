#!/usr/bin/env bash
# Fixed-interval experiment-box progress watchdog (standing rule, 2026-08-05):
# one ssh connection per host per check (no tight polling), every INTERVAL
# seconds, over all HOSTS. Emits one compact STATUS line per host per check;
# ALERT lines on: unreachable box, stalled progress (round counts + END lines
# frozen while containers run), or new failed runs (NO_REPORT count increased).
INTERVAL="${1:-600}"
shift
HOSTS=("${@:-moltres-claude}")
[ ${#HOSTS[@]} -eq 0 ] && HOSTS=(moltres-claude)
declare -A prev_key prev_fails
check_host() {
  local host="$1"
  s=$(ssh -o ConnectTimeout=20 -o BatchMode=yes "$host" '
    cd ~ 2>/dev/null || { echo "NO_HOME"; exit 0; }
    # Scan EVERY campaign root: patched-src campaigns (e.g. the FedLUAR
    # head-to-head) run from an isolated tree and would otherwise be invisible.
    G=$(ls -d ~/fl-framework*/campaigns 2>/dev/null)
    [ -z "$G" ] && { echo "REPO_MISSING"; exit 0; }
    ends=0; fails=0; dones=""; rounds=""
    for c in $G; do
      ends=$(( ends + $(cat $c/*/*.DONE 2>/dev/null | grep -c " END ") ))
      fails=$(( fails + $(cat $c/*/*.DONE 2>/dev/null | grep -c "NO_REPORT") ))
      dones="$dones$(grep -l "DONE 20" $c/*/*.DONE 2>/dev/null | xargs -n1 basename 2>/dev/null | tr "\n" ",")"
    done
    # NB: campaigns/<campaign>/<arm>/seed<N>/run.log — three levels under a
    # campaign root. Dropping one level silently empties `active`, which is
    # the stall signal (bitten 2026-08-05 20:38).
    for f in $(ls -t $(for c in $G; do echo $c/*/*/seed*/run.log; done) 2>/dev/null | head -3); do
      rounds="$rounds$(basename $(dirname $(dirname $f)))/$(basename $(dirname $f)):$(grep -c "val_acc=" $f 2>/dev/null) "
    done
    cont=$(docker ps -q 2>/dev/null | wc -l)
    load=$(cut -d" " -f1 /proc/loadavg)
    echo "ends=$ends fails=$fails containers=$cont load=$load active=[$rounds] campaign_done=[$dones]"
  ' 2>/dev/null)
  local tag="${host%-claude}"
  local now=$(date '+%H:%M')
  if [ -z "$s" ]; then
    echo "[$now][$tag] ALERT unreachable (ssh failed/timeout)"
    prev_key[$host]=""
  else
    local ends fails cont rounds key
    ends=$(echo "$s" | grep -o 'ends=[0-9]*')
    fails=$(echo "$s" | grep -o 'fails=[0-9]*' | grep -o '[0-9]*')
    cont=$(echo "$s" | grep -o 'containers=[0-9]*' | grep -o '[0-9]*')
    rounds=$(echo "$s" | grep -o 'active=\[[^]]*\]')
    key="$ends|$rounds"
    if [ "$key" = "${prev_key[$host]:-__unset__}" ] && [ "${cont:-0}" -gt 0 ]; then
      echo "[$now][$tag] ALERT stalled: no new rounds/END lines since last check — $s"
    elif [ "${fails:-0}" -gt "${prev_fails[$host]:--1}" ] && [ "${prev_fails[$host]:--1}" -ge 0 ]; then
      echo "[$now][$tag] ALERT new failed run(s): $s"
    else
      echo "[$now][$tag] STATUS $s"
    fi
    prev_key[$host]="$key"
    prev_fails[$host]="${fails:-0}"
  fi
}
while true; do
  # One merged event line per cycle (token-frugal: a single notification
  # wakes the agent once, not once per host).
  cycle_out=""
  for h in "${HOSTS[@]}"; do cycle_out+="$(check_host "$h") | "; done
  echo "${cycle_out% | }"
  sleep "$INTERVAL"
done
