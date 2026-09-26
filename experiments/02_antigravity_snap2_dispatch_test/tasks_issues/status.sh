#!/bin/bash
# Usage: status.sh <id> [attempt]... -- run on skampere2; prints one summary line per task
B=/lfs/skampere2/0/brando9/uu-agy-logs/issues
A=${ATTEMPT:-1}
for id in "$@"; do
  L=$B/issues-$id/attempt$A; J=$L/agy.jsonl
  now=$(date +%s); st=$(cat $L/start 2>/dev/null || echo $now); en=$(cat $L/end 2>/dev/null || echo "")
  age=$(( now - $(stat -c %Y $J 2>/dev/null || echo $now) ))
  steps=$(grep -c '"step_update"' $J 2>/dev/null)
  last=$(grep '"tool_name"' $J 2>/dev/null | tail -1 | python3 -c 'import sys,json
l=sys.stdin.read().strip()
try:
  d=json.loads(l)["step_update"]; p=d.get("tool_info",{}).get("parameters",{}); print(d.get("tool_name"), str(p.get("CommandLine") or p.get("TargetFile") or p)[:140].replace("\n"," "))
except Exception as e: print("-")')
  wt=/lfs/skampere2/0/brando9/uu-worktrees/issues-$id
  ch=$(git -C $wt status --short 2>/dev/null | tr '\n' ' ' | cut -c1-200)
  echo "[$id] exit=$(cat $L/exit_code 2>/dev/null || echo run) elapsed=$(( ${en:-$now} - st ))s idle=${age}s events=$steps last=[$last] changes=[$ch]"
done
