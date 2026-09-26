#!/usr/bin/env bash
# Launch one coding-agent run on skampere2 in a detached tmux session.
# Usage: launch.sh <agent:agy|grok|cursor> <issue> <model>
set -euo pipefail
AGENT=$1; N=$2; MODEL=$3
export PATH=/dfs/scratch0/brando9/bin:$PATH
ROOT=/lfs/skampere2/0/brando9/uu-worktrees
EXP=$ROOT/_bugs_runs
case $AGENT in agy) BR=agy/bugs-$N; WT=$ROOT/bugs-$N;; *) BR=$AGENT/bugs-$N; WT=$ROOT/bugs-$N-$AGENT;; esac
SESSION=agy-bugs-$N; [ "$AGENT" = agy ] || SESSION=$AGENT-bugs-$N
cd /lfs/skampere2/0/brando9/ultimate-utils
[ -d "$WT" ] || git worktree add -q -b "$BR" "$WT" origin/main
RUN=$EXP/$AGENT-$N; mkdir -p "$RUN"
sed -e "s|__WT__|$WT|g" -e "s|__BRANCH__|$BR|g" -e "s|__SESSION_PREFIX__|uu-bugs-$AGENT-$N|g" "$EXP/task_$N.md" > "$RUN/task.md"
case $AGENT in
  agy)    CMD="agy --dangerously-skip-permissions --model $MODEL --output-format stream-json -p \"\$(cat $RUN/task.md)\"";;
  grok)   CMD="grok --always-approve --permission-mode bypassPermissions -m $MODEL --cwd $WT --output-format streaming-json --prompt-file $RUN/task.md";;
  cursor) CMD="cursor-agent -p --force --trust --model $MODEL --workspace $WT --output-format stream-json \"\$(cat $RUN/task.md)\"";;
esac
tmux new-session -d -s "$SESSION" "cd $WT && export PATH=$PATH && date +%s > $RUN/start && ( $CMD ) > $RUN/out.jsonl 2> $RUN/stderr; echo \$? > $RUN/exit; date +%s > $RUN/end"
echo "launched $SESSION wt=$WT run=$RUN"
