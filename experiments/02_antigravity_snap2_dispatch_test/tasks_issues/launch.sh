#!/bin/bash
# Usage: launch.sh <id> [model] [attempt] [agent: agy|grok]  -- run on skampere2
set -euo pipefail
ID=$1; MODEL=${2:-gemini-3.8-flash-high}; ATTEMPT=${3:-1}; AGENT=${4:-agy}
export PATH=/dfs/scratch0/brando9/bin:$PATH
REPO=/lfs/skampere2/0/brando9/ultimate-utils
WT=/lfs/skampere2/0/brando9/uu-worktrees/issues-$ID
BR=$AGENT/issues-$ID
BASE=/lfs/skampere2/0/brando9/uu-agy-logs/issues
LOG=$BASE/issues-$ID/attempt$ATTEMPT
mkdir -p "$LOG"
if [ ! -d "$WT" ]; then
  git -C "$REPO" fetch -q origin
  git -C "$REPO" worktree add -q -b "$BR" "$WT" origin/main
fi
sed "s|agy/issues-$ID|$BR|g" "$BASE/tasks/issue_$ID.md" > "$LOG/task.md"
if [ -f "$BASE/tasks/feedback_$ID.md" ] && [ "$ATTEMPT" -gt 1 ]; then cat "$BASE/tasks/feedback_$ID.md" >> "$LOG/task.md"; fi
case $AGENT in
  agy)  CMD="agy --dangerously-skip-permissions --model \"$MODEL\" --output-format stream-json -p \"\$(cat \"$LOG/task.md\")\"";;
  grok) CMD="grok --always-approve --permission-mode bypassPermissions -m \"$MODEL\" --cwd \"$WT\" --output-format streaming-json --prompt-file \"$LOG/task.md\"";;
esac
cat > "$LOG/run.sh" <<RUN
#!/bin/bash
export PATH=/dfs/scratch0/brando9/bin:\$PATH
cd "$WT"
date +%s > "$LOG/start"
$CMD > "$LOG/agy.jsonl" 2> "$LOG/agy.stderr"
echo \$? > "$LOG/exit_code"
date +%s > "$LOG/end"
RUN
S="$AGENT-issues-$ID"
tmux new-session -d -s "$S" "bash $LOG/run.sh"
echo "launched $S agent=$AGENT model=$MODEL log=$LOG wt=$WT branch=$BR"
