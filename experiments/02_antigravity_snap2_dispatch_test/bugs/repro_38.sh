#!/usr/bin/env bash
# Repro for issue #38: run a start_watcher.sh with the system python3 and an
# isolated session/job dir, then check whether the tmux session is still alive.
# Usage: bash repro_38.sh <path/to/start_watcher.sh> <label> [extra watcher args...]
# Env: REPRO_PATH (PATH for the launch; default: caller PATH), PYTHON (interpreter override).
# Never touches the real 'job_watcher' session or ~/dfs/job_queue.
set -u
SCRIPT=$1; LABEL=$2; shift 2
SESSION="uu-agy-bugs-repro38-${LABEL}"
JOBDIR="/lfs/skampere2/0/brando9/uu-worktrees/_scratch_bugs/jobq-${LABEL}"
mkdir -p "$(dirname "$JOBDIR")"
tmux kill-session -t "$SESSION" 2>/dev/null
# The original script hard-codes session and job dir; make an isolated copy.
COPY=$(mktemp /tmp/start_watcher_${LABEL}.XXXX.sh)
sed -e "s|^SESSION_NAME=.*|SESSION_NAME=\"\${WATCHER_SESSION:-${SESSION}}\"|" \
    -e "s|^JOB_DIR=.*|JOB_DIR=\"\${WATCHER_JOB_DIR:-${JOBDIR}}\"|" "$SCRIPT" > "$COPY"
# Keep the copy next to the original so relative paths (py_src) resolve the same.
cp "$COPY" "$(dirname "$SCRIPT")/.repro_copy.sh"
echo "== env: PYTHON=${PYTHON:-<unset>} PYTHONPATH=${PYTHONPATH:-<unset>} VIRTUAL_ENV=${VIRTUAL_ENV:-<unset>}"
echo "== running $SCRIPT (isolated copy), session=$SESSION"
env -u PYTHONPATH -u VIRTUAL_ENV PATH="${REPRO_PATH:-$PATH}" WATCHER_SESSION="$SESSION" WATCHER_JOB_DIR="$JOBDIR" \
  UUTILS_WATCHER_NOTIFY_DRY_RUN=1 \
  bash "$(dirname "$SCRIPT")/.repro_copy.sh" --poll 5 "$@" 2>&1
RC=$?
echo "== script exit code: $RC"
sleep 8
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "== RESULT: tmux session '$SESSION' ALIVE after 8s"
  tmux capture-pane -p -t "$SESSION" | grep -v '^$' | tail -8
  tmux kill-session -t "$SESSION"
else
  echo "== RESULT: tmux session '$SESSION' DEAD after 8s"
fi
ls "$JOBDIR/logs" 2>/dev/null && tail -5 "$JOBDIR"/logs/* 2>/dev/null
rm -f "$(dirname "$SCRIPT")/.repro_copy.sh" "$COPY"
