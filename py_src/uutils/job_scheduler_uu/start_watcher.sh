#!/usr/bin/env bash
# start_watcher.sh — Launch the DFS job-queue watcher inside a tmux session.
#
# Usage (from any node that shares the DFS):
#   bash start_watcher.sh              # uses defaults
#   bash start_watcher.sh --poll 10    # custom poll interval
#   bash start_watcher.sh --timeout 7200  # 2-hour timeout
#
# What it does:
#   1. Ensures ~/dfs/job_queue/{pending,running,completed,failed,logs}/ exist
#   2. Starts a tmux session named "job_watcher" running the Python daemon
#   3. If the session already exists, prints a warning and exits
#
# To stop:  tmux kill-session -t job_watcher
# To view:  tmux attach -t job_watcher
set -euo pipefail

JOB_DIR="${HOME}/dfs/job_queue"
SESSION_NAME="job_watcher"

# Forward all CLI args to the Python daemon.
# Use an array to preserve quoting of arguments with spaces.
EXTRA_ARGS=("$@")

# Check if tmux session already running.
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "[WARN] tmux session '${SESSION_NAME}' already exists on $(hostname)."
    echo "       Attach with:  tmux attach -t ${SESSION_NAME}"
    echo "       Kill with:    tmux kill-session -t ${SESSION_NAME}"
    exit 0
fi

# Ensure directories exist (harmless if they already do).
mkdir -p "${JOB_DIR}"/{pending,running,completed,failed,logs}

# Determine the Python to use.  Prefer the venv if activated, else system python3.
PYTHON="${PYTHON:-python3}"

echo "[INFO] Starting job watcher on $(hostname) in tmux session '${SESSION_NAME}'"
echo "       Job dir: ${JOB_DIR}"
echo "       Python:  $(command -v "${PYTHON}")"
if [ ${#EXTRA_ARGS[@]} -eq 0 ]; then
    echo "       Extra:   <none>"
else
    echo "       Extra:   ${EXTRA_ARGS[*]}"
fi

# Build the command string for tmux.  We must quote the job-dir in case it
# contains spaces, and properly escape each extra argument.
TMUX_CMD="${PYTHON} -m uutils.job_scheduler_uu.scheduler --job-dir '${JOB_DIR}'"
for arg in ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}; do
    # Escape single quotes within the argument for the tmux shell.
    escaped_arg="${arg//\'/\'\\\'\'}"
    TMUX_CMD="${TMUX_CMD} '${escaped_arg}'"
done

tmux new-session -d -s "${SESSION_NAME}" "${TMUX_CMD}"

echo "[OK]   Watcher running.  Attach with:  tmux attach -t ${SESSION_NAME}"
