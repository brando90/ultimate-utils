#!/usr/bin/env bash
# start_watcher.sh — Launch the DFS job-queue watcher inside a tmux session.
#
# Environment variables:
#   PYTHON                        - Override the Python interpreter to use
#   WATCHER_SESSION               - Override the tmux session name (default: job_watcher)
#   WATCHER_JOB_DIR               - Override the job directory (default: ~/dfs/job_queue)
#   WATCHER_STARTUP_CHECK_SECS    - Seconds to wait before checking liveness (default: 5)
#   UUTILS_WATCHER_NOTIFY_DRY_RUN - Set to disable lifecycle emails
#
# The daemon logs stdout/stderr to ${WATCHER_JOB_DIR}/logs/watcher_tmux_$(hostname -s).log.
#
# Usage (from any node that shares the DFS):
#   bash start_watcher.sh              # uses defaults (1 job at a time)
#   bash start_watcher.sh --max-concurrent 4  # run up to 4 jobs in parallel
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

# Smart-mode dispatch uses claude-code, which must always run on the Max
# subscription (OAuth). If ANTHROPIC_API_KEY is set in the parent shell,
# claude-code would prefer it over OAuth and silently switch to the key.
# Scrub it (and cousins) before the watcher inherits them.
unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN CLAUDE_CODE_USE_BEDROCK CLAUDE_CODE_USE_VERTEX

SESSION_NAME="${WATCHER_SESSION:-job_watcher}"
JOB_DIR="${WATCHER_JOB_DIR:-${HOME}/dfs/job_queue}"

# Compute the repo's py_src from the script's own location and prepend to PYTHONPATH
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PY_SRC="$(cd "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"
export PYTHONPATH="${PY_SRC}${PYTHONPATH:+:${PYTHONPATH}}"

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

# Determine the Python to use. Prefer $PYTHON, then venv, then probing known paths.
if [ -n "${PYTHON:-}" ]; then
    CHOSEN_PYTHON="$PYTHON"
elif [ -n "${VIRTUAL_ENV:-}" ]; then
    CHOSEN_PYTHON="${VIRTUAL_ENV}/bin/python"
else
    CHOSEN_PYTHON=""
    for py in python3 "${HOME}/uv_envs/veribench/bin/python" /usr/bin/python3; do
        if command -v "$py" >/dev/null 2>&1; then
            if "$py" -c "import uutils.job_scheduler_uu.scheduler" >/dev/null 2>&1; then
                CHOSEN_PYTHON="$py"
                break
            fi
            # Keep the first valid executable as fallback so preflight can print its error
            if [ -z "$CHOSEN_PYTHON" ]; then
                CHOSEN_PYTHON="$py"
            fi
        fi
    done
    if [ -z "$CHOSEN_PYTHON" ]; then
        CHOSEN_PYTHON="python3"
    fi
fi

# Resolve the interpreter to an absolute path
if command -v "$CHOSEN_PYTHON" >/dev/null 2>&1; then
    CHOSEN_PYTHON="$(command -v "$CHOSEN_PYTHON")"
fi

# Preflight: check if the chosen interpreter can import the scheduler
if ! err_msg=$("$CHOSEN_PYTHON" -c "import uutils.job_scheduler_uu.scheduler" 2>&1); then
    last_line=$(echo "$err_msg" | tail -n 1)
    echo "[FAIL] Python interpreter ${CHOSEN_PYTHON} cannot import uutils.job_scheduler_uu.scheduler."
    echo "       Error: ${last_line}"
    echo "       Fix: set PYTHON=, activate a venv, or install dill."
    exit 1
fi

echo "[INFO] Starting job watcher on $(hostname) in tmux session '${SESSION_NAME}'"
echo "       Job dir: ${JOB_DIR}"
echo "       Python:  ${CHOSEN_PYTHON}"
if [ ${#EXTRA_ARGS[@]} -eq 0 ]; then
    echo "       Extra:   <none>"
else
    echo "       Extra:   ${EXTRA_ARGS[*]}"
fi

LOG_FILE="${JOB_DIR}/logs/watcher_tmux_$(hostname -s).log"

# Build the command string for tmux. We must quote each exported value safely
# and escape the arguments.
TMUX_CMD=""

# tmux does not pass the caller's environment. Explicitly export PATH, PYTHONPATH, etc.
printf -v Q_PATH '%q' "${PATH}"
TMUX_CMD+="export PATH=${Q_PATH}; "

printf -v Q_PYTHONPATH '%q' "${PYTHONPATH}"
TMUX_CMD+="export PYTHONPATH=${Q_PYTHONPATH}; "

if [ -n "${UUTILS_WATCHER_NOTIFY_DRY_RUN:-}" ]; then
    printf -v Q_DRY_RUN '%q' "${UUTILS_WATCHER_NOTIFY_DRY_RUN}"
    TMUX_CMD+="export UUTILS_WATCHER_NOTIFY_DRY_RUN=${Q_DRY_RUN}; "
fi

printf -v Q_PYTHON '%q' "${CHOSEN_PYTHON}"
TMUX_CMD+="${Q_PYTHON} -m uutils.job_scheduler_uu.scheduler --job-dir "

printf -v Q_JOB_DIR '%q' "${JOB_DIR}"
TMUX_CMD+="${Q_JOB_DIR}"

for arg in "${EXTRA_ARGS[@]}"; do
    printf -v Q_ARG '%q' "${arg}"
    TMUX_CMD+=" ${Q_ARG}"
done

# Send daemon's stdout+stderr to log file and keep exit status by redirecting
printf -v Q_LOG_FILE '%q' "${LOG_FILE}"
TMUX_CMD+=" >> ${Q_LOG_FILE} 2>&1"

tmux new-session -d -s "${SESSION_NAME}" "${TMUX_CMD}"

# Liveness check
sleep "${WATCHER_STARTUP_CHECK_SECS:-5}"

if ! tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "[FAIL] Watcher session died early."
    if [ -f "${LOG_FILE}" ]; then
        echo "       Last 20 lines of log (${LOG_FILE}):"
        tail -n 20 "${LOG_FILE}"
    fi
    exit 1
fi

echo "[OK]   Watcher running.  Attach with:  tmux attach -t ${SESSION_NAME}"
