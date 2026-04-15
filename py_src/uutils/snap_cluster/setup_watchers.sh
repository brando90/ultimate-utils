#!/bin/bash
# setup_watchers.sh -- Bootstrap watcher daemons on SNAP servers via SSH.
#
# This script SSHes into each SNAP server and starts a tmux session running
# the watcher daemon. The watcher polls the shared AFS/DFS task queue for
# pending tasks and executes them (e.g., Claude Code with full permissions).
#
# Prerequisites:
#   - SSH keys set up for passwordless login to SNAP servers
#   - AFS/DFS mounted on all servers (it is by default on SNAP)
#   - Python environment with uutils installed on each server
#   - Claude Code CLI installed on each server (for claude_code executor)
#
# Usage:
#   bash setup_watchers.sh                          # Start on all SNAP servers
#   bash setup_watchers.sh --servers "snap1 snap3"  # Start on specific servers
#   bash setup_watchers.sh --stop                   # Stop all watchers
#   bash setup_watchers.sh --status                 # Check watcher status
#
# Configuration (edit these or pass as env vars):
QUEUE_DIR="${QUEUE_DIR:-$HOME/afs_task_queue}"
# Real SNAP server names (edit to match your access): compute, GPU, etc.
# See https://ilwiki.stanford.edu/doku.php?id=snap-servers:snap-servers
# Common servers: rambo, madmax1-7, hyperion1-3, turing1-3, ampere1-9, etc.
# Entry point: shark.stanford.edu
SNAP_SERVERS="${SNAP_SERVERS:-ampere1 ampere2 ampere3 ampere4 ampere5 ampere6 ampere7 ampere8 ampere9}"
TMUX_SESSION_NAME="${TMUX_SESSION_NAME:-snap_task_watcher}"
POLL_INTERVAL="${POLL_INTERVAL:-5}"
# Python environment activation command -- edit for your setup
# Options: "conda activate snap_cluster_setup" or "source ~/.virtualenvs/snap_cluster_setup/bin/activate"
PYTHON_ENV_CMD="${PYTHON_ENV_CMD:-source ~/.virtualenvs/snap_cluster_setup/bin/activate}"

# Parse arguments
ACTION="start"
CUSTOM_SERVERS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --stop)
            ACTION="stop"
            shift
            ;;
        --status)
            ACTION="status"
            shift
            ;;
        --servers)
            CUSTOM_SERVERS="$2"
            shift 2
            ;;
        --queue_dir)
            QUEUE_DIR="$2"
            shift 2
            ;;
        --poll_interval)
            POLL_INTERVAL="$2"
            shift 2
            ;;
        --env_cmd)
            PYTHON_ENV_CMD="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--stop|--status] [--servers 'snap1 snap3'] [--queue_dir PATH] [--poll_interval N]"
            echo ""
            echo "  --stop              Stop all watcher daemons"
            echo "  --status            Check watcher status on all servers"
            echo "  --servers LIST      Space-separated list of servers (default: snap1-snap9)"
            echo "  --queue_dir PATH    Shared queue directory (default: ~/afs_task_queue)"
            echo "  --poll_interval N   Seconds between polls (default: 5)"
            echo "  --env_cmd CMD       Python env activation command"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

SERVERS="${CUSTOM_SERVERS:-$SNAP_SERVERS}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "  SNAP Cluster Task Watcher Manager"
echo "========================================"
echo "  Action:     $ACTION"
echo "  Queue dir:  $QUEUE_DIR"
echo "  Servers:    $SERVERS"
echo "  Session:    $TMUX_SESSION_NAME"
echo "========================================"
echo ""

# Ensure queue directory structure exists (on shared filesystem, only need to do once)
if [ "$ACTION" = "start" ]; then
    mkdir -p "$QUEUE_DIR"/{pending,running,completed,failed,logs}
    echo "Queue directory initialized: $QUEUE_DIR"
    echo ""
fi

for server in $SERVERS; do
    echo -n "[$server] "

    case $ACTION in
        start)
            # Check if watcher is already running
            EXISTING=$(ssh -o ConnectTimeout=5 "$server" "tmux has-session -t $TMUX_SESSION_NAME 2>/dev/null && echo 'yes' || echo 'no'" 2>/dev/null)

            if [ "$EXISTING" = "yes" ]; then
                echo -e "${YELLOW}Watcher already running (tmux session '$TMUX_SESSION_NAME' exists). Skipping.${NC}"
                continue
            fi

            # Start watcher in a new tmux session
            ssh -o ConnectTimeout=5 "$server" bash -l <<REMOTE_CMD 2>/dev/null
                # Activate Python environment
                $PYTHON_ENV_CMD 2>/dev/null || true

                # AFS reauth (if SU_PASSWORD is set)
                if [ -n "\$SU_PASSWORD" ]; then
                    echo "\$SU_PASSWORD" | /afs/cs/software/bin/reauth 2>/dev/null || true
                fi

                # Start tmux session with watcher
                tmux new-session -d -s "$TMUX_SESSION_NAME" \
                    "bash -l -c '$PYTHON_ENV_CMD 2>/dev/null; python -m uutils.snap_cluster.watcher --queue_dir $QUEUE_DIR --poll_interval $POLL_INTERVAL --server_name $server 2>&1 | tee $QUEUE_DIR/logs/watcher_${server}.log'"
REMOTE_CMD

            if [ $? -eq 0 ]; then
                echo -e "${GREEN}Watcher started in tmux session '$TMUX_SESSION_NAME'${NC}"
            else
                echo -e "${RED}Failed to start watcher (SSH connection failed?)${NC}"
            fi
            ;;

        stop)
            ssh -o ConnectTimeout=5 "$server" "tmux kill-session -t $TMUX_SESSION_NAME 2>/dev/null" 2>/dev/null
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}Watcher stopped${NC}"
            else
                echo -e "${YELLOW}No watcher running (or SSH failed)${NC}"
            fi
            ;;

        status)
            RESULT=$(ssh -o ConnectTimeout=5 "$server" bash -l <<'REMOTE_STATUS' 2>/dev/null
                if tmux has-session -t snap_task_watcher 2>/dev/null; then
                    echo "RUNNING"
                else
                    echo "STOPPED"
                fi
REMOTE_STATUS
            )

            if [ "$RESULT" = "RUNNING" ]; then
                echo -e "${GREEN}Watcher RUNNING${NC}"
            elif [ "$RESULT" = "STOPPED" ]; then
                echo -e "${RED}Watcher STOPPED${NC}"
            else
                echo -e "${YELLOW}Unreachable (SSH timeout)${NC}"
            fi
            ;;
    esac
done

echo ""

# Show queue status if starting or checking status
if [ "$ACTION" = "start" ] || [ "$ACTION" = "status" ]; then
    echo "Queue status:"
    echo "  Pending:   $(ls -1 "$QUEUE_DIR/pending/" 2>/dev/null | wc -l) tasks"
    echo "  Running:   $(ls -1 "$QUEUE_DIR/running/" 2>/dev/null | wc -l) tasks"
    echo "  Completed: $(ls -1 "$QUEUE_DIR/completed/" 2>/dev/null | wc -l) tasks"
    echo "  Failed:    $(ls -1 "$QUEUE_DIR/failed/" 2>/dev/null | wc -l) tasks"
    echo ""
    echo "To submit a task:"
    echo "  python -m uutils.snap_cluster.submit --prompt 'Your task here' --queue_dir $QUEUE_DIR"
    echo ""
    echo "To check status:"
    echo "  python -m uutils.snap_cluster.submit --status --queue_dir $QUEUE_DIR"
fi
