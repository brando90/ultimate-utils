#!/usr/bin/env bash
# ==============================================================================
# Purpose: Idempotent setup script for uutils automation services on SNAP.
# Usage:   bash setup_snap_automation.sh [--check | --apply]
#
# GUARANTEE: This script NEVER modifies the user's crontab and NEVER modifies
#            or reads the contents of any files under ~/keys.
# ==============================================================================

set -euo pipefail

MODE="check"
if [[ $# -gt 0 ]]; then
    if [[ "$1" == "--apply" ]]; then
        MODE="apply"
    elif [[ "$1" == "--check" ]]; then
        MODE="check"
    else
        echo "Usage: $0 [--check | --apply]"
        exit 1
    fi
fi

# Variables
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHONPATH_DIR="$REPO_ROOT/py_src"
LOG_DIR="${HOME}/logs/uutils_automation"
CONFIG_DIR="${HOME}/.config/uutils_automation"

HAS_FAIL=0

print_status() {
    local status=$1
    local msg=$2
    if [[ "$status" == "PASS" ]]; then
        echo -e "[ \033[32mPASS\033[0m ] $msg"
    elif [[ "$status" == "WARN" ]]; then
        echo -e "[ \033[33mWARN\033[0m ] $msg"
    elif [[ "$status" == "FAIL" ]]; then
        echo -e "[ \033[31mFAIL\033[0m ] $msg"
        HAS_FAIL=1
    fi
}

echo "=== SNAP Automation Setup ($MODE mode) ==="

# 1. Hostname
HOSTNAME=$(hostname)
echo "Hostname: $HOSTNAME"

# 2. Python Candidates
echo "Checking Python candidates..."
CANDIDATES=("/usr/bin/python3" "${HOME}/uv_envs/veribench/bin/python")
if command -v python3 >/dev/null 2>&1; then
    PY3_PATH=$(command -v python3)
    if [[ ! " ${CANDIDATES[@]} " =~ " ${PY3_PATH} " ]]; then
        CANDIDATES+=("$PY3_PATH")
    fi
fi

for py in "${CANDIDATES[@]}"; do
    if [[ -x "$py" ]]; then
        # Check if uutils can be imported
        # Override PYTHONPATH specifically to include repo root py_src,
        # but respect system provided one in the shell if any (like snap's /dfs/scratch0/brando9/lib/...)
        if PYTHONPATH="$PYTHONPATH_DIR:${PYTHONPATH:-}" "$py" -c "import uutils" >/dev/null 2>&1; then
            print_status "PASS" "$py (imports uutils successfully)"
        else
            print_status "WARN" "$py (exists but fails to import uutils)"
        fi
    else
        print_status "WARN" "$py (not found or not executable)"
    fi
done

# 3. Credential files existence
echo "Checking credential files..."
CRED_FILES=(
    "keys/discord_webhook_url.txt"
    "keys/gmail_app_password.txt"
    "keys/twitter_api_config.json"
    "keys/whatsapp_api_config.json"
    "keys/slack_bot_token.txt"
    "keys/zuliprc"
    "keys/instagram_credentials.json"
    "keys/facebook_credentials.json"
)

for cred in "${CRED_FILES[@]}"; do
    if [[ -f "${HOME}/${cred}" ]]; then
        print_status "PASS" "~/${cred} (found)"
    else
        print_status "WARN" "~/${cred} (missing - required if you use this service)"
    fi
done

# 4. logrotate and flock availability
echo "Checking dependencies..."
if command -v logrotate >/dev/null 2>&1 || [[ -x "/usr/sbin/logrotate" ]]; then
    print_status "PASS" "logrotate is available"
else
    print_status "FAIL" "logrotate not found"
fi

if command -v flock >/dev/null 2>&1; then
    print_status "PASS" "flock is available"
else
    print_status "FAIL" "flock not found"
fi

# 5. Crontab
echo "Checking current crontab for uutils/watcher..."
# read-only crontab -l, ignore error if no crontab
CRON_OUT=$(crontab -l 2>/dev/null || true)
if [[ -z "$CRON_OUT" ]]; then
    print_status "WARN" "No crontab found for user"
else
    # Find lines matching uutils or watcher
    MATCHES=$(echo "$CRON_OUT" | grep -Ei 'uutils|watcher' || true)
    if [[ -n "$MATCHES" ]]; then
        echo "Found related crontab entries:"
        echo "$MATCHES" | sed 's/^/  /'
        print_status "PASS" "Crontab entries for uutils/watcher exist"
    else
        print_status "WARN" "No crontab entries mentioning uutils or watcher found"
    fi
fi

if [[ "$MODE" == "apply" ]]; then
    echo "=== Applying Setup ==="

    if [[ -d "$LOG_DIR" ]]; then
        print_status "PASS" "Log directory already exists at $LOG_DIR"
    else
        mkdir -p "$LOG_DIR"
        print_status "PASS" "Created log directory at $LOG_DIR"
    fi

    if [[ -d "$CONFIG_DIR" ]]; then
        print_status "PASS" "Config directory already exists at $CONFIG_DIR"
    else
        mkdir -p "$CONFIG_DIR"
        print_status "PASS" "Created config directory at $CONFIG_DIR"
    fi

    TEMPLATE_PATH="$(dirname "${BASH_SOURCE[0]}")/logrotate.conf.template"
    if [[ -f "$TEMPLATE_PATH" ]]; then
        TEMP_CONF=$(mktemp)
        sed "s|@LOG_DIR@|$LOG_DIR|g" "$TEMPLATE_PATH" > "$TEMP_CONF"

        if [[ -f "$CONFIG_DIR/logrotate.conf" ]]; then
            if cmp -s "$TEMP_CONF" "$CONFIG_DIR/logrotate.conf"; then
                print_status "PASS" "logrotate.conf already exists and matches template"
            else
                echo "Diff between existing and new logrotate.conf:"
                diff "$CONFIG_DIR/logrotate.conf" "$TEMP_CONF" || true
                mv "$CONFIG_DIR/logrotate.conf" "$CONFIG_DIR/logrotate.conf.bak"
                mv "$TEMP_CONF" "$CONFIG_DIR/logrotate.conf"
                print_status "PASS" "Updated logrotate.conf (old version backed up to .bak)"
            fi
        else
            mv "$TEMP_CONF" "$CONFIG_DIR/logrotate.conf"
            print_status "PASS" "Generated logrotate config at $CONFIG_DIR/logrotate.conf"
        fi

        if [[ -f "$CONFIG_DIR/logrotate.state" ]]; then
            print_status "PASS" "logrotate state file already exists"
        else
            touch "$CONFIG_DIR/logrotate.state"
            print_status "PASS" "Created logrotate state file at $CONFIG_DIR/logrotate.state"
        fi
    else
        print_status "FAIL" "Could not find logrotate.conf.template"
    fi
fi

echo "=== Summary ==="
if [[ $HAS_FAIL -eq 1 ]]; then
    echo "Check completed with FAILURES."
    exit 1
else
    echo "Check completed successfully (No FAILs)."
    exit 0
fi
