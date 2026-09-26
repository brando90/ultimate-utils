# SNAP Automation Server Setup

This directory contains scripts and configurations for setting up `uutils` automation services on the SNAP cluster.

## 1. SNAP Node Selection
- **Which node to use:** The skampere machines are shared SNAP servers that all mount `/dfs/scratch0/brando9`. Install the crontab on exactly *one* host, preferably `skampere2`, which already has the `krenew.sh` and `@reboot` watcher lines, so jobs do not run twice.
- **`/dfs` Sharing:** Because all skampere hosts share the `/dfs/scratch0/brando9` storage, credentials and logs placed in shared directories are accessible across all nodes.

## 2. Python Environment Choice
- **System Python (`/usr/bin/python3`):** It is Python 3.12 but it does *not* have `dill` and might lack other packages.
- **Virtual Environment:** It is highly recommended to use `~/uv_envs/veribench/bin/python` (Python 3.11). This environment has the necessary dependencies installed. Ensure you point your cron jobs to this executable.

## 3. Human Setup Checklist (Issue #29)

### Done by the automation scripts:
- [x] Create idempotent setup script to check dependencies and generate logrotate configs.
- [x] Provide `crontab.example` with `flock` examples for concurrency control.
- [x] Provide `logrotate.conf.template` to manage log growth.
- [x] Document the manual steps, node selection, and debugging processes.

### Remaining steps for the human:
- [ ] **Run the setup script:** Execute `bash scripts/snap_automation/setup_snap_automation.sh --apply` to generate configuration files and log directories.
- [ ] **Copy credentials:** Manually create and copy the required credential files to `~/keys/` and set their permissions:
  - `~/keys/discord_webhook_url.txt`
  - `~/keys/gmail_app_password.txt`
  - `~/keys/twitter_api_config.json`
  - `~/keys/whatsapp_api_config.json`
  - `~/keys/slack_bot_token.txt`, `~/keys/zuliprc`, `~/keys/instagram_credentials.json`, `~/keys/facebook_credentials.json`
  - SMS: `~/keys/twilio_credentials.json` or `~/keys/tasker_autoremote_key.txt`
  - Run `chmod 600 ~/keys/*` to ensure safety.
- [ ] **Install crontab entries:** Run `crontab -e` and paste the relevant lines from `scripts/snap_automation/crontab.example`.
- [ ] **Write the periodic jobs you want** (for example a Google Drive sync or a daily summary). No periodic job script exists yet; the messaging modules are send-once CLIs and libraries (see `crontab.example`).
- [ ] **Test each service individually on SNAP:** blocked until the matching credentials exist in `~/keys`. Every module defaults to dry-run, so test with the dry-run CLI first, then once with `--send`.

## 4. Monitoring and Debugging

### Monitoring
- **Crontab:** Run `crontab -l` to see what jobs are actively scheduled.
- **Logs:** Check `~/logs/uutils_automation/` for execution logs (e.g., `tail -f ~/logs/uutils_automation/logrotate.log`).
- **Tmux:** Use `tmux ls` to see running terminal sessions (useful if any long-running scripts like the watcher were manually started).

### Debugging Failures
- **Kerberos/AFS Token Expiry:** If scripts suddenly lose permission to read files on shared drives, your Kerberos token likely expired. Ensure your crontab includes the renewal script: `0 */4 * * * /dfs/scratch0/brando9/bin/krenew.sh`.
- **Stale Lock Files:** The `crontab.example` uses `flock -n` to prevent overlapping runs. If a service crashes hard, a lockfile in `~/.config/uutils_automation/*.lock` might be left behind but `flock` ties it to the process ID, so stale locks usually don't block new runs. However, if jobs aren't starting, verify if the lock file is being held by a zombie process (`lsof ~/.config/uutils_automation/*.lock`).
- **Missing Dependencies:** Ensure you are using the correct Python binary (`~/uv_envs/veribench/bin/python`). Check the logs in `~/logs/uutils_automation/` for any `ModuleNotFoundError`.
