# Task: issue #29 — SNAP automation server for uutils services (agent-doable parts only)

You are working in a git worktree of the `ultimate-utils` Python library on the SNAP cluster host skampere2. Your working directory is `__WT__` (branch `__BRANCH__`). Work only inside this directory. Do not commit or push. Do not modify the user's crontab, `~/keys`, `~/.bashrc`, any tmux session you did not create, or anything outside this worktree except scratch files under `/lfs/skampere2/0/brando9/uu-worktrees/_scratch_bugs/`.

## Hard safety limits
Never send an email, Slack/Zulip/Discord/WhatsApp/Twitter message, or any network post; never log into anything; never read, print, or copy files under `~/keys` (you may only test whether a path exists with `test -f`). Never install crontab entries. Never call any LLM provider API. If a step needs credentials, write it as a documented manual step for the human.

## Context (verified on skampere2)
- Hosts share `/dfs/scratch0/brando9` (HOME on skampere2 is `/lfs/skampere2/0/brando9`; `~/dfs` → `/dfs/scratch0/brando9`). System python is `/usr/bin/python3` 3.12 without `dill`; a working venv is `~/uv_envs/veribench/bin/python` (3.11). `/usr/sbin/logrotate` exists. `tmux` 3.4.
- The user's crontab already has `0 */4 * * * /dfs/scratch0/brando9/bin/krenew.sh` (Kerberos/AFS renewal) and `@reboot /dfs/scratch0/brando9/bin/start_watcher_at_reboot.sh` (job-queue watcher). The job-queue watcher lives in `py_src/uutils/job_scheduler_uu/` (another agent is fixing `start_watcher.sh` in parallel; do not edit that directory).
- Messaging modules in the package: `py_src/uutils/emailing.py`, `discord_uu.py`, `twitter_uu.py`, `whatsapp_uu.py`. Read them to learn which credential file each expects; there is no Google Drive or Slack/Zulip module in this repo today.

## Deliverables
1. `scripts/snap_automation/setup_snap_automation.sh` — idempotent, safe to re-run, `set -euo pipefail`, `bash -n` clean, with `--check` (default: report only, change nothing) and `--apply` (create `~/logs/uutils_automation/` and write a user-level logrotate config + state file under `~/.config/uutils_automation/`; still never touches crontab or keys). `--check` reports: hostname, python candidates and whether each imports `uutils` with `PYTHONPATH=<repo>/py_src`, whether each expected credential path exists (existence only, no contents), whether `logrotate` is available, current crontab lines that mention uutils/watcher (read-only `crontab -l`), and prints PASS/WARN/FAIL per item with an exit code of 0 when nothing FAILs.
2. `scripts/snap_automation/crontab.example` — commented example entries, one per service, each wrapped with `flock -n` to prevent overlap and each redirecting to `~/logs/uutils_automation/<service>.log`, plus the logrotate invocation (`logrotate --state <state> <conf>`). Commented out by default; for services with no code in the repo, say so in the comment instead of inventing a module.
3. `scripts/snap_automation/logrotate.conf.template` — weekly, rotate 8, compress, missingok, notifempty, copytruncate, for `~/logs/uutils_automation/*.log` (the setup script substitutes the real path).
4. `scripts/snap_automation/README.md` — which SNAP node to use and why (login vs compute; skampere hosts; `/dfs` sharing), Python/venv choice, the one-time human steps (credential copy with `chmod 600`, `crontab -e` to install chosen lines), how to monitor (`crontab -l`, log paths, `tmux ls`) and debug failures (Kerberos/AFS token expiry via krenew, missing deps, stale lock files). Mark clearly which checklist items of issue #29 are done by this change and which remain for the human.

## How to test (run these and include the output)
`bash -n` on the script; `bash scripts/snap_automation/setup_snap_automation.sh --check`; `HOME=/lfs/skampere2/0/brando9/uu-worktrees/_scratch_bugs/fakehome29 bash scripts/snap_automation/setup_snap_automation.sh --apply` then run it again (idempotency) and `logrotate -d --state <state> <generated conf>` against the generated config (debug mode only, rotates nothing).

## Deliverable format
Leave the files uncommitted in the worktree. Print a short report at the end: files created, the test commands and their output, and the exact list of #29 checklist items still requiring the human.

TL;DR: add a safe check/apply setup script, example crontab, logrotate template and README for running uutils automation on SNAP; test them on skampere2 without touching crontab, keys, or sending anything; list what still needs the human.

---
# REVIEW FEEDBACK on your first attempt (files already in this worktree — revise them, do not start over)

The coordinator rejected the first attempt for these concrete reasons:
1. `crontab.example` hard-codes this temporary worktree (`/lfs/skampere2/0/brando9/uu-worktrees/bugs-29/...`) and a hard-coded HOME. Use `$HOME/ultimate-utils` (cron runs commands through /bin/sh, so `$HOME` expands in the command field; variable assignments at the top of a crontab do not expand). Never mention the worktree path in any committed file.
2. Entry 1 runs `job_scheduler_uu/watcher.py` every 5 minutes, but that file does not exist and the watcher is a long-lived daemon, not a periodic job. Replace it with a comment that documents the real mechanism: the existing `@reboot /dfs/scratch0/brando9/bin/start_watcher_at_reboot.sh` line, and `bash $HOME/ultimate-utils/py_src/uutils/job_scheduler_uu/start_watcher.sh` for a manual start (tmux session `job_watcher`).
3. Entries 3–6 schedule `discord_uu.py`, `twitter_uu.py`, `whatsapp_uu.py`, and `emailing.py` as cron jobs. These are libraries, and their `__main__` blocks are dry-run smoke tests, not services. Remove those entries. Instead, add one generic, commented template line with a placeholder (`<service>_job.py`) and state plainly that no periodic service entrypoint exists yet for email, Discord, Twitter, WhatsApp, Slack, Zulip, or Google Drive.
4. README: do not claim skampere hosts are "login nodes" or talk about SLURM preemption. Those claims are unverified. Say instead: the skampere machines are shared SNAP servers that all mount `/dfs/scratch0/brando9`. Install the crontab on exactly one host, preferably skampere2, which already has the `krenew.sh` and `@reboot` watcher lines, so jobs do not run twice. Update the checklist to list every issue #29 item as done or remaining. "Test each service individually on SNAP" stays remaining: it is blocked until credentials exist in `~/keys` and service entrypoints exist.
5. `setup_snap_automation.sh`: add a header comment (purpose, usage, the guarantee that it never touches crontab or `~/keys` contents). Use `${BASH_SOURCE[0]}` for paths. Also check that `flock` is available. In `--apply`, say "already exists" instead of "Created" when a directory or config already exists, and do not overwrite an existing `logrotate.conf` unless it differs from the rendered template (show a diff and keep the old one as `.bak`). Note: the SNAP shell sets `PYTHONPATH=/dfs/scratch0/brando9/lib/python3.12/site-packages`, so keep the explicit `PYTHONPATH=<repo>/py_src` override in the import check.

Re-run the tests from the task (bash -n, --check, --apply twice in the fake HOME, logrotate -d) and print their output.

TL;DR: remove invented cron jobs and the worktree path, document the real watcher mechanism, don't make unverified node claims, make --apply honestly idempotent, and re-test.
