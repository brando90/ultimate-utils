# Task: fix issue #38 — start_watcher.sh reports [OK] while the watcher dies

You are working in a git worktree of the `ultimate-utils` Python library on the SNAP cluster host skampere2. Your working directory is `__WT__` (branch `__BRANCH__`). Work only inside this directory. Do not commit, push, or touch any other directory, tmux session, or git clone.

## The bug
`py_src/uutils/job_scheduler_uu/start_watcher.sh` launches the job-queue daemon (`python -m uutils.job_scheduler_uu.scheduler`) inside a tmux session. It uses `${PYTHON:-python3}`, never sets `PYTHONPATH`, and prints `[OK] Watcher running` right after `tmux new-session`. On skampere2 the default `/usr/bin/python3` (3.12) lacks the `dill` dependency and `uutils` is not importable, so the daemon crashes within a second, the tmux session disappears, and the script still exits 0 with `[OK]`. Reproduced on skampere2: script exit code 0, session dead after 8 s.

## Required fix (edit only start_watcher.sh, plus a short note in its header comment)
1. Keep the variable names `SESSION_NAME` and `JOB_DIR`, but let the environment override them: `WATCHER_SESSION` (default `job_watcher`) and `WATCHER_JOB_DIR` (default `${HOME}/dfs/job_queue`).
2. Compute the repo's `py_src` from the script's own location (`BASH_SOURCE`, not `$HOME/ultimate-utils`) and prepend it to `PYTHONPATH` for the daemon.
3. Choose the Python interpreter: `$PYTHON` if set; else `$VIRTUAL_ENV/bin/python` if a venv is active; else the first of `python3` (on PATH), `$HOME/uv_envs/veribench/bin/python`, `/usr/bin/python3` that can `import uutils.job_scheduler_uu.scheduler` with that PYTHONPATH.
4. Preflight: if the chosen interpreter cannot import `uutils.job_scheduler_uu.scheduler`, print `[FAIL]` with the interpreter path, the last line of the import error, and how to fix it (set `PYTHON=`, activate a venv, or install `dill`), and exit 1 without creating a tmux session.
5. Send the daemon's stdout+stderr to `${JOB_DIR}/logs/watcher_tmux_$(hostname -s).log` (append) so a crash is visible after the session dies; keep the daemon's exit status (use `set -o pipefail` inside the tmux command or redirect instead of piping).
6. Liveness check: after `tmux new-session`, sleep `${WATCHER_STARTUP_CHECK_SECS:-5}` seconds, then `tmux has-session`. If the session is gone, print `[FAIL]` plus the last 20 lines of that log and exit 1. Print `[OK]` only after the check passes.
7. Keep existing behaviour: the `unset ANTHROPIC_API_KEY ...` line, the "session already exists" early exit, argument forwarding with quoting, `mkdir -p` of the queue dirs.
8. Do not rely on shell aliases or functions anywhere.

## How to test (on this host) — safety rules
- Never use the session name `job_watcher` and never use `~/dfs/job_queue`. Always set `WATCHER_SESSION=__SESSION_PREFIX__-t<N>` and `WATCHER_JOB_DIR=/lfs/skampere2/0/brando9/uu-worktrees/_scratch_bugs/__SESSION_PREFIX__-jobq`.
- The daemon sends a "started" email through a coding-agent CLI if one is on PATH. To guarantee no email is sent, run every test with `PATH=/usr/local/bin:/usr/bin:/bin` (no agent CLIs there) and `UUTILS_WATCHER_NOTIFY_DRY_RUN=1`.
- Kill every tmux session you start (`tmux kill-session -t <name>`) before you finish.
- Test cases: (a) `PYTHON=/usr/bin/python3` must fail fast with `[FAIL]` and exit 1 and leave no session; (b) `PYTHON=$HOME/uv_envs/veribench/bin/python` must print `[OK]`, exit 0, and the session must still be alive 10 s later; (c) no `PYTHON` set and no venv: the auto-selection must find a working interpreter and succeed; (d) `bash -n` on the script passes.

## Deliverable
Leave the edited file uncommitted in the worktree. At the end, print a short report: what you changed, the exact test commands you ran, and their output (exit codes and the [OK]/[FAIL] lines).

TL;DR: make start_watcher.sh pick a working Python, set PYTHONPATH from its own location, fail loudly on import errors, log the daemon, and only print [OK] after confirming the tmux session survived; test only with isolated session names and no agent CLIs on PATH.

---
# REVIEW FEEDBACK on your first attempt (already in this worktree — revise it, do not start over)

Your first attempt is close, but the coordinator rejected it for these concrete reasons:
1. **tmux does not pass the caller's environment.** When a tmux server is already running (it is on skampere2), `tmux new-session` gives the new session the *server's* global environment, not the environment of the shell that ran start_watcher.sh. So `PATH`, `VIRTUAL_ENV` and `UUTILS_WATCHER_NOTIFY_DRY_RUN` from the caller are lost, and a bare interpreter name is re-resolved against the server's PATH. Fix: (a) resolve the chosen interpreter to an **absolute path** (`command -v`) before the preflight, and use that absolute path in the tmux command and in the `Python:` line; (b) inside the tmux command, explicitly `export PATH=…` and `export PYTHONPATH=…` with the caller's values, and also export `UUTILS_WATCHER_NOTIFY_DRY_RUN` when it is set. Quote each value safely with bash `printf '%q'` (not hand-written single quotes).
2. Delete the stray `test_script.sh` from the worktree root. Only `py_src/uutils/job_scheduler_uu/start_watcher.sh` may change.
3. Rewrite the header comment: no "Modifications for issue #38" changelog; instead document the behaviour and the environment variables `PYTHON`, `WATCHER_SESSION`, `WATCHER_JOB_DIR`, `WATCHER_STARTUP_CHECK_SECS`, and the log path.

Safety for re-testing: after making fix 1, run every test with `PATH=/usr/local/bin:/usr/bin:/bin` so no agent CLI is visible to the daemon, and check the daemon log. It must say `No agent binary for lifecycle email`. If a log line ever says `Dispatched lifecycle email`, stop immediately and report it. Re-run test cases (a)–(d) and add (e): start with `PATH=/usr/local/bin:/usr/bin:/bin`, then print the daemon process's real PATH with `tr '\0' '\n' < /proc/<pid>/environ | grep ^PATH=` to prove the caller's PATH reached it. Kill all your tmux sessions at the end.

TL;DR: resolve the interpreter to an absolute path, explicitly export PATH/PYTHONPATH (and the dry-run var) inside the tmux command using printf %q, delete test_script.sh, clean up the header comment, and re-test safely with a no-agent PATH.
