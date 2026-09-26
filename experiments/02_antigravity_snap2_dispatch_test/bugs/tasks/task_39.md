# Task: fix issue #39 — watcher agent launch fails with "Exec format error: clauded"

You are working in a git worktree of the `ultimate-utils` Python library on the SNAP cluster host skampere2. Your working directory is `__WT__` (branch `__BRANCH__`). Work only inside this directory. Do not commit, push, or touch any other directory, tmux session, or git clone. Never send an email or run a coding-agent CLI with a real prompt.

## The bug
`py_src/uutils/job_scheduler_uu/scheduler.py` picks a coding-agent CLI in `_find_agent_binary()` with `shutil.which(name)` and then launches the *bare name* (`["clauded", "-p", prompt]`) via `subprocess.Popen`, both for the daemon lifecycle email (`_send_daemon_lifecycle_email`) and for smart-mode jobs. On SNAP, `/dfs/scratch0/brando9/bin/clauded` is a text file whose first line is `#\!/bin/bash` (a backslash before the `!`), so it is not a valid shebang and `execve` fails with `[Errno 8] Exec format error`. Reproduced on skampere2 with `PATH=/dfs/scratch0/brando9/bin:/usr/local/bin:/usr/bin:/bin`:
```
INFO Smart-job agent: clauded (/dfs/scratch0/brando9/bin/clauded)
WARNING Failed to dispatch lifecycle email: [Errno 8] Exec format error: 'clauded'
```
When a valid `clauded` appears later on PATH, CPython silently skips the broken one, so `shutil.which` reports one file while another is executed. Interactive shells hide all of this because `clauded` is often a shell alias; the fix must never depend on aliases, shell functions, or `shell=True`.

## Required fix (scheduler.py, plus a new test file)
1. Add `_is_exec_ready(path) -> bool`: true only if the path is a regular file, `os.access(path, os.X_OK)`, and its first bytes are a real shebang `#!` or a native binary magic (ELF `\x7fELF`; Mach-O `\xcf\xfa\xed\xfe`, `\xfe\xed\xfa\xcf`, `\xca\xfe\xba\xbe`).
2. Add `_which_exec_ready(name) -> Optional[str]`: walk `os.environ["PATH"]` in order and return the first absolute path for `name` that is exec-ready. For each executable candidate that is *not* exec-ready, log one WARNING naming the file and its first line (e.g. `skipping /x/clauded: not exec-ready (first line '#\\!/bin/bash' is not a shebang)`).
3. `_find_agent_binary()` keeps its priority (clauded > codex > claude) but uses `_which_exec_ready`, puts the resolved **absolute path** as `cmd[0]`, and falls through to the next agent when a name has no exec-ready candidate. Keep the return type `(display_name, cmd_prefix)`.
4. Dry run for notifications: if the environment variable `UUTILS_WATCHER_NOTIFY_DRY_RUN` is `1`/`true`/`yes`, `_send_daemon_lifecycle_email` must not start any process; it logs `DRY-RUN lifecycle email via <abs path>: <event>` instead. Also add a CLI flag `--no-lifecycle-email` to the daemon's argparse that disables the lifecycle email entirely (log one INFO line), and thread it through to where `_send_daemon_lifecycle_email` is called.
5. If `Popen` still raises `OSError`, the WARNING must name the absolute path, errno, and the hint "check the file's shebang line".
6. Update docstrings/comments to match. Do not change the smart-job prompt text or any email addresses.
7. Add `tests/test_job_scheduler_agent_resolution.py` (pytest, uses `tmp_path` and `monkeypatch`, no network, no real agents): (a) a broken no-shebang `clauded` first on PATH and a valid `#!/bin/sh` `clauded` later → resolves to the valid one's absolute path; (b) only a broken `clauded` plus a valid `codex` → picks codex; (c) nothing exec-ready → `None`; (d) with `UUTILS_WATCHER_NOTIFY_DRY_RUN=1` the lifecycle email never calls `subprocess.Popen` (monkeypatch it to raise); (e) the resolved absolute path from (a) actually executes with `subprocess.run([path, "--version"])` without `OSError` (the stub prints something and exits 0).

## How to test
Run `PYTHONPATH=$PWD/py_src $HOME/uv_envs/veribench/bin/python -m pytest -q tests/test_job_scheduler_agent_resolution.py` and `... -m pytest -q tests/` (existing tests must still pass). Also run `python -m py_compile` on scheduler.py. Do not run `clauded`, `codex`, or `claude` yourself.

## Deliverable
Leave the changes uncommitted in the worktree. At the end, print a short report: what you changed, the test commands you ran, and their output.

TL;DR: resolve agent CLIs to absolute paths of files that are really executable (valid shebang or binary), skip and warn about broken wrappers, add a notification dry-run and a --no-lifecycle-email flag, and cover it with pytest; never send email or call a real agent.
