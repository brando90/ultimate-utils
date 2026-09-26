# Experiment 02 — bugs scope (`uu-agy-bugs`): issues #38, #39, #29 via Antigravity on SNAP-2

**TLDR:** Antigravity (`agy` 1.2.7, `gemini-3.1-pro-high`) on skampere2 produced all three landed changes: #39 on the first attempt, #38 and #29 on the second after specific review feedback. The Claude coordinator verified each on skampere2. #38 and #39 are closed; #29 landed its agent-doable part and stays open, blocked on Brando. Comparison workers: Cursor (`composer-2.5`) solved #39 correctly in 69 s; Grok (`grok-4.7`) hit its free-tier limit and produced nothing usable.

Coordinator: Claude Code, `claude-opus-5-5`, Mac. Host: `skampere2.stanford.edu`, `PATH=/dfs/scratch0/brando9/bin:$PATH`.
Run logs on skampere2: `/lfs/skampere2/0/brando9/uu-worktrees/_bugs_runs/<agent>-<issue>/{task.md,out.jsonl,stderr,exit,start,end}`.
Prompts and repro scripts in this repo: `bugs/tasks/task_{38,39,29}.md`, `bugs/launch.sh`, `bugs/repro_38.sh`, `bugs/repro_39.py`, `bugs/before.log`.

## Client checks (09-26-2026, skampere2)

| Client | Version | Auth / ping | Model chosen |
|---|---|---|---|
| Antigravity `agy` | 1.2.7 | `agy -p "reply with OK"` → `OK` in 5.5 s | `gemini-3.1-pro-high` (Pro tier, high reasoning; also listed: `gemini-3.8-flash-high`, `gemini-3.7-flash-*`, `gemini-3.6-flash-*`, `gemini-3.1-pro-low`) |
| Grok `grok` | 1.0.34 | `grok models` prints "You are not authenticated" but `grok -p "reply with OK"` → `OK` | `grok-4.7` |
| Cursor `cursor-agent` | 2026.09.18-9a7762b | `status`: logged in as brandojazz@gmail.com; ping → `OK` in 43 s | `composer-2.5` (Cursor's own model) |

**Recipe finding:** the dispatch recipe's `agy -p --dangerously-skip-permissions --model … "<prompt>"` fails immediately: `Error: -p took "--dangerously-skip-permissions" as its prompt`. Working form: `agy --dangerously-skip-permissions --model <id> --output-format stream-json -p "<prompt>"`. The first three `agy` launches (logs in `agy-<n>/attempt0/`) died this way; they are dispatch errors, not model failures, and were relaunched at 16:29 PDT. Sent to the other two coordinators.

## Before the fix (reproduced on skampere2, origin/main `424ab60`)

- **#38:** the unmodified `start_watcher.sh` under system `python3` (isolated session/job dir) printed `[OK] Watcher running`, exited 0, and the tmux session was **dead after 8 s**. System `/usr/bin/python3` is 3.12 without `dill`.
- **#39:** `/dfs/scratch0/brando9/bin/clauded` starts with `#\!/bin/bash` (backslash before `!`), so it has no valid shebang. With a cron-like `PATH=/dfs/scratch0/brando9/bin:/usr/local/bin:/usr/bin:/bin`: `WARNING Failed to dispatch lifecycle email: [Errno 8] Exec format error: 'clauded'`. With the good `/afs/cs.stanford.edu/u/brando9/bin/clauded` later on `PATH`, CPython silently skips the broken file, so the bug hides in interactive shells while `shutil.which` still reports the broken path.

Full output: `bugs/before.log`.

## Tasks

| Issue | Agent / model | Wall time | Outcome | Commit | Log |
|---|---|---|---|---|---|
| #39 | Antigravity `gemini-3.1-pro-high` | 122 s (1 attempt) | **landed**; coordinator re-ran `pytest tests/` (21 passed; 32 after rebase), `py_compile`, `git diff --check`, secret scan, after-repro | [`03a55d4`](https://github.com/brando90/ultimate-utils/commit/03a55d4) | `_bugs_runs/agy-39/` |
| #39 | Cursor `composer-2.5` (comparison) | 69 s | correct and slightly more polished (5 tests, one INFO line for `--no-lifecycle-email`); not landed because the experiment measures Antigravity | — | `_bugs_runs/cursor-39/` |
| #38 | Antigravity `gemini-3.1-pro-high` | attempt 1: 167 s; attempt 2: 375 s | attempt 1 **rejected** (bare `python3` in the tmux command, relied on the caller's env, which a running tmux server does not pass on; stray `test_script.sh`); attempt 2 **landed** unchanged by the coordinator | [`210ecea`](https://github.com/brando90/ultimate-utils/commit/210ecea) | `_bugs_runs/agy-38/{attempt1,}` |
| #38 | Grok `grok-4.7` (comparison) | 352 s | **Grok failed**: exit 1, `You've reached your free Grok Build usage limit for now. Get SuperGrok…`; partial uncommitted edit only | — | `_bugs_runs/grok-38/` |
| #29 | Antigravity `gemini-3.1-pro-high` | attempt 1: 121 s; attempt 2: 97 s | attempt 1 **rejected** (hard-coded temp worktree path, nonexistent `watcher.py` cron job, library smoke tests scheduled as services, unverified login-node/SLURM claims); attempt 2 **landed with coordinator edits** (commented out the logrotate cron line and used `/usr/sbin/logrotate`, fixed the manual-start note, updated module/credential lists for modules other coordinators landed meanwhile). Issue stays **open, blocked on Brando** (credentials, choosing periodic jobs, installing the crontab) | [`49a58c0`](https://github.com/brando90/ultimate-utils/commit/49a58c0) | `_bugs_runs/agy-29/{attempt1,}` |

## After the fix

- **#39** (`03a55d4`, same minimal PATH as the before run): `WARNING skipping /dfs/scratch0/brando9/bin/clauded: not exec-ready (first line '#\\!/bin/bash' is not a shebang)` → `Smart-job agent: codex (/usr/local/bin/codex)`; dry-run: `DRY-RUN lifecycle email via /usr/local/bin/codex`, nothing launched; non-dry phase: the repro's guard replaced the prompt with `--version`, which executed without ENOEXEC. No email was sent.
- The host wrapper `/dfs/scratch0/brando9/bin/clauded` is still broken; not changed (shared host file outside the repo). The #39 comment recommends replacing it with the working `/afs/cs.stanford.edu/u/brando9/bin/clauded`.

- **#38** (`210ecea`; isolated sessions, `--no-lifecycle-email`, `PATH=/usr/bin:/bin`; full output `bugs/after_38.log`): `PYTHON=/usr/bin/python3` → `[FAIL] ... No module named 'dill'`, exit 1, no session; veribench venv → `[OK]`, exit 0, session **alive** after 8 s; auto-select → veribench venv, `[OK]`, alive. `bash -n` and `git diff --check` clean.
- **#29** (`49a58c0`; `bugs/after_29.log`): `bash -n` ok; `--check` exit 0 (read-only; `gmail_app_password.txt` present, other credential files missing); `--apply` twice in a scratch HOME, second run reports every file as existing; `logrotate -d` parses the rendered config; real crontab unchanged (2 lines).

## Issues

- #39: closed by `03a55d4`; evidence comment https://github.com/brando90/ultimate-utils/issues/39#issuecomment-5850966043
- #38: closed by `210ecea`; evidence comment https://github.com/brando90/ultimate-utils/issues/38#issuecomment-5851027429
- #29: **open**, blocker comment https://github.com/brando90/ultimate-utils/issues/29#issuecomment-5851038550 (credentials in `~/keys`, choose the first periodic job, install the crontab on one host, live-test with `--send`)

## Safety notes

- A tmux server is already running on skampere2, so new sessions inherit the **server's** global environment, not the caller's. A `PATH=` or `UUTILS_WATCHER_NOTIFY_DRY_RUN=` set by the caller never reaches the daemon. Antigravity's first #38 test started a real watcher (main-branch code). Its startup email failed only because the broken `clauded` wrapper raised ENOEXEC (`_scratch_bugs/uu-bugs-agy-38-jobq/logs/watcher_tmux_skampere2.log`). No run logged `Dispatched lifecycle email` except the guarded `--version` repro.
- Grok's `grok models` says "You are not authenticated", yet `-p` works until the free-tier quota runs out. The `uu-agy-issues` coordinator saw the same limit on #24 and #32.

## Verdict

**Yes, the SNAP-2 Antigravity agents work end to end, with coordinator review.** Headless `agy` on skampere2 authenticated, answered in seconds, edited its own worktree, and ran the tests it was told to run. All three tasks landed from Antigravity output with `gemini-3.1-pro-high`:
- **#39:** first attempt, 122 s.
- **#38 and #29:** second attempt each (375 s and 97 s).

Quality was **good on a precise spec, weaker on judgement**:
- **#39** (a fully specified code change with tests) was right the first time.
- **#38** missed an environment subtlety: tmux sessions don't inherit the caller's environment.
- **#29** (open-ended docs and ops work) invented a nonexistent `watcher.py` job and unverified cluster claims. It needed one rejection plus small coordinator edits.

Every rejection was fixed in one re-dispatch, so none needed a coordinator takeover.

**Dispatch lessons:**
1. The recipe's `agy -p --dangerously-skip-permissions ...` form fails instantly. Put the flags first and `-p "<prompt>"` last.
2. Grok on skampere2 is on the free tier: it hit its usage limit after about 6 min, and all three coordinators saw the same.
3. Cursor (`composer-2.5`) was the fastest correct worker (69 s on #39).
4. An agent test that starts a daemon on a shared host needs a hard kill switch that doesn't rely on environment variables (here `--no-lifecycle-email`). A running tmux server drops the caller's environment, and only the broken `clauded` wrapper stopped one test watcher from emailing.

Worktrees `bugs-{38,39,29}` and `bugs-38-grok` / `bugs-39-cursor` and run logs are kept under `/lfs/skampere2/0/brando9/uu-worktrees/` for inspection.
