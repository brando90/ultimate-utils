# Antigravity task: ultimate-utils issue #40

You are working in a git worktree of the Python library `ultimate-utils` (package `uutils`, source under `py_src/uutils/`, tests under `tests/`).
Worktree: `/lfs/skampere2/0/brando9/uu-worktrees/issues-40` (branch `agy/issues-40`). Work only inside this directory. Do not touch any other directory, repo, tmux session or process on this machine.

## Hard safety rules (a violation means the work is rejected)
1. **Dry-run is the default everywhere.** Every function or CLI that could send, post, upload, message, or log in to any external service takes `dry_run: bool = True` (CLI: acts only with an explicit `--send` / `--execute` flag). In dry-run mode the code makes **no network request, no login, reads no credential file**, and prints/logs/returns a description of what it would do. It must work on a machine with no `~/keys/` files at all.
2. **Never run a non-dry-run path yourself.** Do not send any email, SMS, WhatsApp, Slack, Zulip, Facebook or Instagram message, do not upload anything, do not log in to any account, do not create files under `~/keys/`.
3. **No secrets.** Credentials are only referenced by path (e.g. `~/keys/<service>...`) or env var name; never write a token, key, password or real phone number/address into the repo. Use obviously fake examples (`+15550000000`, `xoxb-FAKE`, `example@example.com`).
4. **No direct LLM-provider API code.** Never import or call `anthropic`, `openai`, `google.genai`, `litellm`, or raw HTTP to their APIs. If an LLM is needed, shell out to the `clauded -p "<prompt>"` CLI via `subprocess` (only outside dry-run).
5. Third-party SDKs are optional: prefer plain `requests` for HTTP; import any optional dependency lazily inside the function that needs it, so `import uutils.<module>` works without it. Declare optional deps in a new group under `[project.optional-dependencies]` in `pyproject.toml` if you need one; do not change the required `dependencies` list.

## Engineering requirements
- Match the style of existing modules such as `py_src/uutils/whatsapp_uu.py` and `py_src/uutils/emailing.py`: a module docstring with setup steps (what the human must do once, where credentials go, how to run), type hints, small functions.
- Write pytest tests in `tests/` that run **offline** (monkeypatch `requests`/`subprocess`; use `tmp_path` for files). Include at least one test proving that calling the public API with default arguments makes no network call (e.g. patch `requests.post`/`requests.get`/`subprocess.run` to raise and assert the default call still succeeds).
- Run the tests with exactly this command from the worktree root and make them pass:
  `PYTHONPATH=py_src /lfs/skampere2/0/brando9/uu-agy-issues-venv/bin/python -m pytest -q <your test files>`
  (Do not `pip install` anything into that venv and do not `pip install -e` the package.)
- Run `git diff --check` and fix whitespace errors.
- **Do not commit and do not push.** Leave your changes as uncommitted edits/new files in the worktree; the coordinator reviews and commits them.

## The task
GitHub issue #40 "Email-triggered Claude agent: reply from allowlisted addresses to run a Claude session and get an answer back in-thread" — **re-scoped**: the repo owner forbids direct LLM-provider API code, so the Claude session must run through the `clauded -p` command-line tool via `subprocess`, not the `anthropic` SDK or `claude-agent-sdk`.

An unmerged branch `origin/claude/email-reply-automation-a1mAv` already implemented it under `experiments/email_reply_bot/` (allowlist, SPF/DKIM/DMARC header checks, Reply-To/Return-Path anti-spoof, rate limit, idempotent store, threaded reply, 54 tests). Bring it onto this branch at the new path `experiments/03_email_reply_bot/`:
`git fetch origin; mkdir -p experiments/03_email_reply_bot; git archive origin/claude/email-reply-automation-a1mAv experiments/email_reply_bot | tar -x --strip-components=2 -C experiments/03_email_reply_bot`
Then fix every reference to the old path (`experiments.email_reply_bot` module paths, docs).

Required changes:
1. Replace `src/real_llm.py`'s `AnthropicClient` with `ClaudeCLIClient` implementing the existing `LLMClient.run(*, prompt, system, workdir=None) -> str` protocol by running `["clauded", "-p", <system + "\n\n" + prompt>]` with `subprocess.run(..., cwd=workdir, capture_output=True, text=True, timeout=...)`, returning stripped stdout and raising a clear error on non-zero exit or timeout. Remove `anthropic` from `requirements.txt`, from `PLAN.md`/`README.md`/`ISSUE.md` wording where it describes the implementation (say `clauded -p` instead), and from config (`llm.model`/`max_tokens` become optional CLI settings such as `llm.timeout_s`).
2. **Safe defaults in `src/main.py`**: with default arguments it must not log in to Gmail, not call `clauded`, and not send mail. Add `--live` (required to build the real Gmail client and real `ClaudeCLIClient`) and `--send` (required, together with `--live`, for outbound mail; otherwise pipeline `dry_run=True`). Without `--live`, `main` should run the pipeline once over local `.eml` files given by `--fixtures DIR` (default: the bundled `tests/fixtures`) with a fake Gmail client and a fake echo LLM, print each accept/reject decision, and exit 0 — a safe demo of the security filter.
3. Keep all the existing tests passing (adjust imports for the new path) and add tests for: `ClaudeCLIClient` command construction with a monkeypatched `subprocess.run`; `main([])` with defaults touches neither `subprocess.run` nor the Gmail client builder nor any send (patch them to raise) and returns 0; `--send` without `--live` is rejected or has no effect.
4. Update `README.md` with: what it does, the allowlist, how to run the offline demo, the one-time human steps for live use (Gmail OAuth client + token under `~/keys/`, an always-on host, `clauded` logged in), and that the daemon only acts with `--live --send`.
Run tests with: `cd experiments/03_email_reply_bot && PYTHONPATH=. /lfs/skampere2/0/brando9/uu-agy-issues-venv/bin/python -m pytest -q tests` (they must not need Google libraries; `pyyaml` is installed).

## When you are done
Reply with: the list of files you created/changed, the exact test command you ran and its final pass/fail line, and anything you could not do. Do not claim something works unless you ran it.

TL;DR: Implement the task above in this worktree only, with dry-run as the default that makes no network call or login, offline pytest tests that pass with the given command, no secrets, no LLM-provider API code, and leave the changes uncommitted for review.
