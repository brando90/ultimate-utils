# Antigravity task: ultimate-utils issue #27

You are working in a git worktree of the Python library `ultimate-utils` (package `uutils`, source under `py_src/uutils/`, tests under `tests/`).
Worktree: `/lfs/skampere2/0/brando9/uu-worktrees/issues-27` (branch `agy/issues-27`). Work only inside this directory. Do not touch any other directory, repo, tmux session or process on this machine.

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
GitHub issue #27 "Slack automation: messaging, channel monitoring, and notifications".

Create `py_src/uutils/slack_uu.py` with a `SlackClient` that talks to the official Slack Web API (`https://slack.com/api/<method>`) using plain `requests` (no `slack-sdk` needed).
- `SlackClient(token: str | None = None, dry_run: bool = True)` and `SlackClient.from_token(token_file: str = "~/keys/slack_bot_token.txt", dry_run: bool = True)`; `from_token` reads the file (or env var `SLACK_BOT_TOKEN`) only when `dry_run=False`.
- Methods: `send_message(channel, text, thread_ts=None)` (`chat.postMessage`), `upload_file(channel, file_path, title="")` (use the current external-upload flow `files.getUploadURLExternal` + upload + `files.completeUploadExternal`; `files.upload` is deprecated), `list_channels()` (`conversations.list`, handle cursor pagination), `get_channel_history(channel, limit=50, oldest=None)` (`conversations.history`), `get_unread_messages(channel, since_ts)` (history newer than `since_ts`).
- Every Slack call goes through one private `_call(method, **params)` that, in dry-run, returns `{"ok": True, "dry_run": True, "method": ..., "params": ...}` and prints a one-line description, and otherwise sends the request with `Authorization: Bearer <token>` and raises a clear error when the response has `"ok": false`.
- A convenience `notify(text, channel, dry_run=True)` for "experiment finished" notifications.
- A small CLI: `python -m uutils.slack_uu send --channel C --text T [--send]` (dry-run unless `--send`).
- Module docstring: one-time setup (create a Slack app, bot scopes `chat:write`, `channels:history`, `channels:read`, `files:write`, install to workspace, save bot token to `~/keys/slack_bot_token.txt` with `chmod 600`, invite bot to channels), and usage examples.
- Tests: `tests/test_slack_uu.py` (dry-run default makes no HTTP call; `_call` builds correct URL/headers/params with a mocked `requests`; pagination in `list_channels`; error on `ok: false`).

## When you are done
Reply with: the list of files you created/changed, the exact test command you ran and its final pass/fail line, and anything you could not do. Do not claim something works unless you ran it.

TL;DR: Implement the task above in this worktree only, with dry-run as the default that makes no network call or login, offline pytest tests that pass with the given command, no secrets, no LLM-provider API code, and leave the changes uncommitted for review.
