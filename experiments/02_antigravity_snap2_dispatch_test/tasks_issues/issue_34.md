# Antigravity task: ultimate-utils issue #34

You are working in a git worktree of the Python library `ultimate-utils` (package `uutils`, source under `py_src/uutils/`, tests under `tests/`).
Worktree: `/lfs/skampere2/0/brando9/uu-worktrees/issues-34` (branch `agy/issues-34`). Work only inside this directory. Do not touch any other directory, repo, tmux session or process on this machine.

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
GitHub issue #34 "WhatsApp + Claude integration: auto-reply to messages via Anthropic API" — **re-scoped**: the repo owner forbids direct LLM-provider API code, so the Claude part must use the `clauded -p` command-line tool through `subprocess` instead of the `anthropic` SDK.

An unmerged branch `origin/claude/whatsapp-integration-iRBAL` (one commit, f091f32) added to `py_src/uutils/whatsapp_uu.py`: `mark_as_read`, `WhatsAppClaudeBot`, `_verify_webhook_signature`, `_extract_messages`, `create_whatsapp_webhook_app()` (Flask), `run_whatsapp_bot()`, plus `playground/whatsapp_claude/test_whatsapp_claude_bot.py` and a `whatsapp` optional-dependency group in `pyproject.toml`. See it with `git fetch origin; git show f091f32 --stat; git show f091f32`.

Port that work onto the current `py_src/uutils/whatsapp_uu.py` (keep all existing functions working unchanged) with these changes:
1. **Remove every trace of the Anthropic SDK / API key** (`import anthropic`, `_load_anthropic_key`, `ANTHROPIC_API_KEY`, `~/keys/anthropic_api_key.txt`, `anthropic` in pyproject). Replace the reply generator with a pluggable `reply_fn: Callable[[str], str]`; the default real one is `clauded_reply(prompt: str, timeout: int = 300) -> str` which runs `["clauded", "-p", prompt]` with `subprocess.run(..., capture_output=True, text=True, timeout=timeout)` and returns stdout stripped (clear error on non-zero exit). The prompt given to `clauded` includes a short system instruction plus the recent conversation history for that contact (bounded, e.g. last 20 messages).
2. **Safety**: `WhatsAppClaudeBot(..., dry_run: bool = True, allowed_phones: set[str] | None = None, max_replies_per_hour: int = 10)`. Messages from phones not in `allowed_phones` (after normalisation) are ignored (default `None`/empty => ignore everyone). A per-phone rate limit applies. In dry-run the bot never calls `clauded`, never sends or marks-as-read over the network; it returns/prints the prompt it would send and the reply target.
3. Flask stays an optional lazy import; `create_whatsapp_webhook_app(bot=..., ...)` must verify the Meta `X-Hub-Signature-256` signature when an app secret is configured, reply 200 quickly and do the Claude call in a background thread (Meta requires a response within ~5 s). `run_whatsapp_bot()` defaults to dry-run and must refuse to start with `dry_run=False` unless `allowed_phones` is non-empty.
4. Put tests in `tests/test_whatsapp_claude_bot.py` as proper offline pytest tests (you may adapt the ones from the branch's playground script; do not add the playground script itself): allowlist rejection, rate limit, dry-run makes no `subprocess.run`/`requests.post` call, `clauded_reply` builds the right command with a monkeypatched `subprocess.run`, `_extract_messages` on a sample Meta payload, signature verification (valid/invalid). Flask-dependent tests must `pytest.importorskip("flask")`.
5. Update the module docstring: setup (Meta WhatsApp Business app, config JSON in `~/keys/whatsapp_api_config.json`, HTTPS webhook exposure), that Claude replies go through the `clauded` CLI (Claude Code subscription, no API key), and the allowlist/dry-run safety defaults.

## When you are done
Reply with: the list of files you created/changed, the exact test command you ran and its final pass/fail line, and anything you could not do. Do not claim something works unless you ran it.

TL;DR: Implement the task above in this worktree only, with dry-run as the default that makes no network call or login, offline pytest tests that pass with the given command, no secrets, no LLM-provider API code, and leave the changes uncommitted for review.
