# Antigravity task: ultimate-utils issue #32

You are working in a git worktree of the Python library `ultimate-utils` (package `uutils`, source under `py_src/uutils/`, tests under `tests/`).
Worktree: `/lfs/skampere2/0/brando9/uu-worktrees/issues-32` (branch `agy/issues-32`). Work only inside this directory. Do not touch any other directory, repo, tmux session or process on this machine.

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
GitHub issue #32 "Automate SMS/texting via Google Messages".

Create `py_src/uutils/sms_uu.py` with two backends, both via plain `requests`, both dry-run by default. (The Google Messages Web browser-automation option needs a paired phone and is out of scope here; mention it in the docstring as future work.)
- Twilio backend (reliable, sends from a separate Twilio number): credentials JSON `~/keys/twilio_credentials.json` with `account_sid`, `auth_token`, `from_number`, optional `self_number`. `send_sms(to, message)` = POST `https://api.twilio.com/2010-04-01/Accounts/<sid>/Messages.json` with basic auth `(sid, token)` and form fields `To`, `From`, `Body`.
- Tasker + AutoRemote backend (Android, sends from Brando's real number via the phone): key file `~/keys/tasker_autoremote_key.txt`; `send_sms(to, message)` = GET `https://autoremotejoaomgcd.appspot.com/sendmessage` with params `key` and `message` formatted as `sms=:=<to>=:=<message>` (document that a matching Tasker profile must exist on the phone).
- A `SMSClient` with `SMSClient.from_twilio(credentials_file=..., dry_run=True)`, `SMSClient.from_autoremote(key_file=..., self_number="", dry_run=True)`, methods `send_sms(to, message) -> dict` and `send_self_reminder(message) -> dict` (to `self_number`; clear error if missing). Credential files are read only when `dry_run=False`.
- Phone number validation/normalisation to E.164 (`+` and digits; strip spaces/dashes/parentheses; reject empty or too-short numbers) with a clear `ValueError`.
- Dry-run returns `{"dry_run": True, "backend": ..., "to": ..., "message": ...}` and prints a preview.
- CLI: `python -m uutils.sms_uu send --backend twilio|autoremote --to NUMBER --message M [--send]`.
- Module docstring: one-time setup for each backend (Twilio account + number, costs about $1/month plus per-message fees; Tasker + AutoRemote on Android), where credentials go (`chmod 600`), usage.
- Tests: `tests/test_sms_uu.py` (dry-run default no HTTP; Twilio request URL/auth/form with mocked `requests`; AutoRemote URL/params; normalisation cases; `send_self_reminder` without `self_number` errors).

## When you are done
Reply with: the list of files you created/changed, the exact test command you ran and its final pass/fail line, and anything you could not do. Do not claim something works unless you ran it.

TL;DR: Implement the task above in this worktree only, with dry-run as the default that makes no network call or login, offline pytest tests that pass with the given command, no secrets, no LLM-provider API code, and leave the changes uncommitted for review.
