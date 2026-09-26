# Antigravity task: ultimate-utils issue #30 and #31

You are working in a git worktree of the Python library `ultimate-utils` (package `uutils`, source under `py_src/uutils/`, tests under `tests/`).
Worktree: `/lfs/skampere2/0/brando9/uu-worktrees/issues-30-31` (branch `agy/issues-30-31`). Work only inside this directory. Do not touch any other directory, repo, tmux session or process on this machine.

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
  `env -u PYTHONPATH PYTHONPATH=py_src /lfs/skampere2/0/brando9/uu-agy-issues-venv/bin/python -m pytest -q -p no:cacheprovider <your test files>`
  (The venv already has pytest, requests, pyyaml, numpy, pandas, dill, networkx, lark. Do not `pip install` anything, do not `pip install -e` the package, and do **not** add `sitecustomize.py`, `conftest.py` path hacks or any other file that edits `sys.path`; if an import is missing, report it instead.)
- Run `git diff --check` and fix whitespace errors.
- **Do not commit and do not push.** Leave your changes as uncommitted edits/new files in the worktree; the coordinator reviews and commits them.

## The task
GitHub issues #30 "Add email automation for Stanford Bachata Club Gmail" and #31 "Add email automation for Stanford Lean AI Lab Gmail".

`py_src/uutils/emailing.py` already has `send_email_smtp(to, subject, body, smtp_user, smtp_pass, smtp_pass_file, smtp_host, smtp_port, from_addr, cc, bcc, attachments)` (read it first). Add account-profile support on top of it in a new module `py_src/uutils/club_emailing.py` (do not change the behaviour of existing functions in `emailing.py`).
- An `EmailAccount` dataclass: `name`, `address_file`, `app_password_file`, `members_file`, `smtp_host="smtp.gmail.com"`, `smtp_port=587`. Two built-in profiles in a dict `ACCOUNTS`:
  - `"bachata"`: address from `~/keys/bachata_gmail_address.txt`, app password `~/keys/bachata_gmail_app_password.txt`, members `~/keys/bachata_members.csv`.
  - `"leanai"`: address from `~/keys/leanai_gmail_address.txt`, app password `~/keys/leanai_gmail_app_password.txt`, members `~/keys/leanai_members.csv`.
  Do **not** hardcode any real Gmail address (the real addresses are unknown and live in those key files). Env-var overrides `UUTILS_<NAME>_GMAIL_ADDRESS` / `UUTILS_<NAME>_GMAIL_APP_PASSWORD_FILE` are fine.
- `send_account_email(account, to, subject, body, cc="", attachments=None, dry_run=True)`: in dry-run, reads no key file, sends nothing, prints and returns a preview dict (account, to, subject, first lines of body, attachments). Otherwise resolves the address and calls `emailing.send_email_smtp(..., smtp_user=address, smtp_pass_file=..., from_addr=address)`.
- `load_members(members_file) -> list[str]`: CSV with an `email` column (skip blanks/duplicates, case-insensitive dedupe).
- `send_announcement(account, subject, body, recipients_file=None, dry_run=True, bcc_all=True)`: sends ONE email to the account's own address with members in BCC (so members do not see each other). Dry-run: prints the recipient count and preview only; it may read the members file only if one is passed explicitly via `recipients_file` (so a dry-run preview of a real list is possible), otherwise does not read it.
- Templates: `TEMPLATES` dict + `render_template(name, **kwargs) -> tuple[subject, body]` with at least `bachata_practice_reminder` (date, time, location), `bachata_welcome` (name), `leanai_experiment_finished` (experiment_name, hostname, summary), `leanai_meeting_reminder` (date, time, location/link). Missing kwargs raise a clear `KeyError`/`ValueError`.
- CLI: `python -m uutils.club_emailing preview --account bachata --template bachata_practice_reminder --date ... --time ... --location ...` (offline) and `send --account A --to X --subject S --body B [--send]`.
- Module docstring: one-time setup per account (log into the club Gmail, enable 2-Step Verification, create an App Password, save it plus the address to the key files with `chmod 600`) and examples.
- Tests: `tests/test_club_emailing.py` (dry-run default makes no SMTP call and reads no key file: patch `smtplib.SMTP`, `smtplib.SMTP_SSL` and `uutils.emailing.send_email_smtp` to raise; non-dry-run path calls `send_email_smtp` with the right arguments using fake key files in `tmp_path` and a monkeypatched `send_email_smtp`; `load_members` dedupe; template rendering and missing-kwarg error).

## When you are done
Reply with: the list of files you created/changed, the exact test command you ran and its final pass/fail line, and anything you could not do. Do not claim something works unless you ran it.

TL;DR: Implement the task above in this worktree only, with dry-run as the default that makes no network call or login, offline pytest tests that pass with the given command, no secrets, no LLM-provider API code, and leave the changes uncommitted for review.
