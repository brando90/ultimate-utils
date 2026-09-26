# Antigravity task: ultimate-utils issue #28

You are working in a git worktree of the Python library `ultimate-utils` (package `uutils`, source under `py_src/uutils/`, tests under `tests/`).
Worktree: `/lfs/skampere2/0/brando9/uu-worktrees/issues-28` (branch `agy/issues-28`). Work only inside this directory. Do not touch any other directory, repo, tmux session or process on this machine.

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
GitHub issue #28 "Zulip automation: messaging, stream monitoring, and notifications".

Create `py_src/uutils/zulip_uu.py` with a `ZulipClient` that talks to the official Zulip REST API (`<site>/api/v1/...`, HTTP basic auth with bot email + API key) using plain `requests` (the `zulip` pip package is not required).
- `ZulipClient(site: str = "", email: str = "", api_key: str = "", dry_run: bool = True)` and `ZulipClient.from_zuliprc(zuliprc_file: str = "~/keys/zuliprc", dry_run: bool = True)` which parses the standard `[api]` section (`email`, `key`, `site`) with `configparser`, only when `dry_run=False`.
- Methods: `send_message(stream, topic, content)` (POST `/messages`, type `stream`), `send_dm(user_email, content)` (type `direct`), `get_messages(stream=None, topic=None, limit=50)` (GET `/messages` with a JSON `narrow`, `anchor="newest"`, `num_before=limit`, `num_after=0`), `get_streams()` (GET `/streams`), `upload_file(file_path) -> str` (POST `/user_uploads`, returns the URL), `get_unread_count()` (summarise unread messages per stream/topic, e.g. from GET `/messages` with narrow `is:unread`).
- All calls go through one private `_request(method, path, **kwargs)`; in dry-run it returns `{"result": "success", "dry_run": True, ...}` and prints a one-line description; otherwise it sends the request and raises a clear error when `result != "success"`.
- A convenience `notify(content, stream, topic, dry_run=True)`.
- CLI: `python -m uutils.zulip_uu send --stream S --topic T --content C [--send]` (dry-run unless `--send`).
- Module docstring: one-time setup (Zulip Settings -> Personal -> Bots -> Add bot, download its zuliprc, save to `~/keys/zuliprc`, `chmod 600`) and usage examples.
- Tests: `tests/test_zulip_uu.py` (dry-run default makes no HTTP call; zuliprc parsing from a `tmp_path` fake file; correct URL/auth/payload for send_message and get_messages with mocked `requests`; error on non-success).

## When you are done
Reply with: the list of files you created/changed, the exact test command you ran and its final pass/fail line, and anything you could not do. Do not claim something works unless you ran it.

TL;DR: Implement the task above in this worktree only, with dry-run as the default that makes no network call or login, offline pytest tests that pass with the given command, no secrets, no LLM-provider API code, and leave the changes uncommitted for review.
