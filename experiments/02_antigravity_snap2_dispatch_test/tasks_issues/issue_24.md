# Antigravity task: ultimate-utils issue #24

You are working in a git worktree of the Python library `ultimate-utils` (package `uutils`, source under `py_src/uutils/`, tests under `tests/`).
Worktree: `/lfs/skampere2/0/brando9/uu-worktrees/issues-24` (branch `agy/issues-24`). Work only inside this directory. Do not touch any other directory, repo, tmux session or process on this machine.

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
GitHub issue #24 "Automate Facebook posting with captions from Google Drive images".

Create `py_src/uutils/facebook_uu.py` using the **official Facebook Graph API** via plain `requests` (do not add `facebook-sdk`; it is unmaintained).
- Important limitation to document in the module docstring: since 2018 the Graph API cannot publish to a **personal profile timeline**; programmatic posting works for **Pages** you admin (Page access token). The module therefore targets a Page id.
- Credentials JSON at `~/keys/facebook_credentials.json` with keys `page_id`, `page_access_token`, optional `api_version` (default `"v21.0"`); loaded only when `dry_run=False`.
- `FacebookClient(page_id="", page_access_token="", api_version="v21.0", dry_run=True)` and `FacebookClient.from_credentials(credentials_file=..., dry_run=True)`.
- `post_text(message) -> dict` (POST `/<page_id>/feed`), `post_image(image_path_or_url, caption="") -> dict` (POST `/<page_id>/photos`; local file -> multipart `source`, URL -> `url` param), `post_images_batch(images, captions=None) -> list[dict]`.
- Local-folder pipeline point (e.g. a folder synced from Google Drive), no new deps: `plan_posts_from_folder(folder, caption_suffix=".txt") -> list[dict]` pairs `.jpg/.jpeg/.png` images with sidecar caption `<name>.txt` files. Reads local files only.
- Dry-run returns dicts like `{"dry_run": True, "would_post": ..., "caption": ...}` and prints a preview.
- CLI: `python -m uutils.facebook_uu preview --folder DIR` (offline) and `post --image PATH_OR_URL --caption C [--send]`, `post-text --message M [--send]`.
- Module docstring: one-time setup (Meta developer app, Page, `pages_manage_posts` + `pages_read_engagement` permissions, long-lived Page token, save JSON with `chmod 600`) and usage.
- Tests: `tests/test_facebook_uu.py` (dry-run default no HTTP; correct endpoint/params for text, URL image and local-file image with mocked `requests`; folder planning in `tmp_path`).

## When you are done
Reply with: the list of files you created/changed, the exact test command you ran and its final pass/fail line, and anything you could not do. Do not claim something works unless you ran it.

TL;DR: Implement the task above in this worktree only, with dry-run as the default that makes no network call or login, offline pytest tests that pass with the given command, no secrets, no LLM-provider API code, and leave the changes uncommitted for review.
