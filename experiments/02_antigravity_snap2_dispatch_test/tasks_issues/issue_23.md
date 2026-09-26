# Antigravity task: ultimate-utils issue #23

You are working in a git worktree of the Python library `ultimate-utils` (package `uutils`, source under `py_src/uutils/`, tests under `tests/`).
Worktree: `/lfs/skampere2/0/brando9/uu-worktrees/issues-23` (branch `agy/issues-23`). Work only inside this directory. Do not touch any other directory, repo, tmux session or process on this machine.

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
GitHub issue #23 "Automate Instagram posting with captions from Google Drive images".

Create `py_src/uutils/instagram_uu.py` using the **official Instagram Graph API** content-publishing flow via plain `requests` (no instagrapi, no unofficial APIs).
- Credentials JSON at `~/keys/instagram_credentials.json` with keys `ig_user_id`, `access_token`, optional `api_version` (default `"v21.0"`); loaded only when `dry_run=False`.
- `InstagramClient(ig_user_id="", access_token="", api_version="v21.0", dry_run=True)` and `InstagramClient.from_credentials(credentials_file=..., dry_run=True)`.
- `post_image(image_url, caption="") -> dict`: step 1 POST `https://graph.facebook.com/<ver>/<ig_user_id>/media` with `image_url`, `caption`, `access_token` -> creation id; step 2 POST `.../<ig_user_id>/media_publish` with `creation_id`. Important API constraint: the Graph API needs a **publicly reachable image URL**, not a local path; if a local path is passed, raise a clear `ValueError` explaining this (in dry-run just report it in the plan).
- `post_images_batch(image_urls, captions=None) -> list[dict]` (captions default to "" and must match length).
- `refresh_long_lived_token()` helper documented (GET `.../refresh_access_token` style or `oauth/access_token` with `fb_exchange_token`), dry-run by default.
- Drive/pipeline integration point without new deps: `plan_posts_from_folder(folder, caption_suffix=".txt") -> list[dict]` that lists images (`.jpg/.jpeg/.png`) in a local folder (e.g. a folder synced from Google Drive) and pairs each with a sidecar caption file `<name>.txt` if present. This only reads local files.
- Dry-run returns dicts like `{"dry_run": True, "would_post": ..., "caption": ...}` and prints a preview so captions can be reviewed before posting.
- CLI: `python -m uutils.instagram_uu preview --folder DIR` (always offline) and `post --image-url URL --caption C [--send]`.
- Module docstring: one-time setup (Instagram Professional account linked to a Facebook Page, Meta app, long-lived token, save JSON to `~/keys/instagram_credentials.json` with `chmod 600`), the public-URL constraint, and usage.
- Tests: `tests/test_instagram_uu.py` (dry-run default no HTTP; two-step publish flow calls the right URLs in order with mocked `requests`; local-path error; folder planning with sidecar captions in `tmp_path`).

## When you are done
Reply with: the list of files you created/changed, the exact test command you ran and its final pass/fail line, and anything you could not do. Do not claim something works unless you ran it.

TL;DR: Implement the task above in this worktree only, with dry-run as the default that makes no network call or login, offline pytest tests that pass with the given command, no secrets, no LLM-provider API code, and leave the changes uncommitted for review.
