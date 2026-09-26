# Antigravity task: ultimate-utils issue #25

You are working in a git worktree of the Python library `ultimate-utils` (package `uutils`, source under `py_src/uutils/`, tests under `tests/`).
Worktree: `/lfs/skampere2/0/brando9/uu-worktrees/issues-25` (branch `agy/issues-25`). Work only inside this directory. Do not touch any other directory, repo, tmux session or process on this machine.

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
GitHub issue #25 "Automate phone images -> Google Drive upload pipeline".

An earlier, unmerged branch already wrote a Google Drive module. Bring it onto this branch and make it safe:
1. Get the file: `git show origin/claude/google-drive-integration-54fTy:py_src/uutils/gdrive_uu.py > py_src/uutils/gdrive_uu.py` (run `git fetch origin` first if needed). Also inspect `git show origin/claude/google-drive-integration-54fTy -- .gitignore pyproject.toml` and carry over only the Drive-related `.gitignore` lines and the Google optional dependencies (as a new `gdrive` group under `[project.optional-dependencies]`; do not touch the required `dependencies` or the version number).
2. Make sure `import uutils.gdrive_uu` works without any Google packages installed (lazy imports inside functions) and does no auth or network at import.
3. Add `dry_run: bool = True` to every function/method that authenticates, downloads, uploads or syncs (`sync_folder`, `sync_drive_folder`, `download_images_from_drive`, `upload_file(s)`, `download_file(s)`, `get_gdrive_client`, ...). In dry-run nothing authenticates or touches the network: return/print a plan (which credentials file would be used, which folder id, which local destination).
4. Add the phone -> Drive -> local pipeline entry point: `python -m uutils.gdrive_uu sync-phone --folder-id ID --dest DIR [--credentials FILE] [--execute]` which, without `--execute`, only prints the plan; with `--execute` it downloads new images from that Drive folder to DIR (read-only on Drive; never deletes or uploads).
5. Extend the module docstring with the phone-side setup (iOS: Google Drive app -> Settings -> Backup / or Photos upload to a chosen folder; Android: Drive/Photos backup to a Drive folder; Google Photos and Drive no longer auto-sync, so use a Drive folder), sharing that folder with the service-account email, finding the folder id in the URL, and an example cron line for periodic sync (`--execute`), noting the machine must be on at that time.
6. Tests: `tests/test_gdrive_uu.py` (module imports without Google libs; every public entry point with default args makes no network/auth call; CLI `sync-phone` without `--execute` prints a plan and exits 0; the image-filtering helper, if any, works on fake file listings). Use `monkeypatch`/fakes; do not install Google libraries.

## When you are done
Reply with: the list of files you created/changed, the exact test command you ran and its final pass/fail line, and anything you could not do. Do not claim something works unless you ran it.

TL;DR: Implement the task above in this worktree only, with dry-run as the default that makes no network call or login, offline pytest tests that pass with the given command, no secrets, no LLM-provider API code, and leave the changes uncommitted for review.
