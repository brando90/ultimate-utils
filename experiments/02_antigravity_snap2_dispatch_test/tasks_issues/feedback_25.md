
## Coordinator review of your first attempt (fix these; everything else was good and should stay)
Your previous attempt's edits are still in this worktree (`py_src/uutils/gdrive_uu.py`, `tests/test_gdrive_uu.py`, `.gitignore`, `pyproject.toml`). The safety behaviour, read-only default scope, CLI, docs, `.gitignore` and `gdrive` extra are good. Fix the data model only:
1. **Remove `PlanDict` and `PlanList`.** `GDriveClient` must not subclass `dict` (a client that compares equal to / is truthy like a dict and supports `client["scopes"]` is confusing), and list methods must not return a `list` subclass whose `__getitem__`/`__contains__`/`get` answer string keys with plan metadata (`files["action"]` silently returning plan data is a trap).
2. Replace with plain types and a consistent contract, documented in the module docstring:
   - `GDriveClient` is a plain class. Keep its attributes (`dry_run`, `scopes`, `credentials_file`, `token_file`) and add a `describe() -> dict` method if you need to show the config.
   - Methods that return a list in live mode (`list_files`, `list_images`, `list_folders`, `download_files`, `upload_files`, `sync_folder`) return a plain empty `list` in dry-run, after printing/logging the plan.
   - Methods that return one object in live mode (`download_file`, `upload_file`) and module-level entry points that describe an action (`sync_drive_folder`, `download_images_from_drive`) return a plain `dict` plan with `"dry_run": True` in dry-run.
3. Update the tests to the new contract (assert on printed plan text via `capsys` and on plain `list`/`dict` return values) and keep the no-network/no-auth-by-default tests. Use the exact test command from the task.
