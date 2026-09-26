# Task: fix test isolation in tests/test_todo_fixes.py (review feedback)

You are in a git worktree of `ultimate-utils` on branch `agy/expts-todos`, at the repo root. Your previous commit fixed three TODO bugs and added `tests/test_todo_fixes.py`. The source fixes were accepted. One problem in the test was rejected in review:

`test_fix2_collect_hist_arbitrary_classes` does `sys.modules[mod] = MagicMock()` for `data_utils`, `utils`, `maps`, `nn_models` and never undoes it. Those fake modules leak into every later test in the same pytest session (a fake top-level `utils` module is especially dangerous).

Fix: take pytest's `monkeypatch` fixture in that test and use `monkeypatch.setitem(sys.modules, mod, MagicMock())` only for names not already in `sys.modules`, so they are removed automatically after the test. Also make sure `uutils.torch_uu.mit_trainer_code` itself does not stay cached with the fakes bound: after the test, remove it with `monkeypatch.delitem(sys.modules, 'uutils.torch_uu.mit_trainer_code', raising=False)` registered before the import (so monkeypatch restores the pre-test state). Change nothing else in the source files.

Verify: `PYTHONPATH=py_src python -m pytest tests/test_todo_fixes.py -q` passes, and add a check in the same run by executing `PYTHONPATH=py_src python -c "import pytest,sys; rc=pytest.main(['-q','tests/test_todo_fixes.py']); print('LEAK' if 'utils' in sys.modules and type(sys.modules['utils']).__name__=='MagicMock' else 'NO_LEAK'); sys.exit(rc)"` which must print `NO_LEAK`. Run `git diff --check`.

Commit with message `tests: undo sys.modules fakes in collect_hist test via monkeypatch` (a new commit; do not amend, do not push). Print the pytest summary line and the NO_LEAK line in your final reply.

TL;DR: In tests/test_todo_fixes.py, replace the permanent sys.modules MagicMock insertions with monkeypatch so nothing leaks, verify NO_LEAK, commit locally as a new commit, do not push.
