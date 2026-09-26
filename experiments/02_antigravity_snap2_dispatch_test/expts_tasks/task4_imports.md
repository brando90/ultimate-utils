# Task: add missing imports for undefined names in ultimate-utils (imports only)

You are in a git worktree of `ultimate-utils` on branch `agy/expts-imports`, at the repo root. Work only here; do not push. Do not edit anything under `py_src/uutils/job_scheduler_uu/`.

## Problem

`pyflakes py_src` reports `undefined name` errors. Some are simply missing imports. Fix ONLY these names, and only by adding an import:

- `Type`, `Optional`-style typing names -> `from typing import Type` (merge into an existing `from typing import ...` line if the file has one)
- `logging` -> `import logging`
- `np` -> `import numpy as np`
- `torchvision` -> `import torchvision`
- `Optimizer` -> `from torch.optim import Optimizer`

Find them with: `pip install --user pyflakes 2>/dev/null; python -m pyflakes py_src 2>/dev/null | grep -E "undefined name '(Type|logging|np|torchvision|Optimizer)'"` (or `uvx pyflakes` / `pipx run pyflakes` if pip is blocked). For each hit, first check the name is not meant to be something else (e.g. a local variable typo); if it is anything other than the obvious standard import, skip it and list it as skipped in your final reply.

Put each import with the file's other top-level imports. If the name is used inside a function that already has local imports, a top-level import is still fine. Do NOT touch any other undefined name (`get_transform`, `RuleIdx`, `sigmoid`, etc.) and do not change any other code.

## Verification you must run

1. The same pyflakes command prints nothing for those five names.
2. The total pyflakes `undefined name` count dropped by exactly the number of names you fixed, and no new pyflakes messages of any kind appeared in changed files (compare `pyflakes` output on `git show origin/main:<file>` vs the new file).
3. `python -m py_compile` passes on every changed file.
4. `git diff` contains only added import lines (or edits to an existing `from typing import` line). `git diff --check` is clean.

Commit with message `uutils: add missing imports flagged by pyflakes (Type, logging, np, torchvision, Optimizer)`. Do not push. In your final reply list each fixed file:name and each skipped one with the reason, plus before/after undefined-name counts.

TL;DR: Add only the obvious missing imports for pyflakes undefined names Type/logging/np/torchvision/Optimizer, prove nothing else changed and the count dropped, commit locally, do not push.
