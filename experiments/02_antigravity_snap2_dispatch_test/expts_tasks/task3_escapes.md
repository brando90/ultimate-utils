# Task: fix invalid escape sequences (SyntaxWarning) in ultimate-utils without changing any string value

You are in a git worktree of `ultimate-utils` on branch `agy/expts-escapes`, at the repo root. Work only here; do not push. Do not edit anything under `py_src/uutils/job_scheduler_uu/`.

## Problem

Python 3.12+ emits `SyntaxWarning: invalid escape sequence` for these (future Python makes them errors):

```
py_src/uutils/__init__.py:1487 '\p'   and :1495 '\p'
py_src/uutils/evals/prompts_evals.py:61 '\R'
py_src/uutils/evals/utils.py:347 '\%'
py_src/uutils/evals/data_eval_utils.py:310 '\s'
py_src/uutils/plot/__init__.py:449 '\p'   and :458 '\p'
py_src/uutils/torch_uu/mit_trainer_code.py:61 '\i'
py_src/uutils/torch_uu/metrics/confidence_intervals.py:4 and :210 '\i'
py_src/uutils/torch_uu/metrics/complexity/task2vec_norm_complexity.py:4 '\i'
py_src/uutils/torch_uu/metrics/diversity/diversity.py:205, :410, :432, :447 '\i'
py_src/uutils/torch_uu/dataset/delaunay_uu.py:312 '\i'
```
(Line numbers are where the offending string or docstring is; confirm with the command below.)

## Rule: string VALUES must not change

For each offending string literal choose ONE of:
- prefix it with `r` (or `r` combined with an existing `f`, i.e. `rf`) — ONLY if the literal contains no other backslash escape that is meaningful (`\n`, `\t`, `\\`, `\'`, `\"`, `\x..`, `\u....`, line-continuation backslash, etc.);
- otherwise double the offending backslash (`\p` -> `\\p`).

Do not change anything else (no reformatting, no other edits).

## Verification you must run

1. `python -W error::SyntaxWarning -m py_compile <each file above>` succeeds for every file listed.
2. Value-equality check: for each changed file, compare every string constant in the AST before and after:
```
python - <<'PY'
import ast, subprocess, sys, warnings
warnings.simplefilter('ignore')
files = subprocess.check_output(['git','diff','--name-only','origin/main','--','*.py']).decode().split()
bad = 0
for f in files:
    old = subprocess.check_output(['git','show',f'origin/main:{f}']).decode()
    new = open(f).read()
    s = lambda src: [n.value for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    if s(old) != s(new): print('VALUE CHANGED', f); bad += 1
print('STRINGS_IDENTICAL' if not bad else 'FAIL'); sys.exit(bad)
PY
```
   It must print `STRINGS_IDENTICAL`.
3. `git diff --check` is clean.

Commit with message `uutils: fix invalid escape sequences (SyntaxWarning) without changing string values`. Do not push. In your final reply print the STRINGS_IDENTICAL line and the list of files changed.

TL;DR: Make every listed invalid escape sequence valid (raw-string prefix or doubled backslash) so `python -W error::SyntaxWarning -m py_compile` passes, prove all AST string values are identical to origin/main, commit locally, do not push.
