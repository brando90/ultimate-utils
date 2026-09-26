# Task: fix three small TODO/FIXME bugs in ultimate-utils, with tests

You are working in a git worktree of the `ultimate-utils` repo on branch `agy/expts-todos`. Your current directory is the repo root. Package source is `py_src/uutils/`; tests live in `tests/` (pytest). Do all work here; do not touch any other directory, repo, or branch, and do not push. Do not edit anything under `py_src/uutils/job_scheduler_uu/`.

Run Python as: `PYTHONPATH=py_src python -m pytest tests/test_todo_fixes.py -q`. numpy, scipy and torch are installed. Keep each fix minimal; do not refactor surrounding code.

## Fix 1 — `py_src/uutils/evals/data_eval_utils.py`

Two branches in the path-dispatch function near the top do `raise NotImplemented` (one with `# TODO`). `NotImplemented` is a constant, not an exception, so this actually raises `TypeError`. Change both to `raise NotImplementedError(f'... not supported yet: {path}')` with a short message naming the dataset, and drop the `# TODO` comment on that line.

## Fix 2 — `py_src/uutils/torch_uu/mit_trainer_code.py`, `collect_hist`

`hist = np.zeros((N, 10))  ## TODO fix hack, don't hardcode # of classes`. Make it work for any number of output units: allocate `hist` lazily from the first batch's `outputs.shape[1]` (keep the dtype float, keep `N = len(dataloader.dataset)` rows). Remove the TODO comment. Behaviour for 10 classes must be unchanged.

## Fix 3 — `py_src/uutils/torch_uu/metrics/diversity/task2vec_based_metrics/task_similarity.py`, `get_normalized_embeddings`

`# FIXME: compute variance using only valid embeddings`. Missing (`None`) embeddings are replaced by zero vectors before computing `normalization = np.sqrt((F ** 2).mean(axis=0, keepdims=True))`, so the zeros deflate the normalization. When `normalization is None`, compute it from only the rows that came from non-`None` embeddings. Keep the returned `F` shape and the zero rows for missing embeddings. When there are no `None`s the result must be identical to before. Remove the FIXME comment.

Read how `get_variance` is used in that file to construct valid test inputs (a Task2Vec embedding object with `hessian` and `scale` attributes; a small `types.SimpleNamespace(hessian=np.array([...]), scale=np.array([...]))` should work, but check the code).

## Tests

Create `tests/test_todo_fixes.py` with focused pytest tests:
- Fix 1: calling the dispatch function with a path containing `Putnam_MATH_variation_static2` raises `NotImplementedError` (not `TypeError`). Find the function name in the file. If importing the module pulls in heavy optional deps that are not installed, use `pytest.importorskip` for them rather than editing imports.
- Fix 2: a tiny `torch.nn.Linear(4, 3)` over a `TensorDataset` of 7 samples with batch size 3 gives `hist.shape == (7, 3)` and matches `net(X)` row by row.
- Fix 3: with embeddings `[e0, None, e2]`, the normalization equals the one computed from `[e0, e2]` alone, row 1 of `F` is all zeros; and with no `None`s the output matches the old formula.

Run the tests until they pass, then run `git diff --check`.

## Commit

Commit all changes with message:
`uutils: fix three small TODO bugs (NotImplemented raise, collect_hist class count, task2vec normalization) with tests`
Do not push. In your final reply, print the pytest summary line.

TL;DR: In branch agy/expts-todos, fix `raise NotImplemented` in evals/data_eval_utils.py, the hardcoded 10 classes in mit_trainer_code.collect_hist, and the FIXME normalization over valid embeddings in task_similarity.get_normalized_embeddings; add tests/test_todo_fixes.py, make it pass, commit locally, do not push.
