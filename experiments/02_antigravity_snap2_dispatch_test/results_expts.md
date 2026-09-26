# Results: `uu-agy-expts` coordinator (experiments folder and small TODOs)

Coordinator: Claude Code (`claude-opus-5-5`) on Brando's Mac. Workers: Antigravity (`agy`) on `skampere2.stanford.edu`. Started 09-26-2026.

## Client check (skampere2, 09-26-2026)

- `agy --version`: `1.2.7` at `/dfs/scratch0/brando9/bin/agy`.
- `agy models`: newest/strongest Gemini listed is `gemini-3.8-flash-high` (others: 3.8/3.7/3.6 Flash low–high, `gemini-3.1-pro-high/low`). Used `gemini-3.8-flash-high` for every task.
- Ping: `agy --model gemini-3.8-flash-high -p "reply with OK"` returned `OK` in 7.7 s. Auth worked; no API key exported.
- Recipe bug: the documented `agy -p --dangerously-skip-permissions --model <id> ...` form fails immediately with `Error: -p took "--model" as its prompt`. `-p` takes the next token as its prompt, so flags must come before `-p "<prompt>"`.

## Triage of `experiments/`

| Item | Decision | Reason |
|---|---|---|
| `01_self_hosted_openclaw/` | Blocked on Brando | Phase 0 needs his answers to six open questions, then Gmail OAuth consent and WhatsApp QR pairing from his phone. Issue #41 was already closed (plan captured). Smallest next step: Brando answers questions 1–6 in `cc_prompt.md`. |
| `2024/september/09_to_13/` | Archive (stale) | vLLM/DSPy/Unsloth install notes, 09-13-2024 to 09-17-2024. |
| `2024/september/vllm_lora_test.py` | Archive (stale) | Placeholder adapter path, never run. |
| `2024/october/16_*_vs_r2.py` | Archive (finished) | Toy R² vs rank / effective-dimension plots. |
| `2025/january/8_spectral_theory_expts.py` | Archive (stale) | Empty file. |

## Tasks

| # | Task | Model | Wall time | Outcome | Commit | Log (skampere2) |
|---|---|---|---|---|---|---|
| 1 | Archive `2024/`, `2025/` under `experiments/archive/`; exp01 blocked row | Antigravity `gemini-3.8-flash-high` | 209 s | Landed first try (9 files, 100% renames; README index; exp01 status row) | [738ffb8](https://github.com/brando90/ultimate-utils/commit/738ffb8) | `logs/expts-archive/agy.jsonl` |
| 2 | Fix three TODO bugs with tests (`raise NotImplemented`, `collect_hist` 10-class hardcode, task2vec normalization FIXME) | Antigravity `gemini-3.8-flash-high` | 396 s + 104 s redo | Landed after one rejection: source fixes right first time; test leaked `sys.modules` MagicMock fakes, fixed via `monkeypatch` on redo. 20 passed incl. existing suite, NO_LEAK | [6f9459e](https://github.com/brando90/ultimate-utils/commit/6f9459e) | `logs/expts-todos/agy.jsonl`, `agy_redo.jsonl` |
| 3 | Invalid escape sequences (17 `SyntaxWarning`s in 10 files), values unchanged | Grok `grok-4.7` → Antigravity `gemini-3.8-flash-high` | Grok 82 s (quota failure); Antigravity 499 s | Grok hit "free Grok Build usage limit" (exit 1, no changes, Grok-reported $0.25). Antigravity landed first try: every AST string constant identical, no SyntaxWarning left | [0ba8221](https://github.com/brando90/ultimate-utils/commit/0ba8221) | `logs/expts-escapes/grok.jsonl`, `agy.jsonl` |
| 4 | Missing imports for pyflakes undefined names (`Type`, `logging`, `np`, `torchvision`, `Optimizer`) | Grok `grok-4.7` → Antigravity `gemini-3.8-flash-high` | Grok 155 s (quota failure); Antigravity 151 s | Grok hit the same usage limit (exit 1, no changes, $0.09). Antigravity landed first try: imports only, 9 undefined-name errors cleared (25 → 16 in 4 files), no new pyflakes messages | [150717d](https://github.com/brando90/ultimate-utils/commit/150717d) | `logs/expts-imports/grok.jsonl`, `agy.jsonl` |

Log paths are relative to `/lfs/skampere2/0/brando9/uu-worktrees/` on skampere2.

Task prompts: `expts_tasks/`.

Grok (`grok 1.0.34`, `grok-4.7`) was added after Brando asked to use it; launch form: `grok --always-approve --permission-mode bypassPermissions -m grok-4.7 --cwd <wt> --output-format streaming-json --prompt-file task.md` (confirmed by `uu-agy-bugs`).

Remaining pyflakes undefined names (about 65, e.g. `get_transform` ×16, `RuleIdx` ×5, `interpolated_net`) are in legacy research code whose intent is unclear; skipped as not clearly fixable.

TODOs considered but skipped as unclear or not testable: significant figures in `torch_uu.get_mean_std_pairs`, the deliberately disabled `_dont_get_cifar10`, and the open-ended research/training TODOs in `evals/` and `hf_uu/train/`.

## Verdict (09-26-2026)

**Antigravity on SNAP-2 works end to end.** `gemini-3.8-flash-high` completed 4 of 4 tasks, and all 4 landed on `main` (`738ffb8`, `6f9459e`, `0ba8221`, `150717d`). 3 were accepted on the first attempt; 1 was rejected once (a test leaked `sys.modules` fakes) and fixed correctly on the redo. Wall times were 104–499 s per run. No auth problems, hangs or refusals. Its prompts must be tightly specified and its output must be checked by a command (tests, AST equality, pyflakes diff): the rejected test was plausible-looking but wrong, and the escapes agent fast-forwarded its own branch onto new upstream commits without being asked (harmless here).

**Grok on SNAP-2 does not work for sustained use on the free tier.** `grok-4.7` launched headless fine but both runs died within 82–155 s on "You've reached your free Grok Build usage limit" (Grok-reported $0.25 + $0.09) with no changes, and the other two coordinators' Grok runs share that quota. Needs a paid SuperGrok plan (Brando's decision; not purchased).

**Blocked on Brando:** experiment 01 (self-hosted OpenClaw) needs his answers to six open questions, then Gmail OAuth consent and WhatsApp QR pairing; recorded in its `cc_prompt.md` status table.

**Not attempted:** the remaining ~54 pyflakes undefined names across the repo and the open-ended research TODOs, because the right fix is not clear from the code.

Worktrees and logs stay on skampere2 under `/lfs/skampere2/0/brando9/uu-worktrees/` as evidence.
