# Task: archive stale experiments in ultimate-utils (git moves only)

You are working in a git worktree of the `ultimate-utils` repo on branch `agy/expts-archive`. Your current directory is the repo root. Do all work here; do not touch any other directory, repo, or branch, and do not push.

## Background

`experiments/` holds dated scratch experiments from 2024–2025 that are finished or abandoned. The repo has no archive convention yet, so we are creating one: `experiments/archive/`. Experiment evidence must never be deleted or edited, only moved.

## Steps

1. Move with `git mv` (preserve history, change no file contents):
   - `experiments/2024` -> `experiments/archive/2024`
   - `experiments/2025` -> `experiments/archive/2025`
   Move whole directories. Do not rename, edit, or delete any file inside them (including the empty file `2025/january/8_spectral_theory_expts.py`). Ignore untracked files such as `.DS_Store` and `.png` files that git does not track; leave them where they are.
2. Create `experiments/archive/README.md` with a short intro line ("Finished or stale experiments, moved here unchanged with `git mv`; nothing was deleted.") and a Markdown table with columns `Path | Date | Outcome (one line)`, one row per archived item:
   - `2024/september/09_to_13/` (09-13-2024 to 09-17-2024): setup notes for vLLM + DSPy and Unsloth on skampere1; vLLM 0.4.1 with torch 2.2.1 installed, flash-attn install failed; DSPy local-server client left unresolved. Stale.
   - `2024/september/vllm_lora_test.py` (09-2024): vLLM LoRA-adapter inference snippet with a placeholder adapter path; never run against a real adapter. Stale.
   - `2024/october/16_rank_vs_r2.py` and `2024/october/16_ed_vs_r2.py` (10-16-2024): synthetic linear-regression toys, R² versus feature rank and versus effective dimensionality (participation ratio); plots produced, question answered informally. Finished.
   - `2025/january/8_spectral_theory_expts.py` (01-2025): empty placeholder file, never started. Stale.
   Write the Outcome cells yourself from the facts above, one line each. Dates use MM-DD-YYYY.
3. In `experiments/01_self_hosted_openclaw/cc_prompt.md`, add ONE new row at the TOP of the table under "## Status & Log" (directly below the `|------|...` separator line), leaving every other line unchanged:
   `| 09-26-2026 | uu-agy-expts coordinator (Antigravity) | — | blocked on Brando | Not started. Phase 0 needs Brando's answers to the six open questions above, then his Gmail OAuth consent and a WhatsApp QR pairing from his phone; no agent can do these. Smallest next step: Brando answers questions 1–6. |`
4. Run `git status` and `git diff --cached -M --stat` and confirm every moved file shows as a 100% rename.
5. Commit everything with message:
   `experiments: archive 2024/ and 2025/ under experiments/archive/; mark exp01 blocked on Brando`
   Do not push.

## Rules

- Use `git mv`; never `rm`, never rewrite file contents of moved files.
- Do not edit anything outside `experiments/archive/`, the two moved directories, and that one table row in `cc_prompt.md`.
- No secrets, no network calls, no pushes.

TL;DR: `git mv` experiments/2024 and experiments/2025 into experiments/archive/, write experiments/archive/README.md with a one-line outcome per item, add one "blocked on Brando" status row to exp01's cc_prompt.md, verify 100% renames, commit locally, do not push.
