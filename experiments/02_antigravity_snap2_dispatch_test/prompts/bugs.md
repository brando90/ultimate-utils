# Coordinator uu-agy-bugs: watcher and SNAP automation issues via Antigravity on SNAP-2

Read `CLAUDE.md` in this repo and `~/agents-config/INDEX_RULES.md` first. Your repo is `~/ultimate-utils` on this Mac and `/lfs/skampere2/0/brando9/ultimate-utils` on skampere2.

Your scope is these GitHub issues in `brando90/ultimate-utils`:
- #38 `job_scheduler_uu/start_watcher.sh` silently dies without venv/PYTHONPATH setup.
- #39 Watcher lifecycle email fails with "Exec format error: clauded" on startup (an alias is not an executable; the fix must not depend on shell aliases).
- #29 Set up SNAP cluster automation server for uutils services: finish only the parts an agent can do and verify on skampere2; leave the rest open with a precise blocker comment.

Read each issue with `gh issue view <n> --comments`. #38 and #39 are reproducible on skampere2, so the fix must be demonstrated there (show the failing run before and the working run after, in the ledger). Do not send real emails while testing; use a dry-run or print mode, or stub the sender.

Ledger: `experiments/02_antigravity_snap2_dispatch_test/results_bugs.md`.

## Common contract (all three coordinators)

**Purpose.** Brando wants to know whether his Antigravity (Google Gemini) coding agents on SNAP-2 work end to end. You are the coordinator: the Antigravity agents do the work; you dispatch, verify, land, and record.

**Dispatch recipe.**
1. `ssh skampere2.stanford.edu`; put `/dfs/scratch0/brando9/bin` first on `PATH`. The repo is `/lfs/skampere2/0/brando9/ultimate-utils` (default branch `main`).
2. First check the client: `agy --version`, `agy models` (pick the strongest Gemini model listed and record its exact ID), and one trivial `agy -p "reply with OK"` call. If auth fails, record the exact error and stop dispatching to that host (do not export `GEMINI_API_KEY`, do not try a login flow that needs Brando's browser; report it).
3. Give each Antigravity task its own git worktree and branch (`agy/<your-scope>-<issue>`) under `/lfs/skampere2/0/brando9/uu-worktrees/`, created from a fresh `git fetch` of `origin/main`, so the three coordinators never share an index. The main SNAP clone has someone's uncommitted one-line edit in `py_src/uutils/collaborators.py` and is behind `origin/main`: do not pull, reset, stash, or commit in that clone; only add worktrees from it.
4. Run it detached in a tmux session on skampere2 named `agy-<your-scope>-<n>`, e.g. `agy -p --dangerously-skip-permissions --model <id> --output-format stream-json "$(cat task.md)" > agy.jsonl 2> agy.stderr`. Keep the task prompt self-contained, end it with a TL;DR, and write it to the experiment folder. Up to 3 concurrent Antigravity agents per coordinator.
5. Wait with a bounded check loop (no busy polling in this chat; sleep 5–10 min between checks); stop a run that has made no progress for 45 min and record it.

**Verification before landing (you do this, deterministically).** Read the full diff; run the relevant tests or the script itself on skampere2; `git diff --check`; scan for secrets (keys, tokens, `.env`, anything under `~/keys`). Reject and re-dispatch once with specific feedback if the change is wrong. If Antigravity fails twice on a task, you may finish it yourself, but label that task "coordinator-finished, Antigravity failed" in the ledger.

**Landing.** Rebase the branch on `origin/main`, fast-forward `main`, push. Never force-push. Other coordinators push to the same repo: pull/rebase and retry on rejection. Commit messages name the Antigravity model ID and end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

**Issues.** Close an issue only when the landed commit resolves it; the closing comment links the commit and says what was verified. An issue that cannot be finished without Brando (a login, a phone, an account, sending messages as him, spending money) stays open with one comment stating the concrete blocker and the smallest next step; do not close it. Obsolete or duplicate issues may be closed with a comment explaining why.

**Hard limits.** No direct LLM-provider API code or API keys (Hard Rule 9: an issue asking for "Anthropic API" code is re-scoped to the `clauded -p` CLI or left open). Never send email, SMS, WhatsApp, Slack, Zulip, or social-media posts, and never log into any account. No secrets in commits. Do not touch other repos or other people's runs on SNAP.

**Ledger.** Keep your `results_<scope>.md` in `experiments/02_antigravity_snap2_dispatch_test/` live: per task, the issue, Antigravity model ID, wall time, outcome (landed / rejected / Antigravity failed / blocked on Brando), commit link, and log path. Commit it with explicit pathspecs only (`git commit -- <paths>`). Finish with a short verdict: did the SNAP-2 Antigravity agents work, and how well.

**Stop condition.** Stop when every task in your scope is landed, blocked on Brando with a comment, or failed twice with the failure recorded. End your final message with a TL;DR.

TL;DR: Coordinate Antigravity (Gemini) agents on skampere2 to finish the bugs scope above, verify each change yourself, push verified work to main, close only resolved issues with commit links, and record whether the SNAP-2 Antigravity agents worked.
