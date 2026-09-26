# Coordinator uu-agy-issues: triage and finish the automation-feature issues via Antigravity on SNAP-2

Read `CLAUDE.md` in this repo and `~/agents-config/INDEX_RULES.md` first. Your repo is `~/ultimate-utils` on this Mac and `/lfs/skampere2/0/brando9/ultimate-utils` on skampere2.

Your scope is the open GitHub issues in `brando90/ultimate-utils` numbered #23–#28, #30–#34, #40, and #42 (social-media posting, Drive upload, WhatsApp, Slack, Zulip, SMS, club/lab Gmail, Stanford admin, email-triggered agent, CardinalEngage). Most need an account or a login that only Brando can provide. For each issue:
1. Read it with `gh issue view <n> --comments` and check whether the repo already contains work for it.
2. Decide: finishable now by an agent (for example a documented, tested script skeleton with a dry-run mode that sends nothing), blocked on Brando, or obsolete/duplicate.
3. Dispatch the finishable ones to Antigravity on skampere2, verify, land, and close with the commit link. For blocked ones, add one comment with the concrete blocker and smallest next step and leave the issue open. Close obsolete or duplicate ones with the reason.

Nothing you land may send a message, post, or log in when run with default arguments; dry-run must be the default. #34 asks for Anthropic API code: re-scope it to the `clauded -p` CLI or leave it open with that note.

Ledger: `experiments/02_antigravity_snap2_dispatch_test/results_issues.md`.

## Common contract (all three coordinators)

**Purpose.** Brando wants to know whether his Antigravity (Google Gemini) coding agents on SNAP-2 work end to end. You are the coordinator: the Antigravity agents do the work; you dispatch, verify, land, and record.

**Dispatch recipe.**
1. `ssh skampere2.stanford.edu`; put `/dfs/scratch0/brando9/bin` first on `PATH`. The repo is `/lfs/skampere2/0/brando9/ultimate-utils` (default branch `master`, not `main`).
2. First check the client: `agy --version`, `agy models` (pick the strongest Gemini model listed and record its exact ID), and one trivial `agy -p "reply with OK"` call. If auth fails, record the exact error and stop dispatching to that host (do not export `GEMINI_API_KEY`, do not try a login flow that needs Brando's browser; report it).
3. Give each Antigravity task its own git worktree and branch (`agy/<your-scope>-<issue>`) under `/lfs/skampere2/0/brando9/uu-worktrees/`, so the three coordinators never share an index.
4. Run it detached in a tmux session on skampere2 named `agy-<your-scope>-<n>`, e.g. `agy -p --dangerously-skip-permissions --model <id> --output-format stream-json "$(cat task.md)" > agy.jsonl 2> agy.stderr`. Keep the task prompt self-contained, end it with a TL;DR, and write it to the experiment folder. Up to 3 concurrent Antigravity agents per coordinator.
5. Wait with a bounded check loop (no busy polling in this chat; sleep 5–10 min between checks); stop a run that has made no progress for 45 min and record it.

**Verification before landing (you do this, deterministically).** Read the full diff; run the relevant tests or the script itself on skampere2; `git diff --check`; scan for secrets (keys, tokens, `.env`, anything under `~/keys`). Reject and re-dispatch once with specific feedback if the change is wrong. If Antigravity fails twice on a task, you may finish it yourself, but label that task "coordinator-finished, Antigravity failed" in the ledger.

**Landing.** Rebase the branch on `origin/master`, fast-forward `master`, push. Never force-push. Other coordinators push to the same repo: pull/rebase and retry on rejection. Commit messages name the Antigravity model ID and end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

**Issues.** Close an issue only when the landed commit resolves it; the closing comment links the commit and says what was verified. An issue that cannot be finished without Brando (a login, a phone, an account, sending messages as him, spending money) stays open with one comment stating the concrete blocker and the smallest next step; do not close it. Obsolete or duplicate issues may be closed with a comment explaining why.

**Hard limits.** No direct LLM-provider API code or API keys (Hard Rule 9: an issue asking for "Anthropic API" code is re-scoped to the `clauded -p` CLI or left open). Never send email, SMS, WhatsApp, Slack, Zulip, or social-media posts, and never log into any account. No secrets in commits. Do not touch other repos or other people's runs on SNAP.

**Ledger.** Keep your `results_<scope>.md` in `experiments/02_antigravity_snap2_dispatch_test/` live: per task, the issue, Antigravity model ID, wall time, outcome (landed / rejected / Antigravity failed / blocked on Brando), commit link, and log path. Commit it with explicit pathspecs only (`git commit -- <paths>`). Finish with a short verdict: did the SNAP-2 Antigravity agents work, and how well.

**Stop condition.** Stop when every task in your scope is landed, blocked on Brando with a comment, or failed twice with the failure recorded. End your final message with a TL;DR.

TL;DR: Coordinate Antigravity (Gemini) agents on skampere2 to finish the issues scope above, verify each change yourself, push verified work to master, close only resolved issues with commit links, and record whether the SNAP-2 Antigravity agents worked.
