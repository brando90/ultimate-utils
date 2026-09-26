# Experiment 02: do Antigravity (Gemini) agents on SNAP-2 finish real ultimate-utils work?

**TLDR:** Three Claude Code coordinators (Mac, tmux session `uu-agy-snap2`, Remote Control on) each dispatch Antigravity (`agy`) agents on `skampere2.stanford.edu` to close finishable GitHub issues and experiments in this repo, verify the result, and push to `main`. The question is whether the SNAP Antigravity agents work end to end, not only whether they answer a ping. Started 09-26-2026.

| Coordinator | Scope | Ledger |
|---|---|---|
| `uu-agy-bugs` | Issues #38, #39 (watcher bugs), #29 (SNAP automation server) | `results_bugs.md` |
| `uu-agy-issues` | Triage and finish/close issues #23–#28, #30–#34, #40, #42 | `results_issues.md` |
| `uu-agy-expts` | Experiments folder and repo TODOs that can be finished or archived | `results_expts.md` |

Success criterion per task: an Antigravity agent on SNAP-2 produced the change, the coordinator verified it (diff, tests or run, secret scan), it landed on `main`, and any closed issue links the commit. Failures of the Antigravity client itself (auth, hang, refusal, bad edits) are results, recorded with their log paths.
