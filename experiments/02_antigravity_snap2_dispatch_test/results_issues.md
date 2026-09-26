# Results — coordinator `uu-agy-issues` (issues #23–#28, #30–#34, #40, #42)

Live ledger, started 09-26-2026. The coordinator is Claude Opus 5.5 (`claude-opus-5-5`) running on Brando's Mac. The workers are Antigravity (`agy`) agents on `skampere2.stanford.edu`.

## Client check (skampere2, 09-26-2026 16:25 PDT)

- `agy --version` → `1.2.11` (`/dfs/scratch0/brando9/bin/agy`).
- `agy models` lists Gemini 3.8/3.7/3.6 Flash (high/medium/low), Gemini 3.1 Pro (high/low), and non-Gemini entries. The coordinator chose **`gemini-3.8-flash-high`** as the strongest Gemini model: it is the newest generation at the highest effort. The older `gemini-3.1-pro-high` is the fallback.
- The ping `agy --model gemini-3.8-flash-high -p "reply with OK"` returned `OK` in 4.7 s, so auth works.
- Recipe correction: `-p` takes the *next* argument as its prompt. The shared recipe's `agy -p --dangerously-skip-permissions --model <id> …` form fails with `-p took "--model" as its prompt`. Put `-p "<prompt>"` last instead.

## Setup

- Task prompts: `tasks_issues/issue_<id>.md`. Each is built from `_header.md` (safety and test rules), `body_<id>.md` and `_footer.md` (ends with a TL;DR).
- Launcher: `tasks_issues/launch.sh`. Each task gets a worktree at `/lfs/skampere2/0/brando9/uu-worktrees/issues-<id>` on branch `agy/issues-<id>`, created from `origin/main`, and runs in the tmux session `agy-issues-<id>`.
- Logs on skampere2: `/lfs/skampere2/0/brando9/uu-agy-logs/issues/issues-<id>/attempt<k>/{task.md,agy.jsonl,agy.stderr,exit_code,start,end}`.
- Test venv: `/lfs/skampere2/0/brando9/uu-agy-issues-venv` (pytest, requests, pyyaml). Tests run with `PYTHONPATH=py_src`, so no editable install of the heavy dependencies is needed.
- The agent leaves its edits uncommitted. The coordinator reviews the diff, reruns the tests, runs `git diff --check`, scans for secrets and then commits.

## Triage

| Issue | Topic | Decision |
|---|---|---|
| #23 | Instagram posting | Finishable: Graph API client, dry-run default |
| #24 | Facebook posting | Finishable: Page Graph API client, dry-run default |
| #25 | Phone → Drive pipeline | Finishable: port `gdrive_uu.py` from the unmerged branch, add a dry-run `sync-phone` command and phone setup docs |
| #26 | WhatsApp unread check | **Blocked on Brando:** QR pairing or phone-side Tasker, plus a choice of approach |
| #27 | Slack | Finishable: Web API client, dry-run default |
| #28 | Zulip | Finishable: REST client, dry-run default |
| #30 + #31 | Bachata / Lean AI Gmail | Finishable together: account profiles and templates on `emailing.py`, dry-run default |
| #32 | SMS | Code finishable (Twilio + AutoRemote, dry-run); the issue stays open for Brando's backend choice and paid account |
| #33 | Stanford admin | **Blocked on Brando:** SSO + Duo login needed to see the real pages |
| #34 | WhatsApp + Claude | Re-scoped from the Anthropic API to the `clauded -p` CLI; port the unmerged branch with an allowlist and dry-run |
| #40 | Email-triggered agent | Re-scoped from the Anthropic SDK to the `clauded -p` CLI; port the unmerged branch to `experiments/03_email_reply_bot/`; safe offline default |
| #42 | CardinalEngage | **Blocked on Brando:** the issue's own gate ("nothing ships before notes exist"), #33 auth, and the API key needing a re-scope |

## Tasks

Wall time is from the run's `start` to its `end` file on skampere2. The log directory is under `/lfs/skampere2/0/brando9/uu-agy-logs/issues/`. Commit links point to `brando90/ultimate-utils`.

| Issue | Agent / model | Attempt | Wall time | Outcome | Commit | Log |
|---|---|---|---|---|---|---|
| #28 Zulip | agy / gemini-3.8-flash-high | 1 | 360 s | **landed**, closed. The coordinator dropped the agent's `sitecustomize.py` sys.path hack. | [01b431e](https://github.com/brando90/ultimate-utils/commit/01b431e) | `issues-28/attempt1/` |
| #40 email agent | agy / gemini-3.8-flash-high | 1 | 380 s | **landed**; left open, blocked on Brando (Gmail OAuth, always-on host). Re-scoped to `clauded -p`. The coordinator dropped two sys.path hacks and a root test that imported the experiment as `src`. | [b2f968d](https://github.com/brando90/ultimate-utils/commit/b2f968d) | `issues-40/attempt1/` |
| #27 Slack | agy / gemini-3.8-flash-high | 1 | 317 s | **rejected**: JSON bodies are wrong for Slack read and upload methods, plus a `sitecustomize.py` hack | — | `issues-27/attempt1/` |
| #27 Slack | agy / gemini-3.8-flash-high | 2 | 115 s | **landed**, closed; both review points fixed | [da1831a](https://github.com/brando90/ultimate-utils/commit/da1831a) | `issues-27/attempt2/` |
| #32 SMS | grok / grok-4.7 | 1 | 269 s | Files written, then exit 1 on the Grok free-tier usage limit. **Landed** after the coordinator removed a `sys.modules` stub in the test; left open, blocked on Brando (backend choice, paid Twilio or Tasker) | [42d682a](https://github.com/brando90/ultimate-utils/commit/42d682a) | `issues-32/attempt1-grok/` |
| #24 Facebook | grok / grok-4.7 | 0 | 208 s | **Grok failed**: exit 1 on the free-tier usage limit, no edits | — | `issues-24/attempt0-grok/` |
| #24 Facebook | agy / gemini-3.8-flash-high | 1 | 208 s | **landed**, closed | [4371fcd](https://github.com/brando90/ultimate-utils/commit/4371fcd) | `issues-24/attempt1/` |
| #23 Instagram | agy / gemini-3.8-flash-high | 1 | 229 s | **landed**, closed | [914059e](https://github.com/brando90/ultimate-utils/commit/914059e) | `issues-23/attempt1/` |
| #34 WhatsApp + Claude | agy / gemini-3.8-flash-high | 1 | 310 s | **rejected**: flipped existing `send_whatsapp_*` defaults to dry-run (breaks downstream callers); ran `clauded` with full tools on third-party messages (prompt-injection risk); no webhook-auth requirement when live | — | `issues-34/attempt1/` |
| #34 WhatsApp + Claude | agy / gemini-3.8-flash-high | 2 | 200 s | **landed**, all 3 points fixed; left open, blocked on Brando (Meta account, public webhook, auto-replies as him) | [35d3e16](https://github.com/brando90/ultimate-utils/commit/35d3e16) | `issues-34/attempt2/` |
| #25 Drive | agy / gemini-3.8-flash-high | 1 | 290 s | **rejected**: `GDriveClient` subclassed `dict`; list results answered string keys with plan metadata (safety behaviour was fine) | — | `issues-25/attempt1/` |
| #25 Drive | agy / gemini-3.8-flash-high | 2 | 218 s | **landed** after the coordinator resolved a `pyproject.toml` extras conflict with #34; left open, blocked on Brando (folder ID, phone end-to-end test) | [832956d](https://github.com/brando90/ultimate-utils/commit/832956d) | `issues-25/attempt2/` |
| #30 + #31 club Gmail | agy / gemini-3.8-flash-high | 1 | 298 s | **rejected**: BCC announcements undeliverable (`emailing.SMTPNotifier` sent the comma-joined BCC as one recipient) | — | `issues-30-31/attempt1/` |
| #30 + #31 club Gmail | agy / gemini-3.8-flash-high | 2 | 133 s | **landed**, both closed; includes a backward-compatible BCC fix in `emailing.py` | [43fa23a](https://github.com/brando90/ultimate-utils/commit/43fa23a) | `issues-30-31/attempt2/` |
| #26 WhatsApp unread | — | — | — | blocked on Brando ([comment](https://github.com/brando90/ultimate-utils/issues/26#issuecomment-5850908411)) | — | — |
| #33 Stanford admin | — | — | — | blocked on Brando ([comment](https://github.com/brando90/ultimate-utils/issues/33#issuecomment-5850908523)) | — | — |
| #42 CardinalEngage | — | — | — | blocked on Brando ([comment](https://github.com/brando90/ultimate-utils/issues/42#issuecomment-5850908634)) | — | — |

## Final issue states (09-26-2026, 17:20 PDT)

- **Closed with a landed commit (6):** #23 Instagram, #24 Facebook, #27 Slack, #28 Zulip, #30 Bachata Gmail, #31 Lean AI Gmail. Each closing comment links the commit, says what was verified, and lists the remaining one-time credential setup.
- **Code landed, left open because it is blocked on Brando (4):** #25 (Drive folder ID and phone end-to-end test), #32 (backend choice plus a paid Twilio number or Android Tasker), #34 (Meta WhatsApp Business account, public webhook, and replying to real contacts as him), #40 (Gmail OAuth consent and an always-on host). Each has one comment with the commit, the blocker and the smallest next step.
- **Blocked on Brando with no code (3):** #26 (WhatsApp QR pairing or phone), #33 (Stanford SSO + Duo login), #42 (his CardinalEngage notes, the #33 session, and an Anthropic-key design that has to be re-scoped).
- **Re-scoped under Hard Rule 9:** #34 and #40 now reach Claude only through the `clauded` CLI. The direct `anthropic` SDK code on the old branches was not ported.
- **Nothing sends, posts or logs in by default:** every new public API defaults to `dry_run=True`, and every CLI needs `--send`/`--execute` (#40: `--live --send`). The only exceptions are pre-existing functions, left unchanged so downstream callers keep working.

## Findings

- **Antigravity (`agy` 1.2.11, `gemini-3.8-flash-high`) works end to end on skampere2.** All 12 agy runs exited 0, in 115–380 s each (median about 260 s). They read the repo, wrote modules with docstrings and offline pytest suites, ran the tests themselves and reported honestly, except for the undisclosed `sitecustomize.py` hacks. 9 tasks covering 10 issues were attempted by agy or Grok; all 9 landed.
- **First-attempt quality: 4 of 8 agy tasks landed on the first try** (#23, #24, #28, #40). The other 4 (#25, #27, #30+#31, #34) were rejected once on coordinator review for real defects: Slack API encoding, an unsafe tool surface plus changed public defaults, a confusing dict-subclass design, and an undeliverable BCC. Tests did not catch any of these because the agent had written tests to match its own assumptions. **All 4 were fixed on the second attempt** with targeted feedback, so none needed coordinator finishing. Deterministic review of the diff remains necessary.
- **The shell environment trips the agents.** The SNAP login shell exports `PYTHONPATH=/dfs/scratch0/brando9/lib/python3.12/site-packages`, so pip skipped dependencies in the test venv. Three of the first four agy runs silently wrote `sitecustomize.py` files that inject host-specific `site-packages` paths. After the fix (clean venv, `env -u PYTHONPATH` in the prompt, an explicit ban) no later run added one.
- **Grok is not usable today.** Both `grok-4.7` runs hit the free-tier "Grok Build usage limit" (208 s and 269 s). #32's run had already written good files, which landed after a one-line test fix. The other two coordinators saw the same limit from 16:37 PDT.
- **Recipe bug:** `agy -p <flags>` fails because `-p` takes the next argument as the prompt. Use `agy <flags> -p "<prompt>"`.
- **Claude Code on skampere2 is logged out** (`claude -p` → "OAuth session expired"). That blocks any live `clauded -p` use there.
- **Safety check the coordinator verified on the Mac:** `claude -p --tools ""` still exposes the claude.ai MCP connectors (Gmail send, Drive share). Only `--tools "" --strict-mcp-config` with the prompt on stdin gives a model with no tools. The #34 WhatsApp bot uses that form.

## Verdict

**Yes, the SNAP-2 Antigravity agents work end to end, and they are good but need a reviewer.** Every dispatch ran headless without hangs or auth failures. All 10 attempted issues ended with code on `main`: 8 of the 9 tasks were written by Antigravity and 1 by Grok before its quota ran out. That closed 6 issues and unblocked 4 more to a single human step. About half of the first attempts had a real defect that only a careful diff review caught. One round of specific feedback fixed all of them, in 115–218 s.
