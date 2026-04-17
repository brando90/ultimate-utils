# Feature: Email-triggered Claude agent — reply from allowlisted addresses to run a Claude session and get an answer back

## Summary

Overnight experiment pipelines (e.g. the SolveAll.org conjecture runs) already send status/report emails to `brandojazz@gmail.com`. Today, acting on those reports requires Brando to open a laptop and manually start a Claude Code session. This issue proposes a small always-on daemon that watches a dedicated inbox, and when a reply comes in from one of Brando's own trusted addresses, launches a headless Claude Agent SDK session with the reply body as the prompt and emails the answer back in-thread.

In short: **reply to an experiment email with a question or instruction → Claude runs → the answer is emailed back in the same thread.**

## Motivation

- Brando often gets experiment reports on mobile and wants to ask follow-ups ("audit file 14 again", "rerun with seed=7", "summarize category A") without SSHing into a box.
- A reply-to-act interface is dramatically lower friction than phone-based terminals.
- The Claude Agent SDK already supports headless / programmatic use, so the glue is mostly "inbox watcher + SDK call + SMTP reply".

## Non-goals

- Not a general-purpose chatbot. Not exposed to the public internet.
- Not a replacement for the interactive Claude Code CLI on a workstation.
- Not a long-running persistent agent — each reply spawns a fresh session (v1).

## Hard security requirements

**Only take action on replies whose verified `From:` address is in this allowlist:**

- `brando.science@gmail.com` (and any alias under that account)
- `brandojazz@gmail.com`
- `brando9@stanford.edu`

Everything else must be **silently ignored** (no reply, no processing, no logging of content beyond a hash). Specifically:

- Reject any message where SPF, DKIM, or DMARC fails for the claimed sender domain (gmail.com / stanford.edu).
- Reject if the `From:` header doesn't match one of the three allowlisted addresses exactly (case-insensitive, after normalization).
- Reject mail where `Reply-To` or `Return-Path` disagrees with `From` (anti-spoof).
- Optionally also require a shared secret token in the subject or body (e.g. `@claude <token>`) as defense-in-depth.
- Rate-limit: at most N actions/hour per sender (configurable, default 10).
- Audit log: every accepted and every rejected message is logged with timestamp, sender, message-id, decision, and reason. Bodies of rejected messages are NOT stored.

## User experience

1. Nightly pipeline emails a report to `brandojazz@gmail.com`.
2. Brando hits "Reply" on his phone from `brando.science@gmail.com` and writes: `@claude please re-audit file 14 and tell me which lemmas are actually proved vs. stubbed.`
3. The watcher sees the reply within ~30s, verifies allowlist + auth headers, extracts the instruction, and spawns a Claude Agent SDK session in the repo/working directory referenced by the thread (or a default workspace).
4. Claude runs (reads files, runs tools, etc.) and produces an answer.
5. The daemon emails the answer back to Brando in the same thread (correct `In-Reply-To` / `References` headers).
6. Full transcript is archived to disk for later review.

## Acceptance criteria

- [ ] Replying from any of the three allowlisted addresses triggers a Claude session and produces a threaded reply within 5 minutes (subject to Claude runtime).
- [ ] Replying from any other address produces **no** outgoing email and no Claude invocation.
- [ ] Spoofed mail (forged `From:` with failing DKIM/SPF) is rejected.
- [ ] The service survives a restart with no duplicate replies (idempotent on `Message-ID`).
- [ ] A dry-run / `--no-send` mode exists for testing.
- [ ] Audit log is written to a rotating file and is human-readable.
- [ ] Secrets (OAuth token, Anthropic key) are loaded from `~/keys/` or env vars — never committed.

## Out of scope (future work)

- Attachments in/out.
- Multi-turn conversations across multiple replies (v1 treats each reply as independent).
- Web UI / dashboard.
- Slack or SMS triggers (same architecture, different ingestion adapter).
