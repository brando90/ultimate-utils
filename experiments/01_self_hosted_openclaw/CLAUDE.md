# Experiment 01: Self-Hosted OpenClaw for In-App Admin Triage

> **For Claude Code (or any coding agent picking this up cold):** read this file end-to-end before taking any action. Then check the **Status & Log** section at the bottom for the latest state — the plan below is the original spec; the log is ground truth for what's already done.

## Goal (one sentence)

Stand up a self-hosted [OpenClaw](https://github.com/steipete/claw-bot) instance — backed by the user's Claude Max subscription — that reads Brando's Gmail, drafts replies to admin tasks, pings him in WhatsApp (or Discord/Telegram as fallback) for approval, and sends approved replies from Gmail. Goal: **zero app-switching to triage admin email.**

## Why (context for the agent)

- Brando's `uutils` has `emailing.py`, `discord_uu.py`, `whatsapp_uu.py` — these are **one-way notification senders** called from job schedulers and watchers. They do not handle inbound messages and are not agents.
- The real friction is *responding* to admin email. Email-as-notification is fine; email-as-task-inbox-you-have-to-context-switch-to is the pain point.
- Building inbound webhook handling + agent loop inside `uutils` is out of scope (utility library, not a service). See README §"Notifications vs. Interactive Agents" and issue [#41](https://github.com/brando90/ultimate-utils/issues/41).
- OpenClaw already bundles Baileys (WhatsApp), grammY (Telegram), Discord.py, AppleScript bridges, agent loop, persistent memory. Self-hosting keeps secrets local; **Claude Max makes the model calls effectively free**, so the only ongoing cost is electricity / VPS rental on a box Brando already owns.
- Hosted alternative (`myclaw.ai`, $33–$66/mo) is rejected: Brando has the skills + hardware to self-host, and the recurring cost provides no benefit when Max already covers the model.

## Non-goals

- Don't extend `uutils` with inbound webhook receivers or bot frameworks.
- Don't build WhatsApp notifications for job-finished pings (email already handles this).
- Don't pay for `myclaw.ai` hosting unless self-host is proven impossible.
- Don't try to make Claude Code itself the in-app agent — it's a coding CLI, not a personal assistant runtime.

## Plan (5 phases)

### Phase 0 — Pre-flight
- [ ] Confirm with user which host machine to use. Options, in order of preference:
  1. **Mac mini at home** — best (iMessage works, persistent, low power).
  2. **A SNAP node Brando already has access to** (e.g. mercury2 / ampere1). Easy but shared.
  3. **Cheap VPS** (Hetzner ~$5/mo) — fallback if no Mac/SNAP works.
- [ ] Verify Claude Max subscription auth works on chosen host (`claude --version`, signed in).
- [ ] `ls -la ~/keys/` on the host — note which keys already exist (Anthropic, Gmail OAuth, etc.) before asking the user for new ones.

### Phase 1 — Install & boot OpenClaw
- [ ] Clone `https://github.com/steipete/claw-bot` into a stable location (e.g. `~/openclaw/`).
- [ ] Read its README. Follow the official setup — do **not** improvise; the project is fast-moving.
- [ ] Configure model = Claude (use Brando's Max auth, not a separate API key). If OpenClaw doesn't natively support Max-subscription auth, fall back to an API key from `~/keys/anthropic_bm_key_koyejolab.txt` and **flag this to the user** so he can decide whether to provision a separate key or contribute upstream.
- [ ] Run a smoke test: chat with the agent in its default channel (likely Telegram or web UI). Confirm it can execute a trivial command (e.g. `ls`).

### Phase 2 — Wire Gmail
- [ ] Use Gmail API (OAuth desktop app) — not IMAP scraping. Token stored at `~/keys/gmail_openclaw_token.json`, mode 600.
- [ ] Scopes needed: `gmail.readonly` + `gmail.send` + `gmail.modify` (to mark read / label triaged).
- [ ] Test: agent lists 5 most recent unread admin emails (filter by sender domains: stanford.edu, financial aid offices, conferences, etc. — ask Brando for his admin filters).

### Phase 3 — Wire WhatsApp (primary inbound channel)
- [ ] Use OpenClaw's bundled Baileys integration. Pair via QR code from Brando's phone.
- [ ] Verify two-way: agent can send a message AND respond to a reply from Brando's phone.
- [ ] Fallback if Baileys is unstable on this host: switch to **Telegram** (grammY) — same UX, more reliable infra, no QR re-pair friction. Don't sink more than a day on Baileys.

### Phase 4 — Approval flow (the actual feature)
- [ ] Define the agent prompt: "Read unread admin emails, classify (admin / spam / personal / research). For admin only, draft a reply. DM Brando in WhatsApp with: subject line + 1-line summary + draft. Wait for `approve`, `edit: <new text>`, or `skip`. On approve, send via Gmail and label the email `triaged-by-claw`."
- [ ] Idempotency: the agent must not double-process the same email. Use a Gmail label (`triaged-by-claw`) as the durable marker.
- [ ] Rate limit: max one approval request per minute to avoid spamming Brando.

### Phase 5 — Hybrid: expose `uutils` as tools
Once Phases 0–4 work end-to-end:
- [ ] Decide the integration mechanism — likely an MCP server that wraps `uutils` functions, or a thin shell-tool registry. Check OpenClaw's tool-extension mechanism first.
- [ ] Surface (initial set, expand as needed):
  - `uutils.job_scheduler_uu` queue status — "are any of my jobs running?"
  - `uutils.emailing.send_email` — let the agent send notifications/results emails on Brando's behalf
  - W&B run summary lookup (via existing `uutils.logging_uu`)
- [ ] **Do not** wrap every utility — only ones the agent realistically needs. YAGNI.

## Definition of done

The experiment is "done" when, on the host machine, **all of the following hold for at least 7 consecutive days without manual intervention:**

1. OpenClaw is running 24/7 (systemd service, launchd, or tmux+autorestart — agent's choice).
2. Brando triaged ≥10 real admin emails entirely from WhatsApp (or Telegram) — never opened Gmail web UI for them.
3. Zero false-positive sends (no replies dispatched without explicit `approve`).
4. The Gmail label `triaged-by-claw` is consistently applied; no duplicates pinged.
5. Brando reports the friction reduction is real (subjective check-in — ask him).

If any of (1)–(4) fail and aren't fixable in a day, document the blocker in **Status & Log** below and recommend either (a) switching channels (WhatsApp → Telegram), (b) falling back to a minimal DIY Discord bot + Claude Agent SDK, or (c) parking the experiment.

## Hard rules for the executing agent

These come from `~/agents-config/INDEX_RULES.md`. Re-read that file at the start of any session working on this experiment.

- **Never commit secrets.** Tokens go in `~/keys/`, mode 600.
- **Refresh agents-config first** (`git -C ~/agents-config pull`) and re-read `INDEX_RULES.md`.
- **Run QA** before reporting any non-trivial milestone done (Hard Rule #3).
- **Email Brando** at `brando.science@gmail.com` (CC `brando9@stanford.edu`) when each phase completes — this counts as a "big task" (Hard Rule #13).
- **Just do it** (Guideline #14) — don't draft when told to send. But for OpenClaw, **stop and ask** before any of these destructive steps:
  - Sending a reply from Gmail before the approval flow is verified end-to-end.
  - Pairing Baileys to a WhatsApp account other than Brando's.
  - Granting OpenClaw shell access on a SNAP node shared with other users.
- **Dual TLDR** (Hard Rule #4) on every response.

## References

- Issue: [#41](https://github.com/brando90/ultimate-utils/issues/41) — full plan + options analysis
- README §"Notifications vs. Interactive Agents": `~/ultimate-utils/README.md`
- Existing notification modules: `py_src/uutils/{emailing,discord_uu,whatsapp_uu}.py`
- OpenClaw: https://github.com/steipete/claw-bot
- MyClaw (rejected hosted option): https://myclaw.ai
- Branch: `claude/add-whatsapp-discord-integration-twCHN`

## Open questions for Brando

Before Phase 0, the agent should confirm:
1. Which host machine? (Mac mini home / SNAP node / VPS)
2. Primary inbound channel: WhatsApp or Telegram? (Telegram is more reliable; WhatsApp is what he uses more.)
3. Admin-email filter: which sender domains / labels count as "admin" vs. ignore?
4. OK to use his Anthropic API key as fallback if OpenClaw can't reuse Max-subscription auth?

## Status & Log

Append-only. Most recent entry on top. Each entry: date, who, what changed, what's next.

| Date | Author | Phase | Status | Notes |
|------|--------|-------|--------|-------|
| 2026-04-26 | claude-code (planning) | — | spec drafted | This file created. No setup actions taken yet. Awaiting Brando's answers to the four open questions. |
