# Experiment 01: Self-Hosted OpenClaw — Claude Code / Codex Prompt

> **For the executing agent (Claude Code or Codex) picking this up cold:** read this file end-to-end, then check **Status & Log** at the bottom — the plan is the original spec, the log is ground truth.
>
> **Two instances will be deployed**, both running the user's "smartest Codex" model (Brando has Codex Pro/Max, so model calls are effectively free):
> - **Instance A:** SNAP `mercury2`, inside a long-lived tmux session
> - **Instance B:** Local machine (Mac mini at home)
>
> Both must auto-restart on crash without manual intervention.

## Goal (one sentence)

Stand up two self-hosted [OpenClaw](https://github.com/steipete/claw-bot) instances (mercury2 + local Mac), both backed by the user's Codex Pro subscription, that read Brando's Gmail, draft replies to admin tasks, ping him in WhatsApp (or Telegram fallback) for approval, and send approved replies from Gmail. Goal: **zero app-switching to triage admin email.**

## Why (context for the agent)

- Brando's `uutils` has `emailing.py`, `discord_uu.py`, `whatsapp_uu.py` — these are **one-way notification senders** called from job schedulers and watchers. They do not handle inbound messages and are not agents.
- The real friction is *responding* to admin email. Email-as-notification is fine; email-as-task-inbox-you-have-to-context-switch-to is the pain point.
- Building inbound webhook handling + agent loop inside `uutils` is out of scope (utility library, not a service). See `~/ultimate-utils/README.md` §"Notifications vs. Interactive Agents" and issue [brando90/ultimate-utils#41](https://github.com/brando90/ultimate-utils/issues/41).
- OpenClaw already bundles Baileys (WhatsApp), grammY (Telegram), Discord.py, AppleScript bridges, agent loop, persistent memory. Self-hosting keeps secrets local; **Codex Pro makes the model calls effectively free**, so the only ongoing cost is electricity / VPS rental on a box Brando already owns.
- Hosted alternative (`myclaw.ai`, $33–$66/mo) is rejected: Brando has the skills + hardware to self-host, and the recurring cost provides no benefit when Codex Pro already covers the model.
- **Why two instances?** Redundancy + experimentation. Mercury2 has cluster-network reliability; local Mac mini has iMessage and survives SNAP maintenance windows. If one drops, the other keeps triaging.

## Non-goals

- Don't extend `uutils` with inbound webhook receivers or bot frameworks.
- Don't build WhatsApp notifications for job-finished pings (email already handles this).
- Don't pay for `myclaw.ai` hosting unless self-host is proven impossible on **both** target machines.
- Don't try to make Claude Code itself the in-app agent — it's a coding CLI, not a personal assistant runtime.
- Don't run both instances pointing at the same Gmail label simultaneously without an idempotency strategy (see Phase 4).

## Plan (5 phases × 2 instances)

Run all phases on **both** instances unless explicitly noted otherwise. Where the steps diverge by host, sub-bullets are tagged `[A:mercury2]` or `[B:local]`.

### Phase 0 — Pre-flight (per instance)
- [ ] Verify host is reachable and the user's environment is sane.
  - `[A:mercury2]` `ssh mercury2` then `tmux ls`. Pick or create a stable tmux session name: `openclaw`.
  - `[B:local]` Confirm Mac mini hostname and that `launchctl` is usable for the user.
- [ ] Verify Codex CLI auth works (`codex --version` and a trivial `codex exec` ping). Codex Pro subscription must be active on this host.
- [ ] `ls -la ~/keys/` on the host — note existing tokens (Anthropic, OpenAI, Gmail OAuth) before asking the user for new ones. Use `~/keys/openai_bm_key_koyejolab.txt` as fallback if Codex Pro CLI auth isn't reusable by OpenClaw.
- [ ] `[A:mercury2]` Confirm SNAP required symlinks exist (per `~/agents-config/INDEX_RULES.md` §"SNAP Required Symlinks") — especially `~/agents-config`, `~/keys`, `~/.claude`, `~/dfs`. If missing, run `~/agents-config/scripts/snap_setup.sh` (or follow `machine/snap-init.md`) before proceeding.

### Phase 1 — Install & boot OpenClaw (per instance)
- [ ] Clone `https://github.com/steipete/claw-bot` to a stable location:
  - `[A:mercury2]` `/dfs/scratch0/<user>/openclaw` with symlink `~/openclaw → /dfs/scratch0/<user>/openclaw` (DFS-backed, survives node reboots).
  - `[B:local]` `~/openclaw`.
- [ ] Read OpenClaw's README. Follow official setup — do **not** improvise; the project moves fast.
- [ ] Configure model = **Codex's smartest available** (the user explicitly wants this; do not silently downgrade). If OpenClaw can't reuse Codex Pro CLI auth, fall back to the OpenAI API key from `~/keys/openai_bm_key_koyejolab.txt` and **flag this to the user** so he can decide whether to provision a separate key or contribute upstream support.
- [ ] Smoke test: chat with the agent in its default channel. Confirm it can execute `ls` and report results.

### Phase 2 — Wire Gmail (shared between instances)
Gmail is one inbox; configure once, reuse on both hosts.
- [ ] Use Gmail API (OAuth desktop app) — not IMAP scraping. Token at `~/keys/gmail_openclaw_token.json`, mode 600. **Same token file mirrored to both hosts** (copy via `scp`; do not re-OAuth twice — keep the audit trail simple).
- [ ] Scopes: `gmail.readonly` + `gmail.send` + `gmail.modify` (to mark read / label triaged).
- [ ] Test on each host: agent lists 5 most recent unread admin emails. Filter by sender domains — ask Brando for his admin filter list (likely stanford.edu, financial-aid offices, conference orgs, etc.).

### Phase 3 — Wire WhatsApp / Telegram (per instance)
- [ ] **Primary:** OpenClaw's bundled Baileys (WhatsApp). Pair via QR code from Brando's phone.
  - Pair **once** and copy the Baileys session files to the second host. WhatsApp's multi-device limit is currently 4 — two OpenClaw instances + Brando's phone + maybe his Mac WhatsApp = at the limit. Verify before pairing.
- [ ] **Fallback:** Telegram (grammY) — same UX, more reliable infra, no QR re-pair friction. If Baileys is flaky on either host, switch that host to Telegram. Don't sink more than a day on Baileys per host.
- [ ] Verify two-way: each agent can send AND respond to a reply from Brando's phone.

### Phase 4 — Approval flow + dual-instance idempotency
This is the actual feature. **Critical:** with two instances reading the same inbox, you MUST prevent double-processing.

- [ ] Agent prompt: "Read unread admin emails. Classify (admin / spam / personal / research). For admin only: draft a reply, DM Brando in WhatsApp with subject + 1-line summary + draft. Wait for `approve`, `edit: <text>`, or `skip`. On approve, send via Gmail and apply label `triaged-by-claw`."
- [ ] **Idempotency strategy (mandatory before going live):** when an instance picks up an email to draft, it immediately applies a Gmail label `claw-claimed-by-<hostname>`. Other instances skip emails with any `claw-claimed-by-*` label. Use Gmail's atomic label-add as a poor-man's lock. On final send, replace the claim label with `triaged-by-claw`. If a claim is older than 30 minutes with no resolution (instance crashed mid-draft), the other instance may steal it.
- [ ] Rate limit: max one approval request per minute *per instance*, max two per minute total — don't spam Brando.
- [ ] Cross-instance health: each instance pings the other every 15 min via a shared file in `~/dfs/openclaw_heartbeat/<hostname>.txt`. If one is silent for >30 min, the other emails Brando.

### Phase 5 — Auto-restart resilience (per instance)
**Required for "definition of done."** Both instances must self-heal without Brando logging in.

- [ ] `[A:mercury2]` tmux session `openclaw` runs a watchdog loop:
  ```bash
  cd ~/openclaw && while true; do
    date '+%F %T starting openclaw' >> ~/openclaw/watchdog.log
    ./run.sh 2>&1 | tee -a ~/openclaw/run.log
    date '+%F %T openclaw exited code=$? — restart in 5s' >> ~/openclaw/watchdog.log
    sleep 5
  done
  ```
  Use `tmux new-session -d -s openclaw 'bash -c "<watchdog loop>"'` so it survives ssh disconnect. Verify with `tmux ls` after logout/login.
- [ ] `[A:mercury2]` Add to user's crontab on mercury2: `*/10 * * * * pgrep -f "openclaw/run.sh" >/dev/null || tmux new-session -d -s openclaw '...'` — re-launches the tmux session if the whole thing dies (rare but possible after node reboot).
- [ ] `[B:local]` Use **launchd** for true 24/7 across reboots:
  - Write `~/Library/LaunchAgents/ai.openclaw.plist` with `KeepAlive=true`, `RunAtLoad=true`, `ProgramArguments` pointing at `~/openclaw/run.sh`.
  - `launchctl load -w ~/Library/LaunchAgents/ai.openclaw.plist`
  - Verify: `launchctl list | grep openclaw` shows the job.
  - Kill the process manually and confirm launchd restarts it within 10 seconds.
- [ ] Both: log rotation (`logrotate` on Linux, manual `find ~/openclaw/*.log -mtime +14 -delete` cron on Mac) — don't fill the disk with 6 months of agent chatter.

### Phase 6 — Hybrid: expose `uutils` as tools
Once Phases 0–5 work end-to-end on both instances:
- [ ] Decide integration mechanism — likely an MCP server wrapping `uutils`, or a thin shell-tool registry. Check OpenClaw's tool-extension mechanism first.
- [ ] Surface (initial set, expand as needed):
  - `uutils.job_scheduler_uu` queue status — "are any of my jobs running?"
  - `uutils.emailing.send_email` — let the agent send notifications/results emails on Brando's behalf
  - W&B run summary lookup (via existing `uutils.logging_uu`)
- [ ] **Do not** wrap every utility — only ones the agent realistically needs. YAGNI.

## Definition of done

The experiment is "done" when, **across both instances, all of the following hold for at least 7 consecutive days without manual intervention:**

1. Both instances are running 24/7 (mercury2 tmux+watchdog, local launchd).
2. Each instance has been killed at least once during the 7-day window and **auto-restarted within 1 minute** (verify in logs).
3. Brando triaged ≥10 real admin emails entirely from WhatsApp (or Telegram) — never opened Gmail web UI for them.
4. Zero false-positive sends (no replies dispatched without explicit `approve`).
5. Zero double-processed emails (Gmail label audit shows each `triaged-by-claw` came from exactly one instance).
6. The Gmail label flow (`claw-claimed-by-<host>` → `triaged-by-claw`) is consistently applied; stale claims auto-expire correctly.
7. Brando reports the friction reduction is real (subjective check-in — ask him).

If any of (1)–(6) fail and aren't fixable in a day, document the blocker in **Status & Log** below and recommend either (a) consolidating to one instance, (b) switching channels (WhatsApp → Telegram), or (c) parking the experiment.

## Hard rules for the executing agent

These come from `~/agents-config/INDEX_RULES.md`. Re-read that file at the start of any session working on this experiment.

- **Refresh agents-config first** (`git -C ~/agents-config pull`) and re-read `INDEX_RULES.md` (Hard Rule #5).
- **Never commit secrets.** Tokens go in `~/keys/`, mode 600 (Hard Rule #1).
- **Run QA** before reporting any non-trivial milestone done (Hard Rule #3).
- **Email Brando** at `brando.science@gmail.com` (CC `brando9@stanford.edu`) when each phase completes — counts as a "big task" (Hard Rule #13).
- **Dual TLDR** (Hard Rule #4) on every response.
- **Just do it** (Guideline #14) — but for OpenClaw, **stop and ask** before any of these destructive steps:
  - Sending a reply from Gmail before the approval flow is verified end-to-end on **both** instances.
  - Pairing Baileys to a WhatsApp account other than Brando's.
  - Granting OpenClaw shell access on a SNAP node shared with other users (mercury2 is shared — coordinate with the SNAP machine doc `machine/mercury2.md`).
  - Modifying Brando's `~/Library/LaunchAgents/` outside the OpenClaw plist.

## References

- Issue: [brando90/ultimate-utils#41](https://github.com/brando90/ultimate-utils/issues/41) — full plan + options analysis
- README §"Notifications vs. Interactive Agents": `~/ultimate-utils/README.md`
- Existing notification modules: `~/ultimate-utils/py_src/uutils/{emailing,discord_uu,whatsapp_uu}.py`
- OpenClaw: https://github.com/steipete/claw-bot
- MyClaw (rejected hosted option): https://myclaw.ai
- Branch (uutils-side stub): `claude/add-whatsapp-discord-integration-twCHN`
- Mercury2 machine doc: `~/agents-config/machine/mercury2.md`
- Mac machine doc: `~/agents-config/machine/mac.md`

## Open questions for Brando

Before Phase 0, the agent should confirm:
1. Mercury2 tmux session name — `openclaw` ok, or pick another?
2. Local host — confirm Mac mini hostname & user account.
3. Primary inbound channel: WhatsApp or Telegram? (Telegram is more reliable; WhatsApp matches existing usage.)
4. Admin-email filter: which sender domains / labels count as "admin" vs. ignore?
5. WhatsApp multi-device count — is there room for two OpenClaw pairings on top of phone + Mac WhatsApp?
6. OK to use `~/keys/openai_bm_key_koyejolab.txt` as fallback if OpenClaw can't reuse Codex Pro CLI auth?

## Status & Log

Append-only. Most recent entry on top. Each entry: date, who, what changed, what's next.

| Date | Author | Phase | Status | Notes |
|------|--------|-------|--------|-------|
| 2026-04-26 | claude-code (planning) | — | spec drafted | This file created in `~/agents-config/experiments/01_self_hosted_openclaw/cc_prompt.md`. Two-instance plan (mercury2 tmux + local Mac launchd), Codex Pro as model, auto-restart required. No setup actions taken yet. Awaiting Brando's answers to the six open questions. |
