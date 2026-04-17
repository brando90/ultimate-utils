# Implementation plan: email-triggered Claude agent

Tracking issue: `ISSUE.md` (same directory).

Branch: `claude/email-reply-automation-a1mAv`.

Target location in repo: `experiments/email_reply_bot/`.

## Allowlist (single source of truth)

```
ALLOWED_SENDERS = {
    "brando.science@gmail.com",
    "brandojazz@gmail.com",
    "brando9@stanford.edu",
}
```

Normalization before compare: `addr.strip().lower()`, strip `+suffix` aliases (`brando.science+foo@gmail.com` → `brando.science@gmail.com`), strip display-name.

## Architecture

```
  Gmail mailbox (brandojazz@gmail.com)
          │
          │ 1. Pub/Sub push (preferred) or IMAP IDLE (fallback)
          ▼
  ┌───────────────────────┐
  │  watcher.py           │  — long-running daemon
  │   - fetch new msg     │
  │   - verify SPF/DKIM   │
  │   - allowlist check   │
  │   - dedupe by Msg-ID  │
  └──────────┬────────────┘
             │ 2. accepted message → task queue (sqlite)
             ▼
  ┌───────────────────────┐
  │  dispatcher.py        │
  │   - build prompt      │
  │   - spawn Claude SDK  │
  │     session (headless)│
  │   - capture stdout    │
  └──────────┬────────────┘
             │ 3. answer text + transcript
             ▼
  ┌───────────────────────┐
  │  replier.py           │
  │   - Gmail API send    │
  │   - In-Reply-To /     │
  │     References        │
  │   - audit log write   │
  └───────────────────────┘
```

Each stage is a separate module so they can be tested independently and swapped (e.g. IMAP → Pub/Sub later).

## Files to create

```
experiments/email_reply_bot/
├── ISSUE.md                  (feature issue text)
├── PLAN.md                   (this file)
├── README.md                 (setup / run instructions)
├── pyproject.toml  or
│   requirements.txt          (isolated deps; not added to uutils core)
├── config.example.yaml       (allowlist, paths, rate limits — no secrets)
├── src/
│   ├── __init__.py
│   ├── allowlist.py          (verify_sender, normalize_addr)
│   ├── auth_headers.py       (SPF/DKIM/DMARC verification)
│   ├── gmail_client.py       (OAuth + fetch + send; thin wrapper)
│   ├── watcher.py            (main loop)
│   ├── dispatcher.py         (spawn Claude SDK session)
│   ├── replier.py            (compose + send threaded reply)
│   ├── store.py              (sqlite: seen message-ids, rate limits, audit log)
│   └── prompts.py            (system prompt template for the agent)
├── tests/
│   ├── test_allowlist.py
│   ├── test_auth_headers.py
│   ├── test_dedupe.py
│   └── fixtures/             (sample raw MIME: legit, spoofed, stranger)
└── scripts/
    ├── run_watcher.sh        (launch with .env loaded)
    └── install_service.sh    (systemd unit installer, optional)
```

## Dependencies

- `google-api-python-client`, `google-auth`, `google-auth-oauthlib` (Gmail API)
- `dkimpy` or `authheaders` (header verification — or trust Gmail's `Authentication-Results`)
- `anthropic` + Claude Agent SDK (`claude-agent-sdk`) for the session
- `pyyaml` for config
- `pytest` for tests

Install locally into a venv under `experiments/email_reply_bot/.venv` — do not pollute uutils' core deps.

## Milestones

### M1 — Local dry run (no network) — ~half day
- [ ] `allowlist.py` with `verify_sender(addr) -> bool`, unit-tested for aliases, case, `+tag`, display-name stripping.
- [ ] `auth_headers.py` that parses Gmail's `Authentication-Results:` header and returns `{spf, dkim, dmarc}` verdicts.
- [ ] Fixtures: three raw `.eml` files — (a) legit `brando.science`, (b) spoofed `brando.science` w/ failing DKIM, (c) stranger.
- [ ] `store.py` with sqlite for seen-ids + audit log.
- [ ] `dispatcher.py` that, given a prompt string, invokes Claude Agent SDK headlessly and returns the final assistant message. Start with the simplest possible single-turn call.

### M2 — Gmail round-trip (read-only first) — ~half day
- [ ] OAuth flow: create a GCP project, enable Gmail API, generate `credentials.json`, run a one-time consent to produce `token.json`. Store outside the repo (`~/.config/email_reply_bot/`).
- [ ] `gmail_client.fetch_unseen()` — pulls unseen messages in a specific label (`claude-inbox`).
- [ ] End-to-end test with `--no-send`: incoming mail → allowlist check → dispatcher → print reply to stdout.

### M3 — Send reply — ~half day
- [ ] `replier.send_threaded_reply(orig_msg, body_text)` — builds MIME with `In-Reply-To`, `References`, `Subject: Re: …`, uses `threadId` for Gmail.
- [ ] Idempotency: reject if `Message-ID` already in `store.seen`.
- [ ] Rate limit per sender.

### M4 — Hardening & deploy — ~half day
- [ ] Rotating audit log (`logging.handlers.RotatingFileHandler`).
- [ ] Config file with sane defaults.
- [ ] Systemd unit (or `tmux` script) for the always-on host.
- [ ] README with full setup walkthrough.
- [ ] Kill switch: a file like `/tmp/email_bot_pause` pauses processing.

### M5 — Nice-to-haves (later)
- Gmail Pub/Sub push instead of polling (sub-second latency).
- Per-thread memory (SQLite keyed by `threadId`) so follow-ups carry context.
- Attachment support (inbound screenshots → vision; outbound log files).
- Multi-inbox support (watch `brando.science` too).

## Security checklist (must all be true before enabling send)

- [ ] Allowlist enforced at watcher level — rejected messages never reach dispatcher.
- [ ] DKIM pass required; fall back to rejecting if `Authentication-Results` is missing.
- [ ] No secrets in repo — `credentials.json`, `token.json`, Anthropic API key all in `~/.config/email_reply_bot/` or env vars.
- [ ] `config.example.yaml` in repo; real `config.yaml` gitignored.
- [ ] Outbound reply includes a footer noting it's an automated Claude response.
- [ ] Sandbox: dispatcher runs Claude with a restricted working directory and no unrestricted shell; allow-list the tools the agent can call.
- [ ] Optional shared-secret token required in message body (default off, togglable).
- [ ] Audit log reviewed after first week of operation.

## Test strategy

- **Unit**: allowlist normalization edge cases, header parsing, idempotency, rate limit.
- **Integration (offline)**: feed canned `.eml` through watcher → verify expected dispatch / rejection.
- **End-to-end (manual)**: send real mail from each of the three allowlisted accounts + one stranger account; confirm only the three trigger replies.
- **Regression**: keep fixtures of past bad inputs (spoofed sender, missing DKIM, tag alias, display-name trickery) and assert they're still rejected.

## Open questions

1. Which Gmail account hosts the watcher — `brandojazz@gmail.com` or a new dedicated address like `brando-claude-bot@gmail.com`?
2. Where does the daemon run — the Stanford server, a home machine, or a small VM?
3. Should the agent's working directory be inferred from the email thread (e.g. a tag like `[repo: solveall]`) or fixed to one default workspace in v1?
4. How much of the Claude Agent SDK's tool surface do we expose — read-only, or full shell? Start read-only.
5. Do we want a mirror-copy of every accepted inbound message saved to disk for later replay?
