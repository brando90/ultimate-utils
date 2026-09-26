# Implementation plan: email-triggered Claude agent

Tracking issue: `ISSUE.md` (same directory).

Target location in repo: `experiments/03_email_reply_bot/`.

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
  │   - spawn clauded -p  │
  │     session           │
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

## Files

```
experiments/03_email_reply_bot/
├── ISSUE.md                  (feature issue text)
├── PLAN.md                   (this file)
├── README.md                 (setup / run instructions)
├── requirements.txt          (isolated deps; not added to uutils core)
├── config.example.yaml       (allowlist, paths, rate limits — no secrets)
├── src/
│   ├── __init__.py
│   ├── allowlist.py          (verify_sender, normalize_addr)
│   ├── auth_headers.py       (SPF/DKIM/DMARC verification)
│   ├── gmail_client.py       (GmailClient Protocol + MIME builder)
│   ├── dispatcher.py         (LLMClient Protocol + prompt builder)
│   ├── pipeline.py           (end-to-end handler)
│   ├── real_gmail.py         (Gmail API-backed client)
│   ├── real_llm.py           (clauded -p subprocess runner)
│   ├── store.py              (sqlite: seen message-ids, rate limits, audit log)
│   └── main.py               (daemon and offline demo entry point)
└── tests/
    ├── conftest.py
    ├── test_allowlist.py
    ├── test_auth_headers.py
    ├── test_message.py
    ├── test_pipeline.py
    ├── test_real_llm.py
    ├── test_main.py
    ├── test_store.py
    └── fixtures/             (sample raw MIME: legit, spoofed, stranger)
```

## Dependencies

- `google-api-python-client`, `google-auth`, `google-auth-oauthlib` (Gmail API)
- `pyyaml` for config
- `pytest` for tests
- `clauded` CLI tool installed and logged in locally (invoked via `subprocess` with `clauded -p`)

Install locally into a venv under `experiments/03_email_reply_bot/.venv` — do not pollute uutils' core deps.

## Milestones

### M1 — Local dry run (no network) — ~half day
- [x] `allowlist.py` with `verify_sender(addr) -> bool`, unit-tested for aliases, case, `+tag`, display-name stripping.
- [x] `auth_headers.py` that parses Gmail's `Authentication-Results:` header and returns `{spf, dkim, dmarc}` verdicts.
- [x] Fixtures: raw `.eml` files — legit, spoofed, stranger, aliases.
- [x] `store.py` with sqlite for seen-ids + audit log.
- [x] `dispatcher.py` / `real_llm.py`: runs `clauded -p` headlessly via subprocess and returns stripped stdout.

### M2 — Gmail round-trip (read-only first) — ~half day
- [x] OAuth flow: create a GCP project, enable Gmail API, generate `credentials.json`, run a one-time consent to produce `token.json`. Store under `~/keys/`.
- [x] `gmail_client.fetch_unseen()` — pulls unseen messages in a specific label (`INBOX`).
- [x] Safe demo mode and `--live` mode with `--dry-run` default.

### M3 — Send reply — ~half day
- [x] Outbound threaded reply MIME with `In-Reply-To`, `References`, `Subject: Re: …`, uses `threadId` for Gmail.
- [x] Idempotency: reject if `Message-ID` already in `store.seen`.
- [x] Rate limit per sender.

### M4 — Hardening & deploy — ~half day
- [x] Sane defaults: offline demo runs by default on fixtures without `--live`.
- [x] Config file with sane defaults.
- [x] README with full setup walkthrough and offline demo instructions.
- [x] Kill switch: a file like `/tmp/email_bot_pause` pauses processing.

### M5 — Nice-to-haves (later)
- Gmail Pub/Sub push instead of polling (sub-second latency).
- Per-thread memory (SQLite keyed by `threadId`) so follow-ups carry context.
- Attachment support (inbound screenshots → vision; outbound log files).
- Multi-inbox support (watch `brando.science` too).

## Security checklist (must all be true before enabling send)

- [x] Allowlist enforced at watcher level — rejected messages never reach dispatcher.
- [x] DKIM pass required; fall back to rejecting if `Authentication-Results` is missing.
- [x] No secrets in repo — `credentials.json`, `token.json` in `~/keys/` or env vars; `clauded` logged in locally.
- [x] `config.example.yaml` in repo; real `config.yaml` gitignored.
- [x] Outbound reply includes a footer noting it's an automated Claude response.
- [x] Sandbox: dispatcher runs Claude via `clauded -p` with configured working directory.
- [x] Dry-run default: outbound sending only active when both `--live` and `--send` are explicitly passed.
