# 03_email_reply_bot

A secure, allowlist-gated email reply daemon. Watches a Gmail inbox, accepts replies only from trusted addresses, executes a headless Claude session via the local `clauded -p` CLI (invoked via `subprocess`), and emails the answer back in-thread.

No direct LLM API code or third-party SDKs are used. Dry-run is the default everywhere.

See [`ISSUE.md`](ISSUE.md) for background and security specifications, and [`PLAN.md`](PLAN.md) for architecture details.

## What it does

When experiment runs email status updates to `brandojazz@gmail.com`, hitting "Reply" from an authorized address with an instruction or question runs a headless Claude session via `clauded -p` and sends the result back in the same email thread with valid `In-Reply-To` and `References` headers.

### Sender allowlist

Only these three verified addresses can trigger a Claude session:
- `brando.science@gmail.com` (including `+tag` aliases)
- `brandojazz@gmail.com`
- `brando9@stanford.edu`

All incoming messages undergo strict fail-closed security filtering:
- Normalized address must match the allowlist.
- SPF, DKIM, and DMARC verification results in `Authentication-Results` must pass.
- `Reply-To` and `Return-Path` headers must match `From:` (anti-spoof protection).
- Unseen messages are deduplicated by `Message-ID` in SQLite.
- Senders are rate-limited (default: 10 accepted replies per hour).
- Any unverified or unauthorized message is silently rejected with no response sent.

## Offline demo (safe default)

By default, running `src.main` performs an offline demo over bundled `.eml` fixtures. It connects to **no** network, touches **no** credentials, invokes **no** subprocess, and sends **no** mail:

```bash
cd experiments/03_email_reply_bot
python -m src.main
```

Expected demo output:
```text
[alias_tag.eml] accepted=True reason=accepted
[legit_brando9_stanford.eml] accepted=True reason=accepted
[legit_brando_science.eml] accepted=True reason=accepted
[legit_brandojazz.eml] accepted=True reason=accepted
[reply_to_mismatch.eml] accepted=False reason=reply-to mismatch
[spoofed_dkim_fail.eml] accepted=False reason=auth headers: spf=softfail
[stranger.eml] accepted=False reason=sender not in allowlist
```

You can also pass a custom directory of `.eml` files:
```bash
python -m src.main --fixtures /path/to/eml/dir
```

## Setup for live use

Live operation requires explicit human setup on an always-on host:

### 1. Always-on host
Deploy on a persistent server or VM with network access to Gmail.

### 2. Claude CLI (`clauded`)
Install and authenticate the `clauded` CLI tool:
```bash
clauded login
```
Verify that `clauded -p "ping"` runs and produces output.

### 3. Gmail OAuth credentials
1. In the Google Cloud Console, create a project and enable the **Gmail API**.
2. Configure an OAuth Consent Screen and create credentials for a **Desktop App**.
3. Download the client secret JSON to `~/keys/gmail_oauth_credentials.json` and restrict permissions:
   ```bash
   chmod 600 ~/keys/gmail_oauth_credentials.json
   ```
4. On first live run, complete the one-time browser OAuth consent to generate `~/keys/gmail_oauth_token.json` (chmod 600).

### 4. Configuration
Copy the template and adjust paths:
```bash
cp config.example.yaml config.yaml
```

## Running the daemon

The daemon requires `--live` to instantiate real Gmail and `clauded` clients, and **only acts with `--live --send`**:

```bash
# Dry-run live watcher (fetches and checks mail, but sends no outbound email):
python -m src.main --live --config config.yaml

# Live daemon with outbound email sending enabled:
python -m src.main --live --send --config config.yaml

# Process a single batch and exit:
python -m src.main --live --send --config config.yaml --once
```

Emergency pause:
```bash
touch /tmp/email_bot_pause   # pauses polling
rm /tmp/email_bot_pause      # resumes polling
```

## Running tests

All unit and integration tests run offline using in-memory fakes and monkeypatched subprocesses. No credentials, live Gmail, or `clauded` binary required:

```bash
cd experiments/03_email_reply_bot
PYTHONPATH=. /lfs/skampere2/0/brando9/uu-agy-issues-venv/bin/python -m pytest -q tests
```
