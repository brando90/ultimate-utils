# email_reply_bot

Small always-on daemon that watches a Gmail inbox, accepts replies only from
Brando's three trusted addresses (`brando.science@gmail.com`,
`brandojazz@gmail.com`, `brando9@stanford.edu`), runs Claude against the reply
body, and emails the answer back in-thread.

See [`ISSUE.md`](ISSUE.md) for motivation and security requirements, and
[`PLAN.md`](PLAN.md) for the implementation roadmap.

## Layout

```
src/
  allowlist.py       address normalization + allowlist check
  auth_headers.py    SPF/DKIM/DMARC verdict parser for Gmail's Authentication-Results header
  message.py         RFC-822 parsing + quoted-reply stripping
  store.py           SQLite: seen message-ids, rate-limit, audit log
  gmail_client.py    GmailClient Protocol + MIME reply builder (no Google deps)
  dispatcher.py      LLMClient Protocol + prompt builder (no Anthropic deps)
  pipeline.py        end-to-end handler (testable against fakes)
  real_gmail.py      Gmail API-backed GmailClient (lazy imports googleapiclient)
  real_llm.py        Anthropic-backed LLMClient (lazy imports anthropic)
  main.py            daemon entry point (CLI, polling loop, pause file)

tests/
  test_allowlist.py
  test_auth_headers.py
  test_message.py
  test_store.py
  test_pipeline.py      full-pipeline integration tests with fake clients
  fixtures/*.eml        realistic raw MIME (legit, stranger, spoofed, alias)
```

## Install (deployment host)

```bash
cd experiments/email_reply_bot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## One-time Gmail OAuth setup

1. In GCP console, create a project, enable the Gmail API.
2. Create an OAuth client of type "Desktop" → download as `credentials.json`.
3. `mkdir -p ~/.config/email_reply_bot && mv ~/Downloads/credentials.json ~/.config/email_reply_bot/`
4. First run of the daemon will pop a browser consent window and write
   `~/.config/email_reply_bot/token.json`.

## Run

```bash
cp config.example.yaml config.yaml          # edit paths if needed
python -m src.main --config config.yaml --once --dry-run    # smoke test
python -m src.main --config config.yaml                     # daemon
touch /tmp/email_bot_pause                                  # pause (resume by deleting)
```

## Tests

```bash
cd experiments/email_reply_bot
python -m pytest -q                     # no google/anthropic deps required
```

All tests run offline against the `GmailClient` / `LLMClient` protocols via
in-memory fakes — no live network, no OAuth, no Anthropic key required.
