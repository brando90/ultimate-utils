GitHub issue #27 "Slack automation: messaging, channel monitoring, and notifications".

Create `py_src/uutils/slack_uu.py` with a `SlackClient` that talks to the official Slack Web API (`https://slack.com/api/<method>`) using plain `requests` (no `slack-sdk` needed).
- `SlackClient(token: str | None = None, dry_run: bool = True)` and `SlackClient.from_token(token_file: str = "~/keys/slack_bot_token.txt", dry_run: bool = True)`; `from_token` reads the file (or env var `SLACK_BOT_TOKEN`) only when `dry_run=False`.
- Methods: `send_message(channel, text, thread_ts=None)` (`chat.postMessage`), `upload_file(channel, file_path, title="")` (use the current external-upload flow `files.getUploadURLExternal` + upload + `files.completeUploadExternal`; `files.upload` is deprecated), `list_channels()` (`conversations.list`, handle cursor pagination), `get_channel_history(channel, limit=50, oldest=None)` (`conversations.history`), `get_unread_messages(channel, since_ts)` (history newer than `since_ts`).
- Every Slack call goes through one private `_call(method, **params)` that, in dry-run, returns `{"ok": True, "dry_run": True, "method": ..., "params": ...}` and prints a one-line description, and otherwise sends the request with `Authorization: Bearer <token>` and raises a clear error when the response has `"ok": false`.
- A convenience `notify(text, channel, dry_run=True)` for "experiment finished" notifications.
- A small CLI: `python -m uutils.slack_uu send --channel C --text T [--send]` (dry-run unless `--send`).
- Module docstring: one-time setup (create a Slack app, bot scopes `chat:write`, `channels:history`, `channels:read`, `files:write`, install to workspace, save bot token to `~/keys/slack_bot_token.txt` with `chmod 600`, invite bot to channels), and usage examples.
- Tests: `tests/test_slack_uu.py` (dry-run default makes no HTTP call; `_call` builds correct URL/headers/params with a mocked `requests`; pagination in `list_channels`; error on `ok: false`).
