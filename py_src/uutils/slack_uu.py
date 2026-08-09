"""Slack messaging -- webhooks (send) and bot (read, reply, list channels).

Webhook usage (one-way, send only):
    from uutils.slack_uu import send_slack_message
    send_slack_message("Hello from uutils!")

Bot usage (read, reply, send via bot):
    from uutils.slack_uu import read_messages, send_bot_message, reply_to_message
    messages = read_messages("CHANNEL_ID")
    send_bot_message("CHANNEL_ID", "Hello via bot!")
    reply_to_message("CHANNEL_ID", "THREAD_TS", "Got it!")

CLI usage:
    python -m uutils.slack_uu --list-channels
    python -m uutils.slack_uu --read --channel-id C0123456789 --limit 5
    python -m uutils.slack_uu --send "Hello!" --channel-id C0123456789
    python -m uutils.slack_uu --reply "Thanks!" --channel-id C0123456789 --thread-ts 1234567890.123456

Setup (Webhook -- send only):
    1. Go to https://api.slack.com/apps -> "Create New App" -> "From scratch"
    2. Go to "Incoming Webhooks" in the left sidebar -> toggle ON
    3. Click "Add New Webhook to Workspace" -> select a channel -> "Allow"
    4. Copy the webhook URL
    5. Save it:
       echo 'https://hooks.slack.com/services/T.../B.../xxx' > ~/keys/slack_webhook_url.txt
       chmod 600 ~/keys/slack_webhook_url.txt

Setup (Bot -- read, reply, list channels):
    1. Go to https://api.slack.com/apps -> "Create New App" -> "From scratch"
    2. Go to "OAuth & Permissions" in the left sidebar
    3. Under "Bot Token Scopes", add these scopes:
       - channels:read       (list public channels)
       - channels:history    (read messages in public channels)
       - chat:write          (send messages)
       - groups:read         (list private channels the bot is in)
       - groups:history      (read messages in private channels)
    4. Click "Install to Workspace" at the top -> "Allow"
    5. Copy the "Bot User OAuth Token" (starts with xoxb-)
    6. Save it:
       echo 'xoxb-...' > ~/keys/slack_bot_token.txt
       chmod 600 ~/keys/slack_bot_token.txt
    7. Invite the bot to channels:
       In Slack, go to a channel and type: /invite @YourBotName

Testing:
    # Dry-run (no credentials needed):
    python -m uutils.slack_uu --smoke-test

    # Integration test with a real workspace:
    python -m uutils.slack_uu --list-channels
    python -m uutils.slack_uu --read --channel-id C0123456789 --limit 5
    python -m uutils.slack_uu --send "Test from uutils" --channel-id C0123456789

Refs:
    - Slack webhook docs: https://api.slack.com/messaging/webhooks
    - Slack Web API: https://api.slack.com/web
    - Slack Bot scopes: https://api.slack.com/scopes
    - chat.postMessage: https://api.slack.com/methods/chat.postMessage
    - conversations.history: https://api.slack.com/methods/conversations.history
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import requests

log = logging.getLogger(__name__)

DEFAULT_WEBHOOK_FILE = "~/keys/slack_webhook_url.txt"
DEFAULT_BOT_TOKEN_FILE = "~/keys/slack_bot_token.txt"
SLACK_API_BASE = "https://slack.com/api"
MAX_MESSAGE_LENGTH = 4000  # Slack's text limit per message
RATE_LIMIT_RETRY_ATTEMPTS = 3


# ── Credential resolution ─────────────────────────────────────────────


def _resolve_webhook_url(
    webhook_url: str = "",
    webhook_url_file: str = "",
) -> str:
    """Resolve webhook URL from argument, file, or default file location."""
    if webhook_url:
        return webhook_url
    fpath = Path(webhook_url_file or DEFAULT_WEBHOOK_FILE).expanduser()
    if fpath.is_file():
        url = fpath.read_text().strip()
        if url:
            return url
    raise ValueError(
        f"No Slack webhook URL: provide webhook_url, or save one to {DEFAULT_WEBHOOK_FILE}\n"
        "Create a webhook: https://api.slack.com/apps -> Incoming Webhooks"
    )


def _resolve_bot_token(
    bot_token: str = "",
    bot_token_file: str = "",
) -> str:
    """Resolve bot token from argument, env var, file, or default file location."""
    if bot_token:
        return bot_token
    env_token = os.environ.get("SLACK_BOT_TOKEN", "")
    if env_token:
        return env_token
    fpath = Path(bot_token_file or DEFAULT_BOT_TOKEN_FILE).expanduser()
    if fpath.is_file():
        token = fpath.read_text().strip()
        if token:
            return token
    raise ValueError(
        f"No Slack bot token: provide bot_token, set SLACK_BOT_TOKEN env var, "
        f"or save one to {DEFAULT_BOT_TOKEN_FILE}\n"
        "Create a bot: https://api.slack.com/apps -> OAuth & Permissions"
    )


# ── HTTP helpers ───────────────────────────────────────────────────────


def _retry_after_seconds(resp: requests.Response) -> float:
    """Extract retry-after from Slack rate limit response."""
    retry_after = resp.headers.get("Retry-After", 1.0)
    try:
        return max(float(retry_after), 0.0)
    except (TypeError, ValueError):
        return 1.0


def _slack_api(
    method: str,
    bot_token: str,
    json_payload: dict | None = None,
    params: dict | None = None,
) -> dict:
    """Make an authenticated request to the Slack Web API.

    Slack returns 200 with {"ok": false, "error": "..."} for app-level errors,
    and uses HTTP 429 for rate limits. This handles both.
    """
    url = f"{SLACK_API_BASE}/{method}"
    headers = {
        "Authorization": f"Bearer {bot_token}",
        "Content-Type": "application/json; charset=utf-8",
    }
    for attempt in range(RATE_LIMIT_RETRY_ATTEMPTS):
        resp = requests.post(
            url, headers=headers,
            json=json_payload, params=params, timeout=30,
        )
        if resp.status_code == 429:
            retry_after = _retry_after_seconds(resp)
            log.warning("Slack rate limited, retrying in %.1fs (attempt %d/%d)",
                        retry_after, attempt + 1, RATE_LIMIT_RETRY_ATTEMPTS)
            time.sleep(retry_after)
            continue
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok"):
            error = data.get("error", "unknown_error")
            raise RuntimeError(f"Slack API error ({method}): {error}")
        return data
    resp.raise_for_status()
    data = resp.json()
    if not data.get("ok"):
        raise RuntimeError(f"Slack API error ({method}): {data.get('error', 'unknown_error')}")
    return data


def _post_webhook_with_retry(url: str, payload: dict) -> None:
    """POST to a Slack incoming webhook with rate-limit retry."""
    for attempt in range(RATE_LIMIT_RETRY_ATTEMPTS):
        resp = requests.post(url, json=payload, timeout=30)
        if resp.status_code == 429:
            retry_after = _retry_after_seconds(resp)
            log.warning("Slack rate limited, retrying in %.1fs (attempt %d/%d)",
                        retry_after, attempt + 1, RATE_LIMIT_RETRY_ATTEMPTS)
            time.sleep(retry_after)
            continue
        resp.raise_for_status()
        return
    resp.raise_for_status()


def _split_message(message: str) -> list[str]:
    """Split a message into chunks that fit Slack's limit."""
    if len(message) <= MAX_MESSAGE_LENGTH:
        return [message]
    chunks = []
    while message:
        if len(message) <= MAX_MESSAGE_LENGTH:
            chunks.append(message)
            break
        split_at = message.rfind("\n", 0, MAX_MESSAGE_LENGTH)
        if split_at <= 0:
            split_at = MAX_MESSAGE_LENGTH
        chunks.append(message[:split_at])
        message = message[split_at:].lstrip("\n")
    return chunks


# ── Webhook functions (send only) ─────────────────────────────────────


def send_slack_message(
    message: str,
    webhook_url: str = "",
    webhook_url_file: str = "",
    username: str = "",
    icon_emoji: str = "",
    channel: str = "",
    dry_run: bool = False,
) -> None:
    """Send a plain text message to Slack via incoming webhook.

    Args:
        message: Text content to send (auto-split if >4000 chars).
        webhook_url: Slack webhook URL. If empty, reads from webhook_url_file.
        webhook_url_file: Path to file containing webhook URL.
        username: Override the webhook's default username.
        icon_emoji: Override the webhook's default icon (e.g., ":robot_face:").
        channel: Override the webhook's default channel (e.g., "#general").
        dry_run: If True, print instead of sending.
    """
    if dry_run:
        log.info("[DRY-RUN] Slack message: %s", message[:200])
        print(f"[DRY-RUN] Slack message ({len(message)} chars): {message[:200]}...")
        return

    url = _resolve_webhook_url(webhook_url, webhook_url_file)
    chunks = _split_message(message)
    for chunk in chunks:
        payload: dict = {"text": chunk}
        if username:
            payload["username"] = username
        if icon_emoji:
            payload["icon_emoji"] = icon_emoji
        if channel:
            payload["channel"] = channel
        _post_webhook_with_retry(url, payload)
    log.info("Slack webhook message sent (%d chunk(s), %d chars total)", len(chunks), len(message))


def send_slack_block_message(
    blocks: list[dict],
    text: str = "",
    webhook_url: str = "",
    webhook_url_file: str = "",
    username: str = "",
    dry_run: bool = False,
) -> None:
    """Send a rich Block Kit message to Slack via incoming webhook.

    Args:
        blocks: List of Slack Block Kit block dicts.
            See: https://api.slack.com/block-kit
        text: Fallback text for notifications (recommended).
        webhook_url: Slack webhook URL.
        webhook_url_file: Path to file containing webhook URL.
        username: Override the webhook's default username.
        dry_run: If True, print instead of sending.
    """
    if dry_run:
        log.info("[DRY-RUN] Slack block message: %d blocks", len(blocks))
        print(f"[DRY-RUN] Slack block message: {len(blocks)} block(s)")
        return

    url = _resolve_webhook_url(webhook_url, webhook_url_file)
    payload: dict = {"blocks": blocks}
    if text:
        payload["text"] = text
    if username:
        payload["username"] = username
    _post_webhook_with_retry(url, payload)
    log.info("Slack block message sent (%d blocks)", len(blocks))


# ── Bot functions (REST-based via Slack Web API) ──────────────────────


def list_channels(
    bot_token: str = "",
    bot_token_file: str = "",
    include_private: bool = False,
    dry_run: bool = False,
) -> list[dict]:
    """List Slack channels the bot can see.

    Args:
        bot_token: Bot token directly.
        bot_token_file: Path to file containing bot token.
        include_private: If True, include private channels (requires groups:read scope).
        dry_run: If True, print instead of making API call.

    Returns:
        List of channel dicts with keys: id, name, is_private, num_members, topic, etc.
    """
    if dry_run:
        print("[DRY-RUN] Would list Slack channels")
        return []
    token = _resolve_bot_token(bot_token, bot_token_file)
    types = "public_channel,private_channel" if include_private else "public_channel"
    result = _slack_api("conversations.list", token, params={"types": types, "limit": 200})
    channels = result.get("channels", [])
    log.info("Found %d channel(s)", len(channels))
    return channels


def read_messages(
    channel_id: str,
    limit: int = 10,
    bot_token: str = "",
    bot_token_file: str = "",
    dry_run: bool = False,
) -> list[dict]:
    """Read recent messages from a Slack channel.

    Args:
        channel_id: The Slack channel ID (e.g., C0123456789).
        limit: Number of messages to fetch (max 100, default 10).
        bot_token: Bot token directly.
        bot_token_file: Path to file containing bot token.
        dry_run: If True, print instead of making API call.

    Returns:
        List of message dicts (newest first) with keys: text, user, ts, thread_ts, etc.
    """
    if dry_run:
        print(f"[DRY-RUN] Would read {limit} messages from channel {channel_id}")
        return []
    token = _resolve_bot_token(bot_token, bot_token_file)
    limit = min(max(limit, 1), 100)
    result = _slack_api("conversations.history", token, params={
        "channel": channel_id,
        "limit": limit,
    })
    messages = result.get("messages", [])
    log.info("Read %d message(s) from channel %s", len(messages), channel_id)
    return messages


def read_thread(
    channel_id: str,
    thread_ts: str,
    limit: int = 50,
    bot_token: str = "",
    bot_token_file: str = "",
    dry_run: bool = False,
) -> list[dict]:
    """Read replies in a Slack thread.

    Args:
        channel_id: The Slack channel ID containing the thread.
        thread_ts: Timestamp of the parent message (the thread root).
        limit: Number of replies to fetch (max 100, default 50).
        bot_token: Bot token directly.
        bot_token_file: Path to file containing bot token.
        dry_run: If True, print instead of making API call.

    Returns:
        List of message dicts in the thread (including the parent).
    """
    if dry_run:
        print(f"[DRY-RUN] Would read thread {thread_ts} in channel {channel_id}")
        return []
    token = _resolve_bot_token(bot_token, bot_token_file)
    limit = min(max(limit, 1), 100)
    result = _slack_api("conversations.replies", token, params={
        "channel": channel_id,
        "ts": thread_ts,
        "limit": limit,
    })
    messages = result.get("messages", [])
    log.info("Read %d message(s) in thread %s", len(messages), thread_ts)
    return messages


def send_bot_message(
    channel_id: str,
    message: str,
    bot_token: str = "",
    bot_token_file: str = "",
    dry_run: bool = False,
) -> dict | None:
    """Send a message to a Slack channel using the bot.

    Args:
        channel_id: The Slack channel ID.
        message: Text content to send (auto-split if >4000 chars).
        bot_token: Bot token directly.
        bot_token_file: Path to file containing bot token.
        dry_run: If True, print instead of sending.

    Returns:
        API response dict for the last sent chunk, or None for dry-run.
    """
    if dry_run:
        log.info("[DRY-RUN] Bot message to channel %s: %s", channel_id, message[:200])
        print(f"[DRY-RUN] Bot message to channel {channel_id} ({len(message)} chars): {message[:200]}...")
        return None
    token = _resolve_bot_token(bot_token, bot_token_file)
    chunks = _split_message(message)
    result = None
    for chunk in chunks:
        result = _slack_api("chat.postMessage", token, json_payload={
            "channel": channel_id,
            "text": chunk,
        })
    log.info("Bot message sent to channel %s (%d chunk(s), %d chars total)",
             channel_id, len(chunks), len(message))
    return result


def reply_to_message(
    channel_id: str,
    thread_ts: str,
    reply: str,
    bot_token: str = "",
    bot_token_file: str = "",
    broadcast: bool = False,
    dry_run: bool = False,
) -> dict | None:
    """Reply to a message thread in a Slack channel.

    Args:
        channel_id: The Slack channel ID containing the thread.
        thread_ts: Timestamp of the parent message to reply to.
        reply: Text content of the reply.
        bot_token: Bot token directly.
        bot_token_file: Path to file containing bot token.
        broadcast: If True, also post the reply to the channel (not just the thread).
        dry_run: If True, print instead of sending.

    Returns:
        API response dict, or None for dry-run.
    """
    if dry_run:
        log.info("[DRY-RUN] Reply in thread %s in channel %s: %s",
                 thread_ts, channel_id, reply[:200])
        print(f"[DRY-RUN] Reply in thread {thread_ts}: {reply[:200]}...")
        return None
    token = _resolve_bot_token(bot_token, bot_token_file)
    payload = {
        "channel": channel_id,
        "text": reply[:MAX_MESSAGE_LENGTH],
        "thread_ts": thread_ts,
    }
    if broadcast:
        payload["reply_broadcast"] = True
    result = _slack_api("chat.postMessage", token, json_payload=payload)
    log.info("Reply sent in thread %s in channel %s", thread_ts, channel_id)
    return result


def add_reaction(
    channel_id: str,
    timestamp: str,
    emoji: str,
    bot_token: str = "",
    bot_token_file: str = "",
    dry_run: bool = False,
) -> dict | None:
    """Add an emoji reaction to a Slack message.

    Args:
        channel_id: The Slack channel ID.
        timestamp: Timestamp of the message to react to.
        emoji: Emoji name without colons (e.g., "thumbsup", "white_check_mark").
        bot_token: Bot token directly.
        bot_token_file: Path to file containing bot token.
        dry_run: If True, print instead of sending.

    Returns:
        API response dict, or None for dry-run.
    """
    if dry_run:
        print(f"[DRY-RUN] Would react :{emoji}: to message {timestamp} in {channel_id}")
        return None
    token = _resolve_bot_token(bot_token, bot_token_file)
    result = _slack_api("reactions.add", token, json_payload={
        "channel": channel_id,
        "timestamp": timestamp,
        "name": emoji,
    })
    log.info("Reaction :%s: added to %s in %s", emoji, timestamp, channel_id)
    return result


def upload_file(
    channel_id: str,
    file_path: str | Path,
    message: str = "",
    bot_token: str = "",
    bot_token_file: str = "",
    thread_ts: str = "",
    dry_run: bool = False,
) -> dict | None:
    """Upload a file to a Slack channel.

    Args:
        channel_id: The Slack channel ID.
        file_path: Path to the file to upload.
        message: Optional message to accompany the file.
        bot_token: Bot token directly.
        bot_token_file: Path to file containing bot token.
        thread_ts: If provided, upload into this thread.
        dry_run: If True, print instead of uploading.

    Returns:
        API response dict, or None for dry-run.
    """
    fp = Path(file_path).expanduser()
    if not fp.is_file():
        raise FileNotFoundError(f"File not found: {fp}")

    if dry_run:
        log.info("[DRY-RUN] Slack file upload: %s", fp)
        print(f"[DRY-RUN] Slack file upload: {fp.name} ({fp.stat().st_size} bytes)")
        return None

    token = _resolve_bot_token(bot_token, bot_token_file)

    # Step 1: Get an upload URL
    get_url_result = _slack_api("files.getUploadURLExternal", token, params={
        "filename": fp.name,
        "length": fp.stat().st_size,
    })
    upload_url = get_url_result["upload_url"]
    file_id = get_url_result["file_id"]

    # Step 2: Upload the file content
    with open(fp, "rb") as f:
        resp = requests.post(upload_url, files={"file": (fp.name, f)}, timeout=60)
        resp.raise_for_status()

    # Step 3: Complete the upload and share to channel
    complete_payload = {
        "files": [{"id": file_id, "title": fp.name}],
        "channel_id": channel_id,
    }
    if message:
        complete_payload["initial_comment"] = message[:MAX_MESSAGE_LENGTH]
    if thread_ts:
        complete_payload["thread_ts"] = thread_ts
    result = _slack_api("files.completeUploadExternal", token, json_payload=complete_payload)
    log.info("Slack file uploaded: %s (%d bytes) to %s", fp.name, fp.stat().st_size, channel_id)
    return result


# ── Smoke test & CLI ──────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Slack utilities -- webhooks and bot operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Dry-run smoke tests (no credentials needed)
  python -m uutils.slack_uu --smoke-test

  # List channels the bot can see
  python -m uutils.slack_uu --list-channels

  # Read last 5 messages from a channel
  python -m uutils.slack_uu --read --channel-id C0123456789 --limit 5

  # Read a thread
  python -m uutils.slack_uu --read-thread --channel-id C0123456789 --thread-ts 1234567890.123456

  # Send a message to a channel
  python -m uutils.slack_uu --send "Hello!" --channel-id C0123456789

  # Reply in a thread
  python -m uutils.slack_uu --reply "Thanks!" --channel-id C0123456789 --thread-ts 1234567890.123456

  # Upload a file
  python -m uutils.slack_uu --upload ~/results/plot.png --channel-id C0123456789
""",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Run dry-run smoke tests")
    parser.add_argument("--list-channels", action="store_true", help="List channels the bot can see")
    parser.add_argument("--read", action="store_true", help="Read recent messages from a channel")
    parser.add_argument("--read-thread", action="store_true", help="Read replies in a thread")
    parser.add_argument("--send", type=str, default="", help="Send a message to a channel")
    parser.add_argument("--reply", type=str, default="", help="Reply in a thread")
    parser.add_argument("--react", type=str, default="", help="Add reaction emoji (e.g., thumbsup)")
    parser.add_argument("--upload", type=str, default="", help="Upload a file to a channel")
    parser.add_argument("--channel-id", type=str, default="", help="Channel ID (e.g., C0123456789)")
    parser.add_argument("--thread-ts", type=str, default="", help="Thread timestamp (for --reply, --read-thread)")
    parser.add_argument("--message-ts", type=str, default="", help="Message timestamp (for --react)")
    parser.add_argument("--limit", type=int, default=10, help="Number of messages to read (default: 10)")
    parser.add_argument("--broadcast", action="store_true", help="Also post reply to channel")
    parser.add_argument("--dry-run", action="store_true", help="Print instead of sending")
    parser.add_argument("--bot-token-file", type=str, default="", help="Path to bot token file")

    args = parser.parse_args()

    if args.smoke_test:
        print("=== Slack dry-run smoke tests ===")
        send_slack_message("Hello from uutils!", dry_run=True)
        send_slack_block_message([{"type": "section", "text": {"type": "mrkdwn", "text": "Test"}}], dry_run=True)
        long_msg = "x" * 8500
        chunks = _split_message(long_msg)
        assert len(chunks) == 3, f"Expected 3 chunks, got {len(chunks)}"
        assert all(len(c) <= MAX_MESSAGE_LENGTH for c in chunks)
        print(f"Message splitting: 8500 chars -> {len(chunks)} chunks OK")
        # Bot function dry-run tests
        list_channels(dry_run=True)
        read_messages("C000000000", dry_run=True)
        read_thread("C000000000", "1234567890.123456", dry_run=True)
        send_bot_message("C000000000", "Test bot message", dry_run=True)
        reply_to_message("C000000000", "1234567890.123456", "Test reply", dry_run=True)
        add_reaction("C000000000", "1234567890.123456", "thumbsup", dry_run=True)
        print("All dry-run smoke tests passed!")
    elif args.list_channels:
        channels = list_channels(bot_token_file=args.bot_token_file, dry_run=args.dry_run)
        for ch in channels:
            prefix = "🔒" if ch.get("is_private") else " #"
            print(f"  {prefix}{ch['name']}  (id: {ch['id']}, members: {ch.get('num_members', '?')})")
    elif args.read:
        if not args.channel_id:
            parser.error("--read requires --channel-id")
        messages = read_messages(args.channel_id, limit=args.limit,
                                 bot_token_file=args.bot_token_file, dry_run=args.dry_run)
        for msg in reversed(messages):  # oldest first
            user = msg.get("user", "?")
            text = msg.get("text", "")
            ts = msg.get("ts", "")
            thread_indicator = " [thread]" if msg.get("thread_ts") else ""
            print(f"  [{ts}] {user}: {text[:200]}{thread_indicator}")
            print(f"    (ts: {ts})")
    elif args.read_thread:
        if not args.channel_id or not args.thread_ts:
            parser.error("--read-thread requires --channel-id and --thread-ts")
        messages = read_thread(args.channel_id, args.thread_ts, limit=args.limit,
                               bot_token_file=args.bot_token_file, dry_run=args.dry_run)
        for msg in messages:
            user = msg.get("user", "?")
            text = msg.get("text", "")
            ts = msg.get("ts", "")
            print(f"  [{ts}] {user}: {text[:200]}")
    elif args.send:
        if not args.channel_id:
            parser.error("--send requires --channel-id")
        send_bot_message(args.channel_id, args.send,
                         bot_token_file=args.bot_token_file, dry_run=args.dry_run)
        if not args.dry_run:
            print("Message sent.")
    elif args.reply:
        if not args.channel_id or not args.thread_ts:
            parser.error("--reply requires --channel-id and --thread-ts")
        reply_to_message(args.channel_id, args.thread_ts, args.reply,
                         broadcast=args.broadcast,
                         bot_token_file=args.bot_token_file, dry_run=args.dry_run)
        if not args.dry_run:
            print("Reply sent.")
    elif args.react:
        if not args.channel_id or not args.message_ts:
            parser.error("--react requires --channel-id and --message-ts")
        add_reaction(args.channel_id, args.message_ts, args.react,
                     bot_token_file=args.bot_token_file, dry_run=args.dry_run)
        if not args.dry_run:
            print("Reaction added.")
    elif args.upload:
        if not args.channel_id:
            parser.error("--upload requires --channel-id")
        upload_file(args.channel_id, args.upload,
                    bot_token_file=args.bot_token_file,
                    thread_ts=args.thread_ts, dry_run=args.dry_run)
        if not args.dry_run:
            print("File uploaded.")
    else:
        parser.print_help()
