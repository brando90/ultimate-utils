"""Discord messaging -- webhooks (send) and bot (read, reply, list channels).

Webhook usage (one-way, send only):
    from uutils.discord_uu import send_discord_message
    send_discord_message("Hello from uutils!")

Bot usage (read, reply, send via bot):
    from uutils.discord_uu import read_messages, send_bot_message, reply_to_message
    messages = read_messages("CHANNEL_ID")
    send_bot_message("CHANNEL_ID", "Hello via bot!")
    reply_to_message("CHANNEL_ID", "MESSAGE_ID", "Got it!")

CLI usage:
    python -m uutils.discord_uu --list-guilds
    python -m uutils.discord_uu --list-channels --guild-id 123456
    python -m uutils.discord_uu --read --channel-id 123456 --limit 5
    python -m uutils.discord_uu --send "Hello!" --channel-id 123456
    python -m uutils.discord_uu --reply "Thanks!" --channel-id 123 --message-id 456

Setup (Webhook -- send only):
    1. In Discord: Server Settings -> Integrations -> Webhooks -> New Webhook
    2. Copy the webhook URL
    3. Save it:
       echo 'https://discord.com/api/webhooks/...' > ~/keys/discord_webhook_url.txt
       chmod 600 ~/keys/discord_webhook_url.txt

Setup (Bot -- read, reply, list channels):
    1. Go to https://discord.com/developers/applications
    2. Click "New Application", give it a name, click "Create"
    3. Go to "Bot" in the left sidebar
    4. Click "Reset Token" to generate a bot token -- copy it immediately
    5. Under "Privileged Gateway Intents", enable "Message Content Intent"
    6. Save the token:
       echo 'YOUR_BOT_TOKEN' > ~/keys/discord_bot_token.txt
       chmod 600 ~/keys/discord_bot_token.txt
    7. Invite the bot to your server:
       - Go to "OAuth2" -> "URL Generator" in the left sidebar
       - Select scopes: "bot"
       - Select bot permissions: "Read Messages/View Channels", "Send Messages",
         "Read Message History"
       - Copy the generated URL and open it in your browser
       - Select your server and authorize

Testing:
    # Dry-run (no credentials needed):
    python -m uutils.discord_uu --smoke-test

    # Integration test with a real server:
    python -m uutils.discord_uu --list-guilds
    python -m uutils.discord_uu --read --channel-id YOUR_CHANNEL_ID --limit 5
    python -m uutils.discord_uu --send "Test from uutils" --channel-id YOUR_CHANNEL_ID

Refs:
    - Discord webhook docs: https://discord.com/developers/docs/resources/webhook
    - Discord bot docs: https://discord.com/developers/docs/intro
    - discord.py: https://discordpy.readthedocs.io/en/stable/
    - Embed structure: https://discord.com/developers/docs/resources/message#embed-object
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

DEFAULT_WEBHOOK_FILE = "~/keys/discord_webhook_url.txt"
DEFAULT_BOT_TOKEN_FILE = "~/keys/discord_bot_token.txt"
DISCORD_API_BASE = "https://discord.com/api/v10"
MAX_MESSAGE_LENGTH = 2000
RATE_LIMIT_RETRY_ATTEMPTS = 3


def _retry_after_seconds(resp: requests.Response) -> float:
    try:
        retry_after = resp.json().get("retry_after", 1.0)
    except ValueError:
        retry_after = resp.headers.get("Retry-After", 1.0)
    try:
        return max(float(retry_after), 0.0)
    except (TypeError, ValueError):
        return 1.0


def _rewind_request_files(files: dict) -> None:
    for value in files.values():
        file_obj = value[1] if isinstance(value, tuple) else value
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)


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
        f"No Discord webhook URL: provide webhook_url, or save one to {DEFAULT_WEBHOOK_FILE}\n"
        "Create a webhook: Discord Server Settings → Integrations → Webhooks → New Webhook"
    )


def _resolve_bot_token(
    bot_token: str = "",
    bot_token_file: str = "",
) -> str:
    """Resolve bot token from argument, env var, file, or default file location."""
    if bot_token:
        return bot_token
    env_token = os.environ.get("DISCORD_BOT_TOKEN", "")
    if env_token:
        return env_token
    fpath = Path(bot_token_file or DEFAULT_BOT_TOKEN_FILE).expanduser()
    if fpath.is_file():
        token = fpath.read_text().strip()
        if token:
            return token
    raise ValueError(
        f"No Discord bot token: provide bot_token, set DISCORD_BOT_TOKEN env var, "
        f"or save one to {DEFAULT_BOT_TOKEN_FILE}\n"
        "Create a bot: https://discord.com/developers/applications → New Application → Bot"
    )


def _bot_request(
    method: str,
    endpoint: str,
    bot_token: str,
    json_payload: dict | None = None,
    params: dict | None = None,
) -> dict | list:
    """Make an authenticated request to the Discord REST API."""
    url = f"{DISCORD_API_BASE}{endpoint}"
    headers = {
        "Authorization": f"Bot {bot_token}",
        "Content-Type": "application/json",
    }
    for attempt in range(RATE_LIMIT_RETRY_ATTEMPTS):
        resp = requests.request(
            method, url, headers=headers,
            json=json_payload, params=params, timeout=30,
        )
        if resp.status_code == 429:
            retry_after = _retry_after_seconds(resp)
            log.warning("Discord rate limited, retrying in %.1fs (attempt %d/%d)",
                        retry_after, attempt + 1, RATE_LIMIT_RETRY_ATTEMPTS)
            time.sleep(retry_after)
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()
    return resp.json()


def _post_with_retry(url: str, **kwargs) -> requests.Response:
    """POST to Discord with rate-limit retry."""
    files = kwargs.get("files")
    for attempt in range(RATE_LIMIT_RETRY_ATTEMPTS):
        if files:
            _rewind_request_files(files)
        resp = requests.post(url, timeout=30, **kwargs)
        if resp.status_code == 429:
            retry_after = _retry_after_seconds(resp)
            log.warning("Discord rate limited, retrying in %.1fs (attempt %d/%d)",
                        retry_after, attempt + 1, RATE_LIMIT_RETRY_ATTEMPTS)
            time.sleep(retry_after)
            continue
        resp.raise_for_status()
        return resp
    resp.raise_for_status()
    return resp


def _split_message(message: str) -> list[str]:
    """Split a message into chunks that fit Discord's 2000-char limit."""
    if len(message) <= MAX_MESSAGE_LENGTH:
        return [message]
    chunks = []
    while message:
        if len(message) <= MAX_MESSAGE_LENGTH:
            chunks.append(message)
            break
        # Try to split at a newline
        split_at = message.rfind("\n", 0, MAX_MESSAGE_LENGTH)
        if split_at <= 0:
            split_at = MAX_MESSAGE_LENGTH
        chunks.append(message[:split_at])
        message = message[split_at:].lstrip("\n")
    return chunks


def send_discord_message(
    message: str,
    webhook_url: str = "",
    webhook_url_file: str = "",
    username: str = "",
    dry_run: bool = False,
) -> None:
    """Send a plain text message to a Discord channel via webhook.

    Args:
        message: Text content to send (auto-split if >2000 chars).
        webhook_url: Discord webhook URL. If empty, reads from webhook_url_file.
        webhook_url_file: Path to file containing webhook URL (default: ~/keys/discord_webhook_url.txt).
        username: Override the webhook's default username.
        dry_run: If True, print instead of sending.
    """
    if dry_run:
        log.info("[DRY-RUN] Discord message: %s", message[:200])
        print(f"[DRY-RUN] Discord message ({len(message)} chars): {message[:200]}...")
        return

    url = _resolve_webhook_url(webhook_url, webhook_url_file)
    chunks = _split_message(message)
    for chunk in chunks:
        payload: dict = {"content": chunk}
        if username:
            payload["username"] = username
        _post_with_retry(url, json=payload)
    log.info("Discord message sent (%d chunk(s), %d chars total)", len(chunks), len(message))


def send_discord_embed(
    title: str,
    description: str = "",
    color: int = 0x5865F2,
    fields: list[dict] | None = None,
    footer: str = "",
    thumbnail_url: str = "",
    webhook_url: str = "",
    webhook_url_file: str = "",
    username: str = "",
    dry_run: bool = False,
) -> None:
    """Send a rich embed message to a Discord channel via webhook.

    Args:
        title: Embed title.
        description: Embed description/body text.
        color: Embed sidebar color as int (default: Discord blurple).
        fields: List of {"name": str, "value": str, "inline": bool} dicts.
        footer: Footer text.
        thumbnail_url: URL for thumbnail image.
        webhook_url: Discord webhook URL.
        webhook_url_file: Path to file containing webhook URL.
        username: Override webhook's default username.
        dry_run: If True, print instead of sending.
    """
    embed: dict = {"title": title, "color": color}
    if description:
        embed["description"] = description
    if fields:
        embed["fields"] = fields
    if footer:
        embed["footer"] = {"text": footer}
    if thumbnail_url:
        embed["thumbnail"] = {"url": thumbnail_url}

    if dry_run:
        log.info("[DRY-RUN] Discord embed: %s", json.dumps(embed, indent=2))
        print(f"[DRY-RUN] Discord embed: {title}")
        return

    url = _resolve_webhook_url(webhook_url, webhook_url_file)
    payload: dict = {"embeds": [embed]}
    if username:
        payload["username"] = username
    _post_with_retry(url, json=payload)
    log.info("Discord embed sent: %s", title)


def send_discord_file(
    file_path: str | Path,
    message: str = "",
    webhook_url: str = "",
    webhook_url_file: str = "",
    username: str = "",
    dry_run: bool = False,
) -> None:
    """Upload a file to a Discord channel via webhook.

    Args:
        file_path: Path to the file to upload.
        message: Optional text message to accompany the file.
        webhook_url: Discord webhook URL.
        webhook_url_file: Path to file containing webhook URL.
        username: Override webhook's default username.
        dry_run: If True, print instead of sending.
    """
    fp = Path(file_path).expanduser()
    if not fp.is_file():
        raise FileNotFoundError(f"File not found: {fp}")

    if dry_run:
        log.info("[DRY-RUN] Discord file upload: %s", fp)
        print(f"[DRY-RUN] Discord file upload: {fp.name} ({fp.stat().st_size} bytes)")
        return

    url = _resolve_webhook_url(webhook_url, webhook_url_file)
    payload: dict = {}
    if message:
        payload["content"] = message[:MAX_MESSAGE_LENGTH]
    if username:
        payload["username"] = username

    with open(fp, "rb") as f:
        files = {"file": (fp.name, f)}
        _post_with_retry(url, data=payload, files=files)
    log.info("Discord file sent: %s (%d bytes)", fp.name, fp.stat().st_size)


# ── Bot functions (REST-based, no event loop needed) ──────────────────


def list_bot_guilds(
    bot_token: str = "",
    bot_token_file: str = "",
    dry_run: bool = False,
) -> list[dict]:
    """List Discord servers (guilds) the bot has been invited to.

    Returns:
        List of guild dicts with keys: id, name, icon, owner, permissions, etc.
    """
    if dry_run:
        print("[DRY-RUN] Would list bot guilds")
        return []
    token = _resolve_bot_token(bot_token, bot_token_file)
    guilds = _bot_request("GET", "/users/@me/guilds", token)
    log.info("Found %d guild(s)", len(guilds))
    return guilds


def list_channels(
    guild_id: str,
    bot_token: str = "",
    bot_token_file: str = "",
    dry_run: bool = False,
) -> list[dict]:
    """List text channels in a Discord server (guild) that the bot can see.

    Args:
        guild_id: The Discord server (guild) ID.
        bot_token: Bot token directly.
        bot_token_file: Path to file containing bot token.
        dry_run: If True, print instead of making API call.

    Returns:
        List of channel dicts filtered to text channels only (type 0).
    """
    if dry_run:
        print(f"[DRY-RUN] Would list channels for guild {guild_id}")
        return []
    token = _resolve_bot_token(bot_token, bot_token_file)
    all_channels = _bot_request("GET", f"/guilds/{guild_id}/channels", token)
    text_channels = [ch for ch in all_channels if ch.get("type") == 0]
    log.info("Found %d text channel(s) in guild %s", len(text_channels), guild_id)
    return text_channels


def read_messages(
    channel_id: str,
    limit: int = 10,
    bot_token: str = "",
    bot_token_file: str = "",
    dry_run: bool = False,
) -> list[dict]:
    """Read recent messages from a Discord channel.

    Args:
        channel_id: The Discord channel ID.
        limit: Number of messages to fetch (max 100, default 10).
        bot_token: Bot token directly.
        bot_token_file: Path to file containing bot token.
        dry_run: If True, print instead of making API call.

    Returns:
        List of message dicts (newest first) with keys: id, content, author, timestamp, etc.
    """
    if dry_run:
        print(f"[DRY-RUN] Would read {limit} messages from channel {channel_id}")
        return []
    token = _resolve_bot_token(bot_token, bot_token_file)
    limit = min(max(limit, 1), 100)
    messages = _bot_request(
        "GET", f"/channels/{channel_id}/messages", token,
        params={"limit": limit},
    )
    log.info("Read %d message(s) from channel %s", len(messages), channel_id)
    return messages


def send_bot_message(
    channel_id: str,
    message: str,
    bot_token: str = "",
    bot_token_file: str = "",
    dry_run: bool = False,
) -> dict | None:
    """Send a message to a Discord channel using the bot.

    Args:
        channel_id: The Discord channel ID.
        message: Text content to send (auto-split if >2000 chars).
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
        result = _bot_request(
            "POST", f"/channels/{channel_id}/messages", token,
            json_payload={"content": chunk},
        )
    log.info("Bot message sent to channel %s (%d chunk(s), %d chars total)",
             channel_id, len(chunks), len(message))
    return result


def reply_to_message(
    channel_id: str,
    message_id: str,
    reply: str,
    bot_token: str = "",
    bot_token_file: str = "",
    dry_run: bool = False,
) -> dict | None:
    """Reply to a specific message in a Discord channel.

    Args:
        channel_id: The Discord channel ID containing the message.
        message_id: The ID of the message to reply to.
        reply: Text content of the reply.
        bot_token: Bot token directly.
        bot_token_file: Path to file containing bot token.
        dry_run: If True, print instead of sending.

    Returns:
        API response dict, or None for dry-run.
    """
    if dry_run:
        log.info("[DRY-RUN] Reply to message %s in channel %s: %s",
                 message_id, channel_id, reply[:200])
        print(f"[DRY-RUN] Reply to {message_id}: {reply[:200]}...")
        return None
    token = _resolve_bot_token(bot_token, bot_token_file)
    result = _bot_request(
        "POST", f"/channels/{channel_id}/messages", token,
        json_payload={
            "content": reply[:MAX_MESSAGE_LENGTH],
            "message_reference": {"message_id": message_id},
        },
    )
    log.info("Reply sent to message %s in channel %s", message_id, channel_id)
    return result


# ── Event-driven Bot (optional, requires discord.py) ──────────────────


def run_listener_bot(
    on_message_callback=None,
    bot_token: str = "",
    bot_token_file: str = "",
) -> None:
    """Run a persistent Discord bot that listens for messages.

    This starts a blocking event loop. Intended for scripts, not library calls.

    Args:
        on_message_callback: async function(message) called for each message.
            If None, just prints messages to stdout.
        bot_token: Bot token directly.
        bot_token_file: Path to file containing bot token.
    """
    try:
        import discord
    except ImportError:
        raise ImportError(
            "discord.py is required for run_listener_bot(). "
            "Install it: pip install discord.py"
        )

    token = _resolve_bot_token(bot_token, bot_token_file)

    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        log.info("Bot logged in as %s (id: %s)", client.user, client.user.id)
        print(f"Bot logged in as {client.user} (id: {client.user.id})")

    @client.event
    async def on_message(message):
        if message.author == client.user:
            return
        if on_message_callback is not None:
            await on_message_callback(message)
        else:
            print(f"[{message.channel.name}] {message.author}: {message.content}")

    client.run(token)


# ── Smoke test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Discord utilities -- webhooks and bot operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Dry-run smoke tests (no credentials needed)
  python -m uutils.discord_uu --smoke-test

  # List servers the bot is in
  python -m uutils.discord_uu --list-guilds

  # List text channels in a server
  python -m uutils.discord_uu --list-channels --guild-id 123456789

  # Read last 5 messages from a channel
  python -m uutils.discord_uu --read --channel-id 123456789 --limit 5

  # Send a message to a channel
  python -m uutils.discord_uu --send "Hello!" --channel-id 123456789

  # Reply to a specific message
  python -m uutils.discord_uu --reply "Thanks!" --channel-id 123 --message-id 456
""",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Run dry-run smoke tests")
    parser.add_argument("--list-guilds", action="store_true", help="List servers the bot is in")
    parser.add_argument("--list-channels", action="store_true", help="List text channels in a guild")
    parser.add_argument("--read", action="store_true", help="Read recent messages from a channel")
    parser.add_argument("--send", type=str, default="", help="Send a message to a channel")
    parser.add_argument("--reply", type=str, default="", help="Reply to a message")
    parser.add_argument("--guild-id", type=str, default="", help="Guild (server) ID")
    parser.add_argument("--channel-id", type=str, default="", help="Channel ID")
    parser.add_argument("--message-id", type=str, default="", help="Message ID (for --reply)")
    parser.add_argument("--limit", type=int, default=10, help="Number of messages to read (default: 10)")
    parser.add_argument("--dry-run", action="store_true", help="Print instead of sending")
    parser.add_argument("--bot-token-file", type=str, default="", help="Path to bot token file")

    args = parser.parse_args()

    if args.smoke_test:
        print("=== Discord dry-run smoke tests ===")
        send_discord_message("Hello from uutils!", dry_run=True)
        send_discord_embed("Test Embed", "This is a test embed", dry_run=True)
        long_msg = "x" * 4500
        chunks = _split_message(long_msg)
        assert len(chunks) == 3, f"Expected 3 chunks, got {len(chunks)}"
        assert all(len(c) <= MAX_MESSAGE_LENGTH for c in chunks)
        print(f"Message splitting: 4500 chars -> {len(chunks)} chunks OK")
        # Bot function dry-run tests
        list_bot_guilds(dry_run=True)
        list_channels("000000000", dry_run=True)
        read_messages("000000000", dry_run=True)
        send_bot_message("000000000", "Test bot message", dry_run=True)
        reply_to_message("000000000", "111111111", "Test reply", dry_run=True)
        print("All dry-run smoke tests passed!")
    elif args.list_guilds:
        guilds = list_bot_guilds(bot_token_file=args.bot_token_file, dry_run=args.dry_run)
        for g in guilds:
            print(f"  {g['id']}  {g['name']}")
    elif args.list_channels:
        if not args.guild_id:
            guilds = list_bot_guilds(bot_token_file=args.bot_token_file)
            if not guilds:
                print("Bot is not in any guilds. Invite it to a server first.")
                sys.exit(1)
            args.guild_id = guilds[0]["id"]
            print(f"Using first guild: {guilds[0]['name']} ({args.guild_id})")
        channels = list_channels(args.guild_id, bot_token_file=args.bot_token_file, dry_run=args.dry_run)
        for ch in channels:
            print(f"  #{ch['name']}  (id: {ch['id']})")
    elif args.read:
        if not args.channel_id:
            parser.error("--read requires --channel-id")
        messages = read_messages(args.channel_id, limit=args.limit,
                                 bot_token_file=args.bot_token_file, dry_run=args.dry_run)
        for msg in reversed(messages):  # oldest first
            author = msg.get("author", {}).get("username", "?")
            content = msg.get("content", "")
            ts = msg.get("timestamp", "")[:19]
            print(f"  [{ts}] {author}: {content[:200]}")
            print(f"    (message_id: {msg['id']})")
    elif args.send:
        if not args.channel_id:
            parser.error("--send requires --channel-id")
        send_bot_message(args.channel_id, args.send,
                         bot_token_file=args.bot_token_file, dry_run=args.dry_run)
        if not args.dry_run:
            print("Message sent.")
    elif args.reply:
        if not args.channel_id or not args.message_id:
            parser.error("--reply requires --channel-id and --message-id")
        reply_to_message(args.channel_id, args.message_id, args.reply,
                         bot_token_file=args.bot_token_file, dry_run=args.dry_run)
        if not args.dry_run:
            print("Reply sent.")
    else:
        parser.print_help()
