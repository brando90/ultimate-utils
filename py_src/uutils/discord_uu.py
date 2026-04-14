"""Discord messaging via webhooks — send text, embeds, and files from Claude Code.

Quick usage:
    from uutils.discord_uu import send_discord_message
    send_discord_message("Hello from uutils!")

    # With explicit webhook URL
    send_discord_message("Hello!", webhook_url="https://discord.com/api/webhooks/...")

    # Rich embed
    from uutils.discord_uu import send_discord_embed
    send_discord_embed("Deploy Complete", "v0.10.2 is live", color=0x00FF00)

    # File upload
    from uutils.discord_uu import send_discord_file
    send_discord_file("~/results/plot.png", message="Latest results")

Setup:
    1. In Discord: Server Settings → Integrations → Webhooks → New Webhook
    2. Copy the webhook URL
    3. Save it:
       echo 'https://discord.com/api/webhooks/...' > ~/keys/discord_webhook_url.txt
       chmod 600 ~/keys/discord_webhook_url.txt

Refs:
    - Discord webhook docs: https://discord.com/developers/docs/resources/webhook
    - Embed structure: https://discord.com/developers/docs/resources/message#embed-object
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import requests

log = logging.getLogger(__name__)

DEFAULT_WEBHOOK_FILE = "~/keys/discord_webhook_url.txt"
MAX_MESSAGE_LENGTH = 2000
RATE_LIMIT_RETRY_ATTEMPTS = 3


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


def _post_with_retry(url: str, **kwargs) -> requests.Response:
    """POST to Discord with rate-limit retry."""
    for attempt in range(RATE_LIMIT_RETRY_ATTEMPTS):
        resp = requests.post(url, timeout=30, **kwargs)
        if resp.status_code == 429:
            retry_after = resp.json().get("retry_after", 1.0)
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
        if split_at == -1:
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


# ── Smoke test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Dry-run tests — no webhook needed
    print("=== Discord dry-run smoke tests ===")
    send_discord_message("Hello from uutils!", dry_run=True)
    send_discord_embed("Test Embed", "This is a test embed", dry_run=True)

    # Test message splitting
    long_msg = "x" * 4500
    chunks = _split_message(long_msg)
    assert len(chunks) == 3, f"Expected 3 chunks, got {len(chunks)}"
    assert all(len(c) <= MAX_MESSAGE_LENGTH for c in chunks)
    print(f"Message splitting: 4500 chars → {len(chunks)} chunks ✓")

    print("All dry-run smoke tests passed!")
