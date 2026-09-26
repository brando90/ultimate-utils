"""Slack automation — messaging, file upload, channel monitoring, and notifications.

Quick usage:
    from uutils.slack_uu import SlackClient, notify

    # Dry-run mode (default, makes no network requests, requires no tokens)
    client = SlackClient.from_token()
    client.send_message(channel="C01234567", text="Hello from uutils!")

    # Live sending (requires token file or SLACK_BOT_TOKEN environment variable)
    client = SlackClient.from_token(dry_run=False)
    client.send_message(channel="C01234567", text="Live notification from pipeline!")

    # File upload via the modern external-upload flow
    client.upload_file(channel="C01234567", file_path="loss_curve.png", title="Training Loss")

    # Channel monitoring
    channels = client.list_channels()
    history = client.get_channel_history(channel="C01234567", limit=50)
    unread = client.get_unread_messages(channel="C01234567", since_ts="1700000000.000000")

    # Convenience helper for experiment completions
    notify(text="Experiment 01 finished: Val Acc = 94.2%", channel="C01234567")

CLI usage:
    # Dry-run (default, safe, sends nothing):
    python -m uutils.slack_uu send --channel C01234567 --text "Experiment finished"

    # Actual live send (requires explicit --send flag):
    python -m uutils.slack_uu send --channel C01234567 --text "Experiment finished" --send

One-time setup:
    1. Create a Slack app:
       Go to https://api.slack.com/apps and click "Create New App" -> "From scratch".
       Name your app (e.g. "UU-Notifier") and pick your workspace.

    2. Configure Bot Token Scopes:
       In your app settings, navigate to "OAuth & Permissions" -> "Scopes" -> "Bot Token Scopes".
       Add the following scopes:
         - chat:write          (Send messages as @your_bot)
         - channels:history    (View messages and other content in public channels)
         - channels:read       (View basic information about public channels)
         - files:write         (Upload, edit, and delete files)
       If you need private channels, also add:
         - groups:history      (View messages in private channels)
         - groups:read         (View basic information about private channels)

    3. Install App to Workspace:
       At the top of "OAuth & Permissions", click "Install to Workspace" and authorize it.

    4. Save your Bot Token securely:
       Copy the "Bot User OAuth Token" (starts with xoxb-).
       Save it to ~/keys/slack_bot_token.txt:
           mkdir -p ~/keys
           echo 'xoxb-YOUR-TOKEN-HERE' > ~/keys/slack_bot_token.txt
           chmod 600 ~/keys/slack_bot_token.txt
       Alternatively, set the SLACK_BOT_TOKEN environment variable:
           export SLACK_BOT_TOKEN='xoxb-YOUR-TOKEN-HERE'

    5. Invite the bot to target channels:
       In each Slack channel you want the bot to post to or read from, run:
           /invite @your_bot_name

Refs:
    - Slack Web API: https://api.slack.com/web
    - chat.postMessage: https://api.slack.com/methods/chat.postMessage
    - files.getUploadURLExternal: https://api.slack.com/methods/files.getUploadURLExternal
    - files.completeUploadExternal: https://api.slack.com/methods/files.completeUploadExternal
    - conversations.list: https://api.slack.com/methods/conversations.list
    - conversations.history: https://api.slack.com/methods/conversations.history
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger(__name__)

DEFAULT_TOKEN_FILE = "~/keys/slack_bot_token.txt"
SLACK_API_BASE = "https://slack.com/api"


class SlackClient:
    """Client for the official Slack Web API using plain requests.

    Dry-run mode is enabled by default. In dry-run mode, no network requests are made,
    no credential files are read, and descriptions of the attempted actions are returned.
    """

    def __init__(self, token: str | None = None, dry_run: bool = True) -> None:
        self.token = token
        self.dry_run = dry_run

    @classmethod
    def from_token(
        cls,
        token_file: str = DEFAULT_TOKEN_FILE,
        dry_run: bool = True,
    ) -> SlackClient:
        """Instantiate a SlackClient.

        In dry-run mode (default), no file is read and no environment variables are required.
        When dry_run=False, reads SLACK_BOT_TOKEN env var, or token_file.
        """
        if dry_run:
            return cls(token=None, dry_run=True)

        token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
        if not token and token_file:
            path = Path(token_file).expanduser()
            if not path.is_file():
                raise FileNotFoundError(
                    f"Slack bot token file not found at {path} and SLACK_BOT_TOKEN env var not set.\n"
                    f"See module docstring for setup instructions."
                )
            token = path.read_text().strip()

        if not token:
            raise ValueError(
                "Slack bot token is empty. Set SLACK_BOT_TOKEN env var or place token in token_file."
            )

        return cls(token=token, dry_run=False)

    def _call(self, method: str, **params: Any) -> dict:
        """Call a Slack Web API method.

        In dry-run mode, returns a dictionary describing the call and prints a one-line summary.
        Outside dry-run mode, sends a form-encoded POST request with Bearer authorization and raises a
        RuntimeError if Slack returns {"ok": false, ...}.
        """
        clean_params = {k: v for k, v in params.items() if v is not None}

        if self.dry_run:
            desc = f"[DRY-RUN] Slack API call: {method} with params: {clean_params}"
            print(desc)
            log.info(desc)
            return {"ok": True, "dry_run": True, "method": method, "params": clean_params}

        if not self.token:
            raise ValueError("Slack token is required when dry_run=False")

        url = f"{SLACK_API_BASE}/{method}"
        headers = {
            "Authorization": f"Bearer {self.token}",
        }
        # Slack methods expect form-encoded arguments; non-scalar values (e.g. list, dict)
        # such as `files` in files.completeUploadExternal must be JSON-serialized strings.
        form_data = {}
        for k, v in clean_params.items():
            if isinstance(v, (dict, list)):
                form_data[k] = json.dumps(v)
            else:
                form_data[k] = v

        resp = requests.post(url, headers=headers, data=form_data, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if not data.get("ok"):
            err = data.get("error", "unknown_error")
            raise RuntimeError(f"Slack API error calling '{method}': {err} (response: {data})")

        return data

    def send_message(
        self,
        channel: str,
        text: str,
        thread_ts: str | None = None,
    ) -> dict:
        """Send a message to a Slack channel or thread (chat.postMessage).

        Args:
            channel: Slack channel ID or name (e.g. 'C01234567').
            text: Message body text.
            thread_ts: Timestamp of parent message to reply in a thread (optional).

        Returns:
            Slack API response dict (or dry-run description dict).
        """
        params: dict[str, Any] = {"channel": channel, "text": text}
        if thread_ts is not None:
            params["thread_ts"] = thread_ts
        return self._call("chat.postMessage", **params)

    def upload_file(
        self,
        channel: str,
        file_path: str | Path,
        title: str = "",
    ) -> dict:
        """Upload a file using the modern external upload flow.

        Flow: files.getUploadURLExternal -> direct upload -> files.completeUploadExternal.
        Note: The legacy files.upload endpoint is deprecated by Slack.

        Args:
            channel: Slack channel ID where the file will be posted.
            file_path: Path to the local file to upload.
            title: Optional title for the file.

        Returns:
            Slack API response dict from files.completeUploadExternal (or dry-run dict).
        """
        p = Path(file_path).expanduser()
        filename = p.name

        if self.dry_run:
            file_size = p.stat().st_size if p.is_file() else 0
            return self._call(
                "files.getUploadURLExternal",
                filename=filename,
                length=file_size,
                channel=channel,
                title=title or filename,
            )

        if not p.is_file():
            raise FileNotFoundError(f"File to upload not found: {p}")

        file_size = p.stat().st_size

        # Step 1: Request an external upload URL
        url_resp = self._call("files.getUploadURLExternal", filename=filename, length=file_size)
        upload_url = url_resp["upload_url"]
        file_id = url_resp["file_id"]

        # Step 2: Upload file bytes to the presigned upload URL
        with open(p, "rb") as f:
            upload_resp = requests.post(
                upload_url,
                data=f,
                headers={"Content-Type": "application/octet-stream"},
                timeout=60,
            )
            upload_resp.raise_for_status()

        # Step 3: Finalize the upload and share to the channel
        file_spec: dict[str, Any] = {"id": file_id}
        if title:
            file_spec["title"] = title
        else:
            file_spec["title"] = filename

        complete_resp = self._call(
            "files.completeUploadExternal",
            files=[file_spec],
            channel_id=channel,
        )
        return complete_resp

    def list_channels(
        self,
        types: str = "public_channel,private_channel",
        limit: int = 100,
    ) -> list[dict]:
        """List channels in the workspace with automatic cursor pagination (conversations.list).

        Args:
            types: Channel types to include (default: 'public_channel,private_channel').
            limit: Number of items to retrieve per page (default: 100).

        Returns:
            List of channel dicts across all pages (or empty list in dry-run).
        """
        if self.dry_run:
            self._call("conversations.list", types=types, limit=limit)
            return []

        channels: list[dict] = []
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {"types": types, "limit": limit}
            if cursor:
                params["cursor"] = cursor
            resp = self._call("conversations.list", **params)
            channels.extend(resp.get("channels", []))
            cursor = resp.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break
        return channels

    def get_channel_history(
        self,
        channel: str,
        limit: int = 50,
        oldest: str | float | None = None,
    ) -> dict:
        """Fetch message history from a channel (conversations.history).

        Args:
            channel: Channel ID.
            limit: Number of messages to fetch (default: 50).
            oldest: Only messages after this timestamp (optional).

        Returns:
            Slack API response dict (containing 'messages' list outside dry-run).
        """
        params: dict[str, Any] = {"channel": channel, "limit": limit}
        if oldest is not None:
            params["oldest"] = str(oldest)
        return self._call("conversations.history", **params)

    def get_unread_messages(
        self,
        channel: str,
        since_ts: str | float,
        limit: int = 50,
    ) -> dict:
        """Fetch messages in channel newer than since_ts (conversations.history with oldest).

        Args:
            channel: Channel ID.
            since_ts: Timestamp to fetch messages newer than.
            limit: Maximum number of messages to return (default: 50).

        Returns:
            Slack API response dict with messages newer than since_ts.
        """
        return self.get_channel_history(channel=channel, limit=limit, oldest=since_ts)


def notify(
    text: str,
    channel: str,
    dry_run: bool = True,
    token_file: str = DEFAULT_TOKEN_FILE,
) -> dict:
    """Convenience function to send a notification message to a Slack channel.

    Args:
        text: Message text to post (e.g. 'experiment finished').
        channel: Slack channel ID or name.
        dry_run: If True (default), does not send network requests or read credentials.
        token_file: Path to token file if dry_run=False.

    Returns:
        API response dict (or dry-run description dict).
    """
    client = SlackClient.from_token(token_file=token_file, dry_run=dry_run)
    return client.send_message(channel=channel, text=text)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Slack automation CLI (uutils) — dry-run by default."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    send_parser = subparsers.add_parser("send", help="Send a message to a Slack channel")
    send_parser.add_argument("--channel", "-c", required=True, help="Slack channel ID or name")
    send_parser.add_argument("--text", "-t", required=True, help="Message text")
    send_parser.add_argument(
        "--send",
        action="store_true",
        default=False,
        help="Actually send the message (defaults to dry-run unless --send is given)",
    )
    send_parser.add_argument(
        "--token-file",
        default=DEFAULT_TOKEN_FILE,
        help=f"Path to Slack bot token file (default: {DEFAULT_TOKEN_FILE})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "send":
        dry_run = not args.send
        client = SlackClient.from_token(token_file=args.token_file, dry_run=dry_run)
        result = client.send_message(channel=args.channel, text=args.text)
        if dry_run:
            print(f"[DRY-RUN] Message to {args.channel}: {args.text}")
        else:
            print(f"Message sent to {args.channel} (ts: {result.get('ts', '?')})")
        return 0

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
