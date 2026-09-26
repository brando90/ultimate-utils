"""Zulip messaging and automation — send stream messages, DMs, fetch messages, and monitor unreads.

Quick usage:
    from uutils.zulip_uu import ZulipClient, notify

    # Dry-run by default (no network calls, no credentials read):
    notify("Build succeeded!", stream="general", topic="ci")

    # Explicit client usage:
    client = ZulipClient(site="https://example.zulipchat.com", email="bot@example.com", api_key="secret", dry_run=False)
    client.send_message(stream="general", topic="releases", content="v1.0.0 released!")

    # From zuliprc file (only reads credentials when dry_run=False):
    client = ZulipClient.from_zuliprc("~/keys/zuliprc", dry_run=False)
    client.send_message(stream="general", topic="releases", content="v1.0.0 released!")
    client.send_dm(user_email="user@example.com", content="Hello via DM!")
    messages = client.get_messages(stream="general", topic="releases", limit=10)
    unreads = client.get_unread_count()

CLI usage:
    # Dry-run (prints what it would do, makes no network call):
    python -m uutils.zulip_uu send --stream general --topic releases --content "v1.0.0 released!"

    # Real send (requires ~/keys/zuliprc):
    python -m uutils.zulip_uu send --stream general --topic releases --content "v1.0.0 released!" --send

Setup:
    1. Log in to your Zulip realm (e.g., https://your-org.zulipchat.com).
    2. Go to Settings (gear icon) -> Personal settings -> Bots.
    3. Click "Add a new bot":
       - Bot type: Generic bot
       - Bot name: e.g. "Notifier Bot"
       - Bot email: e.g. "notifier-bot@your-org.zulipchat.com"
    4. Download the bot's `zuliprc` file.
    5. Save it to `~/keys/zuliprc`:
       mkdir -p ~/keys
       mv ~/Downloads/zuliprc ~/keys/zuliprc
       chmod 600 ~/keys/zuliprc
    6. Verify the contents of `~/keys/zuliprc`:
       [api]
       email=notifier-bot@your-org.zulipchat.com
       key=YOUR_API_KEY
       site=https://your-org.zulipchat.com

Refs:
    - Zulip REST API: https://zulip.com/api/rest
    - Zulip Messages API: https://zulip.com/api/send-message
    - Zuliprc specification: https://zulip.com/api/configuring-python-bindings
"""
from __future__ import annotations

import argparse
import configparser
import json
import logging
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger(__name__)

DEFAULT_ZULIPRC_FILE = "~/keys/zuliprc"


class ZulipClient:
    """Zulip REST API client with dry-run support."""

    def __init__(
        self,
        site: str = "",
        email: str = "",
        api_key: str = "",
        dry_run: bool = True,
    ) -> None:
        site = site.strip().rstrip("/")
        if site and not site.startswith(("http://", "https://")):
            site = f"https://{site}"
        self.site = site
        self.email = email.strip()
        self.api_key = api_key.strip()
        self.dry_run = dry_run

    @classmethod
    def from_zuliprc(
        cls,
        zuliprc_file: str = DEFAULT_ZULIPRC_FILE,
        dry_run: bool = True,
    ) -> ZulipClient:
        """Initialize ZulipClient from a zuliprc config file.

        When dry_run=True, no credential file is read or parsed.
        """
        if dry_run:
            return cls(site="", email="", api_key="", dry_run=True)

        fpath = Path(zuliprc_file).expanduser()
        if not fpath.is_file():
            raise FileNotFoundError(
                f"Zuliprc file not found at {fpath}\n"
                f"Download your bot's zuliprc from Zulip Settings -> Personal -> Bots and save it to {zuliprc_file}."
            )

        config = configparser.ConfigParser()
        config.read(fpath)
        if "api" not in config:
            raise ValueError(f"Missing [api] section in zuliprc file: {fpath}")

        api_section = config["api"]
        email = api_section.get("email", "")
        api_key = api_section.get("key", "")
        site = api_section.get("site", "")
        return cls(site=site, email=email, api_key=api_key, dry_run=False)

    def _request(self, method: str, path: str, **kwargs: Any) -> dict:
        """Send an authenticated request to Zulip API or simulate in dry-run mode."""
        clean_path = path.strip().lstrip("/")
        if self.dry_run:
            desc_parts = [f"[DRY-RUN] Zulip {method.upper()} /{clean_path}"]
            if "data" in kwargs:
                desc_parts.append(f"data={kwargs['data']}")
            if "params" in kwargs:
                desc_parts.append(f"params={kwargs['params']}")
            if "files" in kwargs:
                desc_parts.append(f"files={list(kwargs['files'].keys())}")
            desc = " ".join(desc_parts)
            print(desc)
            log.info(desc)

            dry_result: dict[str, Any] = {"result": "success", "dry_run": True, "msg": ""}
            if clean_path.endswith("messages") and method.upper() == "GET":
                dry_result["messages"] = []
            elif clean_path.endswith("streams") and method.upper() == "GET":
                dry_result["streams"] = []
            elif clean_path.endswith("user_uploads") and method.upper() == "POST":
                filename = "mock_file"
                if "files" in kwargs and isinstance(kwargs["files"], dict) and "filename" in kwargs["files"]:
                    val = kwargs["files"]["filename"]
                    if isinstance(val, tuple):
                        filename = val[0]
                    elif isinstance(val, str):
                        filename = val
                dry_result["uri"] = f"/user_uploads/dry_run/{filename}"
                dry_result["url"] = f"/user_uploads/dry_run/{filename}"
            elif clean_path.endswith("messages") and method.upper() == "POST":
                dry_result["id"] = 0
            return dry_result

        url = f"{self.site}/{clean_path}" if clean_path.startswith("api/v1/") else f"{self.site}/api/v1/{clean_path}"
        auth = (self.email, self.api_key)
        timeout = kwargs.pop("timeout", 30)

        resp = requests.request(method, url, auth=auth, timeout=timeout, **kwargs)
        try:
            data = resp.json()
        except Exception:
            resp.raise_for_status()
            raise RuntimeError(f"Zulip API returned non-JSON response ({resp.status_code}): {resp.text}")

        if data.get("result") != "success":
            msg = data.get("msg", resp.text)
            code = data.get("code", resp.status_code)
            raise RuntimeError(f"Zulip API error ({code}): {msg}")

        resp.raise_for_status()
        return data

    def send_message(self, stream: str, topic: str, content: str) -> dict:
        """Send a message to a Zulip stream.

        Args:
            stream: Target stream name.
            topic: Topic within the stream.
            content: Message body (Markdown supported).

        Returns:
            API response dictionary.
        """
        data = {
            "type": "stream",
            "to": stream,
            "topic": topic,
            "content": content,
        }
        return self._request("POST", "messages", data=data)

    def send_dm(self, user_email: str | list[str], content: str) -> dict:
        """Send a direct (private) message to one or more user emails.

        Args:
            user_email: Recipient email address or list of email addresses.
            content: Message body (Markdown supported).

        Returns:
            API response dictionary.
        """
        if isinstance(user_email, (list, tuple)):
            to_val = json.dumps(list(user_email))
        elif isinstance(user_email, str) and user_email.strip().startswith("["):
            to_val = user_email.strip()
        else:
            to_val = json.dumps([user_email.strip()])

        data = {
            "type": "direct",
            "to": to_val,
            "content": content,
        }
        return self._request("POST", "messages", data=data)

    def get_messages(
        self,
        stream: str | None = None,
        topic: str | None = None,
        limit: int = 50,
    ) -> dict:
        """Fetch messages from Zulip.

        Args:
            stream: Optional stream name to narrow by.
            topic: Optional topic name to narrow by.
            limit: Maximum number of messages to fetch (default: 50).

        Returns:
            API response dictionary containing 'messages' list.
        """
        params: dict[str, Any] = {
            "anchor": "newest",
            "num_before": limit,
            "num_after": 0,
        }
        narrow: list[dict[str, str]] = []
        if stream is not None:
            narrow.append({"operator": "stream", "operand": stream})
        if topic is not None:
            narrow.append({"operator": "topic", "operand": topic})
        if narrow:
            params["narrow"] = json.dumps(narrow)

        return self._request("GET", "messages", params=params)

    def get_streams(self) -> dict:
        """Retrieve all streams visible to the user/bot.

        Returns:
            API response dictionary containing 'streams' list.
        """
        return self._request("GET", "streams")

    def upload_file(self, file_path: str | Path) -> str:
        """Upload a file to Zulip and return its URL/URI.

        Args:
            file_path: Path to the local file to upload.

        Returns:
            Uploaded file URI or URL string.
        """
        p = Path(file_path).expanduser()
        if self.dry_run:
            res = self._request("POST", "user_uploads", files={"filename": p.name})
            return str(res.get("uri") or res.get("url", ""))

        if not p.is_file():
            raise FileNotFoundError(f"File to upload not found: {p}")

        with open(p, "rb") as fp:
            res = self._request("POST", "user_uploads", files={"filename": (p.name, fp)})
        return str(res.get("uri") or res.get("url", ""))

    def get_unread_count(self) -> dict[str, dict[str, int]]:
        """Summarise unread messages per stream and topic.

        Fetches unread messages using the 'is:unread' narrow and aggregates counts
        by stream (or direct recipient) and topic.

        Returns:
            Dictionary mapping stream/recipient to a dictionary of topic counts.
            Example: {"general": {"announcements": 3, "standup": 1}}
        """
        params = {
            "anchor": "newest",
            "num_before": 1000,
            "num_after": 0,
            "narrow": json.dumps([{"operator": "is", "operand": "unread"}]),
        }
        res = self._request("GET", "messages", params=params)
        unread_counts: dict[str, dict[str, int]] = {}
        for msg in res.get("messages", []):
            display = msg.get("display_recipient")
            if isinstance(display, list):
                stream_name = "direct"
            elif display is not None:
                stream_name = str(display)
            else:
                stream_name = "unknown"

            topic_name = str(msg.get("subject", ""))
            if stream_name not in unread_counts:
                unread_counts[stream_name] = {}
            unread_counts[stream_name][topic_name] = (
                unread_counts[stream_name].get(topic_name, 0) + 1
            )
        return unread_counts


def notify(
    content: str,
    stream: str = "general",
    topic: str = "notifications",
    dry_run: bool = True,
    zuliprc_file: str = DEFAULT_ZULIPRC_FILE,
) -> dict:
    """Convenience function to send a notification message to Zulip.

    Defaults to dry-run mode (makes no network calls, reads no credentials).

    Args:
        content: Notification message content.
        stream: Target stream name (default: "general").
        topic: Topic within the stream (default: "notifications").
        dry_run: If True, prints what would be sent without network call (default: True).
        zuliprc_file: Path to zuliprc configuration file (used only if dry_run=False).

    Returns:
        API response dictionary or dry-run simulation dictionary.
    """
    client = ZulipClient.from_zuliprc(zuliprc_file=zuliprc_file, dry_run=dry_run)
    return client.send_message(stream=stream, topic=topic, content=content)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for Zulip automation."""
    parser = argparse.ArgumentParser(description="Zulip automation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    send_parser = subparsers.add_parser("send", help="Send a message to a stream")
    send_parser.add_argument("--stream", "-s", required=True, help="Target stream name")
    send_parser.add_argument("--topic", "-t", required=True, help="Topic in stream")
    send_parser.add_argument("--content", "-c", required=True, help="Message content")
    send_parser.add_argument(
        "--send",
        action="store_true",
        default=False,
        help="Actually send the message (defaults to dry-run)",
    )
    send_parser.add_argument(
        "--zuliprc",
        default=DEFAULT_ZULIPRC_FILE,
        help="Path to zuliprc file (default: ~/keys/zuliprc)",
    )

    args = parser.parse_args(argv)

    if args.command == "send":
        dry_run = not args.send
        client = ZulipClient.from_zuliprc(zuliprc_file=args.zuliprc, dry_run=dry_run)
        result = client.send_message(stream=args.stream, topic=args.topic, content=args.content)
        if dry_run:
            print("[DRY-RUN] Message would be sent. Pass --send to transmit.")
        else:
            print(f"Message sent successfully: id={result.get('id', '?')}")
        return 0
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv[1:]))
