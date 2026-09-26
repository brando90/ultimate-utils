"""WhatsApp messaging — send/receive messages via Meta Business Cloud API or Twilio,
with Claude-powered auto-replies via the clauded CLI.

Quick usage (send message):
    from uutils.whatsapp_uu import send_whatsapp_message
    send_whatsapp_message("+15550000001", "Hello from uutils!", dry_run=True)

Quick usage (Claude auto-reply bot):
    from uutils.whatsapp_uu import WhatsAppClaudeBot, run_whatsapp_bot

    # In dry-run mode (default, safe, offline):
    run_whatsapp_bot(dry_run=True)

    # To run live (requires allowed_phones and WhatsApp config):
    run_whatsapp_bot(
        dry_run=False,
        allowed_phones={"+15550000001"},
        verify_token="YOUR_WEBHOOK_VERIFY_TOKEN",
        app_secret="YOUR_META_APP_SECRET",
    )

Quick usage (programmatic bot without server):
    bot = WhatsAppClaudeBot(dry_run=True, allowed_phones={"+15550000001"})
    reply = bot.generate_reply("+15550000001", "Hey, what's up?")
    print(reply)

Setup (Meta WhatsApp Business Cloud API — recommended):
    1. Create a Meta Business account: https://business.facebook.com/
    2. Set up WhatsApp Business: https://developers.facebook.com/docs/whatsapp/cloud-api/get-started
    3. Get your access token, phone number ID, and app secret from the Meta developer console
    4. Save config:
       cat > ~/keys/whatsapp_api_config.json << 'JSON'
       {
           "provider": "meta",
           "access_token": "YOUR_ACCESS_TOKEN",
           "phone_number_id": "YOUR_PHONE_NUMBER_ID",
           "api_version": "v21.0",
           "verify_token": "YOUR_WEBHOOK_VERIFY_TOKEN",
           "app_secret": "YOUR_META_APP_SECRET"
       }
       JSON
       chmod 600 ~/keys/whatsapp_api_config.json

    5. Claude replies via `clauded` CLI:
       Auto-replies use the `clauded -p` CLI tool (Claude Code subscription).
       No Anthropic SDK or API key is required or used. The CLI must be installed
       and authenticated in your environment.
       Security: The bot invokes `clauded -p --tools "" --strict-mcp-config` with the
       prompt passed on stdin, stripping all tools and external MCP connectors to prevent
       prompt injection attacks from untrusted incoming WhatsApp messages.

    6. Safety defaults:
       - Dry-run is True by default for all bot operations (`WhatsAppClaudeBot`, `run_whatsapp_bot`,
         `mark_as_read`). No network requests, subprocess calls, or credential reads occur in dry-run mode.
         (Note: low-level message sending functions `send_whatsapp_message` and `send_whatsapp_template`
         default to `dry_run=False` for backwards compatibility with existing callers; pass `dry_run=True`
         when testing offline.)
       - Phone allowlist: WhatsAppClaudeBot ignores all messages from phone numbers not
         explicitly listed in `allowed_phones`. By default, allowed_phones is None/empty,
         so the bot ignores all incoming messages until configured.
       - Rate limiting: A per-phone rate limit applies (default: 10 replies/hour).
       - Live execution guard: `run_whatsapp_bot()` refuses to start with `dry_run=False`
         unless `allowed_phones`, `verify_token`, and `app_secret` are all configured.
       - Server binding: Default host is "127.0.0.1". For security, keep it bound to localhost
         and expose it via a reverse proxy (e.g. nginx, caddy) or ngrok for HTTPS.

    7. Expose your webhook (for receiving messages):
       # Option A: ngrok (for local development)
       ngrok http 5000
       # Option B: deploy behind a reverse proxy with a public HTTPS URL (e.g., nginx/caddy)

    8. Configure the webhook in Meta Developer Console:
       - Webhook URL: https://YOUR_DOMAIN/webhook
       - Verify token: same as verify_token in your config
       - Subscribe to: messages

Setup (Twilio — alternative for sending):
    1. Create Twilio account: https://www.twilio.com/
    2. Enable WhatsApp sandbox: https://www.twilio.com/docs/whatsapp/sandbox
    3. Save config:
       cat > ~/keys/whatsapp_api_config.json << 'JSON'
       {
           "provider": "twilio",
           "account_sid": "YOUR_ACCOUNT_SID",
           "auth_token": "YOUR_AUTH_TOKEN",
           "from_number": "whatsapp:+15550000001"
       }
       JSON
       chmod 600 ~/keys/whatsapp_api_config.json

Refs:
    - Meta Cloud API: https://developers.facebook.com/docs/whatsapp/cloud-api
    - Meta Webhooks: https://developers.facebook.com/docs/whatsapp/cloud-api/webhooks
    - Twilio WhatsApp: https://www.twilio.com/docs/whatsapp/api
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import hmac
import json
import logging
from pathlib import Path
import subprocess
import threading
import time
from typing import Any, Callable

import requests

log = logging.getLogger(__name__)

DEFAULT_CONFIG_FILE = "~/keys/whatsapp_api_config.json"
_PHONE_SEPARATORS = {" ", "-", "(", ")", "."}

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant replying to WhatsApp messages on behalf of the user. "
    "Keep responses concise and natural — as if texting a friend. "
    "Use short paragraphs. Avoid markdown formatting (no ** or ## etc.) since this is WhatsApp. "
    "If the message is casual, be casual. If it's a question, give a clear answer. "
    "If you're unsure what someone means, ask a brief clarifying question."
)

MAX_CONVERSATION_HISTORY = 20


def _load_config(config_file: str = "") -> dict:
    """Load WhatsApp API config from a JSON file."""
    fpath = Path(config_file or DEFAULT_CONFIG_FILE).expanduser()
    if not fpath.is_file():
        raise FileNotFoundError(
            f"WhatsApp config not found at {fpath}\n"
            f"Create it with your API credentials — see module docstring for setup instructions."
        )
    config = json.loads(fpath.read_text())
    provider = config.get("provider", "")
    if provider not in ("meta", "twilio"):
        raise ValueError(f"Unknown WhatsApp provider '{provider}' — must be 'meta' or 'twilio'")
    return config


def _normalize_phone(phone: str) -> str:
    """Ensure phone number has country code prefix (digits only, leading +)."""
    phone = phone.strip()
    if phone.lower().startswith("whatsapp:"):
        phone = phone.split(":", 1)[1].strip()
    if not phone:
        raise ValueError("Phone number is empty — provide a number with country code (e.g., '+15550000000')")

    digits: list[str] = []
    seen_plus = False
    for idx, char in enumerate(phone):
        if char.isdigit():
            digits.append(char)
            continue
        if char == "+":
            if idx != 0 or seen_plus:
                raise ValueError(f"Invalid phone number: {phone!r}")
            seen_plus = True
            continue
        if char in _PHONE_SEPARATORS:
            continue
        raise ValueError(f"Invalid phone number: {phone!r}")

    if not digits:
        raise ValueError("Phone number is empty — provide a number with country code (e.g., '+15550000000')")
    return "+" + "".join(digits)


# ── Meta Business Cloud API ──────────────────────────────────────────

def _send_meta_request(config: dict, payload: dict) -> dict:
    """Send a request to the Meta WhatsApp Business Cloud API."""
    api_version = config.get("api_version", "v21.0")
    phone_number_id = config["phone_number_id"]
    access_token = config["access_token"]
    url = f"https://graph.facebook.com/{api_version}/{phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _send_meta_text(config: dict, to: str, message: str) -> dict:
    """Send a text message via Meta WhatsApp Business Cloud API."""
    payload = {
        "messaging_product": "whatsapp",
        "to": to.lstrip("+"),
        "type": "text",
        "text": {"body": message},
    }
    result = _send_meta_request(config, payload)
    log.info("Meta WhatsApp message sent to %s: %s", to, result.get("messages", [{}])[0].get("id", "?"))
    return result


def _send_meta_template(config: dict, to: str, template_name: str, language: str, components: list | None) -> dict:
    """Send a template message via Meta WhatsApp Business Cloud API."""
    template: dict = {
        "name": template_name,
        "language": {"code": language},
    }
    if components:
        template["components"] = components

    payload = {
        "messaging_product": "whatsapp",
        "to": to.lstrip("+"),
        "type": "template",
        "template": template,
    }
    result = _send_meta_request(config, payload)
    log.info("Meta WhatsApp template '%s' sent to %s", template_name, to)
    return result


# ── Twilio ────────────────────────────────────────────────────────────

def _send_twilio_text(config: dict, to: str, message: str) -> dict:
    """Send a text message via Twilio WhatsApp API."""
    account_sid = config["account_sid"]
    auth_token = config["auth_token"]
    from_number = config["from_number"]

    # Twilio expects "whatsapp:+1234567890" format
    if not to.startswith("whatsapp:"):
        to = f"whatsapp:{to}"
    if not from_number.startswith("whatsapp:"):
        from_number = f"whatsapp:{from_number}"

    url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
    resp = requests.post(
        url,
        auth=(account_sid, auth_token),
        data={"From": from_number, "To": to, "Body": message},
        timeout=30,
    )
    resp.raise_for_status()
    result = resp.json()
    log.info("Twilio WhatsApp message sent to %s: sid=%s", to, result.get("sid", "?"))
    return result


# ── Public API ────────────────────────────────────────────────────────

def send_whatsapp_message(
    to: str,
    message: str,
    config_file: str = "",
    dry_run: bool = False,
) -> dict | None:
    """Send a WhatsApp text message.

    Args:
        to: Recipient phone number with country code (e.g., "+15550000001").
        message: Text message to send.
        config_file: Path to config JSON (default: ~/keys/whatsapp_api_config.json).
        dry_run: If True, print instead of sending.

    Returns:
        API response dict, or None for dry-run.
    """
    to = _normalize_phone(to)

    if dry_run:
        log.info("[DRY-RUN] WhatsApp message to %s: %s", to, message[:200])
        print(f"[DRY-RUN] WhatsApp to {to}: {message[:200]}")
        return None

    config = _load_config(config_file)
    provider = config["provider"]

    if provider == "meta":
        return _send_meta_text(config, to, message)
    elif provider == "twilio":
        return _send_twilio_text(config, to, message)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def send_whatsapp_template(
    to: str,
    template_name: str,
    language: str = "en_US",
    components: list | None = None,
    config_file: str = "",
    dry_run: bool = False,
) -> dict | None:
    """Send a WhatsApp template message (Meta Business API only).

    Template messages are required by Meta for initiating conversations outside the
    24-hour customer service window.

    Args:
        to: Recipient phone number with country code.
        template_name: Name of the approved message template.
        language: Template language code (default: "en_US").
        components: Optional template components (header, body, button parameters).
        config_file: Path to config JSON.
        dry_run: If True, print instead of sending.

    Returns:
        API response dict, or None for dry-run.
    """
    to = _normalize_phone(to)

    if dry_run:
        log.info("[DRY-RUN] WhatsApp template '%s' to %s", template_name, to)
        print(f"[DRY-RUN] WhatsApp template '{template_name}' to {to}")
        return None

    config = _load_config(config_file)
    if config["provider"] != "meta":
        raise ValueError("Template messages are only supported with Meta Business API")

    return _send_meta_template(config, to, template_name, language, components)


def mark_as_read(message_id: str, config_file: str = "", dry_run: bool = True) -> dict | None:
    """Mark a WhatsApp message as read (Meta Business API only).

    Args:
        message_id: The wamid of the message to mark as read.
        config_file: Path to config JSON.
        dry_run: If True, do not send network requests (default: True).

    Returns:
        API response dict, or None for dry-run or non-Meta provider.
    """
    if dry_run:
        log.info("[DRY-RUN] WhatsApp mark_as_read for message_id %s", message_id)
        print(f"[DRY-RUN] WhatsApp mark_as_read for message_id: {message_id}")
        return None

    config = _load_config(config_file)
    if config["provider"] != "meta":
        log.debug("mark_as_read only supported with Meta Business API")
        return None
    payload = {
        "messaging_product": "whatsapp",
        "status": "read",
        "message_id": message_id,
    }
    return _send_meta_request(config, payload)


# ── Claude CLI integration ───────────────────────────────────────────

def clauded_reply(
    prompt: str,
    timeout: int = 300,
    clauded_bin: str = "clauded",
) -> str:
    """Generate a reply using the local clauded CLI with all tools disabled.

    Security rationale:
        WhatsApp messages originate from untrusted external senders. Because
        `clauded` is configured with permissions bypassed (equivalent to
        `claude --dangerously-skip-permissions`), any incoming message could attempt
        prompt-injection to execute shell commands, read/write files, or trigger
        external connector actions (e.g., claude.ai Gmail/Drive MCP connectors).

        To ensure absolute isolation:
        - `--tools ""` disables all standard Claude tools.
        - `--strict-mcp-config` disables MCP connectors that would otherwise remain active.
        - The prompt is passed via stdin (`input=prompt`), NOT as a positional command-line
          argument, because the variadic `--tools` option would consume subsequent
          positional arguments.

    Args:
        prompt: Prompt text to pass to clauded CLI via stdin.
        timeout: Execution timeout in seconds (default: 300).
        clauded_bin: Name or path of the clauded binary (default: "clauded").

    Returns:
        Stripped stdout response.

    Raises:
        RuntimeError: If clauded exits with a non-zero status code.
    """
    cmd = [clauded_bin, "-p", "--tools", "", "--strict-mcp-config"]
    proc = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"clauded failed with exit code {proc.returncode}: {proc.stderr.strip()}"
        )
    return proc.stdout.strip()


class WhatsAppClaudeBot:
    """Manages Claude-powered replies to WhatsApp conversations via clauded CLI.

    Keeps per-contact conversation history in memory and invokes the clauded CLI
    (or a custom reply_fn) to generate context-aware replies.

    Args:
        system_prompt: System prompt shaping Claude's reply style.
        reply_fn: Pluggable function (prompt: str) -> str. Defaults to clauded_reply.
        config_file: Path to WhatsApp API config JSON.
        dry_run: If True, do not call clauded, mark read, or send messages (default: True).
        allowed_phones: Collection of allowed phone numbers. If None or empty,
                        all incoming messages are ignored (default: None).
        max_replies_per_hour: Maximum replies per phone number per rolling hour (default: 10).
        max_history: Max conversation history messages retained per contact (default: 20).
    """

    def __init__(
        self,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        reply_fn: Callable[[str], str] | None = None,
        config_file: str = "",
        dry_run: bool = True,
        allowed_phones: set[str] | list[str] | None = None,
        max_replies_per_hour: int = 10,
        max_history: int = MAX_CONVERSATION_HISTORY,
        clauded_bin: str = "clauded",
    ):
        self.system_prompt = system_prompt
        self.clauded_bin = clauded_bin
        if reply_fn is not None:
            self.reply_fn = reply_fn
        elif clauded_bin != "clauded":
            self.reply_fn = lambda p: clauded_reply(p, clauded_bin=clauded_bin)
        else:
            self.reply_fn = clauded_reply
        self.config_file = config_file
        self.dry_run = dry_run
        self.max_replies_per_hour = max_replies_per_hour
        self.max_history = max_history

        # Normalize allowed phones
        self.allowed_phones: set[str] = set()
        if allowed_phones is not None:
            for p in allowed_phones:
                if p:
                    try:
                        self.allowed_phones.add(_normalize_phone(p))
                    except ValueError:
                        pass

        # Conversation history: phone_number -> list of {"role": str, "content": str}
        self._conversations: dict[str, list[dict[str, str]]] = defaultdict(list)

        # Rate limiting: phone_number -> list of timestamps (float)
        self._reply_timestamps: dict[str, list[float]] = defaultdict(list)

    def is_phone_allowed(self, phone: str) -> bool:
        """Check if phone number is in the allowed_phones set after normalization."""
        try:
            norm = _normalize_phone(phone)
            return norm in self.allowed_phones
        except ValueError:
            return False

    def is_rate_limited(self, phone: str, now: float | None = None) -> bool:
        """Check if phone number has reached max_replies_per_hour."""
        if now is None:
            now = time.time()
        cutoff = now - 3600.0
        try:
            norm = _normalize_phone(phone)
        except ValueError:
            return True
        timestamps = [ts for ts in self._reply_timestamps[norm] if ts > cutoff]
        self._reply_timestamps[norm] = timestamps
        return len(timestamps) >= self.max_replies_per_hour

    def _record_reply(self, phone: str, now: float | None = None) -> None:
        """Record a reply timestamp for rate limiting."""
        if now is None:
            now = time.time()
        try:
            norm = _normalize_phone(phone)
            self._reply_timestamps[norm].append(now)
        except ValueError:
            pass

    def get_history(self, phone: str) -> list[dict[str, str]]:
        """Get conversation history for a contact."""
        norm = _normalize_phone(phone)
        return list(self._conversations[norm])

    def clear_history(self, phone: str) -> None:
        """Clear conversation history for a contact."""
        norm = _normalize_phone(phone)
        self._conversations[norm].clear()

    def add_message(self, phone: str, role: str, content: str) -> None:
        """Add a message to conversation history, trimming oldest if needed."""
        norm = _normalize_phone(phone)
        history = self._conversations[norm]
        history.append({"role": role, "content": content})
        while len(history) > self.max_history:
            history.pop(0)

    def _build_prompt(self, phone: str) -> str:
        """Build the prompt for clauded including system instruction and recent history."""
        norm = _normalize_phone(phone)
        history = self._conversations[norm][-self.max_history:]
        lines = [self.system_prompt, "", "Recent conversation history:"]
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        lines.append("")
        lines.append("Please output only the text of the next assistant reply to the user.")
        return "\n".join(lines)

    def generate_reply(self, phone: str, incoming_message: str) -> str | None:
        """Generate a reply for an incoming WhatsApp message.

        Checks allowlist and rate limit. If allowed, records the message in history.
        In dry-run mode, does NOT call clauded or external services; logs/prints
        and returns a description with the prompt and reply target.

        Args:
            phone: Sender phone number.
            incoming_message: Incoming message text.

        Returns:
            Reply string (or dry-run description), or None if ignored or rate-limited.
        """
        try:
            norm_phone = _normalize_phone(phone)
        except ValueError as e:
            log.warning("Ignoring message with invalid phone %r: %s", phone, e)
            return None

        if not self.is_phone_allowed(norm_phone):
            log.info("Ignoring message from %s (not in allowed_phones)", norm_phone)
            return None

        if self.is_rate_limited(norm_phone):
            log.warning("Rate limit exceeded for %s (%d replies in past hour)", norm_phone, self.max_replies_per_hour)
            return None

        self.add_message(norm_phone, "user", incoming_message)
        prompt = self._build_prompt(norm_phone)

        if self.dry_run:
            desc = f"[DRY-RUN] Target: {norm_phone} | Prompt:\n{prompt}"
            log.info("[DRY-RUN] Generated prompt for %s", norm_phone)
            print(desc)
            self._record_reply(norm_phone)
            self.add_message(norm_phone, "assistant", f"[DRY-RUN reply to {norm_phone}]")
            return desc

        reply = self.reply_fn(prompt)
        self.add_message(norm_phone, "assistant", reply)
        self._record_reply(norm_phone)
        log.info("Claude reply for %s: %s", norm_phone, reply[:100])
        return reply

    def generate_reply_and_send(self, phone: str, incoming_message: str) -> str | None:
        """Generate a reply and send it back via WhatsApp.

        In dry-run mode, does not make network calls or invoke clauded CLI.

        Args:
            phone: Sender phone number.
            incoming_message: Incoming message text.

        Returns:
            The reply text that was (or would be) sent, or None if ignored/rate-limited.
        """
        reply = self.generate_reply(phone, incoming_message)
        if reply is None:
            return None

        send_whatsapp_message(
            to=phone,
            message=reply,
            config_file=self.config_file,
            dry_run=self.dry_run,
        )
        return reply


# ── Webhook server (Flask) ───────────────────────────────────────────

def _verify_webhook_signature(payload: bytes, signature: str, app_secret: str) -> bool:
    """Verify the X-Hub-Signature-256 header from Meta webhooks.

    Args:
        payload: Raw request body bytes.
        signature: Content of X-Hub-Signature-256 header.
        app_secret: Meta app secret.

    Returns:
        True if signature is valid or if app_secret is empty (no verification needed).
        False if app_secret is set but signature is missing or does not match.
    """
    if not app_secret:
        return True
    if not signature:
        return False
    expected = "sha256=" + hmac.new(
        app_secret.encode("utf-8"), payload, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


def _extract_messages(body: dict) -> list[dict[str, Any]]:
    """Extract incoming text messages from a Meta webhook payload.

    Returns a list of dicts with keys: from_phone, message_id, text, timestamp, contact_name.
    """
    messages = []
    for entry in body.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            for msg in value.get("messages", []):
                if msg.get("type") != "text":
                    log.debug("Skipping non-text message type: %s", msg.get("type"))
                    continue
                text_obj = msg.get("text")
                text_body = text_obj.get("body", "") if isinstance(text_obj, dict) else ""
                raw_from = msg.get("from", "")
                try:
                    from_phone = _normalize_phone(raw_from)
                except ValueError:
                    from_phone = "+" + raw_from
                messages.append({
                    "from_phone": from_phone,
                    "message_id": msg.get("id", ""),
                    "text": text_body,
                    "timestamp": msg.get("timestamp", ""),
                    "contact_name": _extract_contact_name(value, raw_from),
                })
    return messages


def _extract_contact_name(value: dict, wa_id: str) -> str:
    """Extract the contact's display name from webhook payload."""
    clean_wa = wa_id.lstrip("+")
    for contact in value.get("contacts", []):
        c_id = str(contact.get("wa_id", "")).lstrip("+")
        if c_id == clean_wa:
            return contact.get("profile", {}).get("name", "")
    return ""


def create_whatsapp_webhook_app(
    bot: WhatsAppClaudeBot | None = None,
    verify_token: str = "",
    app_secret: str = "",
    auto_reply: bool = True,
    auto_mark_read: bool = True,
    on_message: Any = None,
    background: bool = True,
) -> Any:
    """Create a Flask app that serves as a WhatsApp webhook endpoint.

    Args:
        bot: WhatsAppClaudeBot instance. Created automatically with dry_run=True if None.
        verify_token: Token for Meta webhook verification (GET requests).
                      If empty and not in dry-run, reads from config.
        app_secret: Meta app secret for X-Hub-Signature-256 verification.
        auto_reply: If True, automatically generate and send Claude replies.
        auto_mark_read: If True, mark incoming messages as read.
        on_message: Optional callback(phone, text, reply) called after message processing.
        background: If True, process replies in a background thread to return 200 quickly.

    Returns:
        Flask app instance.
    """
    from flask import Flask, jsonify, request

    app = Flask(__name__)

    if bot is None:
        bot = WhatsAppClaudeBot()

    # Resolve verify_token and app_secret from config if not provided and outside dry-run
    if not bot.dry_run:
        if not verify_token or not app_secret:
            try:
                config = _load_config(bot.config_file)
                if not verify_token:
                    verify_token = config.get("verify_token", "")
                if not app_secret:
                    app_secret = config.get("app_secret", "")
            except (FileNotFoundError, ValueError):
                pass

    @app.route("/webhook", methods=["GET"])
    def webhook_verify():
        """Handle Meta webhook verification (challenge-response)."""
        mode = request.args.get("hub.mode")
        token = request.args.get("hub.verify_token")
        challenge = request.args.get("hub.challenge")

        if mode == "subscribe" and token == verify_token:
            log.info("Webhook verified successfully")
            return challenge, 200
        log.warning("Webhook verification failed: mode=%s token_match=%s", mode, token == verify_token)
        return "Forbidden", 403

    @app.route("/webhook", methods=["POST"])
    def webhook_receive():
        """Handle incoming WhatsApp messages from Meta."""
        payload = request.get_data()

        # Verify signature if app_secret is configured
        signature = request.headers.get("X-Hub-Signature-256", "")
        if app_secret and not _verify_webhook_signature(payload, signature, app_secret):
            log.warning("Invalid webhook signature")
            return "Invalid signature", 403

        body = request.get_json(silent=True)
        if not body:
            return "OK", 200

        incoming = _extract_messages(body)
        if not incoming:
            return "OK", 200

        def _process_messages(messages_to_process: list[dict[str, Any]]) -> None:
            for msg in messages_to_process:
                phone = msg["from_phone"]
                text = msg["text"]
                name = msg["contact_name"]
                log.info("Processing from %s (%s): %s", phone, name or "unknown", text[:100])

                # Mark as read
                if auto_mark_read:
                    try:
                        mark_as_read(msg["message_id"], config_file=bot.config_file, dry_run=bot.dry_run)
                    except Exception as e:
                        log.warning("Failed to mark message as read: %s", e)

                # Generate and send Claude reply
                reply = None
                if auto_reply:
                    try:
                        reply = bot.generate_reply_and_send(phone, text)
                        if reply:
                            log.info("Replied to %s: %s", phone, reply[:100])
                    except Exception as e:
                        log.error("Failed to reply to %s: %s", phone, e)

                # Call user callback
                if on_message:
                    try:
                        on_message(phone, text, reply)
                    except Exception as e:
                        log.error("on_message callback error: %s", e)

        if background:
            thread = threading.Thread(target=_process_messages, args=(incoming,), daemon=True)
            thread.start()
            app._last_thread = thread
        else:
            _process_messages(incoming)

        return "OK", 200

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "dry_run": bot.dry_run}), 200

    return app


def run_whatsapp_bot(
    host: str = "127.0.0.1",
    port: int = 5000,
    debug: bool = False,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    reply_fn: Callable[[str], str] | None = None,
    config_file: str = "",
    verify_token: str = "",
    app_secret: str = "",
    auto_reply: bool = True,
    dry_run: bool = True,
    allowed_phones: set[str] | list[str] | None = None,
    max_replies_per_hour: int = 10,
    clauded_bin: str = "clauded",
) -> None:
    """Start the WhatsApp Claude bot webhook server.

    Args:
        host: Bind address (default: "127.0.0.1"; bind to localhost and put behind
              a reverse proxy or ngrok for HTTPS).
        port: Port to listen on (default: 5000).
        debug: Flask debug mode (default: False).
        system_prompt: System prompt shaping Claude's responses.
        reply_fn: Pluggable reply generator function (default: clauded_reply).
        config_file: Path to WhatsApp API config JSON.
        verify_token: Meta webhook verification token. Required if dry_run=False.
        app_secret: Meta app secret for X-Hub-Signature-256 verification. Required if dry_run=False.
        auto_reply: If True, auto-reply to incoming messages.
        dry_run: If True, runs in dry-run mode (default: True).
        allowed_phones: Set of allowed phone numbers. Required if dry_run=False.
        max_replies_per_hour: Max replies per hour per contact (default: 10).
        clauded_bin: Name or path of clauded executable (default: "clauded").
    """
    if not dry_run:
        if not allowed_phones:
            raise ValueError(
                "Refusing to start WhatsApp bot with dry_run=False unless allowed_phones is non-empty."
            )
        norm_allowed = set()
        for p in allowed_phones:
            if p:
                try:
                    norm_allowed.add(_normalize_phone(p))
                except ValueError:
                    pass
        if not norm_allowed:
            raise ValueError(
                "Refusing to start WhatsApp bot with dry_run=False: no valid phone numbers in allowed_phones."
            )

        # Resolve verify_token and app_secret from config if not provided
        if not verify_token or not app_secret:
            try:
                cfg = _load_config(config_file)
                if not verify_token:
                    verify_token = cfg.get("verify_token", "")
                if not app_secret:
                    app_secret = cfg.get("app_secret", "")
            except Exception:
                pass

        if not verify_token:
            raise ValueError(
                "Refusing to start WhatsApp bot with dry_run=False unless verify_token is non-empty "
                "(pass verify_token or specify in config JSON)."
            )
        if not app_secret:
            raise ValueError(
                "Refusing to start WhatsApp bot with dry_run=False unless app_secret is non-empty "
                "(pass app_secret or specify in config JSON for X-Hub-Signature-256 verification)."
            )

    bot = WhatsAppClaudeBot(
        system_prompt=system_prompt,
        reply_fn=reply_fn,
        config_file=config_file,
        dry_run=dry_run,
        allowed_phones=allowed_phones,
        max_replies_per_hour=max_replies_per_hour,
        clauded_bin=clauded_bin,
    )
    app = create_whatsapp_webhook_app(
        bot=bot,
        verify_token=verify_token,
        app_secret=app_secret,
        auto_reply=auto_reply,
    )
    print(f"Starting WhatsApp Claude bot on {host}:{port}")
    print(f"  Dry-run: {dry_run}")
    print(f"  Allowed phones: {len(bot.allowed_phones)}")
    print(f"  Auto-reply: {auto_reply}")
    print(f"  Webhook URL: http://{host}:{port}/webhook")
    print(f"  Health check: http://{host}:{port}/health")
    app.run(host=host, port=port, debug=debug)


# ── Smoke test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== WhatsApp dry-run smoke tests ===")
    send_whatsapp_message("+15550000001", "Hello from uutils!", dry_run=True)
    send_whatsapp_template("+15550000001", "hello_world", dry_run=True)
    mark_as_read("wamid.test123", dry_run=True)

    # Test phone normalization
    assert _normalize_phone("15550000001") == "+15550000001"
    assert _normalize_phone("+15550000001") == "+15550000001"
    assert _normalize_phone("  +44 20 7946 0000  ") == "+442079460000"
    assert _normalize_phone("whatsapp:+15550000001") == "+15550000001"
    assert _normalize_phone("+1 (555) 000-0001") == "+15550000001"
    print("Phone normalization tests passed ✓")

    # Test message extraction
    sample_webhook_payload = {
        "entry": [{
            "changes": [{
                "value": {
                    "messages": [{
                        "from": "15550000001",
                        "id": "wamid.test123",
                        "type": "text",
                        "text": {"body": "Hello!"},
                        "timestamp": "1234567890",
                    }],
                    "contacts": [{
                        "wa_id": "15550000001",
                        "profile": {"name": "Test User"},
                    }],
                },
            }],
        }],
    }
    extracted = _extract_messages(sample_webhook_payload)
    assert len(extracted) == 1
    assert extracted[0]["from_phone"] == "+15550000001"
    assert extracted[0]["text"] == "Hello!"
    assert extracted[0]["contact_name"] == "Test User"
    print("Message extraction tests passed ✓")

    # Test WhatsAppClaudeBot in dry-run
    bot = WhatsAppClaudeBot(dry_run=True, allowed_phones={"+15550000001"})
    bot.add_message("+15550000001", "user", "Hi there")
    bot.add_message("+15550000001", "assistant", "Hello!")
    assert len(bot.get_history("+15550000001")) == 2
    bot.clear_history("+15550000001")
    assert len(bot.get_history("+15550000001")) == 0

    # Allowed vs unallowed
    reply = bot.generate_reply("+15550000001", "Hello dry run")
    assert reply is not None
    assert "[DRY-RUN]" in reply

    rejected = bot.generate_reply("+15550000002", "Not allowed")
    assert rejected is None
    print("Bot dry-run tests passed ✓")

    print("\nAll dry-run smoke tests passed!")
