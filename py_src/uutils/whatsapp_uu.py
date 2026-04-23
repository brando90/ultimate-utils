"""WhatsApp messaging — send/receive messages via Meta Business Cloud API or Twilio,
with optional Claude-powered auto-replies.

Quick usage (send only):
    from uutils.whatsapp_uu import send_whatsapp_message
    send_whatsapp_message("+14155551234", "Hello from uutils!")

Quick usage (Claude auto-reply bot):
    from uutils.whatsapp_uu import run_whatsapp_bot
    run_whatsapp_bot()  # starts Flask webhook server on port 5000

Quick usage (programmatic reply without server):
    from uutils.whatsapp_uu import WhatsAppClaudeBot
    bot = WhatsAppClaudeBot()
    reply = bot.generate_reply("+14155551234", "Hey, what's up?")
    print(reply)
    # bot.generate_reply_and_send("+14155551234", "Hey, what's up?")  # also sends via WhatsApp

Setup (Meta Business Cloud API — recommended):
    1. Create a Meta Business account: https://business.facebook.com/
    2. Set up WhatsApp Business: https://developers.facebook.com/docs/whatsapp/cloud-api/get-started
    3. Get your access token and phone number ID from the Meta developer console
    4. Save config:
       cat > ~/keys/whatsapp_api_config.json << 'JSON'
       {
           "provider": "meta",
           "access_token": "YOUR_ACCESS_TOKEN",
           "phone_number_id": "YOUR_PHONE_NUMBER_ID",
           "api_version": "v21.0",
           "verify_token": "YOUR_WEBHOOK_VERIFY_TOKEN"
       }
       JSON
       chmod 600 ~/keys/whatsapp_api_config.json

    5. Set your Anthropic API key (for Claude replies):
       export ANTHROPIC_API_KEY="sk-ant-..."
       # or save to: ~/keys/anthropic_api_key.txt

    6. Expose your webhook (for receiving messages):
       # Option A: ngrok (for local development)
       ngrok http 5000
       # Option B: deploy to a server with a public URL

    7. Configure the webhook in Meta Developer Console:
       - Webhook URL: https://YOUR_DOMAIN/webhook
       - Verify token: same as verify_token in your config
       - Subscribe to: messages

Setup (Twilio — alternative):
    1. Create Twilio account: https://www.twilio.com/
    2. Enable WhatsApp sandbox: https://www.twilio.com/docs/whatsapp/sandbox
    3. Save config:
       cat > ~/keys/whatsapp_api_config.json << 'JSON'
       {
           "provider": "twilio",
           "account_sid": "YOUR_ACCOUNT_SID",
           "auth_token": "YOUR_AUTH_TOKEN",
           "from_number": "whatsapp:+14155238886"
       }
       JSON
       chmod 600 ~/keys/whatsapp_api_config.json

Refs:
    - Meta Cloud API: https://developers.facebook.com/docs/whatsapp/cloud-api
    - Meta Webhooks: https://developers.facebook.com/docs/whatsapp/cloud-api/webhooks
    - Twilio WhatsApp: https://www.twilio.com/docs/whatsapp/api
    - Anthropic API: https://docs.anthropic.com/en/docs
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger(__name__)

DEFAULT_CONFIG_FILE = "~/keys/whatsapp_api_config.json"
DEFAULT_ANTHROPIC_KEY_FILE = "~/keys/anthropic_api_key.txt"
_PHONE_SEPARATORS = {" ", "-", "(", ")", "."}

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant replying to WhatsApp messages on behalf of the user. "
    "Keep responses concise and natural — as if texting a friend. "
    "Use short paragraphs. Avoid markdown formatting (no ** or ## etc.) since this is WhatsApp. "
    "If the message is casual, be casual. If it's a question, give a clear answer. "
    "If you're unsure what someone means, ask a brief clarifying question."
)

MAX_CONVERSATION_HISTORY = 50  # messages per contact to keep in memory


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
        raise ValueError("Phone number is empty — provide a number with country code (e.g., '+14155551234')")

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
        raise ValueError("Phone number is empty — provide a number with country code (e.g., '+14155551234')")
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
        to: Recipient phone number with country code (e.g., "+14155551234").
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


# ── Mark as read ─────────────────────────────────────────────────────

def mark_as_read(message_id: str, config_file: str = "") -> dict | None:
    """Mark a WhatsApp message as read (Meta Business API only).

    Args:
        message_id: The wamid of the message to mark as read.
        config_file: Path to config JSON.

    Returns:
        API response dict, or None if not using Meta provider.
    """
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


# ── Anthropic / Claude integration ───────────────────────────────────

def _load_anthropic_key() -> str:
    """Load Anthropic API key from env var or key file."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        return key.strip()
    fpath = Path(DEFAULT_ANTHROPIC_KEY_FILE).expanduser()
    if fpath.is_file():
        key = fpath.read_text().strip()
        if key:
            return key
    raise ValueError(
        "No Anthropic API key found. Set ANTHROPIC_API_KEY env var "
        f"or save your key to {DEFAULT_ANTHROPIC_KEY_FILE}"
    )


class WhatsAppClaudeBot:
    """Manages Claude-powered replies to WhatsApp conversations.

    Keeps per-contact conversation history in memory and uses the Anthropic API
    to generate context-aware replies.

    Args:
        system_prompt: System prompt that shapes Claude's reply style.
        model: Anthropic model to use.
        max_tokens: Max tokens per reply.
        config_file: Path to WhatsApp API config JSON.
        anthropic_api_key: Explicit key; if empty, reads from env/file.
        dry_run: If True, don't actually send WhatsApp messages.

    Usage:
        bot = WhatsAppClaudeBot()

        # Generate a reply without sending
        reply = bot.generate_reply("+14155551234", "Hey, what's the weather?")

        # Generate and send
        bot.generate_reply_and_send("+14155551234", "Hey, what's the weather?")
    """

    def __init__(
        self,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
        config_file: str = "",
        anthropic_api_key: str = "",
        dry_run: bool = False,
    ):
        self.system_prompt = system_prompt
        self.model = model
        self.max_tokens = max_tokens
        self.config_file = config_file
        self.dry_run = dry_run

        # Conversation history: phone_number -> list of {"role": ..., "content": ...}
        self._conversations: dict[str, list[dict[str, str]]] = defaultdict(list)

        # Lazy-init the Anthropic client
        self._anthropic_key = anthropic_api_key
        self._client = None

    @property
    def client(self):
        """Lazy-load anthropic client (so import isn't required at module level)."""
        if self._client is None:
            import anthropic
            key = self._anthropic_key or _load_anthropic_key()
            self._client = anthropic.Anthropic(api_key=key)
        return self._client

    def get_history(self, phone: str) -> list[dict[str, str]]:
        """Get conversation history for a contact."""
        return list(self._conversations[phone])

    def clear_history(self, phone: str) -> None:
        """Clear conversation history for a contact."""
        self._conversations[phone].clear()

    def add_message(self, phone: str, role: str, content: str) -> None:
        """Add a message to conversation history, trimming if needed."""
        history = self._conversations[phone]
        history.append({"role": role, "content": content})
        # Trim oldest messages if over limit (keep pairs to maintain alternation)
        while len(history) > MAX_CONVERSATION_HISTORY:
            history.pop(0)

    def generate_reply(self, phone: str, incoming_message: str) -> str:
        """Generate a Claude reply for an incoming WhatsApp message.

        Adds the incoming message to history, calls Claude, adds the reply
        to history, and returns it. Does NOT send the reply via WhatsApp.

        Args:
            phone: Sender phone number (used as conversation key).
            incoming_message: The text message received.

        Returns:
            Claude's reply text.
        """
        phone = _normalize_phone(phone)
        self.add_message(phone, "user", incoming_message)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system_prompt,
            messages=self._conversations[phone],
        )
        reply_text = response.content[0].text
        self.add_message(phone, "assistant", reply_text)
        log.info("Claude reply for %s (%d tokens): %s",
                 phone, response.usage.output_tokens, reply_text[:100])
        return reply_text

    def generate_reply_and_send(self, phone: str, incoming_message: str) -> str:
        """Generate a Claude reply and send it back via WhatsApp.

        Args:
            phone: Sender phone number.
            incoming_message: The text message received.

        Returns:
            The reply text that was sent.
        """
        reply = self.generate_reply(phone, incoming_message)
        send_whatsapp_message(
            to=phone,
            message=reply,
            config_file=self.config_file,
            dry_run=self.dry_run,
        )
        return reply


# ── Webhook server (Flask) ───────────────────────────────────────────

def _verify_webhook_signature(payload: bytes, signature: str, app_secret: str) -> bool:
    """Verify the X-Hub-Signature-256 header from Meta webhooks."""
    if not signature or not app_secret:
        return True  # skip verification if no secret configured
    expected = "sha256=" + hmac.new(
        app_secret.encode(), payload, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


def _extract_messages(body: dict) -> list[dict[str, Any]]:
    """Extract incoming text messages from a Meta webhook payload.

    Returns a list of dicts with keys: from_phone, message_id, text, timestamp.
    """
    messages = []
    for entry in body.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            for msg in value.get("messages", []):
                if msg.get("type") != "text":
                    log.debug("Skipping non-text message type: %s", msg.get("type"))
                    continue
                messages.append({
                    "from_phone": "+" + msg["from"],
                    "message_id": msg["id"],
                    "text": msg["text"]["body"],
                    "timestamp": msg.get("timestamp", ""),
                    "contact_name": _extract_contact_name(value, msg["from"]),
                })
    return messages


def _extract_contact_name(value: dict, wa_id: str) -> str:
    """Extract the contact's display name from webhook payload."""
    for contact in value.get("contacts", []):
        if contact.get("wa_id") == wa_id:
            return contact.get("profile", {}).get("name", "")
    return ""


def create_whatsapp_webhook_app(
    bot: WhatsAppClaudeBot | None = None,
    verify_token: str = "",
    app_secret: str = "",
    auto_reply: bool = True,
    auto_mark_read: bool = True,
    on_message: Any = None,
) -> Any:
    """Create a Flask app that serves as a WhatsApp webhook endpoint.

    Args:
        bot: WhatsAppClaudeBot instance. Created automatically if None.
        verify_token: Token for Meta webhook verification (GET requests).
                      If empty, reads from whatsapp config's "verify_token" field.
        app_secret: Meta app secret for signature verification. Optional.
        auto_reply: If True, automatically generate and send Claude replies.
        auto_mark_read: If True, mark incoming messages as read.
        on_message: Optional callback(phone, text, reply) called after each message.

    Returns:
        Flask app instance.
    """
    from flask import Flask, request, jsonify

    app = Flask(__name__)

    if bot is None:
        bot = WhatsAppClaudeBot()

    # Resolve verify_token from config if not provided
    if not verify_token:
        try:
            config = _load_config(bot.config_file)
            verify_token = config.get("verify_token", "")
        except FileNotFoundError:
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

        # Extract text messages
        incoming = _extract_messages(body)
        for msg in incoming:
            phone = msg["from_phone"]
            text = msg["text"]
            name = msg["contact_name"]
            log.info("Received from %s (%s): %s", phone, name or "unknown", text[:100])

            # Mark as read
            if auto_mark_read:
                try:
                    mark_as_read(msg["message_id"], config_file=bot.config_file)
                except Exception as e:
                    log.warning("Failed to mark message as read: %s", e)

            # Generate and send Claude reply
            reply = ""
            if auto_reply:
                try:
                    reply = bot.generate_reply_and_send(phone, text)
                    log.info("Replied to %s: %s", phone, reply[:100])
                except Exception as e:
                    log.error("Failed to reply to %s: %s", phone, e)

            # Call user callback
            if on_message:
                try:
                    on_message(phone, text, reply)
                except Exception as e:
                    log.error("on_message callback error: %s", e)

        return "OK", 200

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "bot_model": bot.model}), 200

    return app


def run_whatsapp_bot(
    host: str = "0.0.0.0",
    port: int = 5000,
    debug: bool = True,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    model: str = "claude-sonnet-4-20250514",
    config_file: str = "",
    verify_token: str = "",
    app_secret: str = "",
    auto_reply: bool = True,
    dry_run: bool = False,
) -> None:
    """Start the WhatsApp Claude bot webhook server.

    This is the main entry point — run this and point your Meta webhook to
    http://YOUR_HOST:PORT/webhook

    Args:
        host: Bind address (default: 0.0.0.0 for all interfaces).
        port: Port to listen on (default: 5000).
        debug: Flask debug mode.
        system_prompt: System prompt for Claude.
        model: Anthropic model name.
        config_file: Path to WhatsApp API config JSON.
        verify_token: Webhook verification token.
        app_secret: Meta app secret for signature verification.
        auto_reply: If True, auto-reply with Claude. If False, just logs messages.
        dry_run: If True, don't actually send WhatsApp messages.
    """
    bot = WhatsAppClaudeBot(
        system_prompt=system_prompt,
        model=model,
        config_file=config_file,
        dry_run=dry_run,
    )
    app = create_whatsapp_webhook_app(
        bot=bot,
        verify_token=verify_token,
        app_secret=app_secret,
        auto_reply=auto_reply,
    )
    print(f"Starting WhatsApp Claude bot on {host}:{port}")
    print(f"  Model: {model}")
    print(f"  Auto-reply: {auto_reply}")
    print(f"  Dry-run: {dry_run}")
    print(f"  Webhook URL: http://{host}:{port}/webhook")
    print(f"  Health check: http://{host}:{port}/health")
    app.run(host=host, port=port, debug=debug)


# ── Smoke test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== WhatsApp dry-run smoke tests ===")
    send_whatsapp_message("+14155551234", "Hello from uutils!", dry_run=True)
    send_whatsapp_template("+14155551234", "hello_world", dry_run=True)

    # Test phone normalization
    assert _normalize_phone("14155551234") == "+14155551234"
    assert _normalize_phone("+14155551234") == "+14155551234"
    assert _normalize_phone("  +44 20 1234 5678  ") == "+442012345678"
    assert _normalize_phone("whatsapp:+14155551234") == "+14155551234"
    print("Phone normalization tests passed ✓")

    # Test message extraction
    sample_webhook_payload = {
        "entry": [{
            "changes": [{
                "value": {
                    "messages": [{
                        "from": "14155551234",
                        "id": "wamid.test123",
                        "type": "text",
                        "text": {"body": "Hello!"},
                        "timestamp": "1234567890",
                    }],
                    "contacts": [{
                        "wa_id": "14155551234",
                        "profile": {"name": "Test User"},
                    }],
                },
            }],
        }],
    }
    extracted = _extract_messages(sample_webhook_payload)
    assert len(extracted) == 1
    assert extracted[0]["from_phone"] == "+14155551234"
    assert extracted[0]["text"] == "Hello!"
    assert extracted[0]["contact_name"] == "Test User"
    print("Message extraction tests passed ✓")

    # Test WhatsAppClaudeBot conversation management (no API calls)
    bot = WhatsAppClaudeBot(dry_run=True)
    bot.add_message("+14155551234", "user", "Hi there")
    bot.add_message("+14155551234", "assistant", "Hello!")
    assert len(bot.get_history("+14155551234")) == 2
    bot.clear_history("+14155551234")
    assert len(bot.get_history("+14155551234")) == 0
    print("Bot conversation management tests passed ✓")

    print("\nAll dry-run smoke tests passed!")
