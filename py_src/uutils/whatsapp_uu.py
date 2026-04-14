"""WhatsApp messaging — send messages via Meta Business Cloud API or Twilio.

Quick usage:
    from uutils.whatsapp_uu import send_whatsapp_message
    send_whatsapp_message("+14155551234", "Hello from uutils!")

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
           "api_version": "v21.0"
       }
       JSON
       chmod 600 ~/keys/whatsapp_api_config.json

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
    - Twilio WhatsApp: https://www.twilio.com/docs/whatsapp/api
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import requests

log = logging.getLogger(__name__)

DEFAULT_CONFIG_FILE = "~/keys/whatsapp_api_config.json"


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
    # Strip the leading '+' (if any) to check for actual digits
    digits = phone.lstrip("+")
    if not digits:
        raise ValueError("Phone number is empty — provide a number with country code (e.g., '+14155551234')")
    if not phone.startswith("+"):
        phone = "+" + phone
    return phone


# ── Meta Business Cloud API ──────────────────────────────────────────

def _send_meta_text(config: dict, to: str, message: str) -> dict:
    """Send a text message via Meta WhatsApp Business Cloud API."""
    api_version = config.get("api_version", "v21.0")
    phone_number_id = config["phone_number_id"]
    access_token = config["access_token"]

    url = f"https://graph.facebook.com/{api_version}/{phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to.lstrip("+"),
        "type": "text",
        "text": {"body": message},
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    result = resp.json()
    log.info("Meta WhatsApp message sent to %s: %s", to, result.get("messages", [{}])[0].get("id", "?"))
    return result


def _send_meta_template(config: dict, to: str, template_name: str, language: str, components: list | None) -> dict:
    """Send a template message via Meta WhatsApp Business Cloud API."""
    api_version = config.get("api_version", "v21.0")
    phone_number_id = config["phone_number_id"]
    access_token = config["access_token"]

    url = f"https://graph.facebook.com/{api_version}/{phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
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
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    result = resp.json()
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


# ── Smoke test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== WhatsApp dry-run smoke tests ===")
    send_whatsapp_message("+14155551234", "Hello from uutils!", dry_run=True)
    send_whatsapp_template("+14155551234", "hello_world", dry_run=True)

    # Test phone normalization
    assert _normalize_phone("14155551234") == "+14155551234"
    assert _normalize_phone("+14155551234") == "+14155551234"
    assert _normalize_phone("  +44 20 1234 5678  ") == "+44 20 1234 5678"
    print("Phone normalization tests passed ✓")

    print("All dry-run smoke tests passed!")
