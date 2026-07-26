"""End-to-end test for WhatsApp + Claude integration.

Runs in three modes:
    1. Dry-run (default) — tests everything locally, no API calls.
    2. Claude-only — tests Claude reply generation (needs ANTHROPIC_API_KEY).
    3. Full — tests the Flask webhook server with simulated Meta payloads.

Usage:
    # Dry-run (no keys needed)
    python playground/whatsapp_claude/test_whatsapp_claude_bot.py

    # With Claude replies (needs ANTHROPIC_API_KEY)
    python playground/whatsapp_claude/test_whatsapp_claude_bot.py --claude

    # Full webhook server test (needs ANTHROPIC_API_KEY)
    python playground/whatsapp_claude/test_whatsapp_claude_bot.py --full

    # Run the actual bot server (needs all keys)
    python playground/whatsapp_claude/test_whatsapp_claude_bot.py --serve
"""
from __future__ import annotations

import argparse
import json
import sys
import os

# ── Dry-run tests (no API keys needed) ───────────────────────────────

def test_send_dry_run():
    """Test sending messages in dry-run mode."""
    from uutils.whatsapp_uu import send_whatsapp_message, send_whatsapp_template

    print("--- test_send_dry_run ---")
    send_whatsapp_message("+14155551234", "Hello from dry-run test!", dry_run=True)
    send_whatsapp_template("+14155551234", "hello_world", dry_run=True)
    print("PASSED\n")


def test_phone_normalization():
    """Test phone number normalization edge cases."""
    from uutils.whatsapp_uu import _normalize_phone

    print("--- test_phone_normalization ---")
    cases = [
        ("14155551234", "+14155551234"),
        ("+14155551234", "+14155551234"),
        ("  +44 20 1234 5678  ", "+442012345678"),
        ("whatsapp:+14155551234", "+14155551234"),
        ("+1 (415) 555-1234", "+14155551234"),
        ("+49.30.1234.5678", "+493012345678"),
    ]
    for inp, expected in cases:
        result = _normalize_phone(inp)
        assert result == expected, f"_normalize_phone({inp!r}) = {result!r}, expected {expected!r}"
        print(f"  {inp!r:30s} -> {result!r}")
    print("PASSED\n")


def test_message_extraction():
    """Test extracting messages from Meta webhook payloads."""
    from uutils.whatsapp_uu import _extract_messages

    print("--- test_message_extraction ---")

    # Normal text message
    payload = {
        "entry": [{
            "changes": [{
                "value": {
                    "messages": [{
                        "from": "14155551234",
                        "id": "wamid.HBgNMTQxNTU1NTEyMzQ",
                        "type": "text",
                        "text": {"body": "Hey, can you help me with something?"},
                        "timestamp": "1713100800",
                    }],
                    "contacts": [{
                        "wa_id": "14155551234",
                        "profile": {"name": "Alice"},
                    }],
                },
            }],
        }],
    }
    msgs = _extract_messages(payload)
    assert len(msgs) == 1
    assert msgs[0]["from_phone"] == "+14155551234"
    assert msgs[0]["text"] == "Hey, can you help me with something?"
    assert msgs[0]["contact_name"] == "Alice"
    print(f"  Extracted: {msgs[0]}")

    # Non-text message should be skipped
    payload_image = {
        "entry": [{
            "changes": [{
                "value": {
                    "messages": [{
                        "from": "14155551234",
                        "id": "wamid.img123",
                        "type": "image",
                        "image": {"id": "img123"},
                    }],
                    "contacts": [],
                },
            }],
        }],
    }
    msgs = _extract_messages(payload_image)
    assert len(msgs) == 0
    print("  Non-text messages correctly skipped")

    # Empty payload
    msgs = _extract_messages({})
    assert len(msgs) == 0
    print("  Empty payload handled")

    # Multiple messages
    payload_multi = {
        "entry": [{
            "changes": [{
                "value": {
                    "messages": [
                        {"from": "14155551234", "id": "wamid.1", "type": "text",
                         "text": {"body": "First"}},
                        {"from": "14155559876", "id": "wamid.2", "type": "text",
                         "text": {"body": "Second"}},
                    ],
                    "contacts": [
                        {"wa_id": "14155551234", "profile": {"name": "Alice"}},
                        {"wa_id": "14155559876", "profile": {"name": "Bob"}},
                    ],
                },
            }],
        }],
    }
    msgs = _extract_messages(payload_multi)
    assert len(msgs) == 2
    assert msgs[0]["contact_name"] == "Alice"
    assert msgs[1]["contact_name"] == "Bob"
    print(f"  Multi-message: extracted {len(msgs)} messages")

    print("PASSED\n")


def test_bot_conversation_management():
    """Test the WhatsAppClaudeBot conversation history (no API calls)."""
    from uutils.whatsapp_uu import WhatsAppClaudeBot

    print("--- test_bot_conversation_management ---")

    bot = WhatsAppClaudeBot(dry_run=True)
    phone = "+14155551234"

    # Add messages
    bot.add_message(phone, "user", "Hello!")
    bot.add_message(phone, "assistant", "Hi there! How can I help?")
    bot.add_message(phone, "user", "What's the weather like?")
    history = bot.get_history(phone)
    assert len(history) == 3
    assert history[0]["role"] == "user"
    assert history[2]["content"] == "What's the weather like?"
    print(f"  History has {len(history)} messages")

    # Separate conversations per phone
    phone2 = "+442012345678"
    bot.add_message(phone2, "user", "Different convo")
    assert len(bot.get_history(phone)) == 3
    assert len(bot.get_history(phone2)) == 1
    print("  Per-contact histories are isolated")

    # Clear
    bot.clear_history(phone)
    assert len(bot.get_history(phone)) == 0
    assert len(bot.get_history(phone2)) == 1
    print("  History cleared correctly")

    # Test trimming (MAX_CONVERSATION_HISTORY)
    from uutils.whatsapp_uu import MAX_CONVERSATION_HISTORY
    for i in range(MAX_CONVERSATION_HISTORY + 10):
        role = "user" if i % 2 == 0 else "assistant"
        bot.add_message(phone, role, f"Message {i}")
    assert len(bot.get_history(phone)) <= MAX_CONVERSATION_HISTORY
    print(f"  History trimmed to <= {MAX_CONVERSATION_HISTORY} messages")

    print("PASSED\n")


def test_webhook_app_creation():
    """Test creating the Flask webhook app (no server start)."""
    from uutils.whatsapp_uu import WhatsAppClaudeBot, create_whatsapp_webhook_app

    print("--- test_webhook_app_creation ---")

    bot = WhatsAppClaudeBot(dry_run=True)
    app = create_whatsapp_webhook_app(
        bot=bot,
        verify_token="test_token_123",
        auto_reply=False,  # don't try to call Claude
    )
    assert app is not None
    print("  Flask app created successfully")

    # Test webhook verification endpoint
    with app.test_client() as client:
        # Correct token
        resp = client.get("/webhook?hub.mode=subscribe&hub.verify_token=test_token_123&hub.challenge=challenge_abc")
        assert resp.status_code == 200
        assert resp.data == b"challenge_abc"
        print("  Webhook verification (correct token): 200")

        # Wrong token
        resp = client.get("/webhook?hub.mode=subscribe&hub.verify_token=wrong&hub.challenge=x")
        assert resp.status_code == 403
        print("  Webhook verification (wrong token): 403")

        # Health check
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        print(f"  Health check: {data}")

    print("PASSED\n")


def test_webhook_message_handling():
    """Test the webhook POST handler with simulated Meta payloads."""
    from uutils.whatsapp_uu import WhatsAppClaudeBot, create_whatsapp_webhook_app

    print("--- test_webhook_message_handling ---")

    received_messages = []

    def on_message(phone, text, reply):
        received_messages.append({"phone": phone, "text": text, "reply": reply})

    bot = WhatsAppClaudeBot(dry_run=True)
    app = create_whatsapp_webhook_app(
        bot=bot,
        verify_token="test_token",
        auto_reply=False,  # don't call Claude API
        auto_mark_read=False,  # don't call WhatsApp API
        on_message=on_message,
    )

    payload = {
        "entry": [{
            "changes": [{
                "value": {
                    "messages": [{
                        "from": "14155551234",
                        "id": "wamid.test456",
                        "type": "text",
                        "text": {"body": "Can you help me plan a trip?"},
                        "timestamp": "1713100800",
                    }],
                    "contacts": [{
                        "wa_id": "14155551234",
                        "profile": {"name": "Alice"},
                    }],
                },
            }],
        }],
    }

    with app.test_client() as client:
        resp = client.post(
            "/webhook",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code == 200
        print(f"  Webhook POST response: {resp.status_code}")

    assert len(received_messages) == 1
    assert received_messages[0]["phone"] == "+14155551234"
    assert received_messages[0]["text"] == "Can you help me plan a trip?"
    assert received_messages[0]["reply"] == ""  # auto_reply=False
    print(f"  Callback received: {received_messages[0]}")

    print("PASSED\n")


# ── Claude integration tests (needs ANTHROPIC_API_KEY) ───────────────

def test_claude_reply_generation():
    """Test actual Claude reply generation (requires ANTHROPIC_API_KEY)."""
    from uutils.whatsapp_uu import WhatsAppClaudeBot

    print("--- test_claude_reply_generation ---")

    bot = WhatsAppClaudeBot(
        system_prompt="You are a friendly assistant. Keep replies under 50 words.",
        model="claude-sonnet-4-20250514",
        dry_run=True,  # don't send WhatsApp, but do call Claude
    )

    # First message
    reply1 = bot.generate_reply("+14155551234", "Hey! What's 2 + 2?")
    print(f"  User: Hey! What's 2 + 2?")
    print(f"  Claude: {reply1}")
    assert len(reply1) > 0
    assert "4" in reply1 or "four" in reply1.lower()

    # Follow-up (tests conversation context)
    reply2 = bot.generate_reply("+14155551234", "And what's that times 3?")
    print(f"  User: And what's that times 3?")
    print(f"  Claude: {reply2}")
    assert len(reply2) > 0

    # Different contact (separate history)
    reply3 = bot.generate_reply("+442012345678", "Hi there!")
    print(f"  User (+44): Hi there!")
    print(f"  Claude: {reply3}")
    assert len(reply3) > 0

    # Verify histories are separate
    assert len(bot.get_history("+14155551234")) == 4  # 2 user + 2 assistant
    assert len(bot.get_history("+442012345678")) == 2  # 1 user + 1 assistant
    print("  Conversation histories correctly isolated")

    print("PASSED\n")


def test_full_webhook_with_claude():
    """Test webhook with actual Claude replies (requires ANTHROPIC_API_KEY)."""
    from uutils.whatsapp_uu import WhatsAppClaudeBot, create_whatsapp_webhook_app

    print("--- test_full_webhook_with_claude ---")

    replies = []

    def on_message(phone, text, reply):
        replies.append({"phone": phone, "text": text, "reply": reply})

    bot = WhatsAppClaudeBot(
        system_prompt="You are a helpful assistant. Keep replies under 30 words.",
        dry_run=True,  # don't send WhatsApp messages
    )
    app = create_whatsapp_webhook_app(
        bot=bot,
        verify_token="test",
        auto_reply=True,  # WILL call Claude
        auto_mark_read=False,
        on_message=on_message,
    )

    payload = {
        "entry": [{
            "changes": [{
                "value": {
                    "messages": [{
                        "from": "14155551234",
                        "id": "wamid.full_test",
                        "type": "text",
                        "text": {"body": "What's the capital of France?"},
                    }],
                    "contacts": [{"wa_id": "14155551234", "profile": {"name": "Tester"}}],
                },
            }],
        }],
    }

    with app.test_client() as client:
        resp = client.post("/webhook", data=json.dumps(payload), content_type="application/json")
        assert resp.status_code == 200

    assert len(replies) == 1
    assert "paris" in replies[0]["reply"].lower()
    print(f"  Question: What's the capital of France?")
    print(f"  Reply: {replies[0]['reply']}")
    print("PASSED\n")


# ── Server mode ──────────────────────────────────────────────────────

def run_server():
    """Start the actual WhatsApp bot server."""
    from uutils.whatsapp_uu import run_whatsapp_bot
    run_whatsapp_bot(debug=True)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test WhatsApp + Claude integration")
    parser.add_argument("--claude", action="store_true",
                        help="Run Claude integration tests (needs ANTHROPIC_API_KEY)")
    parser.add_argument("--full", action="store_true",
                        help="Run full webhook + Claude tests (needs ANTHROPIC_API_KEY)")
    parser.add_argument("--serve", action="store_true",
                        help="Start the actual WhatsApp bot server")
    args = parser.parse_args()

    if args.serve:
        run_server()
        return

    print("=" * 60)
    print("WhatsApp + Claude Integration Tests")
    print("=" * 60 + "\n")

    # Always run dry-run tests
    test_send_dry_run()
    test_phone_normalization()
    test_message_extraction()
    test_bot_conversation_management()
    test_webhook_app_creation()
    test_webhook_message_handling()

    if args.claude or args.full:
        if not os.environ.get("ANTHROPIC_API_KEY") and not os.path.isfile(
            os.path.expanduser("~/keys/anthropic_api_key.txt")
        ):
            print("SKIPPED: Claude tests require ANTHROPIC_API_KEY env var or ~/keys/anthropic_api_key.txt")
            sys.exit(1)
        test_claude_reply_generation()

    if args.full:
        test_full_webhook_with_claude()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
