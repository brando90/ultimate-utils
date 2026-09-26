"""Offline tests for WhatsApp + Claude integration (uutils.whatsapp_uu).

All tests run completely offline without network requests, real credentials,
or LLM provider SDKs.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from uutils.whatsapp_uu import (
    DEFAULT_SYSTEM_PROMPT,
    WhatsAppClaudeBot,
    _extract_messages,
    _normalize_phone,
    _verify_webhook_signature,
    clauded_reply,
    create_whatsapp_webhook_app,
    mark_as_read,
    run_whatsapp_bot,
    send_whatsapp_message,
    send_whatsapp_template,
)


# ── Phone normalization tests ─────────────────────────────────────────

def test_phone_normalization():
    """Test phone normalization with valid formats and edge cases."""
    cases = [
        ("15550000001", "+15550000001"),
        ("+15550000001", "+15550000001"),
        ("  +44 20 7946 0000  ", "+442079460000"),
        ("whatsapp:+15550000001", "+15550000001"),
        ("+1 (555) 000-0001", "+15550000001"),
        ("+49.30.1234.5678", "+493012345678"),
    ]
    for inp, expected in cases:
        assert _normalize_phone(inp) == expected

    invalid_cases = ["", "   ", "abc", "+", "+1555+0000001", "1555-abc"]
    for inp in invalid_cases:
        with pytest.raises(ValueError):
            _normalize_phone(inp)


# ── Allowlist rejection tests ─────────────────────────────────────────

def test_allowlist_rejection():
    """Messages from numbers not in allowed_phones must be ignored."""
    allowed = "+15550000001"
    rejected = "+15550000002"

    bot = WhatsAppClaudeBot(dry_run=True, allowed_phones={allowed})

    # Allowed number should succeed
    reply_allowed = bot.generate_reply(allowed, "Hello!")
    assert reply_allowed is not None
    assert "[DRY-RUN]" in reply_allowed
    assert len(bot.get_history(allowed)) == 2  # user + assistant

    # Normalized variant of allowed number should also succeed
    reply_variant = bot.generate_reply("15550000001", "Hello again")
    assert reply_variant is not None

    # Number not in allowlist must be ignored
    reply_rejected = bot.generate_reply(rejected, "Hello?")
    assert reply_rejected is None
    assert len(bot.get_history(rejected)) == 0  # not recorded in history

    # generate_reply_and_send should also return None for rejected numbers
    send_rejected = bot.generate_reply_and_send(rejected, "Ignored message")
    assert send_rejected is None


def test_allowlist_default_ignores_everyone():
    """By default (allowed_phones=None or empty), bot ignores all messages."""
    bot_none = WhatsAppClaudeBot(dry_run=True, allowed_phones=None)
    assert bot_none.generate_reply("+15550000001", "Test") is None

    bot_empty = WhatsAppClaudeBot(dry_run=True, allowed_phones=set())
    assert bot_empty.generate_reply("+15550000001", "Test") is None


# ── Rate limit tests ──────────────────────────────────────────────────

def test_rate_limit(monkeypatch: pytest.MonkeyPatch):
    """Verify rate limit per phone number within a rolling hour."""
    phone = "+15550000001"
    bot = WhatsAppClaudeBot(
        dry_run=True,
        allowed_phones={phone},
        max_replies_per_hour=3,
    )

    fake_now = 10000.0

    # First 3 replies should succeed
    for i in range(3):
        fake_now += 10.0
        monkeypatch.setattr(time, "time", lambda t=fake_now: t)
        reply = bot.generate_reply(phone, f"Message {i}")
        assert reply is not None

    # 4th reply within the same hour should be rate-limited
    fake_now += 10.0
    monkeypatch.setattr(time, "time", lambda t=fake_now: t)
    reply_blocked = bot.generate_reply(phone, "Message 4")
    assert reply_blocked is None

    # Fast forward past 1 hour (3601 seconds later)
    fake_now += 3601.0
    monkeypatch.setattr(time, "time", lambda t=fake_now: t)
    reply_unblocked = bot.generate_reply(phone, "Message 5")
    assert reply_unblocked is not None


def test_rate_limit_per_phone_isolation(monkeypatch: pytest.MonkeyPatch):
    """Rate limits must be isolated per phone number."""
    phone1 = "+15550000001"
    phone2 = "+15550000002"
    bot = WhatsAppClaudeBot(
        dry_run=True,
        allowed_phones={phone1, phone2},
        max_replies_per_hour=2,
    )

    fake_now = 1000.0
    monkeypatch.setattr(time, "time", lambda: fake_now)

    assert bot.generate_reply(phone1, "P1-1") is not None
    assert bot.generate_reply(phone1, "P1-2") is not None
    assert bot.generate_reply(phone1, "P1-3") is None  # P1 exhausted

    # P2 should still have its quota
    assert bot.generate_reply(phone2, "P2-1") is not None
    assert bot.generate_reply(phone2, "P2-2") is not None
    assert bot.generate_reply(phone2, "P2-3") is None  # P2 exhausted


# ── Dry-run makes no network or subprocess calls ──────────────────────

def test_dry_run_makes_no_network_or_subprocess_calls(monkeypatch: pytest.MonkeyPatch):
    """Calling the public API with default/dry-run arguments makes no network or subprocess calls."""

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Prohibited call executed during dry run!")

    monkeypatch.setattr("requests.post", forbidden)
    monkeypatch.setattr("requests.get", forbidden)
    monkeypatch.setattr("requests.request", forbidden)
    monkeypatch.setattr("subprocess.run", forbidden)

    # 1. send_whatsapp_message defaults to dry_run=False for backwards compatibility;
    # with explicit dry_run=True, it makes no network calls.
    res_msg = send_whatsapp_message("+15550000001", "Dry run message", dry_run=True)
    assert res_msg is None

    # 2. send_whatsapp_template defaults to dry_run=False for backwards compatibility;
    # with explicit dry_run=True, it makes no network calls.
    res_tmpl = send_whatsapp_template("+15550000001", "hello_world", dry_run=True)
    assert res_tmpl is None

    # 3. mark_as_read defaults to dry_run=True
    res_read = mark_as_read("wamid.fake123")
    assert res_read is None

    # 4. WhatsAppClaudeBot defaults to dry_run=True
    bot = WhatsAppClaudeBot(allowed_phones={"+15550000001"})
    assert bot.dry_run is True

    # 5. bot.generate_reply in dry run
    reply = bot.generate_reply("+15550000001", "Hello bot")
    assert reply is not None
    assert "[DRY-RUN]" in reply

    # 6. bot.generate_reply_and_send in dry run
    reply_sent = bot.generate_reply_and_send("+15550000001", "Hello bot send")
    assert reply_sent is not None
    assert "[DRY-RUN]" in reply_sent


# ── clauded_reply CLI tests ───────────────────────────────────────────

def test_clauded_reply_builds_right_command(monkeypatch: pytest.MonkeyPatch):
    """clauded_reply invokes ['clauded', '-p', '--tools', '', '--strict-mcp-config']
    with prompt on stdin and strips stdout."""
    recorded_calls: list[dict[str, Any]] = []

    class FakeCompletedProcess:
        returncode = 0
        stdout = "  Here is Claude's helpful reply. \n"
        stderr = ""

    def fake_run(cmd: list[str], **kwargs: Any) -> FakeCompletedProcess:
        recorded_calls.append({"cmd": cmd, **kwargs})
        return FakeCompletedProcess()

    monkeypatch.setattr("subprocess.run", fake_run)

    prompt = "Test prompt text"
    result = clauded_reply(prompt, timeout=120)

    assert result == "Here is Claude's helpful reply."
    assert len(recorded_calls) == 1
    call = recorded_calls[0]
    assert call["cmd"] == ["clauded", "-p", "--tools", "", "--strict-mcp-config"]
    assert call["input"] == prompt
    assert call["capture_output"] is True
    assert call["text"] is True
    assert call["timeout"] == 120

    # Test overridable clauded_bin parameter
    recorded_calls.clear()
    custom_bin = "/usr/local/bin/custom_clauded"
    result_custom = clauded_reply(prompt, timeout=60, clauded_bin=custom_bin)
    assert result_custom == "Here is Claude's helpful reply."
    assert len(recorded_calls) == 1
    call_custom = recorded_calls[0]
    assert call_custom["cmd"] == [custom_bin, "-p", "--tools", "", "--strict-mcp-config"]
    assert call_custom["input"] == prompt
    assert call_custom["timeout"] == 60


def test_clauded_reply_error_handling(monkeypatch: pytest.MonkeyPatch):
    """clauded_reply raises RuntimeError on non-zero exit code."""

    class FakeFailedProcess:
        returncode = 127
        stdout = ""
        stderr = "command not found: clauded"

    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: FakeFailedProcess())

    with pytest.raises(RuntimeError) as exc_info:
        clauded_reply("Some prompt")

    assert "exit code 127" in str(exc_info.value)
    assert "command not found: clauded" in str(exc_info.value)


# ── Webhook message extraction tests ──────────────────────────────────

def test_extract_messages():
    """Test extracting text messages and contact names from Meta webhook payloads."""
    payload = {
        "entry": [{
            "changes": [{
                "value": {
                    "messages": [
                        {
                            "from": "15550000001",
                            "id": "wamid.msg1",
                            "type": "text",
                            "text": {"body": "First message"},
                            "timestamp": "1713100001",
                        },
                        {
                            "from": "15550000002",
                            "id": "wamid.msg2",
                            "type": "image",
                            "image": {"id": "img123"},
                        },
                        {
                            "from": "15550000003",
                            "id": "wamid.msg3",
                            "type": "text",
                            "text": {"body": "Second message"},
                            "timestamp": "1713100003",
                        },
                    ],
                    "contacts": [
                        {"wa_id": "15550000001", "profile": {"name": "Alice"}},
                        {"wa_id": "15550000003", "profile": {"name": "Charlie"}},
                    ],
                },
            }],
        }],
    }

    msgs = _extract_messages(payload)
    assert len(msgs) == 2

    assert msgs[0]["from_phone"] == "+15550000001"
    assert msgs[0]["message_id"] == "wamid.msg1"
    assert msgs[0]["text"] == "First message"
    assert msgs[0]["contact_name"] == "Alice"

    assert msgs[1]["from_phone"] == "+15550000003"
    assert msgs[1]["message_id"] == "wamid.msg3"
    assert msgs[1]["text"] == "Second message"
    assert msgs[1]["contact_name"] == "Charlie"


def test_extract_messages_empty_or_malformed():
    """Empty or malformed payloads should return an empty list gracefully."""
    assert _extract_messages({}) == []
    assert _extract_messages({"entry": []}) == []
    assert _extract_messages({"entry": [{"changes": []}]}) == []


# ── Webhook signature verification tests ──────────────────────────────

def test_signature_verification():
    """Test HMAC-SHA256 signature verification."""
    secret = "test_meta_app_secret"
    payload = b'{"entry":[{"id":"123"}]}'

    # Compute valid signature
    sig_hash = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    valid_sig = f"sha256={sig_hash}"

    assert _verify_webhook_signature(payload, valid_sig, secret) is True

    # Tampered signature
    assert _verify_webhook_signature(payload, "sha256=invalidhash123", secret) is False

    # Tampered payload
    assert _verify_webhook_signature(b'{"different": true}', valid_sig, secret) is False

    # Missing signature with secret configured
    assert _verify_webhook_signature(payload, "", secret) is False

    # No secret configured -> skips verification (returns True)
    assert _verify_webhook_signature(payload, "", "") is True
    assert _verify_webhook_signature(payload, "sha256=any", "") is True


# ── Bot conversation management tests ─────────────────────────────────

def test_bot_conversation_management():
    """Test history storage, contact isolation, and trimming."""
    bot = WhatsAppClaudeBot(dry_run=True, max_history=4)
    p1 = "+15550000001"
    p2 = "+15550000002"

    bot.add_message(p1, "user", "msg 1")
    bot.add_message(p1, "assistant", "reply 1")
    bot.add_message(p1, "user", "msg 2")

    bot.add_message(p2, "user", "other msg")

    # Isolation
    assert len(bot.get_history(p1)) == 3
    assert len(bot.get_history(p2)) == 1

    # Trimming when exceeding max_history=4
    bot.add_message(p1, "assistant", "reply 2")
    bot.add_message(p1, "user", "msg 3")
    history_p1 = bot.get_history(p1)
    assert len(history_p1) == 4
    assert history_p1[0]["content"] == "reply 1"  # oldest msg 1 was trimmed

    # Clear history
    bot.clear_history(p1)
    assert len(bot.get_history(p1)) == 0
    assert len(bot.get_history(p2)) == 1


def test_custom_pluggable_reply_fn():
    """Test providing a custom reply_fn to WhatsAppClaudeBot outside dry run."""
    def custom_generator(prompt: str) -> str:
        return f"CUSTOM_REPLY: length={len(prompt)}"

    bot = WhatsAppClaudeBot(
        dry_run=False,
        reply_fn=custom_generator,
        allowed_phones={"+15550000001"},
    )
    reply = bot.generate_reply("+15550000001", "Hello test")
    assert reply is not None
    assert reply.startswith("CUSTOM_REPLY:")
    assert len(bot.get_history("+15550000001")) == 2


def test_run_whatsapp_bot_safety_guard():
    """run_whatsapp_bot must refuse to start with dry_run=False without allowed_phones."""
    with pytest.raises(ValueError) as exc1:
        run_whatsapp_bot(dry_run=False, allowed_phones=None)
    assert "allowed_phones" in str(exc1.value)

    with pytest.raises(ValueError) as exc2:
        run_whatsapp_bot(dry_run=False, allowed_phones=set())
    assert "allowed_phones" in str(exc2.value)

    with pytest.raises(ValueError) as exc3:
        run_whatsapp_bot(dry_run=False, allowed_phones=[""])
    assert "allowed_phones" in str(exc3.value)


def test_bot_with_clauded_bin(monkeypatch: pytest.MonkeyPatch):
    """WhatsAppClaudeBot passes custom clauded_bin to clauded_reply."""
    recorded_calls: list[dict[str, Any]] = []

    class FakeCompletedProcess:
        returncode = 0
        stdout = "bot reply"
        stderr = ""

    def fake_run(cmd: list[str], **kwargs: Any) -> FakeCompletedProcess:
        recorded_calls.append({"cmd": cmd, **kwargs})
        return FakeCompletedProcess()

    monkeypatch.setattr("subprocess.run", fake_run)

    bot = WhatsAppClaudeBot(
        dry_run=False,
        allowed_phones={"+15550000001"},
        clauded_bin="/opt/bin/my_clauded",
    )
    reply = bot.generate_reply("+15550000001", "Hello")
    assert reply == "bot reply"
    assert len(recorded_calls) == 1
    assert recorded_calls[0]["cmd"][0] == "/opt/bin/my_clauded"


def test_run_whatsapp_bot_auth_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """run_whatsapp_bot(dry_run=False) must refuse to start unless allowed_phones,
    verify_token, and app_secret are all non-empty."""
    allowed = {"+15550000001"}

    # Refuse without verify_token
    with pytest.raises(ValueError) as exc_vt:
        run_whatsapp_bot(
            dry_run=False,
            allowed_phones=allowed,
            verify_token="",
            app_secret="sec123",
        )
    assert "verify_token" in str(exc_vt.value)

    # Refuse without app_secret
    with pytest.raises(ValueError) as exc_as:
        run_whatsapp_bot(
            dry_run=False,
            allowed_phones=allowed,
            verify_token="tok123",
            app_secret="",
        )
    assert "app_secret" in str(exc_as.value)

    # Monkeypatch Flask.run so server doesn't actually block
    run_called: list[dict[str, Any]] = []
    from flask import Flask
    monkeypatch.setattr(Flask, "run", lambda self, **kwargs: run_called.append(kwargs))

    # Succeeds with explicit arguments (default host is 127.0.0.1)
    run_whatsapp_bot(
        dry_run=False,
        allowed_phones=allowed,
        verify_token="tok123",
        app_secret="sec123",
    )
    assert len(run_called) == 1
    assert run_called[0]["host"] == "127.0.0.1"

    # Succeeds when loaded from config file
    cfg_data = {
        "provider": "meta",
        "access_token": "fake_token",
        "phone_number_id": "12345",
        "verify_token": "cfg_tok",
        "app_secret": "cfg_sec",
    }
    cfg_file = tmp_path / "whatsapp_config.json"
    cfg_file.write_text(json.dumps(cfg_data))

    run_whatsapp_bot(
        dry_run=False,
        allowed_phones=allowed,
        config_file=str(cfg_file),
    )
    assert len(run_called) == 2


# ── Flask-dependent webhook app tests ─────────────────────────────────

def test_webhook_verification_endpoint():
    """Test Flask webhook GET challenge-response verification."""
    flask = pytest.importorskip("flask")  # skip cleanly if Flask is not installed

    bot = WhatsAppClaudeBot(dry_run=True)
    app = create_whatsapp_webhook_app(
        bot=bot,
        verify_token="test_secret_token",
        auto_reply=False,
    )

    with app.test_client() as client:
        # Correct token and mode
        res_ok = client.get(
            "/webhook?hub.mode=subscribe&hub.verify_token=test_secret_token&hub.challenge=challenge123"
        )
        assert res_ok.status_code == 200
        assert res_ok.data.decode() == "challenge123"

        # Wrong token
        res_bad_token = client.get(
            "/webhook?hub.mode=subscribe&hub.verify_token=wrong_token&hub.challenge=challenge123"
        )
        assert res_bad_token.status_code == 403

        # Health endpoint
        res_health = client.get("/health")
        assert res_health.status_code == 200
        data = res_health.get_json()
        assert data["status"] == "ok"
        assert data["dry_run"] is True


def test_webhook_post_signature_and_background_processing():
    """Test Flask webhook POST with signature verification and background threading."""
    flask = pytest.importorskip("flask")

    secret = "my_app_secret"
    phone = "+15550000001"
    received: list[dict[str, Any]] = []

    def callback(from_phone: str, text: str, reply: str | None) -> None:
        received.append({"phone": from_phone, "text": text, "reply": reply})

    bot = WhatsAppClaudeBot(dry_run=True, allowed_phones={phone})
    app = create_whatsapp_webhook_app(
        bot=bot,
        verify_token="token123",
        app_secret=secret,
        auto_reply=True,
        auto_mark_read=True,
        on_message=callback,
        background=True,
    )

    payload_dict = {
        "entry": [{
            "changes": [{
                "value": {
                    "messages": [{
                        "from": "15550000001",
                        "id": "wamid.test999",
                        "type": "text",
                        "text": {"body": "Webhook test message"},
                    }],
                    "contacts": [{"wa_id": "15550000001", "profile": {"name": "Alice"}}],
                },
            }],
        }],
    }
    raw_payload = json.dumps(payload_dict).encode("utf-8")
    sig_hash = hmac.new(secret.encode("utf-8"), raw_payload, hashlib.sha256).hexdigest()
    valid_sig = f"sha256={sig_hash}"

    with app.test_client() as client:
        # Invalid signature
        res_invalid = client.post(
            "/webhook",
            data=raw_payload,
            headers={"X-Hub-Signature-256": "sha256=invalidsignature", "Content-Type": "application/json"},
        )
        assert res_invalid.status_code == 403

        # Valid signature
        res_valid = client.post(
            "/webhook",
            data=raw_payload,
            headers={"X-Hub-Signature-256": valid_sig, "Content-Type": "application/json"},
        )
        assert res_valid.status_code == 200

    # Wait for the background thread to finish
    if hasattr(app, "_last_thread") and app._last_thread is not None:
        app._last_thread.join(timeout=3.0)

    assert len(received) == 1
    assert received[0]["phone"] == phone
    assert received[0]["text"] == "Webhook test message"
    assert received[0]["reply"] is not None
    assert "[DRY-RUN]" in received[0]["reply"]
