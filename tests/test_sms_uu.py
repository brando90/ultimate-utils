"""Offline tests for uutils.sms_uu.

No network and no ~/keys/ files. ``requests`` and ``subprocess`` are patched.
The real ``uutils/__init__.py`` imports third-party packages that are not
installed in the test environment, so this file registers a lightweight
package stub before loading ``uutils.sms_uu``.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import requests

from uutils.sms_uu import (
    AUTOREMOTE_SEND_URL,
    DEFAULT_AUTOREMOTE_KEY,
    DEFAULT_TWILIO_CREDENTIALS,
    SMSClient,
    TWILIO_MESSAGES_URL,
    main,
    normalize_phone,
)


_TO = "+15550000000"
_FROM = "+15551112222"
_SELF = "+15553334444"
_SID = "ACfakeaccountsid000000000000000000"
_TOKEN = "fake-auth-token"


class _Response:
    def __init__(self, payload: dict | None = None, *, text: str = "", status_code: int = 201):
        self.status_code = status_code
        self.ok = status_code < 400
        self._payload = payload
        self.text = text if payload is None else json.dumps(payload)

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any real HTTP or subprocess use fails the test."""

    def boom(*args, **kwargs):
        raise AssertionError(f"unexpected call args={args!r} kwargs={kwargs!r}")

    monkeypatch.setattr(requests, "post", boom)
    monkeypatch.setattr(requests, "get", boom)
    monkeypatch.setattr(requests, "request", boom)
    monkeypatch.setattr(requests.Session, "request", boom)
    monkeypatch.setattr(subprocess, "run", boom)


def _twilio_file(tmp_path: Path, *, self_number: str | None = _SELF, from_number: str = _FROM) -> Path:
    payload: dict[str, str] = {
        "account_sid": _SID,
        "auth_token": _TOKEN,
        "from_number": from_number,
    }
    if self_number is not None:
        payload["self_number"] = self_number
    path = tmp_path / "twilio_credentials.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _key_file(tmp_path: Path, key: str = "fake-autoremote-key") -> Path:
    path = tmp_path / "tasker_autoremote_key.txt"
    path.write_text(key + "\n", encoding="utf-8")
    return path


def test_module_docstring_covers_setup():
    doc = sys.modules["uutils.sms_uu"].__doc__ or ""
    assert DEFAULT_TWILIO_CREDENTIALS in doc
    assert DEFAULT_AUTOREMOTE_KEY in doc
    assert "chmod 600" in doc
    assert "$1" in doc
    assert "Google Messages" in doc
    assert "sms=:=" in doc
    assert "dry-run" in doc.lower() or "dry_run" in doc


def test_default_dry_run_makes_no_network_call(monkeypatch: pytest.MonkeyPatch):
    """Public constructors default to dry-run and never touch the network."""
    _block_network(monkeypatch)
    twilio = SMSClient.from_twilio()
    auto = SMSClient.from_autoremote()
    assert twilio.dry_run is True
    assert auto.dry_run is True

    twilio_result = twilio.send_sms(_TO, "hello from twilio")
    auto_result = auto.send_sms("1 (555) 000-0000", "hello from the phone")

    assert twilio_result == {
        "dry_run": True,
        "backend": "twilio",
        "to": _TO,
        "message": "hello from twilio",
    }
    assert auto_result == {
        "dry_run": True,
        "backend": "autoremote",
        "to": _TO,
        "message": "hello from the phone",
    }


def test_dry_run_does_not_read_credential_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys):
    _block_network(monkeypatch)
    secret = _twilio_file(tmp_path)
    key = _key_file(tmp_path, key="LEAKME-KEY")
    read: list[Path] = []
    real_read_text = Path.read_text

    def spy(self: Path, *args, **kwargs):
        read.append(self)
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", spy)

    missing = tmp_path / "does-not-exist.json"
    preview = SMSClient.from_twilio(credentials_file=missing).send_sms(_TO, "no file needed")
    reminder = SMSClient.from_autoremote(
        key_file=key, self_number=_SELF,
    ).send_self_reminder("ping")
    also = SMSClient.from_twilio(credentials_file=secret).send_sms(_TO, "file exists but unread")

    assert preview["dry_run"] is True
    assert reminder == {
        "dry_run": True,
        "backend": "autoremote",
        "to": _SELF,
        "message": "ping",
    }
    assert also["backend"] == "twilio"
    assert read == []
    output = capsys.readouterr().out
    assert "LEAKME-KEY" not in output
    assert _TOKEN not in output
    assert "[DRY-RUN] SMS via twilio" in output
    assert "[DRY-RUN] SMS via autoremote" in output


def test_twilio_request_url_auth_and_form(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    calls: list[dict] = []

    def fake_post(url, auth=None, data=None, timeout=None, **kwargs):
        calls.append({"url": url, "auth": auth, "data": data, "timeout": timeout, "kwargs": kwargs})
        return _Response({"sid": "SMfake", "status": "queued"})

    def fake_get(*args, **kwargs):
        raise AssertionError("Twilio send must POST, not GET")

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(requests, "get", fake_get)

    creds = _twilio_file(tmp_path, from_number="+1 (555) 111-2222")
    client = SMSClient.from_twilio(credentials_file=creds, dry_run=False)
    result = client.send_sms("1-555-000-0000", "hello world")

    assert result == {"sid": "SMfake", "status": "queued"}
    assert len(calls) == 1
    assert calls[0]["url"] == TWILIO_MESSAGES_URL.format(account_sid=_SID)
    assert calls[0]["url"] == f"https://api.twilio.com/2010-04-01/Accounts/{_SID}/Messages.json"
    assert calls[0]["auth"] == (_SID, _TOKEN)
    assert calls[0]["data"] == {"To": _TO, "From": _FROM, "Body": "hello world"}
    assert calls[0]["timeout"] == 30
    assert calls[0]["kwargs"] == {}


def test_autoremote_request_url_and_params(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    calls: list[dict] = []

    def fake_get(url, params=None, timeout=None, **kwargs):
        calls.append({"url": url, "params": params, "timeout": timeout, "kwargs": kwargs})
        return _Response(text="OK", status_code=200)

    def fake_post(*args, **kwargs):
        raise AssertionError("AutoRemote send must GET, not POST")

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(requests, "post", fake_post)

    key_path = _key_file(tmp_path, key="fake-autoremote-key")
    client = SMSClient.from_autoremote(key_file=key_path, dry_run=False)
    result = client.send_sms("+1 (555) 000-0000", "bring milk")

    assert result == {"status_code": 200, "text": "OK"}
    assert len(calls) == 1
    assert calls[0]["url"] == AUTOREMOTE_SEND_URL
    assert calls[0]["url"] == "https://autoremotejoaomgcd.appspot.com/sendmessage"
    assert calls[0]["params"] == {
        "key": "fake-autoremote-key",
        "message": "sms=:=+15550000000=:=bring milk",
    }
    assert calls[0]["timeout"] == 30


def test_send_self_reminder_without_self_number_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _block_network(monkeypatch)

    with pytest.raises(ValueError, match="self_number"):
        SMSClient.from_twilio().send_self_reminder("remember")
    with pytest.raises(ValueError, match="self_number"):
        SMSClient.from_autoremote().send_self_reminder("remember")

    # A real Twilio send still errors when the JSON has no self_number, and
    # it must not fall through into an HTTP call (requests is patched to raise).
    creds = _twilio_file(tmp_path, self_number=None)
    client = SMSClient.from_twilio(credentials_file=creds, dry_run=False)
    with pytest.raises(ValueError, match="self_number"):
        client.send_self_reminder("remember")


def test_twilio_self_reminder_uses_number_from_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    seen: dict = {}

    def fake_post(url, auth=None, data=None, timeout=None, **kwargs):
        seen["data"] = data
        return _Response({"sid": "SMself"})

    monkeypatch.setattr(requests, "post", fake_post)
    creds = _twilio_file(tmp_path, self_number="+1 (555) 333-4444")
    client = SMSClient.from_twilio(credentials_file=creds, dry_run=False)
    result = client.send_self_reminder("check the oven")
    assert result == {"sid": "SMself"}
    assert seen["data"]["To"] == _SELF
    assert seen["data"]["Body"] == "check the oven"
    assert seen["data"]["From"] == _FROM


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("+15550000000", "+15550000000"),
        ("15550000000", "+15550000000"),
        ("+1 (555) 000-0000", "+15550000000"),
        ("+1-555-000-0000", "+15550000000"),
        ("  +44 20 7946 0958  ", "+442079460958"),
        ("+12345678", "+12345678"),
        ("+" + "1" * 15, "+" + "1" * 15),
    ],
)
def test_normalize_phone_accepts_e164_shapes(raw: str, expected: str):
    assert normalize_phone(raw) == expected


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ("", "empty"),
        ("   ", "empty"),
        ("+", "empty"),
        ("---", "empty"),
        ("()", "empty"),
        ("123", "too short"),
        ("+1234567", "too short"),
        ("+" + "1" * 16, "too long"),
        ("abc", "Invalid phone number"),
        ("+1 555 000 0000 ext 9", "Invalid phone number"),
        ("++15550000000", "Invalid phone number"),
        ("1555+0000000", "Invalid phone number"),
        (None, "string"),
    ],
)
def test_normalize_phone_rejects_bad_numbers(raw, match: str):
    with pytest.raises(ValueError, match=match):
        normalize_phone(raw)  # type: ignore[arg-type]


def test_invalid_destination_does_not_read_credentials_or_call_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    _block_network(monkeypatch)
    missing = tmp_path / "missing.json"
    client = SMSClient.from_twilio(credentials_file=missing, dry_run=False)
    with pytest.raises(ValueError, match="too short"):
        client.send_sms("123", "hi")


def test_missing_credentials_file_errors_without_network(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _block_network(monkeypatch)
    client = SMSClient.from_twilio(credentials_file=tmp_path / "nope.json", dry_run=False)
    with pytest.raises(FileNotFoundError, match="twilio"):
        client.send_sms(_TO, "hi")
    auto = SMSClient.from_autoremote(key_file=tmp_path / "nope.txt", dry_run=False)
    with pytest.raises(FileNotFoundError, match="AutoRemote"):
        auto.send_sms(_TO, "hi")


def test_cli_defaults_to_dry_run(monkeypatch: pytest.MonkeyPatch, capsys):
    _block_network(monkeypatch)
    code = main(["send", "--backend", "twilio", "--to", _TO, "--message", "hello cli"])
    assert code == 0
    out = capsys.readouterr().out
    assert "[DRY-RUN] SMS via twilio to +15550000000: hello cli" in out
    assert '"dry_run": true' in out
    assert _TOKEN not in out


def test_cli_send_flag_posts_to_twilio(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys):
    calls: list[dict] = []

    def fake_post(url, auth=None, data=None, timeout=None, **kwargs):
        calls.append({"url": url, "auth": auth, "data": data})
        return _Response({"sid": "SMcli"})

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("get")))
    creds = _twilio_file(tmp_path)
    code = main([
        "send",
        "--backend", "twilio",
        "--to", "+1 (555) 000-0000",
        "--message", "from the cli",
        "--send",
        "--credentials-file", str(creds),
    ])
    assert code == 0
    assert calls[0]["auth"] == (_SID, _TOKEN)
    assert calls[0]["data"] == {"To": _TO, "From": _FROM, "Body": "from the cli"}
    assert '"sid": "SMcli"' in capsys.readouterr().out


def test_cli_execute_alias_gets_autoremote(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    calls: list[dict] = []

    def fake_get(url, params=None, timeout=None, **kwargs):
        calls.append({"url": url, "params": params})
        return _Response({"ok": True}, status_code=200)

    monkeypatch.setattr(requests, "get", fake_get)
    key_path = _key_file(tmp_path)
    code = main([
        "send",
        "--backend", "autoremote",
        "--to", _TO,
        "--message", "via execute",
        "--execute",
        "--key-file", str(key_path),
    ])
    assert code == 0
    assert calls[0]["url"] == AUTOREMOTE_SEND_URL
    assert calls[0]["params"]["key"] == "fake-autoremote-key"
    assert calls[0]["params"]["message"] == f"sms=:={_TO}=:=via execute"


def test_cli_send_without_credentials_does_not_call_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys,
):
    _block_network(monkeypatch)
    code = main([
        "send",
        "--backend", "twilio",
        "--to", _TO,
        "--message", "nope",
        "--send",
        "--credentials-file", str(tmp_path / "absent.json"),
    ])
    assert code == 2
    assert "error:" in capsys.readouterr().err


def test_cli_bad_number_is_a_usage_error(monkeypatch: pytest.MonkeyPatch, capsys):
    _block_network(monkeypatch)
    code = main(["send", "--backend", "autoremote", "--to", "12345", "--message", "hi"])
    assert code == 2
    assert "too short" in capsys.readouterr().err
