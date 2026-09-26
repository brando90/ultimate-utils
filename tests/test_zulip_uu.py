"""Offline tests for Zulip automation (uutils.zulip_uu).

All tests run strictly offline by monkeypatching network calls and using tmp_path.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests

from uutils.zulip_uu import ZulipClient, main, notify


# ── Helper for mocking requests.Response ────────────────────────────────


def _make_mock_response(status_code: int = 200, json_data: dict | None = None, text: str = "") -> requests.Response:
    resp = requests.Response()
    resp.status_code = status_code
    if json_data is not None:
        resp._content = json.dumps(json_data).encode("utf-8")
        resp.headers["Content-Type"] = "application/json"
    else:
        resp._content = text.encode("utf-8")
    return resp


# ── 1. Dry-run safety tests (no network calls, no file reads) ───────────


def test_dry_run_default_makes_no_network_call(monkeypatch):
    """Prove that default API calls execute purely offline and read no credentials."""

    def forbidden_call(*args, **kwargs):
        raise RuntimeError("Forbidden: network call attempted during dry-run!")

    monkeypatch.setattr(requests, "request", forbidden_call)
    monkeypatch.setattr(requests, "get", forbidden_call)
    monkeypatch.setattr(requests, "post", forbidden_call)

    # 1. notify convenience function with defaults
    res_notify = notify("Build completed", stream="general", topic="ci")
    assert res_notify["result"] == "success"
    assert res_notify["dry_run"] is True

    # 2. ZulipClient default initialization
    client = ZulipClient()
    assert client.dry_run is True

    # 3. from_zuliprc with dry_run=True must NOT read or require file
    client_rc = ZulipClient.from_zuliprc("/nonexistent/keys/zuliprc", dry_run=True)
    assert client_rc.dry_run is True

    # 4. send_message
    res_msg = client.send_message(stream="general", topic="deploy", content="v1.0.0")
    assert res_msg["result"] == "success"
    assert res_msg["dry_run"] is True

    # 5. send_dm
    res_dm = client.send_dm(user_email="example@example.com", content="Secret ping")
    assert res_dm["result"] == "success"
    assert res_dm["dry_run"] is True

    # 6. get_messages
    res_get = client.get_messages(stream="general", topic="deploy")
    assert res_get["result"] == "success"
    assert res_get["dry_run"] is True
    assert res_get["messages"] == []

    # 7. get_streams
    res_streams = client.get_streams()
    assert res_streams["result"] == "success"
    assert res_streams["dry_run"] is True
    assert res_streams["streams"] == []

    # 8. upload_file with nonexistent file path in dry-run
    res_upload = client.upload_file("/nonexistent/file.txt")
    assert isinstance(res_upload, str)
    assert "dry_run" in res_upload

    # 9. get_unread_count
    unread = client.get_unread_count()
    assert isinstance(unread, dict)
    assert len(unread) == 0

    # 10. CLI with default dry-run
    cli_rc = main(["send", "--stream", "general", "--topic", "t", "--content", "c"])
    assert cli_rc == 0


# ── 2. Zuliprc parsing tests ────────────────────────────────────────────


def test_zuliprc_parsing(tmp_path: Path):
    """Test parsing standard [api] section in zuliprc."""
    fake_rc = tmp_path / "zuliprc"
    fake_rc.write_text(
        "[api]\n"
        "email=bot@example.com\n"
        "key=xoxb-fake-key-12345\n"
        "site=https://example.zulipchat.com\n"
    )

    client = ZulipClient.from_zuliprc(str(fake_rc), dry_run=False)
    assert client.email == "bot@example.com"
    assert client.api_key == "xoxb-fake-key-12345"
    assert client.site == "https://example.zulipchat.com"
    assert client.dry_run is False

    # Missing file when dry_run=False
    with pytest.raises(FileNotFoundError):
        ZulipClient.from_zuliprc(str(tmp_path / "does_not_exist"), dry_run=False)

    # Missing [api] section
    bad_rc = tmp_path / "bad_zuliprc"
    bad_rc.write_text("[other]\nemail=bot@example.com\n")
    with pytest.raises(ValueError, match=r"Missing \[api\] section"):
        ZulipClient.from_zuliprc(str(bad_rc), dry_run=False)


# ── 3. Send message payload and authentication ──────────────────────────


def test_send_message_payload_and_auth(monkeypatch):
    """Verify correct URL, auth, and data payload for send_message."""
    captured: dict[str, object] = {}

    def mock_request(method, url, auth=None, timeout=None, **kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["auth"] = auth
        captured["timeout"] = timeout
        captured["data"] = kwargs.get("data")
        return _make_mock_response(200, {"result": "success", "id": 101})

    monkeypatch.setattr(requests, "request", mock_request)

    client = ZulipClient(
        site="https://example.zulipchat.com",
        email="bot@example.com",
        api_key="fake-secret-key",
        dry_run=False,
    )
    res = client.send_message(stream="releases", topic="v2.0", content="Release notes here")

    assert captured["method"] == "POST"
    assert captured["url"] == "https://example.zulipchat.com/api/v1/messages"
    assert captured["auth"] == ("bot@example.com", "fake-secret-key")
    assert captured["data"] == {
        "type": "stream",
        "to": "releases",
        "topic": "v2.0",
        "content": "Release notes here",
    }
    assert res == {"result": "success", "id": 101}


# ── 4. Send direct message payload ──────────────────────────────────────


def test_send_dm_payload(monkeypatch):
    """Verify correct format for direct message recipients."""
    captured: dict[str, object] = {}

    def mock_request(method, url, auth=None, timeout=None, **kwargs):
        captured["data"] = kwargs.get("data")
        return _make_mock_response(200, {"result": "success", "id": 102})

    monkeypatch.setattr(requests, "request", mock_request)

    client = ZulipClient(
        site="https://example.zulipchat.com",
        email="bot@example.com",
        api_key="fake-key",
        dry_run=False,
    )

    # Single recipient email as string
    client.send_dm(user_email="alice@example.com", content="Direct ping")
    assert captured["data"] == {
        "type": "direct",
        "to": json.dumps(["alice@example.com"]),
        "content": "Direct ping",
    }

    # Multiple recipient emails as list
    client.send_dm(user_email=["alice@example.com", "bob@example.com"], content="Group ping")
    assert captured["data"] == {
        "type": "direct",
        "to": json.dumps(["alice@example.com", "bob@example.com"]),
        "content": "Group ping",
    }


# ── 5. Get messages parameters and narrow ───────────────────────────────


def test_get_messages_parameters(monkeypatch):
    """Verify GET /messages URL, anchor, limits, and narrow filter."""
    captured: dict[str, object] = {}

    def mock_request(method, url, auth=None, timeout=None, **kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["params"] = kwargs.get("params")
        return _make_mock_response(
            200,
            {"result": "success", "messages": [{"id": 1, "content": "hello"}]},
        )

    monkeypatch.setattr(requests, "request", mock_request)

    client = ZulipClient(
        site="https://example.zulipchat.com",
        email="bot@example.com",
        api_key="fake-key",
        dry_run=False,
    )

    # With stream and topic
    res = client.get_messages(stream="engineering", topic="backend", limit=25)
    assert captured["method"] == "GET"
    assert captured["url"] == "https://example.zulipchat.com/api/v1/messages"
    params = captured["params"]
    assert params["anchor"] == "newest"
    assert params["num_before"] == 25
    assert params["num_after"] == 0
    expected_narrow = [
        {"operator": "stream", "operand": "engineering"},
        {"operator": "topic", "operand": "backend"},
    ]
    assert json.loads(params["narrow"]) == expected_narrow
    assert res["messages"] == [{"id": 1, "content": "hello"}]

    # Without filter (stream=None, topic=None)
    client.get_messages(limit=10)
    assert "narrow" not in captured["params"]
    assert captured["params"]["num_before"] == 10


# ── 6. Get streams ──────────────────────────────────────────────────────


def test_get_streams(monkeypatch):
    """Verify GET /streams request."""
    captured: dict[str, object] = {}

    def mock_request(method, url, auth=None, timeout=None, **kwargs):
        captured["method"] = method
        captured["url"] = url
        return _make_mock_response(
            200,
            {"result": "success", "streams": [{"stream_id": 1, "name": "general"}]},
        )

    monkeypatch.setattr(requests, "request", mock_request)

    client = ZulipClient(
        site="https://example.zulipchat.com",
        email="bot@example.com",
        api_key="fake-key",
        dry_run=False,
    )
    res = client.get_streams()
    assert captured["method"] == "GET"
    assert captured["url"] == "https://example.zulipchat.com/api/v1/streams"
    assert res["streams"] == [{"stream_id": 1, "name": "general"}]


# ── 7. Upload file ──────────────────────────────────────────────────────


def test_upload_file(tmp_path: Path, monkeypatch):
    """Verify file upload POST /user_uploads."""
    fake_file = tmp_path / "report.pdf"
    fake_file.write_bytes(b"%PDF-1.4 fake pdf data")

    captured: dict[str, object] = {}

    def mock_request(method, url, auth=None, timeout=None, **kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["files"] = kwargs.get("files")
        return _make_mock_response(
            200,
            {"result": "success", "uri": "/user_uploads/1/abc/report.pdf"},
        )

    monkeypatch.setattr(requests, "request", mock_request)

    client = ZulipClient(
        site="https://example.zulipchat.com",
        email="bot@example.com",
        api_key="fake-key",
        dry_run=False,
    )
    uri = client.upload_file(fake_file)
    assert captured["method"] == "POST"
    assert captured["url"] == "https://example.zulipchat.com/api/v1/user_uploads"
    assert "filename" in captured["files"]
    assert uri == "/user_uploads/1/abc/report.pdf"

    # Missing file when dry_run=False raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        client.upload_file(tmp_path / "nonexistent.txt")


# ── 8. Get unread count summary ─────────────────────────────────────────


def test_get_unread_count(monkeypatch):
    """Verify aggregation of unread messages by stream and topic."""
    mock_messages = [
        {"display_recipient": "general", "subject": "welcome"},
        {"display_recipient": "general", "subject": "welcome"},
        {"display_recipient": "general", "subject": "rules"},
        {"display_recipient": "engineering", "subject": "backend"},
        {"display_recipient": [{"email": "user1@example.com"}], "subject": "", "type": "direct"},
    ]

    def mock_request(method, url, auth=None, timeout=None, **kwargs):
        params = kwargs.get("params", {})
        assert "narrow" in params
        assert json.loads(params["narrow"]) == [{"operator": "is", "operand": "unread"}]
        return _make_mock_response(200, {"result": "success", "messages": mock_messages})

    monkeypatch.setattr(requests, "request", mock_request)

    client = ZulipClient(
        site="https://example.zulipchat.com",
        email="bot@example.com",
        api_key="fake-key",
        dry_run=False,
    )
    counts = client.get_unread_count()
    assert counts == {
        "general": {"welcome": 2, "rules": 1},
        "engineering": {"backend": 1},
        "direct": {"": 1},
    }


# ── 9. Error on non-success ─────────────────────────────────────────────


def test_error_on_non_success(monkeypatch):
    """Verify that result != 'success' raises a clear RuntimeError."""
    def mock_error_400(method, url, auth=None, timeout=None, **kwargs):
        return _make_mock_response(
            400,
            {"result": "error", "msg": "Stream does not exist", "code": "STREAM_DOES_NOT_EXIST"},
        )

    monkeypatch.setattr(requests, "request", mock_error_400)

    client = ZulipClient(
        site="https://example.zulipchat.com",
        email="bot@example.com",
        api_key="fake-key",
        dry_run=False,
    )

    with pytest.raises(RuntimeError, match=r"Zulip API error \(STREAM_DOES_NOT_EXIST\): Stream does not exist"):
        client.send_message(stream="bad_stream", topic="t", content="c")

    def mock_error_200(method, url, auth=None, timeout=None, **kwargs):
        return _make_mock_response(
            200,
            {"result": "error", "msg": "Invalid API key", "code": "UNAUTHORIZED"},
        )

    monkeypatch.setattr(requests, "request", mock_error_200)

    with pytest.raises(RuntimeError, match=r"Zulip API error \(UNAUTHORIZED\): Invalid API key"):
        client.get_streams()


# ── 10. Notify convenience function ─────────────────────────────────────


def test_notify(tmp_path: Path, monkeypatch):
    """Verify notify() convenience wrapper for both dry-run and live modes."""
    # Dry-run
    res_dry = notify("Test ping", stream="dev", topic="ci", dry_run=True)
    assert res_dry["result"] == "success"
    assert res_dry["dry_run"] is True

    # Live mode with mock
    fake_rc = tmp_path / "zuliprc"
    fake_rc.write_text(
        "[api]\n"
        "email=bot@example.com\n"
        "key=fake-key\n"
        "site=https://example.zulipchat.com\n"
    )

    captured: dict[str, object] = {}

    def mock_request(method, url, auth=None, timeout=None, **kwargs):
        captured["data"] = kwargs.get("data")
        return _make_mock_response(200, {"result": "success", "id": 999})

    monkeypatch.setattr(requests, "request", mock_request)

    res_live = notify(
        "Live alert",
        stream="alerts",
        topic="pager",
        dry_run=False,
        zuliprc_file=str(fake_rc),
    )
    assert res_live == {"result": "success", "id": 999}
    assert captured["data"] == {
        "type": "stream",
        "to": "alerts",
        "topic": "pager",
        "content": "Live alert",
    }


# ── 11. CLI tests ───────────────────────────────────────────────────────


def test_cli(tmp_path: Path, monkeypatch, capsys):
    """Verify CLI send subcommand in dry-run and with --send."""
    # 1. CLI dry-run by default
    rc = main(["send", "--stream", "general", "--topic", "test", "--content", "dry test"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "[DRY-RUN]" in captured.out

    # 2. CLI with --send and mock
    fake_rc = tmp_path / "zuliprc"
    fake_rc.write_text(
        "[api]\n"
        "email=bot@example.com\n"
        "key=fake-key\n"
        "site=https://example.zulipchat.com\n"
    )

    called = False

    def mock_request(method, url, auth=None, timeout=None, **kwargs):
        nonlocal called
        called = True
        return _make_mock_response(200, {"result": "success", "id": 777})

    monkeypatch.setattr(requests, "request", mock_request)

    rc = main([
        "send",
        "--stream", "general",
        "--topic", "test",
        "--content", "real test",
        "--send",
        "--zuliprc", str(fake_rc),
    ])
    assert rc == 0
    assert called is True
    captured = capsys.readouterr()
    assert "Message sent successfully: id=777" in captured.out
