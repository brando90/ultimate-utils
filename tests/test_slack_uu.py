"""Tests for uutils.slack_uu — offline unit tests with dry-run verification."""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest
import requests

from uutils.slack_uu import SlackClient, main, notify


class MockResponse:
    """Mock requests.Response for offline testing."""

    def __init__(self, data: dict, status_code: int = 200) -> None:
        self._data = data
        self.status_code = status_code
        self.text = str(data)

    def json(self) -> dict:
        return self._data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


# ── Hard Rule: Default dry-run makes NO network or credential calls ──────────


def test_public_api_defaults_make_no_network_or_credential_calls(monkeypatch, capsys):
    """Calling the public API with default arguments must make zero network requests and read no tokens."""

    def forbidden(*args, **kwargs):
        raise AssertionError("Network access forbidden in dry-run mode!")

    monkeypatch.setattr(requests, "post", forbidden)
    monkeypatch.setattr(requests, "get", forbidden)
    monkeypatch.setattr(requests, "request", forbidden)

    # 1. SlackClient default initialization
    client = SlackClient()
    assert client.dry_run is True
    assert client.token is None

    # 2. SlackClient.from_token with defaults (default token file does not need to exist)
    client_from_token = SlackClient.from_token()
    assert client_from_token.dry_run is True

    # 3. send_message
    res_msg = client.send_message(channel="C01234567", text="Dry run hello")
    assert res_msg["ok"] is True
    assert res_msg["dry_run"] is True
    assert res_msg["method"] == "chat.postMessage"
    assert res_msg["params"] == {"channel": "C01234567", "text": "Dry run hello"}

    # 4. upload_file (file need not even exist in dry-run)
    res_upload = client.upload_file(channel="C01234567", file_path="/nonexistent/plot.png", title="Loss")
    assert res_upload["ok"] is True
    assert res_upload["dry_run"] is True
    assert res_upload["method"] == "files.getUploadURLExternal"
    assert res_upload["params"]["channel"] == "C01234567"
    assert res_upload["params"]["filename"] == "plot.png"

    # 5. list_channels
    channels = client.list_channels()
    assert channels == []

    # 6. get_channel_history
    res_hist = client.get_channel_history(channel="C01234567")
    assert res_hist["ok"] is True
    assert res_hist["dry_run"] is True
    assert res_hist["method"] == "conversations.history"

    # 7. get_unread_messages
    res_unread = client.get_unread_messages(channel="C01234567", since_ts="1700000000.000000")
    assert res_unread["ok"] is True
    assert res_unread["dry_run"] is True
    assert res_unread["method"] == "conversations.history"
    assert res_unread["params"]["oldest"] == "1700000000.000000"

    # 8. notify convenience function
    res_notify = notify("Experiment finished!", "C01234567")
    assert res_notify["ok"] is True
    assert res_notify["dry_run"] is True

    # 9. CLI in default dry-run mode
    exit_code = main(["send", "--channel", "C01234567", "--text", "CLI dry run"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "[DRY-RUN]" in captured.out


# ── Token loading and credential handling ────────────────────────────────────


def test_from_token_reads_env_var(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-fake-env-token")
    client = SlackClient.from_token(token_file="/nonexistent/path.txt", dry_run=False)
    assert client.token == "xoxb-fake-env-token"
    assert client.dry_run is False


def test_from_token_reads_file_when_env_unset(monkeypatch, tmp_path):
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    token_file = tmp_path / "slack_token.txt"
    token_file.write_text("  xoxb-fake-file-token \n")

    client = SlackClient.from_token(token_file=str(token_file), dry_run=False)
    assert client.token == "xoxb-fake-file-token"
    assert client.dry_run is False


def test_from_token_raises_when_file_not_found(monkeypatch, tmp_path):
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    missing = tmp_path / "nonexistent.txt"
    with pytest.raises(FileNotFoundError, match="Slack bot token file not found"):
        SlackClient.from_token(token_file=str(missing), dry_run=False)


def test_from_token_raises_when_token_is_empty(monkeypatch, tmp_path):
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("   \n")
    with pytest.raises(ValueError, match="Slack bot token is empty"):
        SlackClient.from_token(token_file=str(empty_file), dry_run=False)


# ── _call builds correct URL, headers, and params ────────────────────────────


def test_call_builds_correct_url_headers_params(monkeypatch):
    mock_post = MagicMock(return_value=MockResponse({"ok": True, "ts": "12345.67890"}))
    monkeypatch.setattr(requests, "post", mock_post)

    client = SlackClient(token="xoxb-test-token-123", dry_run=False)
    result = client._call("chat.postMessage", channel="C01234567", text="Testing payload")

    assert result["ok"] is True
    assert result["ts"] == "12345.67890"

    assert mock_post.call_count == 1
    call_args, call_kwargs = mock_post.call_args
    assert call_args[0] == "https://slack.com/api/chat.postMessage"
    assert call_kwargs["headers"] == {
        "Authorization": "Bearer xoxb-test-token-123",
    }
    assert call_kwargs["data"] == {"channel": "C01234567", "text": "Testing payload"}
    assert call_kwargs["timeout"] == 30


def test_call_drops_none_and_serializes_non_scalars(monkeypatch):
    mock_post = MagicMock(return_value=MockResponse({"ok": True}))
    monkeypatch.setattr(requests, "post", mock_post)

    client = SlackClient(token="xoxb-test", dry_run=False)
    client._call(
        "test.method",
        scalar_str="hello",
        scalar_int=42,
        none_val=None,
        dict_val={"a": 1, "b": "two"},
        list_val=[1, 2, "three"],
    )

    _, kwargs = mock_post.call_args
    sent_data = kwargs["data"]
    assert "none_val" not in sent_data
    assert sent_data["scalar_str"] == "hello"
    assert sent_data["scalar_int"] == 42
    assert sent_data["dict_val"] == '{"a": 1, "b": "two"}'
    assert sent_data["list_val"] == '[1, 2, "three"]'


def test_call_raises_clear_error_on_ok_false(monkeypatch):
    mock_post = MagicMock(return_value=MockResponse({"ok": False, "error": "channel_not_found"}))
    monkeypatch.setattr(requests, "post", mock_post)

    client = SlackClient(token="xoxb-test-token", dry_run=False)
    with pytest.raises(RuntimeError, match="Slack API error calling 'chat.postMessage': channel_not_found"):
        client.send_message(channel="C_INVALID", text="Hello")


def test_call_requires_token_when_not_dry_run():
    client = SlackClient(token=None, dry_run=False)
    with pytest.raises(ValueError, match="Slack token is required when dry_run=False"):
        client._call("chat.postMessage", channel="C1", text="Hi")


# ── Method tests: send_message, threading, history, unread ───────────────────


def test_send_message_with_thread_ts(monkeypatch):
    mock_post = MagicMock(return_value=MockResponse({"ok": True, "ts": "111.222"}))
    monkeypatch.setattr(requests, "post", mock_post)

    client = SlackClient(token="xoxb-test", dry_run=False)
    client.send_message(channel="C1", text="Thread reply", thread_ts="999.000")

    _, call_kwargs = mock_post.call_args
    assert call_kwargs["data"] == {
        "channel": "C1",
        "text": "Thread reply",
        "thread_ts": "999.000",
    }


def test_get_channel_history_and_unread(monkeypatch):
    mock_post = MagicMock(return_value=MockResponse({"ok": True, "messages": [{"text": "m1"}, {"text": "m2"}]}))
    monkeypatch.setattr(requests, "post", mock_post)

    client = SlackClient(token="xoxb-test", dry_run=False)

    # get_channel_history
    hist = client.get_channel_history(channel="C1", limit=25, oldest="100.5")
    assert hist["ok"] is True
    assert len(hist["messages"]) == 2
    assert mock_post.call_args[1]["data"] == {"channel": "C1", "limit": 25, "oldest": "100.5"}

    # get_unread_messages passes oldest=since_ts
    unread = client.get_unread_messages(channel="C1", since_ts="200.0", limit=10)
    assert unread["ok"] is True
    assert mock_post.call_args[1]["data"] == {"channel": "C1", "limit": 10, "oldest": "200.0"}


# ── Pagination in list_channels ──────────────────────────────────────────────


def test_list_channels_pagination(monkeypatch):
    page1 = {
        "ok": True,
        "channels": [{"id": "C1", "name": "general"}, {"id": "C2", "name": "random"}],
        "response_metadata": {"next_cursor": "cursor_page_2"},
    }
    page2 = {
        "ok": True,
        "channels": [{"id": "C3", "name": "alerts"}],
        "response_metadata": {"next_cursor": ""},
    }

    mock_post = MagicMock(side_effect=[MockResponse(page1), MockResponse(page2)])
    monkeypatch.setattr(requests, "post", mock_post)

    client = SlackClient(token="xoxb-test", dry_run=False)
    channels = client.list_channels(types="public_channel", limit=50)

    assert len(channels) == 3
    assert [c["id"] for c in channels] == ["C1", "C2", "C3"]

    assert mock_post.call_count == 2
    # Verify first page call
    call1_data = mock_post.call_args_list[0][1]["data"]
    assert call1_data == {"types": "public_channel", "limit": 50}
    # Verify second page call included the cursor
    call2_data = mock_post.call_args_list[1][1]["data"]
    assert call2_data == {"types": "public_channel", "limit": 50, "cursor": "cursor_page_2"}


# ── upload_file external-upload flow ─────────────────────────────────────────


def test_upload_file_three_step_flow(monkeypatch, tmp_path):
    test_file = tmp_path / "results.csv"
    test_file.write_text("epoch,loss\n1,0.5\n2,0.3\n")
    file_size = test_file.stat().st_size

    # Step 1 response: files.getUploadURLExternal
    step1_resp = MockResponse({
        "ok": True,
        "upload_url": "https://files.slack.com/upload/v1/abc123upload",
        "file_id": "F_MOCK_123",
    })
    # Step 2 response: direct upload to upload_url
    step2_resp = MockResponse({}, status_code=200)
    # Step 3 response: files.completeUploadExternal
    step3_resp = MockResponse({
        "ok": True,
        "files": [{"id": "F_MOCK_123", "title": "Loss Results"}],
    })

    mock_post = MagicMock(side_effect=[step1_resp, step2_resp, step3_resp])
    monkeypatch.setattr(requests, "post", mock_post)

    client = SlackClient(token="xoxb-test", dry_run=False)
    res = client.upload_file(channel="C_TARGET", file_path=test_file, title="Loss Results")

    assert res["ok"] is True
    assert res["files"][0]["id"] == "F_MOCK_123"

    assert mock_post.call_count == 3

    # Check Step 1: files.getUploadURLExternal
    url1, kwargs1 = mock_post.call_args_list[0][0][0], mock_post.call_args_list[0][1]
    assert url1 == "https://slack.com/api/files.getUploadURLExternal"
    assert kwargs1["data"] == {"filename": "results.csv", "length": file_size}

    # Check Step 2: raw byte upload to upload_url
    url2, kwargs2 = mock_post.call_args_list[1][0][0], mock_post.call_args_list[1][1]
    assert url2 == "https://files.slack.com/upload/v1/abc123upload"
    assert kwargs2["headers"]["Content-Type"] == "application/octet-stream"

    # Check Step 3: files.completeUploadExternal
    url3, kwargs3 = mock_post.call_args_list[2][0][0], mock_post.call_args_list[2][1]
    assert url3 == "https://slack.com/api/files.completeUploadExternal"
    # files must arrive as a JSON string!
    assert kwargs3["data"]["files"] == json.dumps([{"id": "F_MOCK_123", "title": "Loss Results"}])
    assert kwargs3["data"]["channel_id"] == "C_TARGET"


def test_upload_file_dry_run_expanduser(monkeypatch, tmp_path):
    fake_home = tmp_path / "fake_home"
    fake_home.mkdir()
    sample_file = fake_home / "sample.txt"
    sample_file.write_text("dry run file content")
    expected_size = sample_file.stat().st_size

    monkeypatch.setenv("HOME", str(fake_home))
    client = SlackClient(dry_run=True)
    res = client.upload_file(channel="C_TEST", file_path="~/sample.txt")

    assert res["ok"] is True
    assert res["dry_run"] is True
    assert res["params"]["filename"] == "sample.txt"
    assert res["params"]["length"] == expected_size


def test_upload_file_nonexistent_raises_when_not_dry_run(tmp_path):
    client = SlackClient(token="xoxb-test", dry_run=False)
    with pytest.raises(FileNotFoundError, match="File to upload not found"):
        client.upload_file(channel="C1", file_path=tmp_path / "nonexistent.png")


# ── notify helper & CLI tests ────────────────────────────────────────────────


def test_notify_convenience_helper(monkeypatch):
    mock_post = MagicMock(return_value=MockResponse({"ok": True, "ts": "555.666"}))
    monkeypatch.setattr(requests, "post", mock_post)
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-notify-token")

    res = notify(text="Job done", channel="C_NOTIFY", dry_run=False)
    assert res["ok"] is True
    assert res["ts"] == "555.666"

    _, kwargs = mock_post.call_args
    assert kwargs["data"] == {"channel": "C_NOTIFY", "text": "Job done"}


def test_cli_send_live_with_flag(monkeypatch, capsys):
    mock_post = MagicMock(return_value=MockResponse({"ok": True, "ts": "777.888"}))
    monkeypatch.setattr(requests, "post", mock_post)
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-cli-token")

    code = main(["send", "--channel", "C_CLI", "--text", "Hello from CLI", "--send"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Message sent to C_CLI" in captured.out
    assert mock_post.call_count == 1
