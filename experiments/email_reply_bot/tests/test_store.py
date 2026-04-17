import time
from pathlib import Path

import pytest

from src.store import Store


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "state.sqlite")


class TestSeen:
    def test_roundtrip(self, store: Store):
        assert not store.seen("<abc@x>")
        store.mark_seen("<abc@x>")
        assert store.seen("<abc@x>")

    def test_empty_id_is_not_seen(self, store: Store):
        assert not store.seen("")
        store.mark_seen("")  # no-op
        assert not store.seen("")

    def test_idempotent_mark(self, store: Store):
        store.mark_seen("<a>")
        store.mark_seen("<a>")
        assert store.seen("<a>")


class TestAudit:
    def test_log_and_count(self, store: Store):
        store.log(sender="x@y.com", message_id="<1>", decision="accept", body="hi")
        store.log(sender="x@y.com", message_id="<2>", decision="accept", body="hi")
        store.log(sender="x@y.com", message_id="<3>", decision="reject", reason="stranger")
        assert store.count_recent("x@y.com", 3600, decision="accept") == 2
        assert store.count_recent("x@y.com", 3600, decision="reject") == 1

    def test_rate_limit_window(self, store: Store, monkeypatch):
        store.log(sender="x@y.com", message_id="<old>", decision="accept")
        real_time = time.time
        monkeypatch.setattr(time, "time", lambda: real_time() + 7200)
        assert store.count_recent("x@y.com", 3600, decision="accept") == 0

    def test_body_is_hashed_not_stored(self, store: Store, tmp_path: Path):
        store.log(
            sender="x@y.com",
            message_id="<1>",
            decision="accept",
            body="SECRET PAYLOAD DO NOT LEAK",
        )
        contents = (tmp_path / "state.sqlite").read_bytes()
        assert b"SECRET PAYLOAD" not in contents
