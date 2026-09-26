"""Persistent state: seen message-ids, rate limit counters, audit log.

SQLite keeps the daemon idempotent across restarts.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS seen (
    message_id TEXT PRIMARY KEY,
    seen_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    sender TEXT,
    message_id TEXT,
    decision TEXT NOT NULL,
    reason TEXT,
    body_hash TEXT
);

CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit(ts);
CREATE INDEX IF NOT EXISTS idx_audit_sender ON audit(sender);
"""


@dataclass
class Store:
    path: Path

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def seen(self, message_id: str) -> bool:
        if not message_id:
            return False
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT 1 FROM seen WHERE message_id = ?", (message_id,)
            ).fetchone()
            return row is not None

    def mark_seen(self, message_id: str) -> None:
        if not message_id:
            return
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO seen(message_id, seen_at) VALUES (?, ?)",
                (message_id, time.time()),
            )
            conn.commit()

    def count_recent(self, sender: str, window_seconds: int, decision: str = "accept") -> int:
        if not sender:
            return 0
        cutoff = time.time() - window_seconds
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM audit WHERE sender = ? AND decision = ? AND ts >= ?",
                (sender, decision, cutoff),
            ).fetchone()
            return int(row[0])

    def log(
        self,
        *,
        sender: str,
        message_id: str,
        decision: str,
        reason: str = "",
        body: str = "",
    ) -> None:
        body_hash = hashlib.sha256(body.encode("utf-8", "replace")).hexdigest() if body else ""
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO audit(ts, sender, message_id, decision, reason, body_hash) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (time.time(), sender, message_id, decision, reason, body_hash),
            )
            conn.commit()
