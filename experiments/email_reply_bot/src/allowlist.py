"""Sender allowlist.

Only these three addresses are permitted to trigger the Claude agent.
Aliases (`+tag`), display names, and case are normalized before compare.
Everything else is silently rejected.
"""

from __future__ import annotations

from email.utils import parseaddr

ALLOWED_SENDERS: frozenset[str] = frozenset({
    "brando.science@gmail.com",
    "brandojazz@gmail.com",
    "brando9@stanford.edu",
})


def normalize_addr(raw: str) -> str:
    """Extract and normalize an email address.

    Strips display name, lowercases, and removes ``+tag`` aliases.
    Returns ``""`` if parsing fails.
    """
    if not raw:
        return ""
    _display, addr = parseaddr(raw)
    addr = addr.strip().lower()
    if "@" not in addr:
        return ""
    local, _, domain = addr.partition("@")
    local = local.split("+", 1)[0]
    return f"{local}@{domain}"


def verify_sender(raw: str) -> bool:
    """Return True iff the address normalizes to a member of ALLOWED_SENDERS."""
    return normalize_addr(raw) in ALLOWED_SENDERS
