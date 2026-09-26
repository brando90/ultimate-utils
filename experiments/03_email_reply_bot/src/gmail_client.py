"""Gmail transport layer.

Defines a :class:`GmailClient` protocol so the rest of the pipeline can be
driven by a fake in tests. The real implementation (``RealGmailClient``) uses
the Gmail REST API via googleapiclient and is only imported when needed — the
tests do not require any Google libraries to be installed.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from email.message import EmailMessage
from email.utils import make_msgid
from typing import Protocol


@dataclass(frozen=True)
class FetchedMessage:
    message_id: str
    thread_id: str
    raw: bytes


class GmailClient(Protocol):
    def fetch_unseen(self, label: str) -> list[FetchedMessage]: ...

    def send_threaded_reply(
        self,
        *,
        thread_id: str,
        in_reply_to: str,
        references: str,
        to: str,
        subject: str,
        body_text: str,
    ) -> str: ...


def build_reply_mime(
    *,
    from_addr: str,
    to_addr: str,
    subject: str,
    in_reply_to: str,
    references: str,
    body_text: str,
    footer: str = "",
) -> bytes:
    """Build an RFC-822 reply MIME body ready for Gmail's ``raw`` field."""
    msg = EmailMessage()
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg["Subject"] = subject if subject.lower().startswith("re:") else f"Re: {subject}"
    if in_reply_to:
        msg["In-Reply-To"] = in_reply_to
    if references:
        msg["References"] = references
    msg["Message-ID"] = make_msgid()
    content = body_text.rstrip() + ("\n\n" + footer if footer else "")
    msg.set_content(content)
    return bytes(msg)


def encode_raw_for_gmail(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii")
