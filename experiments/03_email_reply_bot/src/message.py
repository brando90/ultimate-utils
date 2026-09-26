"""Parse raw RFC-822 bytes into the fields we care about."""

from __future__ import annotations

from dataclasses import dataclass
from email import message_from_bytes, policy
from email.message import EmailMessage


@dataclass(frozen=True)
class InboundMessage:
    message_id: str
    thread_id: str
    from_addr: str
    to_addrs: tuple[str, ...]
    reply_to: str
    return_path: str
    subject: str
    auth_results: str
    references: str
    body_text: str
    raw: bytes

    @classmethod
    def from_bytes(cls, raw: bytes, thread_id: str = "") -> "InboundMessage":
        msg: EmailMessage = message_from_bytes(raw, policy=policy.default)  # type: ignore[assignment]
        body = _extract_text(msg)
        return cls(
            message_id=(msg.get("Message-ID") or "").strip(),
            thread_id=thread_id or (msg.get("Message-ID") or "").strip(),
            from_addr=(msg.get("From") or "").strip(),
            to_addrs=tuple((msg.get("To") or "").split(",")),
            reply_to=(msg.get("Reply-To") or "").strip(),
            return_path=(msg.get("Return-Path") or "").strip(),
            subject=(msg.get("Subject") or "").strip(),
            auth_results=(msg.get("Authentication-Results") or "").strip(),
            references=(msg.get("References") or "").strip(),
            body_text=body,
            raw=raw,
        )


def _extract_text(msg: EmailMessage) -> str:
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                try:
                    return part.get_content()
                except Exception:
                    continue
        for part in msg.walk():
            if part.get_content_type() == "text/html":
                try:
                    return part.get_content()
                except Exception:
                    continue
        return ""
    try:
        return msg.get_content()
    except Exception:
        return ""


def strip_quoted_reply(body: str) -> str:
    """Remove common quoted-reply markers so the LLM sees only the new text."""
    lines: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith(">"):
            break
        if stripped.startswith("On ") and stripped.rstrip().endswith("wrote:"):
            break
        if stripped in {"--", "---"}:
            break
        lines.append(line)
    return "\n".join(lines).strip()
