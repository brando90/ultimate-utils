"""End-to-end pipeline: fetched message → verified → dispatched → replied.

Depends only on the protocols (``GmailClient``, ``LLMClient``) and the pure
modules (``allowlist``, ``auth_headers``, ``store``, ``message``). This makes
the pipeline fully testable against fakes — no network, no real Claude call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .allowlist import normalize_addr, verify_sender
from .auth_headers import verify_auth_headers
from .dispatcher import LLMClient, SYSTEM_PROMPT, build_prompt
from .gmail_client import FetchedMessage, GmailClient
from .message import InboundMessage, strip_quoted_reply
from .store import Store

log = logging.getLogger(__name__)

REPLY_FOOTER = (
    "—\nThis is an automated reply from the email-reply-bot daemon. "
    "Full transcript archived server-side."
)


@dataclass
class PipelineConfig:
    bot_from_addr: str
    workdir: str | None = None
    rate_limit_per_hour: int = 10
    require_auth_headers: bool = True
    dry_run: bool = False


@dataclass
class PipelineResult:
    accepted: bool
    reason: str
    reply_text: str = ""


class Pipeline:
    def __init__(
        self,
        *,
        gmail: GmailClient,
        llm: LLMClient,
        store: Store,
        config: PipelineConfig,
    ) -> None:
        self.gmail = gmail
        self.llm = llm
        self.store = store
        self.cfg = config

    def handle(self, fetched: FetchedMessage) -> PipelineResult:
        msg = InboundMessage.from_bytes(fetched.raw, thread_id=fetched.thread_id)
        sender = normalize_addr(msg.from_addr)

        if not msg.message_id:
            self.store.log(sender=sender, message_id="", decision="reject", reason="no Message-ID")
            return PipelineResult(False, "no Message-ID")

        if self.store.seen(msg.message_id):
            self.store.log(
                sender=sender,
                message_id=msg.message_id,
                decision="reject",
                reason="duplicate",
            )
            return PipelineResult(False, "duplicate")

        if not verify_sender(msg.from_addr):
            self.store.log(
                sender=sender,
                message_id=msg.message_id,
                decision="reject",
                reason="sender not in allowlist",
            )
            self.store.mark_seen(msg.message_id)
            return PipelineResult(False, "sender not in allowlist")

        reply_to_norm = normalize_addr(msg.reply_to) if msg.reply_to else sender
        return_path_norm = normalize_addr(msg.return_path) if msg.return_path else sender
        if reply_to_norm and reply_to_norm != sender:
            self.store.log(
                sender=sender,
                message_id=msg.message_id,
                decision="reject",
                reason=f"reply-to mismatch ({reply_to_norm} vs {sender})",
            )
            self.store.mark_seen(msg.message_id)
            return PipelineResult(False, "reply-to mismatch")
        if return_path_norm and return_path_norm != sender:
            self.store.log(
                sender=sender,
                message_id=msg.message_id,
                decision="reject",
                reason=f"return-path mismatch ({return_path_norm} vs {sender})",
            )
            self.store.mark_seen(msg.message_id)
            return PipelineResult(False, "return-path mismatch")

        if self.cfg.require_auth_headers:
            ok, reason = verify_auth_headers(sender, msg.auth_results)
            if not ok:
                self.store.log(
                    sender=sender,
                    message_id=msg.message_id,
                    decision="reject",
                    reason=f"auth headers: {reason}",
                )
                self.store.mark_seen(msg.message_id)
                return PipelineResult(False, f"auth headers: {reason}")

        recent = self.store.count_recent(sender, window_seconds=3600, decision="accept")
        if recent >= self.cfg.rate_limit_per_hour:
            self.store.log(
                sender=sender,
                message_id=msg.message_id,
                decision="reject",
                reason=f"rate limit ({recent}/hr)",
            )
            self.store.mark_seen(msg.message_id)
            return PipelineResult(False, "rate limit")

        instruction = strip_quoted_reply(msg.body_text)
        if not instruction:
            self.store.log(
                sender=sender,
                message_id=msg.message_id,
                decision="reject",
                reason="empty instruction",
            )
            self.store.mark_seen(msg.message_id)
            return PipelineResult(False, "empty instruction")

        prompt = build_prompt(subject=msg.subject, body=instruction)
        answer = self.llm.run(prompt=prompt, system=SYSTEM_PROMPT, workdir=self.cfg.workdir)

        self.store.log(
            sender=sender,
            message_id=msg.message_id,
            decision="accept",
            reason="ok",
            body=instruction,
        )
        self.store.mark_seen(msg.message_id)

        if not self.cfg.dry_run:
            # Preserve the inbound References chain so Gmail and other clients
            # thread the reply correctly across many-turn conversations.
            references = (
                f"{msg.references} {msg.message_id}".strip()
                if msg.references
                else msg.message_id
            )
            self.gmail.send_threaded_reply(
                thread_id=msg.thread_id,
                in_reply_to=msg.message_id,
                references=references,
                to=sender,
                subject=msg.subject,
                body_text=answer.rstrip() + "\n\n" + REPLY_FOOTER,
            )

        return PipelineResult(True, "accepted", reply_text=answer)

    def run_once(self, label: str) -> list[PipelineResult]:
        results: list[PipelineResult] = []
        for fetched in self.gmail.fetch_unseen(label):
            try:
                results.append(self.handle(fetched))
            except Exception as exc:
                log.exception("pipeline handler crashed on %s: %s", fetched.message_id, exc)
                results.append(PipelineResult(False, f"handler error: {exc}"))
        return results
