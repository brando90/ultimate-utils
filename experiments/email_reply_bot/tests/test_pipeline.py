"""Full pipeline integration tests with in-memory Gmail and LLM fakes.

This is what actually tests whether the bot "works": feed real RFC-822 fixtures
through the pipeline and assert that only allowlisted, DKIM-passing, non-spoofed
messages produce a threaded outbound reply.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from src.gmail_client import FetchedMessage
from src.pipeline import Pipeline, PipelineConfig
from src.store import Store


FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


@dataclass
class FakeGmail:
    inbox: list[FetchedMessage] = field(default_factory=list)
    sent: list[dict] = field(default_factory=list)

    def fetch_unseen(self, label: str) -> list[FetchedMessage]:
        return list(self.inbox)

    def send_threaded_reply(
        self,
        *,
        thread_id: str,
        in_reply_to: str,
        references: str,
        to: str,
        subject: str,
        body_text: str,
    ) -> str:
        self.sent.append(
            {
                "thread_id": thread_id,
                "in_reply_to": in_reply_to,
                "references": references,
                "to": to,
                "subject": subject,
                "body_text": body_text,
            }
        )
        return f"sent-{len(self.sent)}"


@dataclass
class FakeLLM:
    canned: str = "4."
    calls: list[dict] = field(default_factory=list)

    def run(self, *, prompt: str, system: str, workdir: str | None = None) -> str:
        self.calls.append({"prompt": prompt, "system": system, "workdir": workdir})
        return self.canned


def _build(
    tmp_path: Path,
    *fixture_names: str,
    rate_limit: int = 10,
    require_auth: bool = True,
) -> tuple[Pipeline, FakeGmail, FakeLLM, Store]:
    inbox = [
        FetchedMessage(message_id=f"gm-{i}", thread_id=f"thr-{i}", raw=_load(name))
        for i, name in enumerate(fixture_names)
    ]
    gmail = FakeGmail(inbox=inbox)
    llm = FakeLLM()
    store = Store(tmp_path / "state.sqlite")
    pipeline = Pipeline(
        gmail=gmail,
        llm=llm,
        store=store,
        config=PipelineConfig(
            bot_from_addr="brandojazz@gmail.com",
            rate_limit_per_hour=rate_limit,
            require_auth_headers=require_auth,
        ),
    )
    return pipeline, gmail, llm, store


class TestAcceptAllThreeAllowlisted:
    def test_brando_science(self, tmp_path: Path):
        p, gmail, llm, _ = _build(tmp_path, "legit_brando_science.eml")
        results = p.run_once("INBOX")
        assert len(results) == 1 and results[0].accepted
        assert len(llm.calls) == 1
        assert "re-audit file 14" in llm.calls[0]["prompt"]
        assert len(gmail.sent) == 1
        assert gmail.sent[0]["to"] == "brando.science@gmail.com"
        assert gmail.sent[0]["in_reply_to"] == "<CAB001@mail.gmail.com>"
        assert gmail.sent[0]["thread_id"] == "thr-0"
        assert "4." in gmail.sent[0]["body_text"]

    def test_brandojazz(self, tmp_path: Path):
        p, gmail, llm, _ = _build(tmp_path, "legit_brandojazz.eml")
        results = p.run_once("INBOX")
        assert results[0].accepted
        assert len(gmail.sent) == 1
        assert gmail.sent[0]["to"] == "brandojazz@gmail.com"

    def test_brando9_stanford(self, tmp_path: Path):
        p, gmail, llm, _ = _build(tmp_path, "legit_brando9_stanford.eml")
        results = p.run_once("INBOX")
        assert results[0].accepted
        assert gmail.sent[0]["to"] == "brando9@stanford.edu"

    def test_alias_with_plus_tag(self, tmp_path: Path):
        p, gmail, llm, _ = _build(tmp_path, "alias_tag.eml")
        results = p.run_once("INBOX")
        assert results[0].accepted
        assert gmail.sent[0]["to"] == "brando.science@gmail.com"


class TestRejectStranger:
    def test_stranger_no_reply(self, tmp_path: Path):
        p, gmail, llm, _ = _build(tmp_path, "stranger.eml")
        results = p.run_once("INBOX")
        assert len(results) == 1
        assert not results[0].accepted
        assert "allowlist" in results[0].reason
        assert llm.calls == []
        assert gmail.sent == []

    def test_spoofed_rejected(self, tmp_path: Path):
        p, gmail, llm, _ = _build(tmp_path, "spoofed_dkim_fail.eml")
        results = p.run_once("INBOX")
        assert not results[0].accepted
        assert "auth headers" in results[0].reason
        assert llm.calls == []
        assert gmail.sent == []

    def test_reply_to_mismatch_rejected(self, tmp_path: Path):
        p, gmail, llm, _ = _build(tmp_path, "reply_to_mismatch.eml")
        results = p.run_once("INBOX")
        assert not results[0].accepted
        assert "reply-to" in results[0].reason
        assert llm.calls == []
        assert gmail.sent == []


class TestMixedBatch:
    def test_only_allowlisted_replied(self, tmp_path: Path):
        p, gmail, llm, _ = _build(
            tmp_path,
            "legit_brando_science.eml",
            "stranger.eml",
            "spoofed_dkim_fail.eml",
            "legit_brando9_stanford.eml",
            "reply_to_mismatch.eml",
        )
        results = p.run_once("INBOX")
        accepted = [r for r in results if r.accepted]
        rejected = [r for r in results if not r.accepted]
        assert len(accepted) == 2
        assert len(rejected) == 3
        assert len(gmail.sent) == 2
        assert len(llm.calls) == 2


class TestIdempotency:
    def test_duplicate_message_id_second_time(self, tmp_path: Path):
        p, gmail, llm, _ = _build(tmp_path, "legit_brando_science.eml")
        first = p.run_once("INBOX")
        second = p.run_once("INBOX")
        assert first[0].accepted
        assert not second[0].accepted
        assert "duplicate" in second[0].reason
        assert len(gmail.sent) == 1
        assert len(llm.calls) == 1


class TestDryRun:
    def test_dry_run_calls_llm_but_not_gmail(self, tmp_path: Path):
        p, gmail, llm, _ = _build(tmp_path, "legit_brando_science.eml")
        p.cfg.dry_run = True
        results = p.run_once("INBOX")
        assert results[0].accepted
        assert len(llm.calls) == 1
        assert gmail.sent == []


class TestRateLimit:
    def test_accepts_under_limit(self, tmp_path: Path):
        p, gmail, _, _ = _build(
            tmp_path,
            "legit_brando_science.eml",
            "legit_brandojazz.eml",
            rate_limit=10,
        )
        p.run_once("INBOX")
        assert len(gmail.sent) == 2

    def test_blocks_over_limit(self, tmp_path: Path):
        p, gmail, _, store = _build(tmp_path, "legit_brando_science.eml", rate_limit=1)
        p.run_once("INBOX")
        assert len(gmail.sent) == 1
        p.gmail.inbox = [
            FetchedMessage(
                message_id="gm-2",
                thread_id="thr-2",
                raw=_load("legit_brando_science.eml").replace(
                    b"<CAB001@mail.gmail.com>", b"<CAB001b@mail.gmail.com>"
                ),
            )
        ]
        p.run_once("INBOX")
        assert len(gmail.sent) == 1


class TestOutboundThreading:
    def test_in_reply_to_and_references_set(self, tmp_path: Path):
        p, gmail, _, _ = _build(tmp_path, "legit_brando_science.eml")
        p.run_once("INBOX")
        sent = gmail.sent[0]
        assert sent["in_reply_to"] == "<CAB001@mail.gmail.com>"
        assert sent["references"] == "<CAB001@mail.gmail.com>"
        assert sent["subject"].lower().startswith("re:")

    def test_references_chain_preserved(self, tmp_path: Path):
        """When the inbound has an existing References header (multi-turn
        thread), the outbound should extend it rather than replace it."""
        raw = _load("legit_brando_science.eml").replace(
            b"Subject: Re: Nightly SolveAll report\n",
            b"Subject: Re: Nightly SolveAll report\nReferences: <first@x> <second@x>\n",
        )
        gmail = FakeGmail(inbox=[FetchedMessage("gm-0", "thr-0", raw)])
        llm = FakeLLM()
        store = Store(tmp_path / "state.sqlite")
        p = Pipeline(
            gmail=gmail,
            llm=llm,
            store=store,
            config=PipelineConfig(bot_from_addr="brandojazz@gmail.com"),
        )
        p.run_once("INBOX")
        sent = gmail.sent[0]
        assert "<first@x>" in sent["references"]
        assert "<second@x>" in sent["references"]
        assert "<CAB001@mail.gmail.com>" in sent["references"]


class TestAuditLog:
    def test_rejected_body_not_stored(self, tmp_path: Path):
        p, _, _, _ = _build(tmp_path, "stranger.eml")
        p.run_once("INBOX")
        contents = (tmp_path / "state.sqlite").read_bytes()
        assert b"rm -rf the home directory" not in contents

    def test_accepted_body_hash_stored_not_plaintext(self, tmp_path: Path):
        p, _, _, _ = _build(tmp_path, "legit_brando_science.eml")
        p.run_once("INBOX")
        contents = (tmp_path / "state.sqlite").read_bytes()
        assert b"re-audit file 14" not in contents
        contents_text = contents.decode("utf-8", errors="ignore")
        assert "accept" in contents_text


class TestEmptyInstruction:
    def test_empty_body_rejected(self, tmp_path: Path):
        raw = _load("legit_brando_science.eml")
        raw = raw.replace(
            b"@claude please re-audit file 14 and tell me which lemmas are actually proved.",
            b"> only quoted stuff",
        )
        gmail = FakeGmail(inbox=[FetchedMessage("gm-0", "thr-0", raw)])
        store = Store(tmp_path / "state.sqlite")
        p = Pipeline(
            gmail=gmail,
            llm=FakeLLM(),
            store=store,
            config=PipelineConfig(bot_from_addr="brandojazz@gmail.com"),
        )
        results = p.run_once("INBOX")
        assert not results[0].accepted
        assert "empty" in results[0].reason


class TestMimeBuilding:
    def test_reply_mime_has_headers(self):
        from src.gmail_client import build_reply_mime

        raw = build_reply_mime(
            from_addr="brandojazz@gmail.com",
            to_addr="brando.science@gmail.com",
            subject="hello",
            in_reply_to="<abc@x>",
            references="<abc@x>",
            body_text="hi there",
            footer="--\nautomated",
        )
        assert b"In-Reply-To: <abc@x>" in raw
        assert b"References: <abc@x>" in raw
        assert b"Subject: Re: hello" in raw
        assert b"hi there" in raw
        assert b"automated" in raw

    def test_crlf_injection_in_subject_rejected(self):
        """``EmailMessage`` must refuse to set a header value containing CR/LF,
        so an attacker cannot smuggle extra headers (Bcc, etc) via a crafted
        Subject."""
        from src.gmail_client import build_reply_mime

        with pytest.raises(Exception):
            build_reply_mime(
                from_addr="brandojazz@gmail.com",
                to_addr="brando.science@gmail.com",
                subject="hi\r\nBcc: evil@example.com",
                in_reply_to="<abc@x>",
                references="<abc@x>",
                body_text="body",
            )
