from src.message import InboundMessage, strip_quoted_reply


RAW = b"""\
Message-ID: <CAABCDE+abc@mail.gmail.com>
From: "Brando Miranda" <brando.science@gmail.com>
To: brandojazz@gmail.com
Subject: Re: Nightly SolveAll report
Reply-To: brando.science@gmail.com
Return-Path: <brando.science@gmail.com>
Authentication-Results: mx.google.com; dkim=pass header.i=@gmail.com; spf=pass smtp.mailfrom=brando.science@gmail.com; dmarc=pass header.from=gmail.com
MIME-Version: 1.0
Content-Type: text/plain; charset=UTF-8

@claude please re-audit file 14 and list only truly proved lemmas.

On Thu, Apr 17 2026 at 07:14 AM, Brando <brandojazz@gmail.com> wrote:
> old thread content
>
> more old content
"""


class TestInboundMessage:
    def test_parse_all_fields(self):
        m = InboundMessage.from_bytes(RAW)
        assert m.message_id == "<CAABCDE+abc@mail.gmail.com>"
        assert "brando.science@gmail.com" in m.from_addr
        assert m.subject == "Re: Nightly SolveAll report"
        assert "dkim=pass" in m.auth_results
        assert "re-audit file 14" in m.body_text


class TestStripQuotedReply:
    def test_strip_angle_quote(self):
        body = "new text\n> old text\n> more old"
        assert strip_quoted_reply(body) == "new text"

    def test_strip_on_wrote_marker(self):
        body = "new text\nOn Thu, Apr 17 2026 at 07:14, someone wrote:\nquoted"
        assert strip_quoted_reply(body) == "new text"

    def test_strip_signature_dashes(self):
        body = "actual message\n--\nBrando\nPhD Stanford"
        assert strip_quoted_reply(body) == "actual message"

    def test_nothing_to_strip(self):
        body = "just one line"
        assert strip_quoted_reply(body) == "just one line"

    def test_full_pipeline_on_raw(self):
        m = InboundMessage.from_bytes(RAW)
        instruction = strip_quoted_reply(m.body_text)
        assert "re-audit file 14" in instruction
        assert "old thread content" not in instruction
        assert "wrote:" not in instruction
