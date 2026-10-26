"""Offline tests for the SMTP envelope recipients built by uutils.emailing.

smtplib is mocked, so no connection is opened and no email is sent. Each test
checks the recipient list passed to ``sendmail`` and that no ``Bcc`` header
reaches the message text.
"""
from __future__ import annotations

from email import message_from_string
from unittest.mock import MagicMock, patch

import pytest

from uutils.emailing import send_email_smtp

SENDER = "sender@example.com"
TO = "to@example.com"
FOUR_BCC = [
    "brando.science@gmail.com",
    "brando9@stanford.edu",
    "brandojazz@gmail.com",
    "brando@vals.ai",
]


def _send_and_capture(to: str, cc: str = "", bcc: str = "") -> tuple[list[str], str]:
    """Send through a mocked smtplib.SMTP; return (envelope recipients, message text)."""
    with patch("uutils.emailing.smtplib.SMTP") as smtp_cls, \
            patch("uutils.emailing.smtplib.SMTP_SSL", side_effect=AssertionError("unexpected SMTP_SSL")):
        server = MagicMock()
        smtp_cls.return_value.__enter__.return_value = server
        send_email_smtp(
            to=to, subject="subject", body="body",
            smtp_user=SENDER, smtp_pass="fake-test-password",
            cc=cc, bcc=bcc,
        )
    server.sendmail.assert_called_once()
    from_addr, recipients, msg_text = server.sendmail.call_args.args
    assert from_addr == SENDER
    assert message_from_string(msg_text)["Bcc"] is None
    return recipients, msg_text


def test_single_bcc_unchanged():
    recipients, _ = _send_and_capture(TO, bcc="only@example.com")
    assert recipients == [TO, "only@example.com"]


@pytest.mark.parametrize("sep", [",", ", "])
def test_four_comma_joined_bcc_split_into_envelope(sep: str):
    recipients, msg_text = _send_and_capture(TO, bcc=sep.join(FOUR_BCC))
    assert recipients == [TO, *FOUR_BCC]
    for addr in FOUR_BCC:
        assert addr not in msg_text


def test_bcc_overlapping_to_and_cc_not_duplicated():
    recipients, msg_text = _send_and_capture(
        "brando9@stanford.edu", cc="brandojazz@gmail.com", bcc=",".join(FOUR_BCC),
    )
    assert recipients == [
        "brando9@stanford.edu",
        "brandojazz@gmail.com",
        "brando.science@gmail.com",
        "brando@vals.ai",
    ]
    assert message_from_string(msg_text)["Cc"] == "brandojazz@gmail.com"


@pytest.mark.parametrize("bcc", ["", " , ,"])
def test_empty_bcc_adds_no_recipients(bcc: str):
    recipients, _ = _send_and_capture(TO, bcc=bcc)
    assert recipients == [TO]
