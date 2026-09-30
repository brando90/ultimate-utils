"""Offline tests for uutils.openreview_uu.

A fake client stands in for openreview.api.OpenReviewClient, sockets are blocked, the OPENREVIEW_* env vars
are cleared, and every credentials file lives under tmp_path, so no test touches the network or ~/keys/.
"""
from __future__ import annotations

import ast
import builtins
import json
import logging
import re
import socket
import sys
import traceback
import types
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

import pytest

import uutils.openreview_uu as oru
from uutils.openreview_uu import (
    OpenReviewAuthError,
    fetch_forum,
    load_credentials,
    main,
    make_client,
    redact,
    render_markdown,
    reply_type,
    submission_status,
)

FORUM = "Abc123XyZ9"
VENUE = "ICLR.cc/2027/Conference"
PAPER = f"{VENUE}/Submission35334"
T0 = 1_758_000_000_000  # 09-16-2025 05:20 UTC, in epoch milliseconds like OpenReview's cdate
FAKE_PW = "pw'with\"quote\\ and space"  # quotes/backslash exercise the repr/JSON-escaped redaction paths
FAKE_TOKEN = "fake.jwt.token-value"


@pytest.fixture(autouse=True)
def _offline_and_credential_free(monkeypatch):
    """Fail loudly on any network connection and hide any real OpenReview env credentials."""

    def forbidden(*args, **kwargs):
        raise AssertionError(f"network access attempted: {args!r}")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.delenv("OPENREVIEW_USERNAME", raising=False)
    monkeypatch.delenv("OPENREVIEW_PASSWORD", raising=False)


def _note(note_id, *, invitations, cdate, content, signatures=(f"{PAPER}/Authors",), replyto=FORUM, number=None):
    """Mimic openreview.api.Note: plain attributes, content values wrapped as {'value': ...}."""
    return SimpleNamespace(
        id=note_id, forum=FORUM, replyto=replyto, number=number, invitations=list(invitations),
        signatures=list(signatures), readers=["everyone"], writers=[], cdate=cdate, mdate=cdate + 60_000,
        pdate=None, tcdate=cdate, tmdate=cdate + 60_000, ddate=None, content=content, details=None,
    )


def _submission(venue: str, venueid: str):
    return _note(
        FORUM, invitations=[f"{VENUE}/-/Submission"], cdate=T0, replyto=None, number=35334,
        content={
            "title": {"value": "Judging Lean Specifications"},
            "venue": {"value": venue},
            "venueid": {"value": venueid},
            "pdf": {"value": "/pdf/0123abcd.pdf"},
            "keywords": {"value": ["Lean 4", "LLM judges"]},
            "abstract": {"value": "Line one.\n\nLine *two* with `code`."},
        },
    )


SUBMISSION = _submission("ICLR 2027 Conference Withdrawn Submission", f"{VENUE}/Withdrawn_Submission")
REVIEW_TEXT = "## Summary\nThe paper proposes **judges**.\n\n- weakness: `sorry` everywhere\n"
FEEDBACK = _note("fb1", invitations=[f"{PAPER}/-/Automated_Feedback"], cdate=T0 + 1_000,
                 signatures=[f"{VENUE}/Automated_Feedback_Agent"], content={"feedback": {"value": "Check Table 2."}})
REVIEW_1 = _note("rev1", invitations=[f"{PAPER}/-/Official_Review", f"{VENUE}/-/Edit"], cdate=T0 + 2_000,
                 signatures=[f"{PAPER}/Reviewer_Ab12"],
                 content={"summary": {"value": REVIEW_TEXT}, "rating": {"value": 6}, "confidence": {"value": 4}})
REVIEW_2 = _note("rev2", invitations=[f"{PAPER}/-/Official_Review"], cdate=T0 + 3_000,
                 signatures=[f"{PAPER}/Reviewer_Cd34"],
                 content={"summary": {"value": "Solid."}, "rating": {"value": 3}})
COMMENT = _note("com1", invitations=[f"{PAPER}/-/Official_Comment"], cdate=T0 + 4_000, replyto="rev1",
                content={"title": {"value": "Response"}, "comment": {"value": "Thanks!\n- fixed"}})
WITHDRAWAL = _note("wd1", invitations=[f"{PAPER}/-/Withdrawal"], cdate=T0 + 5_000,
                   content={"withdrawal_confirmation": {"value": "I have read and agree."}})
CHRONOLOGICAL = ["fb1", "rev1", "rev2", "com1", "wd1"]


class FakeClient:
    """Stands in for openreview.api.OpenReviewClient: records calls and serves canned notes."""

    baseurl = oru.API_BASEURL
    token = FAKE_TOKEN

    def __init__(self, submission=SUBMISSION, replies=(COMMENT, REVIEW_2, WITHDRAWAL, FEEDBACK, REVIEW_1)):
        self.submission, self.replies, self.calls = submission, list(replies), []

    def get_note(self, id, details=None):
        self.calls.append(("get_note", id))
        return self.submission

    def get_all_notes(self, forum=None, **kwargs):
        self.calls.append(("get_all_notes", forum))
        return [self.submission, *self.replies]  # like the API, the forum query includes the submission itself


def _install_fake_openreview(monkeypatch, client_cls) -> None:
    api = types.ModuleType("openreview.api")
    api.OpenReviewClient = client_cls
    package = types.ModuleType("openreview")
    package.api = api
    monkeypatch.setitem(sys.modules, "openreview", package)
    monkeypatch.setitem(sys.modules, "openreview.api", api)


def _write_credentials(path: Path, text: str, mode: int = 0o600) -> Path:
    path.write_text(text)
    path.chmod(mode)
    return path


# ── Status and reply parsing ────────────────────────────────────────────


@pytest.mark.parametrize(
    "venueid, venue, expected",
    [
        (f"{VENUE}/Submission", "ICLR 2027 Conference Submission", "active"),
        (VENUE, "ICLR 2027 Poster", "active"),
        (None, None, "active"),
        (f"{VENUE}/Withdrawn_Submission", "ICLR 2027 Conference Withdrawn Submission", "withdrawn"),
        ("TMLR/Withdrawn_Submission", None, "withdrawn"),
        (f"{VENUE}/Desk_Rejected_Submission", "ICLR 2027 Conference Desk Rejected Submission", "desk_rejected"),
        ("TMLR/Desk_Rejected", "Desk rejected by TMLR", "desk_rejected"),
        (f"{VENUE}/Rejected_Submission", "Submitted to ICLR 2027", "rejected"),
    ],
)
def test_submission_status(venueid, venue, expected):
    assert submission_status(venueid, venue) == expected


@pytest.mark.parametrize(
    "note, expected",
    [
        ({"invitations": [f"{PAPER}/-/Official_Review", f"{VENUE}/-/Edit"]}, "Official_Review"),
        ({"invitations": [f"{VENUE}/-/Edit", f"{PAPER}/-/Meta_Review"]}, "Meta_Review"),
        ({"invitations": [f"{VENUE}/-/Edit"]}, "Edit"),
        ({"invitation": f"{PAPER}/-/Public_Comment"}, "Public_Comment"),  # API v1 shape
        ({}, "Unknown"),
    ],
)
def test_reply_type(note, expected):
    assert reply_type(note) == expected


# ── fetch_forum: files, grouping, rendering ─────────────────────────────


def test_fetch_forum_groups_replies_and_writes_raw_json(tmp_path):
    client = FakeClient()
    summary = fetch_forum(FORUM, tmp_path, client=client)

    assert client.calls == [("get_note", FORUM), ("get_all_notes", FORUM)]
    assert summary["status"] == "withdrawn"
    assert summary["title"] == "Judging Lean Specifications"
    assert summary["decision"] is None
    assert summary["n_replies"] == 5
    assert summary["n_replies_by_type"] == {
        "Automated_Feedback": 1, "Official_Comment": 1, "Official_Review": 2, "Withdrawal": 1,
    }

    data = json.loads((tmp_path / FORUM / "forum.json").read_text())
    assert summary["json"] == str(tmp_path / FORUM / "forum.json")
    assert [r["id"] for r in data["replies"]] == CHRONOLOGICAL
    info = data["submission_info"]
    assert (info["withdrawn"], info["desk_rejected"], info["number"]) == (True, False, 35334)
    assert info["venueid"] == f"{VENUE}/Withdrawn_Submission"
    assert info["pdf_url"] == "https://openreview.net/pdf/0123abcd.pdf"
    assert (info["cdate"], info["mdate"], info["pdate"]) == (T0, T0 + 60_000, None)
    review = data["replies"][1]  # raw notes keep invitations, signatures, dates, and every content field
    assert review["invitations"] == [f"{PAPER}/-/Official_Review", f"{VENUE}/-/Edit"]
    assert review["signatures"] == [f"{PAPER}/Reviewer_Ab12"]
    assert (review["cdate"], review["tmdate"], review["replyto"]) == (T0 + 2_000, T0 + 62_000, FORUM)
    assert review["content"] == {"summary": {"value": REVIEW_TEXT}, "rating": {"value": 6}, "confidence": {"value": 4}}
    assert data["submission"]["content"]["keywords"] == {"value": ["Lean 4", "LLM judges"]}


def test_markdown_has_status_header_and_verbatim_replies_in_order(tmp_path):
    fetch_forum(FORUM, tmp_path, client=FakeClient())
    md = (tmp_path / FORUM / "forum.md").read_text()
    data = json.loads((tmp_path / FORUM / "forum.json").read_text())

    assert md == render_markdown(data)  # the JSON dump alone reproduces the Markdown
    assert md.startswith("# Judging Lean Specifications\n")
    assert f"- Status: **withdrawn** (venueid `{VENUE}/Withdrawn_Submission`" in md
    assert "- Decision: none posted" in md
    assert "- Dates: created 09-16-2025 05:20 UTC, modified 09-16-2025 05:21 UTC, published n/a" in md
    assert "- Replies: 5 (Automated_Feedback: 1, Official_Comment: 1, Official_Review: 2, Withdrawal: 1)" in md
    assert re.findall(r"^### \d+\. (\S+) by (\S+)", md, flags=re.M) == [
        ("Automated_Feedback", "Automated_Feedback_Agent"),
        ("Official_Review", "Reviewer_Ab12"),
        ("Official_Review", "Reviewer_Cd34"),
        ("Official_Comment", "Authors"),
        ("Withdrawal", "Authors"),
    ]
    assert f"#### summary\n\n{REVIEW_TEXT}\n" in md  # field text is verbatim, markdown and newlines included
    assert "#### rating\n\n6\n" in md
    assert "### abstract\n\nLine one.\n\nLine *two* with `code`.\n" in md
    assert "### keywords\n\nLean 4, LLM judges\n" in md
    assert "- id `com1`, replyto `rev1`, modified 09-16-2025 05:21 UTC" in md


def test_active_submission_reports_newest_decision(tmp_path):
    submission = _submission("ICLR 2027 Poster", VENUE)
    decisions = [
        _note(f"dec{i}", invitations=[f"{PAPER}/-/Decision"], cdate=T0 + i, signatures=[f"{VENUE}/Program_Chairs"],
              content={"decision": {"value": value}, "comment": {"value": "Congrats."}})
        for i, value in ((2, "Accept (Poster)"), (1, "Reject"))
    ]
    summary = fetch_forum(FORUM, tmp_path, client=FakeClient(submission, decisions))
    assert (summary["status"], summary["decision"], summary["n_replies_by_type"]) == (
        "active", "Accept (Poster)", {"Decision": 2},
    )
    assert "- Decision: Accept (Poster)" in (tmp_path / FORUM / "forum.md").read_text()


@pytest.mark.parametrize("bad_id", ["../escape", "a/b", "", "id with space"])
def test_fetch_forum_rejects_unsafe_forum_ids(tmp_path, bad_id):
    with pytest.raises(ValueError, match="Invalid OpenReview forum id"):
        fetch_forum(bad_id, tmp_path, client=FakeClient())
    assert list(tmp_path.iterdir()) == []


# ── Credentials ─────────────────────────────────────────────────────────


def test_load_credentials_from_env_takes_precedence(monkeypatch, tmp_path):
    cred = _write_credentials(tmp_path / "creds.txt", "file@example.com\nfile-pw\n")
    monkeypatch.setenv("OPENREVIEW_USERNAME", "env@example.com")
    monkeypatch.setenv("OPENREVIEW_PASSWORD", FAKE_PW)
    assert load_credentials(cred) == ("env@example.com", FAKE_PW)
    assert load_credentials(tmp_path / "missing.txt") == ("env@example.com", FAKE_PW)

    monkeypatch.delenv("OPENREVIEW_PASSWORD")  # half-set env falls back to the file
    assert load_credentials(cred) == ("file@example.com", "file-pw")


def test_load_credentials_from_file(tmp_path, caplog):
    cred = _write_credentials(tmp_path / "openreview_credentials.txt", f"\n  me@example.com \n{FAKE_PW}\n\n")
    with caplog.at_level(logging.WARNING, logger=oru.__name__):
        assert load_credentials(cred) == ("me@example.com", FAKE_PW)
    assert caplog.records == []  # mode 600: no permission warning


def test_world_readable_credentials_file_warns_without_leaking(tmp_path, caplog):
    cred = _write_credentials(tmp_path / "creds.txt", f"me@example.com\n{FAKE_PW}\n", mode=0o644)
    with caplog.at_level(logging.WARNING, logger=oru.__name__):
        assert load_credentials(cred)[1] == FAKE_PW
    assert "chmod 600" in caplog.text and FAKE_PW not in caplog.text


def test_missing_credentials_explain_setup_and_create_nothing(tmp_path):
    target = tmp_path / "keys" / "openreview_credentials.txt"
    with pytest.raises(OpenReviewAuthError) as err:
        load_credentials(target)
    message = str(err.value)
    assert "OPENREVIEW_USERNAME" in message and "OPENREVIEW_PASSWORD" in message
    assert str(target) in message and "chmod 600" in message and "read -rs" in message
    assert not target.parent.exists()


def test_malformed_credentials_file_does_not_echo_contents(tmp_path):
    cred = _write_credentials(tmp_path / "creds.txt", f"{FAKE_PW}\n")  # password only, no username line
    with pytest.raises(OpenReviewAuthError, match="1 non-empty line") as err:
        load_credentials(cred)
    assert FAKE_PW not in str(err.value)


# ── Login, lazy import, and secret redaction ────────────────────────────


def test_openreview_py_is_imported_lazily():
    tree = ast.parse(Path(oru.__file__).read_text())
    top_level = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level.add(node.module.split(".")[0])
    assert "openreview" not in top_level


@pytest.mark.parametrize("failure", ["not_installed", "wrong_architecture"])
def test_unusable_openreview_py_gives_install_hint(monkeypatch, failure):
    if failure == "not_installed":
        monkeypatch.setitem(sys.modules, "openreview", None)  # makes `import openreview` raise ImportError
        monkeypatch.setitem(sys.modules, "openreview.api", None)
    else:  # e.g. pycryptodome's x86_64 .so loaded by an arm64 interpreter raises OSError, not ImportError
        real_import = builtins.__import__

        def broken_native_import(name, *args, **kwargs):
            if name.startswith("openreview"):
                raise OSError("Cannot load native module 'Crypto.Hash._BLAKE2s': incompatible architecture")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", broken_native_import)
    with pytest.raises(ImportError, match="pip install openreview-py") as err:
        make_client("me@example.com", FAKE_PW)
    assert FAKE_PW not in str(err.value)


def test_make_client_logs_in_to_api_v2_with_given_credentials(monkeypatch):
    seen = {}

    class Client:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    _install_fake_openreview(monkeypatch, Client)
    assert isinstance(make_client("me@example.com", FAKE_PW), Client)
    assert seen == {"baseurl": "https://api2.openreview.net", "username": "me@example.com", "password": FAKE_PW}


def test_login_failure_never_leaks_password(monkeypatch):
    def failing_client(baseurl, username, password):
        # openreview-py raises OpenReviewException(dict); str() of a dict repr-escapes the quotes and backslash
        raise RuntimeError({"name": "LoginError", "message": f"bad password {password}", "echo": password})

    _install_fake_openreview(monkeypatch, failing_client)
    with pytest.raises(OpenReviewAuthError) as err:
        make_client("me@example.com", FAKE_PW)
    exc = err.value
    rendered = [str(exc), repr(exc), "".join(traceback.format_exception(exc))]
    for text in rendered:
        assert FAKE_PW not in text and repr(FAKE_PW)[1:-1] not in text
    assert "LoginError" in str(exc) and "***" in str(exc) and "me@example.com" in str(exc)
    assert exc.__cause__ is None and exc.__context__ is None


def test_redact_covers_escaped_forms():
    text = f"raw={FAKE_PW} repr={FAKE_PW!r} json={json.dumps(FAKE_PW)} url={quote(FAKE_PW, safe='')} tok={FAKE_TOKEN}"
    cleaned = redact(text, FAKE_PW, FAKE_TOKEN, None, "")
    for form in (FAKE_PW, repr(FAKE_PW)[1:-1], json.dumps(FAKE_PW)[1:-1], quote(FAKE_PW, safe=""), FAKE_TOKEN):
        assert form not in cleaned
    assert cleaned.startswith("raw=*** ")
    assert redact("nothing secret", None, "") == "nothing secret"


# ── CLI ─────────────────────────────────────────────────────────────────


def test_cli_fetches_several_forums_and_prints_only_redacted_summaries(monkeypatch, tmp_path, capsys):
    class Client(FakeClient):
        def __init__(self, baseurl, username, password):
            super().__init__()
            self.baseurl = baseurl

        def get_note(self, id, details=None):
            if id == "Broken1":
                raise RuntimeError(f"403 for {id}; server echoed {FAKE_PW} and {FAKE_TOKEN}")
            return super().get_note(id)

    _install_fake_openreview(monkeypatch, Client)
    monkeypatch.setenv("OPENREVIEW_USERNAME", "me@example.com")
    monkeypatch.setenv("OPENREVIEW_PASSWORD", FAKE_PW)
    code = main(["fetch", "--forum", FORUM, "--forum", "Broken1", "--out", str(tmp_path)])
    out, err = capsys.readouterr()

    assert code == 1 and err == ""
    summaries = json.loads(out)  # stdout is exactly the JSON summary list
    assert [s["forum_id"] for s in summaries] == [FORUM, "Broken1"]
    assert summaries[0]["status"] == "withdrawn" and summaries[0]["n_replies_by_type"]["Official_Review"] == 2
    assert summaries[1]["error"] == "RuntimeError: 403 for Broken1; server echoed *** and ***"
    written = [p.read_text() for p in (tmp_path / FORUM).iterdir()]
    for text in [out, err, *written]:
        assert FAKE_PW not in text and repr(FAKE_PW)[1:-1] not in text and FAKE_TOKEN not in text


def test_cli_without_credentials_exits_2_with_setup_hint(tmp_path, capsys):
    missing = tmp_path / "keys" / "openreview_credentials.txt"
    code = main(["fetch", "--forum", FORUM, "--out", str(tmp_path / "out"), "--credentials-file", str(missing)])
    out, err = capsys.readouterr()
    assert code == 2 and out == ""
    assert "No OpenReview credentials found" in err and "chmod 600" in err
    assert not (tmp_path / "out").exists()
