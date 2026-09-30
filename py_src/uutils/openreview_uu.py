r"""OpenReview forum fetcher. TLDR: dump a submission's status plus every review and comment to JSON + Markdown.

Usage:
    python -m uutils.openreview_uu fetch --forum yoLB5iP7Pn --forum xmHP3UhGQ8 --out ./or_dump
    from uutils.openreview_uu import fetch_forum; fetch_forum("yoLB5iP7Pn", "./or_dump")  # -> summary dict
    Writes <out>/<forum_id>/forum.json (every raw note) and forum.md (status header + replies, oldest first).
    Credentials: OPENREVIEW_USERNAME + OPENREVIEW_PASSWORD env vars, else ~/keys/openreview_credentials.txt
    (line 1 login email, line 2 password, chmod 600). Needs `pip install openreview-py` (imported lazily).

Status comes from the submission's venueid/venue strings: withdrawn, desk_rejected, rejected, or active
(under review or accepted; the newest Decision reply is reported separately). Replies are grouped by the
invitation suffix after '/-/' (Official_Review, Official_Comment, Meta_Review, Decision, Public_Comment, ...).
The password and access token are never printed, logged, or written, and error messages redact them.

One-time setup (type it in your own terminal so the password never enters a chat transcript or shell history):
    mkdir -p ~/keys && (umask 077 && printf 'OpenReview email: ' && read -r u && printf 'OpenReview password: ' && read -rs p && echo && printf '%s\n%s\n' "$u" "$p" > ~/keys/openreview_credentials.txt) && chmod 600 ~/keys/openreview_credentials.txt

Refs:
    - openreview-py (API v2 client): https://github.com/openreview/openreview-py
    - API v2 reference: https://docs.openreview.net/reference/api-v2
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shlex
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, quote_plus

log = logging.getLogger(__name__)

API_BASEURL = "https://api2.openreview.net"
WEB_BASEURL = "https://openreview.net"
DEFAULT_CREDENTIALS_FILE = "~/keys/openreview_credentials.txt"
FORUM_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class OpenReviewAuthError(RuntimeError):
    """Missing/malformed credentials or a failed login. Messages never contain the password or token."""


# ── Credentials and login ───────────────────────────────────────────────


def setup_command(credentials_file: str | Path = DEFAULT_CREDENTIALS_FILE) -> str:
    """zsh/bash one-liner that writes the credentials file (mode 600) without echoing the password."""
    target = Path(credentials_file).expanduser()
    f, d = shlex.quote(str(target)), shlex.quote(str(target.parent))
    return (
        f"mkdir -p {d} && (umask 077 && printf 'OpenReview email: ' && read -r u && "
        f"printf 'OpenReview password: ' && read -rs p && echo && "
        f"printf '%s\\n%s\\n' \"$u\" \"$p\" > {f}) && chmod 600 {f}"
    )


def load_credentials(credentials_file: str | Path = DEFAULT_CREDENTIALS_FILE) -> tuple[str, str]:
    """Return (username, password) from OPENREVIEW_USERNAME/OPENREVIEW_PASSWORD (both set), else the file.

    The file holds the login email on its first non-empty line and the password on the second; surrounding
    whitespace is ignored. Nothing else (keychain, browser stores) is ever consulted, and the file is never created.
    """
    username = os.environ.get("OPENREVIEW_USERNAME", "").strip()
    password = os.environ.get("OPENREVIEW_PASSWORD", "").strip()
    if username and password:
        return username, password

    path = Path(credentials_file).expanduser()
    how = (
        f"Set OPENREVIEW_USERNAME and OPENREVIEW_PASSWORD, or create {path} (line 1: login email, "
        f"line 2: password, permissions 600) by running this in your own terminal:\n    {setup_command(path)}"
    )
    if not path.is_file():
        raise OpenReviewAuthError(f"No OpenReview credentials found. {how}")
    if path.stat().st_mode & 0o077:
        log.warning("%s is readable by other users; run: chmod 600 %s", path, shlex.quote(str(path)))
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 2:
        raise OpenReviewAuthError(f"{path} has {len(lines)} non-empty line(s); expected 2. {how}")
    return lines[0], lines[1]


def redact(text: str, *secrets: str | None) -> str:
    """Replace every secret in text, including its repr/JSON/URL-escaped forms, with '***'."""
    for secret in secrets:
        if secret:
            forms = {secret, repr(secret)[1:-1], json.dumps(secret)[1:-1], quote(secret, safe=""), quote_plus(secret)}
            for form in sorted(forms, key=len, reverse=True):
                text = text.replace(form, "***")
    return text


def _openreview_client_cls() -> Any:
    try:
        from openreview.api import OpenReviewClient
    except (ImportError, OSError) as e:  # OSError: a native dependency built for another CPU architecture
        raise ImportError(f"uutils.openreview_uu needs a working openreview-py (pip install openreview-py): {e}") from e
    return OpenReviewClient


def make_client(
    username: str | None = None,
    password: str | None = None,
    baseurl: str = API_BASEURL,
    credentials_file: str | Path = DEFAULT_CREDENTIALS_FILE,
) -> Any:
    """Return a logged-in openreview.api.OpenReviewClient (API v2); credentials default to load_credentials().

    Accounts with multi-factor authentication need an interactive terminal, where openreview-py prompts for the code.
    """
    if not (username and password):
        username, password = load_credentials(credentials_file)
    client_cls = _openreview_client_cls()
    try:
        return client_cls(baseurl=baseurl, username=username, password=password)
    except Exception as e:
        reason = redact(f"{type(e).__name__}: {e}", password)
    # Raised outside the except block so the unredacted original is neither chained nor kept as __context__.
    raise OpenReviewAuthError(f"OpenReview login failed for {username} at {baseurl}: {reason}")


# ── Parsing ─────────────────────────────────────────────────────────────


def note_to_dict(note: Any) -> dict:
    """Every field of an openreview Note (or a raw note dict) as a plain JSON-ready dict."""
    return dict(note) if isinstance(note, dict) else {k: v for k, v in vars(note).items() if not k.startswith("_")}


def _value(field: Any) -> Any:
    """API v2 wraps content values as {'value': ...}; API v1 stores them bare."""
    return field["value"] if isinstance(field, dict) and "value" in field else field


def _invitations(note: dict) -> list[str]:
    return list(note.get("invitations") or ([note["invitation"]] if note.get("invitation") else []))


def reply_type(note: dict) -> str:
    """Invitation suffix after '/-/' (e.g. 'Official_Review'), preferring any suffix other than the generic 'Edit'."""
    kinds = [inv.rsplit("/-/", 1)[-1] for inv in _invitations(note)]
    return next((k for k in kinds if k != "Edit"), kinds[0] if kinds else "Unknown")


def count_reply_types(replies: list[dict]) -> dict[str, int]:
    return dict(sorted(Counter(reply_type(r) for r in replies).items()))


def submission_status(venueid: str | None, venue: str | None) -> str:
    """'withdrawn', 'desk_rejected', 'rejected', or 'active' (under review, accepted, or unknown)."""
    text = f"{venueid or ''} {venue or ''}".lower().replace("_", " ")
    if "withdrawn" in text:
        return "withdrawn"
    if "desk reject" in text:
        return "desk_rejected"
    if "reject" in text:
        return "rejected"
    return "active"


def submission_info(submission: dict) -> dict:
    """Title, venue, venueid, number, PDF path, dates (epoch ms), and parsed status of a submission note."""
    content = submission.get("content") or {}
    venue, venueid, pdf = (_value(content.get(k)) for k in ("venue", "venueid", "pdf"))
    status = submission_status(venueid, venue)
    return {
        "id": submission.get("id"),
        "number": submission.get("number"),
        "title": _value(content.get("title")),
        "venue": venue,
        "venueid": venueid,
        "status": status,
        "withdrawn": status == "withdrawn",
        "desk_rejected": status == "desk_rejected",
        "pdf": pdf,
        "pdf_url": f"{WEB_BASEURL}{pdf}" if isinstance(pdf, str) and pdf.startswith("/") else pdf,
        "cdate": submission.get("cdate"),
        "mdate": submission.get("mdate"),
        "pdate": submission.get("pdate"),
    }


def latest_decision(replies: list[dict]) -> Any:
    """The 'decision' field of the newest Decision reply, or None."""
    decisions = [_value((r.get("content") or {}).get("decision")) for r in replies if reply_type(r) == "Decision"]
    decisions = [d for d in decisions if d is not None]
    return decisions[-1] if decisions else None


# ── Rendering ───────────────────────────────────────────────────────────


def _fmt_ms(ms: Any) -> str:
    """Epoch milliseconds -> 'MM-DD-YYYY HH:MM UTC'; 'n/a' when missing."""
    if not isinstance(ms, (int, float)) or not ms:
        return "n/a"
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%m-%d-%Y %H:%M UTC")


def _field_text(value: Any) -> str:
    """Verbatim text: strings as-is, scalars and scalar lists inline, anything else as a JSON block."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list) and all(isinstance(v, (str, int, float)) for v in value):
        return ", ".join(str(v) for v in value)
    return "```json\n" + json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n```"


def _render_content(content: dict | None, heading: str) -> list[str]:
    lines: list[str] = []
    for key, field in (content or {}).items():
        lines += [f"{heading} {key}", "", _field_text(_value(field)), ""]
    return lines


def render_markdown(forum: dict) -> str:
    """Readable dump: status header, submission content, then one section per reply, oldest first."""
    info, replies = forum["submission_info"], forum["replies"]
    by_type = ", ".join(f"{k}: {v}" for k, v in forum["n_replies_by_type"].items()) or "none"
    lines = [
        f"# {info['title'] or forum['forum_id']}",
        "",
        f"- Forum: {forum['url']}",
        f"- Status: **{info['status']}** (venueid `{info['venueid']}`, venue `{info['venue']}`)",
        f"- Decision: {forum['decision'] if forum['decision'] is not None else 'none posted'}",
        f"- Submission number: {info['number']}",
        f"- Dates: created {_fmt_ms(info['cdate'])}, modified {_fmt_ms(info['mdate'])}, "
        f"published {_fmt_ms(info['pdate'])}",
        f"- PDF: {info['pdf_url'] or 'none'}",
        f"- Replies: {len(replies)} ({by_type})",
        f"- Fetched: {_fmt_ms(forum['fetched_at'])}",
        "",
        "## Submission content",
        "",
        *_render_content(forum["submission"].get("content"), "###"),
        f"## Replies ({len(replies)}, oldest first)",
        "",
    ]
    for i, reply in enumerate(replies, 1):
        signatures = reply.get("signatures") or []
        who = ", ".join(s.rsplit("/", 1)[-1] for s in signatures) or "unknown"
        invitations = ", ".join(f"`{inv}`" for inv in _invitations(reply)) or "none"
        signed = ", ".join(f"`{s}`" for s in signatures) or "none"
        lines += [
            f"### {i}. {reply_type(reply)} by {who} ({_fmt_ms(reply.get('cdate') or reply.get('tcdate'))})",
            "",
            f"- id `{reply.get('id')}`, replyto `{reply.get('replyto')}`, "
            f"modified {_fmt_ms(reply.get('mdate') or reply.get('tmdate'))}",
            f"- invitations: {invitations}",
            f"- signatures: {signed}",
            "",
            *_render_content(reply.get("content"), "####"),
        ]
    return "\n".join(lines).rstrip() + "\n"


# ── Fetching ────────────────────────────────────────────────────────────


def fetch_forum(forum_id: str, out_dir: str | Path, client: Any = None) -> dict:
    """Fetch a forum (submission + all replies) and write <out_dir>/<forum_id>/forum.json and forum.md.

    Args:
        forum_id: Forum id, i.e. the submission note id (e.g. 'yoLB5iP7Pn').
        out_dir: Directory under which <forum_id>/ is created.
        client: openreview.api.OpenReviewClient, or any object with get_note(id) and get_all_notes(forum=...);
            defaults to make_client(), which logs in with load_credentials().

    Returns:
        Summary dict: forum_id, title, status, venue, venueid, decision, n_replies, n_replies_by_type, file paths.
    """
    if not FORUM_ID_RE.match(forum_id or ""):
        raise ValueError(f"Invalid OpenReview forum id: {forum_id!r}")
    if client is None:
        client = make_client()
    submission = note_to_dict(client.get_note(forum_id))
    notes = [note_to_dict(n) for n in client.get_all_notes(forum=forum_id)]
    replies = sorted(
        (n for n in notes if n.get("id") != forum_id),  # the forum query also returns the submission itself
        key=lambda n: (n.get("cdate") or n.get("tcdate") or 0, n.get("id") or ""),
    )
    info = submission_info(submission)
    forum = {
        "forum_id": forum_id,
        "url": f"{WEB_BASEURL}/forum?id={forum_id}",
        "baseurl": getattr(client, "baseurl", None),
        "fetched_at": int(time.time() * 1000),
        "submission_info": info,
        "decision": latest_decision(replies),
        "n_replies_by_type": count_reply_types(replies),
        "submission": submission,
        "replies": replies,
    }
    dest = Path(out_dir).expanduser() / forum_id
    dest.mkdir(parents=True, exist_ok=True)
    json_path, md_path = dest / "forum.json", dest / "forum.md"
    json_path.write_text(json.dumps(forum, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(forum), encoding="utf-8")
    return {
        "forum_id": forum_id,
        "title": info["title"],
        "status": info["status"],
        "venue": info["venue"],
        "venueid": info["venueid"],
        "decision": forum["decision"],
        "n_replies": len(replies),
        "n_replies_by_type": forum["n_replies_by_type"],
        "json": str(json_path),
        "markdown": str(md_path),
    }


# ── CLI ─────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch OpenReview forums (submission + all replies) to JSON + Markdown."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    fetch = subparsers.add_parser("fetch", help="Fetch one or more forums your account can read")
    fetch.add_argument("--forum", action="append", required=True, help="Forum id; repeat for several forums")
    fetch.add_argument("--out", default="./openreview_dump", help="Output directory (default: ./openreview_dump)")
    fetch.add_argument("--baseurl", default=API_BASEURL, help=f"API v2 base URL (default: {API_BASEURL})")
    fetch.add_argument(
        "--credentials-file",
        default=DEFAULT_CREDENTIALS_FILE,
        help=f"Used when the OPENREVIEW_* env vars are unset (default: {DEFAULT_CREDENTIALS_FILE})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Print only a JSON list of per-forum summaries; exit 1 if any forum failed, 2 if login was impossible."""
    args = _build_parser().parse_args(argv)
    try:
        username, password = load_credentials(args.credentials_file)
        client = make_client(username, password, baseurl=args.baseurl)
    except (OpenReviewAuthError, ImportError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    summaries, failed = [], False
    for forum_id in args.forum:
        try:
            summaries.append(fetch_forum(forum_id, args.out, client=client))
        except Exception as e:
            failed = True
            error = redact(f"{type(e).__name__}: {e}", password, getattr(client, "token", None))
            summaries.append({"forum_id": forum_id, "error": error})
    print(json.dumps(summaries, indent=2, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
