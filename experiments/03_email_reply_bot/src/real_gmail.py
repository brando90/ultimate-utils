"""Gmail API-backed implementation of :class:`GmailClient`.

Kept in its own module so the rest of the package (and the tests) have no
hard dependency on ``google-api-python-client``. Import this only in the
deployment entry point.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from .gmail_client import FetchedMessage, build_reply_mime, encode_raw_for_gmail

_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
]


def _build_service(credentials_path: Path, token_path: Path) -> Any:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds: Credentials | None = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), _SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), _SCOPES)
            creds = flow.run_local_server(port=0)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json())
    return build("gmail", "v1", credentials=creds, cache_discovery=False)


class RealGmailClient:
    def __init__(
        self,
        *,
        user: str,
        bot_from_addr: str,
        credentials_path: Path,
        token_path: Path,
        processed_label_id: str | None = None,
    ) -> None:
        self.user = user
        self.bot_from_addr = bot_from_addr
        self.processed_label_id = processed_label_id
        self._service = _build_service(credentials_path, token_path)

    def fetch_unseen(self, label: str) -> list[FetchedMessage]:
        resp = (
            self._service.users()
            .messages()
            .list(userId=self.user, labelIds=[label], q="-label:claude-processed")
            .execute()
        )
        out: list[FetchedMessage] = []
        for m in resp.get("messages", []):
            raw_resp = (
                self._service.users()
                .messages()
                .get(userId=self.user, id=m["id"], format="raw")
                .execute()
            )
            raw = base64.urlsafe_b64decode(raw_resp["raw"])
            out.append(
                FetchedMessage(
                    message_id=m["id"],
                    thread_id=raw_resp.get("threadId", ""),
                    raw=raw,
                )
            )
        return out

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
        raw = build_reply_mime(
            from_addr=self.bot_from_addr,
            to_addr=to,
            subject=subject,
            in_reply_to=in_reply_to,
            references=references,
            body_text=body_text,
        )
        sent = (
            self._service.users()
            .messages()
            .send(
                userId=self.user,
                body={"raw": encode_raw_for_gmail(raw), "threadId": thread_id},
            )
            .execute()
        )
        return sent.get("id", "")
