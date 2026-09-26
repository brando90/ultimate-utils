"""SMS (Short Message Service) — text via Twilio or Tasker + AutoRemote.

Dry-run is the default. A dry-run makes no network request, does not log in,
and does not read any credential file. Pass ``dry_run=False`` (library) or
``--send`` / ``--execute`` (command line) to actually text someone.

Quick usage:
    from uutils.sms_uu import SMSClient

    # Preview only. Works on a machine with no ~/keys/ files.
    client = SMSClient.from_twilio()
    client.send_sms("+15550000000", "Hello from uutils")

    # Actually send from the Twilio number (reads the credentials file):
    client = SMSClient.from_twilio(dry_run=False)
    client.send_sms("+15550000000", "Hello from uutils")

    # From the Android phone's own number, via Tasker + AutoRemote:
    client = SMSClient.from_autoremote(self_number="+15550000001", dry_run=False)
    client.send_self_reminder("Check the experiment")

Command line (dry-run unless ``--send`` or ``--execute`` is present):
    python -m uutils.sms_uu send --backend twilio --to +15550000000 --message "Hello"
    python -m uutils.sms_uu send --backend autoremote --to +15550000000 --message "Hello" --send

Setup (Twilio — reliable, sends from a separate Twilio number):
    1. Create an account: https://www.twilio.com/try-twilio
    2. Buy a phone number. A local number is about $1 per month, plus a
       per-message fee (see https://www.twilio.com/en-us/sms/pricing).
    3. Copy the Account SID and Auth Token from the Twilio console.
    4. Save credentials (placeholders only — never commit this file):
       mkdir -p ~/keys
       cat > ~/keys/twilio_credentials.json << 'JSON'
       {
           "account_sid": "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
           "auth_token": "your_auth_token",
           "from_number": "+15550000000",
           "self_number": "+15550000001"
       }
       JSON
       chmod 600 ~/keys/twilio_credentials.json
    ``self_number`` is optional. It is only used by ``send_self_reminder``.
    ``from_number`` is the Twilio number the message is sent from.

Setup (Tasker + AutoRemote — Android, sends from your real number):
    1. On the phone, install Tasker (https://tasker.joaoapps.com/) and
       AutoRemote (https://joaoapps.com/autoremote/).
    2. In AutoRemote, copy the personal key (a URL with ``?key=``).
    3. Save that key, and nothing else, to a file:
       mkdir -p ~/keys
       printf '%s\n' 'YOUR_AUTOREMOTE_KEY' > ~/keys/tasker_autoremote_key.txt
       chmod 600 ~/keys/tasker_autoremote_key.txt
    4. Create a Tasker profile that reacts to an AutoRemote message whose
       text is ``sms=:=<to>=:=<message>`` (split on ``=:=``). The first
       field is the literal ``sms``, the second is the destination number,
       the third is the body. The profile's task must be Tasker's "Send
       SMS" action to that number. Without this profile the phone receives
       the AutoRemote event and does not text anyone.
       Avoid the characters ``=:=`` inside the message body; Tasker splits
       on that separator.

Not implemented (future work):
    Google Messages for web (https://messages.google.com/web) can text from
    the phone's own number, but only after that phone scans a QR (quick
    response) code and stays paired in a browser. That browser-automation
    path is out of scope here.

Refs:
    - Twilio Messages: https://www.twilio.com/docs/sms/api/message-resource
    - AutoRemote message URL: https://joaoapps.com/autoremote/personal/
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Literal

log = logging.getLogger(__name__)

BackendName = Literal["twilio", "autoremote"]

DEFAULT_TWILIO_CREDENTIALS = "~/keys/twilio_credentials.json"
DEFAULT_AUTOREMOTE_KEY = "~/keys/tasker_autoremote_key.txt"

TWILIO_MESSAGES_URL = "https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
AUTOREMOTE_SEND_URL = "https://autoremotejoaomgcd.appspot.com/sendmessage"

# E.164 (ITU-T recommendation) is a leading "+" plus 8 to 15 digits.
MIN_E164_DIGITS = 8
MAX_E164_DIGITS = 15
_SEPARATORS = set(" -()")
REQUEST_TIMEOUT_SECONDS = 30

_TWILIO_REQUIRED_KEYS = ("account_sid", "auth_token", "from_number")


# ── Phone numbers ─────────────────────────────────────────────────────

def normalize_phone(phone: str) -> str:
    """Return ``phone`` as E.164: a leading ``+`` and digits only.

    Spaces, dashes, and parentheses are removed. An empty value, a number
    with fewer than 8 or more than 15 digits, or any other character raises
    ``ValueError``. A missing leading ``+`` is added; the caller must already
    include the country code (``15550000000``, not a local number).
    """
    if not isinstance(phone, str):
        raise ValueError(f"Phone number must be a string, got {type(phone).__name__}")
    raw = phone.strip()
    if not raw:
        raise ValueError(
            "Phone number is empty — provide an E.164 number like '+15550000000'"
        )

    digits: list[str] = []
    seen_plus = False
    for idx, char in enumerate(raw):
        if char.isdigit():
            digits.append(char)
            continue
        if char == "+":
            if idx != 0 or seen_plus:
                raise ValueError(
                    f"Invalid phone number {phone!r}: '+' is only allowed once, at the start"
                )
            seen_plus = True
            continue
        if char in _SEPARATORS:
            continue
        raise ValueError(
            f"Invalid phone number {phone!r}: only digits, a leading '+', "
            "spaces, dashes, and parentheses are allowed"
        )

    if not digits:
        raise ValueError(
            "Phone number is empty — provide an E.164 number like '+15550000000'"
        )
    count = len(digits)
    if count < MIN_E164_DIGITS:
        raise ValueError(
            f"Phone number {phone!r} is too short ({count} digits). "
            f"E.164 numbers need at least {MIN_E164_DIGITS} digits, "
            "including the country code (for example '+15550000000')."
        )
    if count > MAX_E164_DIGITS:
        raise ValueError(
            f"Phone number {phone!r} is too long ({count} digits). "
            f"E.164 numbers have at most {MAX_E164_DIGITS} digits."
        )
    return "+" + "".join(digits)


# ── Credentials (read only when dry_run is False) ─────────────────────

def _load_twilio_credentials(path: str | Path) -> dict[str, Any]:
    """Load the Twilio JSON file. Callers must not use this on a dry-run."""
    fpath = Path(path).expanduser()
    if not fpath.is_file():
        raise FileNotFoundError(
            f"Twilio credentials not found at {fpath}. Create "
            f"{DEFAULT_TWILIO_CREDENTIALS} and chmod 600 it. "
            "See the sms_uu module docstring for the one-time setup."
        )
    try:
        data = json.loads(fpath.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Twilio credentials file {fpath} is not valid JSON ({exc.msg})"
        ) from exc
    if not isinstance(data, dict):
        raise ValueError(f"Twilio credentials file {fpath} must contain a JSON object")
    missing = [key for key in _TWILIO_REQUIRED_KEYS if not str(data.get(key, "")).strip()]
    if missing:
        raise ValueError(
            f"Twilio credentials file {fpath} is missing: {', '.join(missing)}"
        )
    return data


def _load_autoremote_key(path: str | Path) -> str:
    """Load the AutoRemote personal key. Callers must not use this on a dry-run."""
    fpath = Path(path).expanduser()
    if not fpath.is_file():
        raise FileNotFoundError(
            f"AutoRemote key not found at {fpath}. Save the key to "
            f"{DEFAULT_AUTOREMOTE_KEY} and chmod 600 it. "
            "See the sms_uu module docstring for the one-time setup."
        )
    key = fpath.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError(f"AutoRemote key file {fpath} is empty")
    return key


def _response_body(resp: Any) -> dict[str, Any]:
    """Return a dict from a ``requests`` response, even when the body is plain text."""
    try:
        payload = resp.json()
    except ValueError:
        return {"status_code": getattr(resp, "status_code", None), "text": getattr(resp, "text", "")}
    if isinstance(payload, dict):
        return payload
    return {"result": payload}


def _requests_module():
    """Import ``requests`` only when a message is actually sent."""
    import requests
    return requests


# ── Client ────────────────────────────────────────────────────────────

class SMSClient:
    """Send an SMS through one backend. Dry-run unless ``dry_run=False``.

    Construct with :meth:`from_twilio` or :meth:`from_autoremote`. Credential
    files are opened only inside a non-dry-run send.
    """

    def __init__(
        self,
        backend: str,
        *,
        dry_run: bool = True,
        credentials_file: str | Path = "",
        key_file: str | Path = "",
        self_number: str = "",
    ) -> None:
        if backend not in ("twilio", "autoremote"):
            raise ValueError(
                f"Unknown SMS backend {backend!r}. Choose 'twilio' or 'autoremote'."
            )
        self.backend: BackendName = backend  # type: ignore[assignment]
        self.dry_run = dry_run
        self.credentials_file = str(credentials_file)
        self.key_file = str(key_file)
        self.self_number = normalize_phone(self_number) if self_number else ""
        self._twilio: dict[str, Any] | None = None
        self._autoremote_key: str | None = None
        self._from_number = ""

    def __repr__(self) -> str:
        return (
            f"SMSClient(backend={self.backend!r}, dry_run={self.dry_run!r}, "
            f"self_number={self.self_number!r})"
        )

    @classmethod
    def from_twilio(
        cls,
        credentials_file: str | Path = DEFAULT_TWILIO_CREDENTIALS,
        dry_run: bool = True,
        self_number: str = "",
    ) -> SMSClient:
        """Client that texts from a Twilio number.

        ``credentials_file`` is the JSON path (default ``~/keys/twilio_credentials.json``).
        It is not read when ``dry_run`` is true. ``self_number`` overrides the
        optional ``self_number`` field in that file; an empty value means
        "use the file" on a real send, and "unset" on a dry-run.
        """
        return cls(
            "twilio",
            dry_run=dry_run,
            credentials_file=credentials_file,
            self_number=self_number,
        )

    @classmethod
    def from_autoremote(
        cls,
        key_file: str | Path = DEFAULT_AUTOREMOTE_KEY,
        self_number: str = "",
        dry_run: bool = True,
    ) -> SMSClient:
        """Client that texts from the Android phone via Tasker + AutoRemote.

        ``key_file`` (default ``~/keys/tasker_autoremote_key.txt``) is not read
        when ``dry_run`` is true. ``self_number`` is the phone's own number,
        used only by :meth:`send_self_reminder`.
        """
        return cls(
            "autoremote",
            dry_run=dry_run,
            key_file=key_file,
            self_number=self_number,
        )

    def send_sms(self, to: str, message: str) -> dict[str, Any]:
        """Text ``message`` to ``to``.

        Dry-run (the default) returns
        ``{"dry_run": True, "backend": ..., "to": ..., "message": ...}``
        and prints a one-line preview. A real send returns the service's
        response JSON (or ``{"status_code", "text"}`` when the body is not JSON).
        """
        if not isinstance(message, str):
            raise ValueError(f"Message must be a string, got {type(message).__name__}")
        normalized = normalize_phone(to)
        if self.dry_run:
            return self._dry_run_result(normalized, message)
        if self.backend == "twilio":
            return self._send_twilio(normalized, message)
        return self._send_autoremote(normalized, message)

    def send_self_reminder(self, message: str) -> dict[str, Any]:
        """Text ``message`` to this client's ``self_number``.

        Raises ``ValueError`` when no self number was configured. On a Twilio
        dry-run the credentials file is not opened, so ``self_number`` must be
        passed to :meth:`from_twilio` (or to :meth:`from_autoremote`). On a
        real Twilio send, ``self_number`` may instead come from the JSON file.
        """
        if self.backend == "twilio" and not self.dry_run and not self.self_number:
            self._ensure_twilio()
        if not self.self_number:
            raise ValueError(
                "self_number is not set, so a self reminder has no destination. "
                "Twilio: add \"self_number\" to "
                f"{DEFAULT_TWILIO_CREDENTIALS}, or pass self_number= to "
                "SMSClient.from_twilio(). AutoRemote: pass self_number= to "
                "SMSClient.from_autoremote()."
            )
        return self.send_sms(self.self_number, message)

    def _dry_run_result(self, to: str, message: str) -> dict[str, Any]:
        preview = message if len(message) <= 200 else message[:200] + "..."
        line = f"[DRY-RUN] SMS via {self.backend} to {to}: {preview}"
        log.info("%s", line)
        print(line)
        return {
            "dry_run": True,
            "backend": self.backend,
            "to": to,
            "message": message,
        }

    def _ensure_twilio(self) -> dict[str, Any]:
        if self._twilio is None:
            data = _load_twilio_credentials(self.credentials_file)
            self._from_number = normalize_phone(str(data["from_number"]))
            raw_self = str(data.get("self_number", "")).strip()
            if not self.self_number and raw_self:
                self.self_number = normalize_phone(raw_self)
            self._twilio = data
        return self._twilio

    def _send_twilio(self, to: str, message: str) -> dict[str, Any]:
        creds = self._ensure_twilio()
        account_sid = str(creds["account_sid"]).strip()
        auth_token = str(creds["auth_token"]).strip()
        url = TWILIO_MESSAGES_URL.format(account_sid=account_sid)
        requests = _requests_module()
        resp = requests.post(
            url,
            auth=(account_sid, auth_token),
            data={"To": to, "From": self._from_number, "Body": message},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        result = _response_body(resp)
        log.info("Twilio SMS sent to %s: sid=%s", to, result.get("sid", "?"))
        return result

    def _send_autoremote(self, to: str, message: str) -> dict[str, Any]:
        if self._autoremote_key is None:
            self._autoremote_key = _load_autoremote_key(self.key_file)
        # The phone's Tasker profile must split this exact separator.
        payload = f"sms=:={to}=:={message}"
        requests = _requests_module()
        resp = requests.get(
            AUTOREMOTE_SEND_URL,
            params={"key": self._autoremote_key, "message": payload},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        result = _response_body(resp)
        log.info("AutoRemote SMS requested for %s", to)
        return result


# ── CLI ───────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m uutils.sms_uu",
        description=(
            "Send an SMS via Twilio or Tasker + AutoRemote. "
            "Does nothing on the network unless --send or --execute is given."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    send = sub.add_parser("send", help="Send one SMS (dry-run unless --send/--execute)")
    send.add_argument("--backend", required=True, choices=("twilio", "autoremote"))
    send.add_argument("--to", required=True, help="Destination number, for example +15550000000")
    send.add_argument("--message", required=True, help="Message body")
    send.add_argument(
        "--send",
        action="store_true",
        help="Actually send. Without this flag (or --execute) the command is a dry-run.",
    )
    send.add_argument(
        "--execute",
        action="store_true",
        help="Alias of --send. Actually send instead of printing a dry-run preview.",
    )
    send.add_argument(
        "--credentials-file",
        default=DEFAULT_TWILIO_CREDENTIALS,
        help=f"Twilio JSON path (default: {DEFAULT_TWILIO_CREDENTIALS}). Read only with --send.",
    )
    send.add_argument(
        "--key-file",
        default=DEFAULT_AUTOREMOTE_KEY,
        help=f"AutoRemote key path (default: {DEFAULT_AUTOREMOTE_KEY}). Read only with --send.",
    )
    send.add_argument(
        "--self-number",
        default="",
        help="Optional. Stored for send_self_reminder; the send command itself uses --to.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Command-line entry. Returns 0 on success, 2 on a user error."""
    args = _build_parser().parse_args(argv)
    dry_run = not (args.send or args.execute)
    try:
        if args.backend == "twilio":
            client = SMSClient.from_twilio(
                credentials_file=args.credentials_file,
                dry_run=dry_run,
                self_number=args.self_number,
            )
        else:
            client = SMSClient.from_autoremote(
                key_file=args.key_file,
                self_number=args.self_number,
                dry_run=dry_run,
            )
        result = client.send_sms(args.to, args.message)
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        # A live send can fail in ``requests`` (imported only on that path).
        # Report it without a traceback full of request internals, and without
        # treating programmer errors in dry-run as a usage mistake.
        if exc.__class__.__module__.split(".")[0] not in {"requests", "urllib3"}:
            raise
        print(f"error: {exc.__class__.__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
