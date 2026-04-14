"""Email utilities — reusable SMTP notifier with BCC, attachments, and dry-run mode.

Quick usage:
    from uutils.emailing import send_email_smtp
    send_email_smtp(
        to="recipient@example.com",
        subject="Hello",
        body="World",
        smtp_user="you@gmail.com",
        smtp_pass_file="~/keys/gmail_app_password.txt",
    )

Class-based usage (for pipelines):
    from uutils.emailing import SMTPNotifier, DryRunNotifier
    notifier = SMTPNotifier(
        host="smtp.gmail.com", port=587,
        user="you@gmail.com",
        password=read_secret("SMTP_PASS", "SMTP_PASS_FILE"),
        from_addr="you@gmail.com",
        bcc="you@gmail.com",
    )
    notifier.notify("recipient@example.com", "Subject", "Body")
    notifier.notify_with_attachments("r@example.com", "Subj", "Body", [Path("paper.pdf")])

Setup: Gmail App Password
    1. Enable 2-Step Verification: https://myaccount.google.com/security
    2. Create app password: https://myaccount.google.com/apppasswords
    3. Save to file: pbpaste > ~/keys/gmail_app_password.txt && chmod 600 ~/keys/gmail_app_password.txt

Refs:
    - Gmail SMTP: https://support.google.com/a/answer/176600
    - App passwords: https://support.google.com/accounts/answer/185833
    - smtplib: https://docs.python.org/3/library/smtplib.html
"""
from __future__ import annotations

import json
import logging
import os
import smtplib
from contextlib import contextmanager
from email.message import Message
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from socket import gethostname

log = logging.getLogger(__name__)

_OUTLOOK_SEND_SCRIPT = """
on run argv
    if (count of argv) < 4 then error "Expected recipient, subject, body, and sender arguments."

    set recipientAddress to item 1 of argv
    set messageSubject to item 2 of argv
    set messageBody to item 3 of argv
    set senderAddress to item 4 of argv
    set attachmentPaths to {}
    if (count of argv) > 4 then set attachmentPaths to items 5 thru -1 of argv

    tell application "Microsoft Outlook"
        if senderAddress is not "" then
            set newMsg to make new outgoing message with properties {subject:messageSubject, content:messageBody, sender:{address:senderAddress}}
        else
            set newMsg to make new outgoing message with properties {subject:messageSubject, content:messageBody}
        end if

        make new to recipient at newMsg with properties {email address:{address:recipientAddress}}
        repeat with attachmentPath in attachmentPaths
            make new attachment at newMsg with properties {file:(POSIX file (contents of attachmentPath))}
        end repeat
        send newMsg
    end tell
end run
"""


# ── Secret reading ─────────────────────────────────────────────────────

def read_secret(env_var: str, file_env_var: str = "", default: str = "") -> str:
    """Read a secret from an env var directly, or from a file path in a *_FILE env var.

    Precedence: env_var value > file contents at file_env_var > default.

    Examples:
        # Direct env var
        os.environ["SMTP_PASS"] = "mypass"
        read_secret("SMTP_PASS")  # -> "mypass"

        # File reference
        os.environ["SMTP_PASS_FILE"] = "/home/user/keys/gmail_app_password.txt"
        read_secret("SMTP_PASS", "SMTP_PASS_FILE")  # -> contents of file
    """
    val = os.environ.get(env_var, "")
    if val:
        return val
    if file_env_var:
        fpath = os.environ.get(file_env_var, "")
        if fpath:
            p = Path(fpath).expanduser()
            if p.is_file():
                return p.read_text().strip()
    return default


def _message_with_hostname(message: str) -> str:
    return f"{message}\nSent from Hostname: {gethostname()}"


def _normalize_attachments(attachments: list[Path | str]) -> list[Path]:
    return [Path(path).expanduser() for path in attachments]


def _existing_attachment_paths(attachments: list[Path | str] | None) -> list[str]:
    existing_paths: list[str] = []
    for path in _normalize_attachments(attachments or []):
        resolved_path = path.resolve()
        if not resolved_path.is_file():
            log.warning("Attachment not found, skipping: %s", resolved_path)
            continue
        existing_paths.append(str(resolved_path))
    return existing_paths


def _read_legacy_password(password_path: str | Path | None) -> str:
    if not password_path:
        raise ValueError("password_path is required for legacy email helpers")
    config = json.loads(Path(password_path).expanduser().read_text())
    password = str(config.get("password", "")).strip()
    if not password:
        raise ValueError("password_path must contain a JSON object with a non-empty 'password'")
    return password


def _attach_files(msg: MIMEMultipart, attachments: list[Path | str]) -> int:
    attached_count = 0
    for file_path in _normalize_attachments(attachments):
        if not file_path.is_file():
            log.warning("Attachment not found, skipping: %s", file_path)
            continue
        part = MIMEApplication(file_path.read_bytes(), Name=file_path.name)
        part["Content-Disposition"] = f'attachment; filename="{file_path.name}"'
        msg.attach(part)
        attached_count += 1
    return attached_count


def _send_legacy_relay(
    subject: str,
    message: str,
    destination: str,
    attachment: Path | None = None,
) -> None:
    from_address = "miranda9@intel-research.net."
    with smtplib.SMTP("smtp.intel-research.net", 25, timeout=30) as server:
        if attachment is None:
            full_message = (
                f"From: {from_address}\n"
                f"To: {destination}\n"
                f"Subject: {subject}\n"
                f"{message}"
            )
            server.sendmail(from_address, destination, full_message)
            return

        msg = MIMEMultipart()
        msg["Subject"] = subject
        msg["From"] = from_address
        msg["To"] = destination
        msg.attach(MIMEText(message, "plain"))
        _attach_files(msg, [attachment])
        server.send_message(msg)


# ── Notifier classes ───────────────────────────────────────────────────

class Notifier:
    """Base class for email notifiers."""

    def notify(self, to_email: str, subject: str, body: str) -> None:
        raise NotImplementedError

    def notify_with_attachments(
        self, to_email: str, subject: str, body: str,
        attachments: list[Path | str],
    ) -> None:
        raise NotImplementedError


class DryRunNotifier(Notifier):
    """Write emails to a local directory instead of sending (for testing)."""

    def __init__(self, outbox_dir: str | Path = "artifacts/email_outbox"):
        self.outbox = Path(outbox_dir)
        self.outbox.mkdir(parents=True, exist_ok=True)

    def _save(self, to_email: str, subject: str, body: str,
              attachments: list[Path | str] | None = None) -> Path:
        import time
        ts = int(time.time())
        safe = to_email.replace("@", "_at_")
        suffix = "_attachments" if attachments else ""
        path = self.outbox / f"{ts}_{safe}{suffix}.txt"
        lines = [f"To: {to_email}", f"Subject: {subject}"]
        if attachments:
            lines.append("Attachments:")
            lines.extend(f"  - {a}" for a in attachments)
        lines.append("")
        lines.append(body)
        path.write_text("\n".join(lines))
        log.info("[DRY-RUN] Email saved to %s", path)
        return path

    def notify(self, to_email: str, subject: str, body: str) -> None:
        self._save(to_email, subject, body)

    def notify_with_attachments(
        self, to_email: str, subject: str, body: str,
        attachments: list[Path | str],
    ) -> None:
        self._save(to_email, subject, body, attachments)


class SMTPNotifier(Notifier):
    """Send real emails via SMTP with optional BCC and attachments.

    Args:
        host: SMTP server hostname (e.g., "smtp.gmail.com")
        port: SMTP port (587 for STARTTLS, 465 for SSL)
        user: SMTP login username (usually your email)
        password: SMTP password or app password
        from_addr: From address shown in emails
        bcc: Optional BCC address (invisible to recipient)
    """

    def __init__(
        self, host: str, port: int, user: str, password: str,
        from_addr: str, bcc: str = "",
    ):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.from_addr = from_addr
        self.bcc = bcc

    @contextmanager
    def _smtp_client(self):
        if self.port == 465:
            with smtplib.SMTP_SSL(self.host, self.port, timeout=30) as server:
                server.login(self.user, self.password)
                yield server
                return

        with smtplib.SMTP(self.host, self.port, timeout=30) as server:
            server.starttls()
            server.login(self.user, self.password)
            yield server

    def _send(self, msg: Message, to_email: str) -> None:
        """Send a message, adding BCC recipient if configured."""
        if not self.user or not self.password or not self.from_addr:
            raise ValueError("SMTPNotifier requires non-empty user, password, and from_addr")
        recipients = [to_email]
        if self.bcc and self.bcc != to_email:
            recipients.append(self.bcc)
        with self._smtp_client() as server:
            server.sendmail(self.from_addr, recipients, msg.as_string())

    def notify(self, to_email: str, subject: str, body: str) -> None:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = self.from_addr
        msg["To"] = to_email
        self._send(msg, to_email)
        log.info("Email sent to %s (bcc: %s): %s",
                 to_email, self.bcc or "none", subject)

    def notify_with_attachments(
        self, to_email: str, subject: str, body: str,
        attachments: list[Path | str],
    ) -> None:
        msg = MIMEMultipart()
        msg["Subject"] = subject
        msg["From"] = self.from_addr
        msg["To"] = to_email
        msg.attach(MIMEText(body))
        attached_count = _attach_files(msg, attachments)
        self._send(msg, to_email)
        log.info("Email with %d attachment(s) sent to %s (bcc: %s): %s",
                 attached_count, to_email, self.bcc or "none", subject)


def get_notifier(
    smtp_host: str = "",
    smtp_port: int = 587,
    smtp_user: str = "",
    smtp_pass: str = "",
    email_from: str = "",
    bcc: str = "",
    dry_run: bool = False,
    outbox_dir: str | Path = "artifacts/email_outbox",
) -> Notifier:
    """Factory: returns SMTPNotifier if credentials are provided, else DryRunNotifier."""
    if dry_run or not smtp_host:
        return DryRunNotifier(outbox_dir)
    return SMTPNotifier(smtp_host, smtp_port, smtp_user, smtp_pass, email_from, bcc=bcc)


# ── Convenience functions ──────────────────────────────────────────────

def send_email_smtp(
    to: str,
    subject: str,
    body: str,
    smtp_user: str = "",
    smtp_pass: str = "",
    smtp_pass_file: str = "",
    smtp_host: str = "smtp.gmail.com",
    smtp_port: int = 587,
    from_addr: str = "",
    bcc: str = "",
    attachments: list[Path | str] | None = None,
) -> None:
    """Send an email via SMTP (Gmail by default). Simple one-shot function.

    Password can be provided directly via smtp_pass, or read from a file
    via smtp_pass_file (e.g., "~/keys/gmail_app_password.txt").

    Args:
        to: Recipient email address.
        subject: Email subject line.
        body: Plain text email body.
        smtp_user: SMTP login (usually your email). Defaults to from_addr.
        smtp_pass: SMTP password directly.
        smtp_pass_file: Path to file containing SMTP password (alternative to smtp_pass).
        smtp_host: SMTP server (default: smtp.gmail.com).
        smtp_port: SMTP port (default: 587 for STARTTLS).
        from_addr: From address. Defaults to smtp_user.
        bcc: Optional BCC address.
        attachments: Optional list of file Paths to attach.
    """
    password = smtp_pass
    if not password and smtp_pass_file:
        p = Path(smtp_pass_file).expanduser()
        if p.is_file():
            password = p.read_text().strip()
    if not password:
        raise ValueError("No SMTP password: provide smtp_pass or smtp_pass_file")
    if not smtp_user:
        smtp_user = from_addr
    if not from_addr:
        from_addr = smtp_user
    if not smtp_user or not from_addr:
        raise ValueError("SMTP sender requires smtp_user and/or from_addr")

    notifier = SMTPNotifier(smtp_host, smtp_port, smtp_user, password, from_addr, bcc=bcc)
    if attachments:
        notifier.notify_with_attachments(to, subject, body, attachments)
    else:
        notifier.notify(to, subject, body)


# ── Outlook AppleScript (macOS) ───────────────────────────────────────

def send_email_via_outlook(
    to: str,
    subject: str,
    body: str,
    sender: str = "",
    attachments: list[Path | str] | None = None,
    dry_run: bool = False,
) -> None:
    """Send an email via the local Microsoft Outlook app (macOS only).

    Uses AppleScript to drive the already-authenticated Outlook — no passwords
    or SMTP config needed. Works with any account configured in Outlook
    (Stanford, Gmail, etc.).

    Args:
        to: Recipient email address.
        subject: Email subject line.
        body: Plain text email body.
        sender: From address (e.g., "brando9@stanford.edu"). If empty, uses
                Outlook's default account.
        attachments: Optional list of file paths to attach.
        dry_run: If True, print instead of sending.
    """
    import platform
    import subprocess

    if platform.system() != "Darwin":
        raise RuntimeError("send_email_via_outlook() requires macOS with Microsoft Outlook")

    if dry_run:
        log.info("[DRY-RUN] Outlook email to %s from %s: %s", to, sender or "(default)", subject)
        print(f"[DRY-RUN] Outlook email to {to} from {sender or '(default)'}: {subject}")
        return

    attachment_paths = _existing_attachment_paths(attachments)
    result = subprocess.run(
        ["osascript", "-e", _OUTLOOK_SEND_SCRIPT, to, subject, body, sender, *attachment_paths],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Outlook AppleScript failed: {result.stderr.strip()}")
    log.info("Email sent via Outlook to %s from %s: %s", to, sender or "(default)", subject)


def send_stanford_email(
    to: str,
    subject: str,
    body: str,
    sender: str = "brando9@stanford.edu",
    attachments: list[Path | str] | None = None,
    dry_run: bool = False,
) -> None:
    """Send an email from a Stanford address via the local Outlook app.

    Convenience wrapper around send_email_via_outlook() with Stanford defaults.

    Args:
        to: Recipient email address.
        subject: Email subject line.
        body: Plain text email body.
        sender: Stanford sender address. Use "brando9@stanford.edu" (default)
                or "brando9@cs.stanford.edu".
        attachments: Optional list of file paths to attach.
        dry_run: If True, print instead of sending.
    """
    send_email_via_outlook(
        to=to, subject=subject, body=body,
        sender=sender, attachments=attachments, dry_run=dry_run,
    )


# ── Legacy functions (backward compatible) ─────────────────────────────

def send_email(subject, message, destination, password_path=None):
    """Legacy: send a plain-text email via Gmail SMTP.

    Prefer send_email_smtp() for new code.
    """
    message = _message_with_hostname(message)
    password = _read_legacy_password(password_path)
    try:
        send_email_smtp(
            to=destination,
            subject=subject,
            body=message,
            smtp_user="slurm.miranda@gmail.com",
            smtp_pass=password,
            from_addr="slurm.miranda@gmail.com",
        )
    except (OSError, smtplib.SMTPConnectError, smtplib.SMTPServerDisconnected) as exc:
        log.warning("Falling back to intel relay after Gmail SMTP connection failure: %s", exc)
        _send_legacy_relay(subject, message, destination)


def send_email_pdf_figs(path_to_pdf, subject, message, destination, password_path=None):
    """Legacy: send an email with a PDF attachment via Gmail SMTP.

    Prefer send_email_smtp(attachments=[...]) for new code.
    """
    pdf_path = Path(path_to_pdf).expanduser()
    message = _message_with_hostname(message)
    password = _read_legacy_password(password_path)
    try:
        send_email_smtp(
            to=destination,
            subject=subject,
            body=message,
            smtp_user="slurm.miranda@gmail.com",
            smtp_pass=password,
            from_addr="slurm.miranda@gmail.com",
            attachments=[pdf_path],
        )
    except (OSError, smtplib.SMTPConnectError, smtplib.SMTPServerDisconnected) as exc:
        log.warning("Falling back to intel relay after Gmail SMTP connection failure: %s", exc)
        _send_legacy_relay(subject, message, destination, attachment=pdf_path)


# ── Tests ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Quick smoke test: dry-run mode (no SMTP needed)
    notifier = DryRunNotifier("/tmp/email_test_outbox")
    notifier.notify("test@example.com", "Test Subject", "Test body")
    notifier.notify_with_attachments(
        "test@example.com", "Test Attachments", "Body",
        [Path("/tmp/nonexistent.pdf")],
    )
    print("Dry-run test passed — check /tmp/email_test_outbox/")

    # Stanford email dry-run test
    send_stanford_email(
        to="test@example.com",
        subject="Stanford Test",
        body="Testing Stanford email dry-run",
        dry_run=True,
    )
    send_stanford_email(
        to="test@example.com",
        subject="Stanford CS Test",
        body="Testing Stanford CS email dry-run",
        sender="brando9@cs.stanford.edu",
        dry_run=True,
    )
    print("Stanford email dry-run tests passed!")

    # Outlook dry-run test
    send_email_via_outlook(
        to="test@example.com",
        subject="Outlook Test",
        body="Testing Outlook dry-run",
        sender="brando9@stanford.edu",
        dry_run=True,
    )
    print("Outlook dry-run test passed!")
