"""Club emailing — profile-based email automation for student clubs and research labs.

Provides account profiles for Stanford Bachata Club and Stanford Lean AI Lab Gmail
accounts, announcement sending with BCC fan-out, template rendering, and offline
dry-run preview.

Quick usage:
    from uutils.club_emailing import send_account_email, send_announcement, render_template

    # 1. Preview an email in dry-run mode (default, no credentials needed):
    send_account_email(
        account="bachata",
        to="dancer@example.com",
        subject="Practice Reminder",
        body="Practice tonight at 7pm in Old Union!",
    )

    # 2. Render from a template:
    subject, body = render_template(
        "bachata_practice_reminder",
        date="Friday, Oct 3",
        time="7:00 PM - 9:00 PM",
        location="Old Union 2nd Floor",
    )
    send_account_email("bachata", to="dancer@example.com", subject=subject, body=body)

    # 3. Send announcement to all club members via BCC:
    send_announcement(
        account="bachata",
        subject=subject,
        body=body,
        recipients_file="~/keys/bachata_members.csv",
        dry_run=True,  # preview recipient count & body
    )

Setup: Gmail App Password for Club Accounts
    For each club Gmail account (e.g. Stanford Bachata Club, Stanford Lean AI Lab):
    1. Log into the club Gmail account in your browser:
       https://mail.google.com
    2. Enable 2-Step Verification on the Google account:
       https://myaccount.google.com/security
    3. Generate an App Password for email:
       https://myaccount.google.com/apppasswords
       - Select 'Mail' and your device, then click 'Generate'
       - Copy the 16-character password (without spaces)
    4. Save credentials to key files with restricted permissions (chmod 600):
       # For Stanford Bachata Club:
       mkdir -p ~/keys
       echo "club_gmail@gmail.com" > ~/keys/bachata_gmail_address.txt
       echo "xxxx xxxx xxxx xxxx" > ~/keys/bachata_gmail_app_password.txt
       chmod 600 ~/keys/bachata_gmail_address.txt ~/keys/bachata_gmail_app_password.txt

       # For Stanford Lean AI Lab:
       echo "lab_gmail@gmail.com" > ~/keys/leanai_gmail_address.txt
       echo "yyyy yyyy yyyy yyyy" > ~/keys/leanai_gmail_app_password.txt
       chmod 600 ~/keys/leanai_gmail_address.txt ~/keys/leanai_gmail_app_password.txt

    5. Members CSV files:
       Save membership rosters to CSV files containing an 'email' column:
       ~/keys/bachata_members.csv
       ~/keys/leanai_members.csv
       chmod 600 ~/keys/bachata_members.csv ~/keys/leanai_members.csv

Environment variable overrides (optional):
    UUTILS_BACHATA_GMAIL_ADDRESS: Club Gmail address override
    UUTILS_BACHATA_GMAIL_APP_PASSWORD_FILE: Path to app password file override
    UUTILS_LEANAI_GMAIL_ADDRESS: Lab Gmail address override
    UUTILS_LEANAI_GMAIL_APP_PASSWORD_FILE: Path to app password file override

CLI usage:
    # Dry-run template preview (offline, reads no keys):
    python -m uutils.club_emailing preview --account bachata --template bachata_practice_reminder --date "Oct 3" --time "7pm" --location "Old Union"

    # Dry-run send preview (offline, reads no keys):
    python -m uutils.club_emailing send --account bachata --to member@example.com --subject "Hi" --body "Hello"

    # Real send (requires credentials and explicit --send flag):
    # python -m uutils.club_emailing send --account bachata --to member@example.com --subject "Hi" --body "Hello" --send

Refs:
    - Gmail App Passwords: https://support.google.com/accounts/answer/185833
    - Gmail SMTP settings: https://support.google.com/a/answer/176600
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import string
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from uutils.emailing import send_email_smtp

log = logging.getLogger(__name__)


# ── Account Dataclass & Profiles ──────────────────────────────────────


@dataclass
class EmailAccount:
    """Email account profile configuration.

    Attributes:
        name: Short name of the account profile (e.g. 'bachata', 'leanai').
        address_file: Path to file containing the Gmail address.
        app_password_file: Path to file containing the 16-character Gmail app password.
        members_file: Path to CSV containing member emails.
        smtp_host: SMTP server hostname (default: smtp.gmail.com).
        smtp_port: SMTP server port (default: 587 for STARTTLS).
    """

    name: str
    address_file: str | Path
    app_password_file: str | Path
    members_file: str | Path
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587


ACCOUNTS: dict[str, EmailAccount] = {
    "bachata": EmailAccount(
        name="bachata",
        address_file="~/keys/bachata_gmail_address.txt",
        app_password_file="~/keys/bachata_gmail_app_password.txt",
        members_file="~/keys/bachata_members.csv",
        smtp_host="smtp.gmail.com",
        smtp_port=587,
    ),
    "leanai": EmailAccount(
        name="leanai",
        address_file="~/keys/leanai_gmail_address.txt",
        app_password_file="~/keys/leanai_gmail_app_password.txt",
        members_file="~/keys/leanai_members.csv",
        smtp_host="smtp.gmail.com",
        smtp_port=587,
    ),
}


def get_account(account: str | EmailAccount) -> EmailAccount:
    """Retrieve an EmailAccount instance by name or return the instance directly."""
    if isinstance(account, EmailAccount):
        return account
    if isinstance(account, str):
        if account in ACCOUNTS:
            return ACCOUNTS[account]
        raise KeyError(
            f"Unknown email account {account!r}. Available accounts: {list(ACCOUNTS.keys())}"
        )
    raise TypeError(f"Expected EmailAccount or str, got {type(account).__name__}")


def resolve_account_address(account: EmailAccount) -> str:
    """Resolve the Gmail address for an account from env var or key file."""
    env_var = f"UUTILS_{account.name.upper()}_GMAIL_ADDRESS"
    val = os.environ.get(env_var, "").strip()
    if val:
        return val
    addr_path = Path(account.address_file).expanduser()
    if not addr_path.is_file():
        raise FileNotFoundError(
            f"Address file for account {account.name!r} not found at {addr_path}.\n"
            f"Create the file with your club Gmail address or set ${env_var}."
        )
    addr = addr_path.read_text(encoding="utf-8").strip()
    if not addr:
        raise ValueError(f"Address file at {addr_path} is empty.")
    return addr


def resolve_account_app_password_file(account: EmailAccount) -> Path:
    """Resolve the app password file path from env var or account config."""
    env_var = f"UUTILS_{account.name.upper()}_GMAIL_APP_PASSWORD_FILE"
    override_path = os.environ.get(env_var, "").strip()
    pass_path = Path(override_path or account.app_password_file).expanduser()
    if not pass_path.is_file():
        raise FileNotFoundError(
            f"App password file for account {account.name!r} not found at {pass_path}.\n"
            f"Create the file with your Gmail app password or set ${env_var}."
        )
    return pass_path


# ── Members Roster ─────────────────────────────────────────────────────


def load_members(members_file: str | Path) -> list[str]:
    """Load member email addresses from a CSV file.

    Expects a CSV with an 'email' column (case-insensitive header match).
    Skips empty/blank entries and duplicate addresses (case-insensitive deduplication).

    Args:
        members_file: Path to the members CSV file.

    Returns:
        List of unique, cleaned email addresses in lowercase.
    """
    path = Path(members_file).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Members file not found: {path}")

    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file at {path} is empty or invalid.")

        email_col = None
        for col in reader.fieldnames:
            if col and col.strip().lower() == "email":
                email_col = col
                break

        if email_col is None:
            raise ValueError(
                f"CSV file at {path} does not contain an 'email' column. "
                f"Available columns: {reader.fieldnames}"
            )

        seen: set[str] = set()
        members: list[str] = []
        for row in reader:
            raw_email = row.get(email_col)
            if not raw_email:
                continue
            email = raw_email.strip()
            if not email:
                continue
            key = email.lower()
            if key not in seen:
                seen.add(key)
                members.append(key)

    return members


# ── Email Templates ───────────────────────────────────────────────────


TEMPLATES: dict[str, dict[str, str]] = {
    "bachata_practice_reminder": {
        "subject": "Stanford Bachata Club: Practice Reminder - {date}",
        "body": (
            "Hi Bachateros,\n\n"
            "This is a reminder for our upcoming practice session:\n\n"
            "  Date: {date}\n"
            "  Time: {time}\n"
            "  Location: {location}\n\n"
            "Looking forward to dancing with everyone!\n\n"
            "Best,\n"
            "Stanford Bachata Club"
        ),
    },
    "bachata_welcome": {
        "subject": "Welcome to Stanford Bachata Club!",
        "body": (
            "Hi {name},\n\n"
            "Welcome to the Stanford Bachata Club! We are thrilled to have you join our dance community.\n\n"
            "Stay tuned for announcements about practices, workshops, and social events.\n\n"
            "Best,\n"
            "Stanford Bachata Club"
        ),
    },
    "leanai_experiment_finished": {
        "subject": "[Lean AI Lab] Experiment Finished: {experiment_name}",
        "body": (
            "Hello,\n\n"
            "Experiment '{experiment_name}' has completed on host {hostname}.\n\n"
            "Summary:\n"
            "{summary}\n\n"
            "Best,\n"
            "Stanford Lean AI Lab"
        ),
    },
    "leanai_meeting_reminder": {
        "subject": "[Lean AI Lab] Meeting Reminder - {date}",
        "body": (
            "Hi everyone,\n\n"
            "This is a reminder for our upcoming Stanford Lean AI Lab meeting:\n\n"
            "  Date: {date}\n"
            "  Time: {time}\n"
            "  Location: {location}\n\n"
            "See you there!\n\n"
            "Best,\n"
            "Stanford Lean AI Lab"
        ),
    },
}


def render_template(template_name: str | None = None, /, **kwargs: Any) -> tuple[str, str]:
    """Render an email template by name with provided keyword arguments.

    Args:
        template_name: Name of the template (e.g. 'bachata_practice_reminder').
                       Can be passed positionally.
        **kwargs: Variables required by the template.

    Returns:
        tuple[str, str]: (subject, body) rendered strings.

    Raises:
        KeyError: If the template is unknown or required variables are missing.
    """
    if template_name is None:
        if "template_name" in kwargs:
            template_name = kwargs.pop("template_name")
        elif "template" in kwargs:
            template_name = kwargs.pop("template")
        elif "name" in kwargs and kwargs["name"] in TEMPLATES:
            template_name = kwargs.pop("name")
        else:
            raise KeyError("Template name must be provided as the first argument.")

    if template_name not in TEMPLATES:
        raise KeyError(
            f"Unknown template {template_name!r}. Available templates: {list(TEMPLATES.keys())}"
        )

    # Normalize location / link alias for meeting reminders
    merged_kwargs = dict(kwargs)
    if template_name == "leanai_meeting_reminder":
        if "location" not in merged_kwargs:
            if "link" in merged_kwargs:
                merged_kwargs["location"] = merged_kwargs["link"]
            elif "location_or_link" in merged_kwargs:
                merged_kwargs["location"] = merged_kwargs["location_or_link"]
        elif "link" not in merged_kwargs and "location" in merged_kwargs:
            merged_kwargs["link"] = merged_kwargs["location"]

    tmpl = TEMPLATES[template_name]
    subject_tmpl = tmpl["subject"]
    body_tmpl = tmpl["body"]

    # Detect required fields
    formatter = string.Formatter()
    required_fields: set[str] = set()
    for _, field, _, _ in formatter.parse(subject_tmpl):
        if field:
            required_fields.add(field)
    for _, field, _, _ in formatter.parse(body_tmpl):
        if field:
            required_fields.add(field)

    missing = [f for f in sorted(required_fields) if f not in merged_kwargs or merged_kwargs[f] is None]
    if missing:
        if template_name == "leanai_meeting_reminder" and "location" in missing:
            raise KeyError(
                f"Template {template_name!r} requires missing argument: 'location' or 'link'"
            )
        raise KeyError(
            f"Template {template_name!r} requires missing argument(s): {', '.join(repr(m) for m in missing)}"
        )

    try:
        subject = subject_tmpl.format(**merged_kwargs)
        body = body_tmpl.format(**merged_kwargs)
    except KeyError as exc:
        raise KeyError(f"Template {template_name!r} missing required key: {exc}") from exc

    return subject, body


# ── Sending Functions ─────────────────────────────────────────────────


def send_account_email(
    account: str | EmailAccount,
    to: str,
    subject: str,
    body: str,
    cc: str = "",
    attachments: list[Path | str] | None = None,
    dry_run: bool = True,
    bcc: str = "",
) -> dict | None:
    """Send an email using an account profile.

    In dry-run mode (default), reads no key files, makes no network calls,
    and returns a preview dictionary.

    Args:
        account: Account profile name or EmailAccount instance.
        to: Recipient email address.
        subject: Email subject.
        body: Email body text.
        cc: Optional CC addresses.
        attachments: Optional list of attachment paths.
        dry_run: If True (default), does not send and reads no secrets.
        bcc: Optional BCC addresses.

    Returns:
        Preview dict if dry_run=True, None otherwise.
    """
    acc = get_account(account)
    lines = body.splitlines()
    body_preview = "\n".join(lines[:10]) if len(lines) > 10 else body

    if dry_run:
        print(f"[DRY-RUN] Account email via {acc.name} to {to}: {subject}")
        if cc:
            print(f"  Cc: {cc}")
        if bcc:
            print(f"  Bcc: {bcc}")
        if attachments:
            print(f"  Attachments: {attachments}")
        print(f"  Body preview:\n{body_preview}")
        log.info("[DRY-RUN] Account email via %s to %s: %s", acc.name, to, subject)

        return {
            "account": acc.name,
            "to": to,
            "subject": subject,
            "body": body_preview,
            "first_lines_of_body": body_preview,
            "cc": cc,
            "bcc": bcc,
            "attachments": [str(a) for a in (attachments or [])],
            "dry_run": True,
        }

    address = resolve_account_address(acc)
    pass_file = resolve_account_app_password_file(acc)

    send_email_smtp(
        to=to,
        subject=subject,
        body=body,
        smtp_user=address,
        smtp_pass_file=str(pass_file),
        smtp_host=acc.smtp_host,
        smtp_port=acc.smtp_port,
        from_addr=address,
        cc=cc,
        bcc=bcc,
        attachments=attachments,
    )
    log.info("Email sent via %s to %s: %s", acc.name, to, subject)
    return None


def send_announcement(
    account: str | EmailAccount,
    subject: str,
    body: str,
    recipients_file: str | Path | None = None,
    dry_run: bool = True,
    bcc_all: bool = True,
) -> dict | None:
    """Send an announcement to club/lab members.

    Sends ONE email to the account's own address with all members in BCC
    (or CC if bcc_all=False) to preserve privacy.

    In dry-run mode (default), reads no key files and sends no emails.
    Only reads members from recipients_file if explicitly provided.

    Args:
        account: Account profile name or EmailAccount instance.
        subject: Email subject.
        body: Email body text.
        recipients_file: Path to members CSV (optional). If None outside dry-run,
                         uses account.members_file.
        dry_run: If True (default), does not send and reads no secrets.
        bcc_all: If True (default), puts members in BCC. If False, puts them in CC.

    Returns:
        Preview dict if dry_run=True, None otherwise.
    """
    acc = get_account(account)
    lines = body.splitlines()
    body_preview = "\n".join(lines[:10]) if len(lines) > 10 else body

    if dry_run:
        recipients: list[str] = []
        if recipients_file is not None:
            recipients = load_members(recipients_file)
        recipient_count = len(recipients)

        print(f"[DRY-RUN] Announcement for {acc.name}: {recipient_count} recipient(s), bcc_all={bcc_all}")
        print(f"[DRY-RUN] Subject: {subject}")
        print(f"[DRY-RUN] Body preview:\n{body_preview}")
        log.info("[DRY-RUN] Announcement for %s: %d recipients", acc.name, recipient_count)

        return {
            "account": acc.name,
            "to": f"<{acc.name} own address>",
            "subject": subject,
            "body": body_preview,
            "first_lines_of_body": body_preview,
            "recipient_count": recipient_count,
            "recipients": recipients,
            "bcc_all": bcc_all,
            "dry_run": True,
        }

    mfile = recipients_file or acc.members_file
    recipients = load_members(mfile)
    address = resolve_account_address(acc)
    pass_file = resolve_account_app_password_file(acc)

    to_addr = address
    recipients_str = ", ".join(recipients)
    bcc_addr = recipients_str if bcc_all else ""
    cc_addr = "" if bcc_all else recipients_str

    send_email_smtp(
        to=to_addr,
        subject=subject,
        body=body,
        smtp_user=address,
        smtp_pass_file=str(pass_file),
        smtp_host=acc.smtp_host,
        smtp_port=acc.smtp_port,
        from_addr=address,
        cc=cc_addr,
        bcc=bcc_addr,
    )
    log.info("Announcement sent via %s to %d recipients: %s", acc.name, len(recipients), subject)
    return None


# ── Command Line Interface ─────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for club emailing."""
    parser = argparse.ArgumentParser(
        prog="python -m uutils.club_emailing",
        description="Club & research lab email automation (Stanford Bachata Club & Lean AI Lab).",
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to execute")

    # preview subcommand
    preview_parser = subparsers.add_parser(
        "preview",
        help="Preview a templated or custom email offline (dry-run, reads no credentials).",
    )
    preview_parser.add_argument(
        "--account",
        default="bachata",
        choices=list(ACCOUNTS.keys()),
        help="Account profile (default: bachata)",
    )
    preview_parser.add_argument(
        "--template",
        choices=list(TEMPLATES.keys()),
        help="Template name to render",
    )
    preview_parser.add_argument("--to", default="preview@example.com", help="Preview recipient")
    preview_parser.add_argument("--subject", default="", help="Custom subject (if not using template)")
    preview_parser.add_argument("--body", default="", help="Custom body (if not using template)")
    preview_parser.add_argument("--date", help="Date parameter for template")
    preview_parser.add_argument("--time", help="Time parameter for template")
    preview_parser.add_argument("--location", help="Location parameter for template")
    preview_parser.add_argument("--link", help="Link parameter for template")
    preview_parser.add_argument("--name", help="Name parameter for template")
    preview_parser.add_argument(
        "--experiment-name", "--experiment_name", dest="experiment_name", help="Experiment name"
    )
    preview_parser.add_argument("--hostname", help="Hostname parameter for template")
    preview_parser.add_argument("--summary", help="Summary parameter for template")

    # send subcommand
    send_parser = subparsers.add_parser(
        "send",
        help="Send an email via an account profile (defaults to dry-run unless --send is given).",
    )
    send_parser.add_argument(
        "--account",
        required=True,
        choices=list(ACCOUNTS.keys()),
        help="Account profile (e.g. bachata, leanai)",
    )
    send_parser.add_argument("--to", help="Recipient email address")
    send_parser.add_argument("--subject", help="Email subject line")
    send_parser.add_argument("--body", help="Email body text")
    send_parser.add_argument("--template", choices=list(TEMPLATES.keys()), help="Optional template name")
    send_parser.add_argument("--cc", default="", help="CC address(es)")
    send_parser.add_argument("--bcc", default="", help="BCC address(es)")
    send_parser.add_argument("--attachments", nargs="*", default=None, help="Attachment file paths")
    send_parser.add_argument(
        "--send", "--execute",
        dest="send",
        action="store_true",
        default=False,
        help="Actually send the email. Without this flag, runs in safe dry-run mode.",
    )
    send_parser.add_argument("--date", help="Date parameter for template")
    send_parser.add_argument("--time", help="Time parameter for template")
    send_parser.add_argument("--location", help="Location parameter for template")
    send_parser.add_argument("--link", help="Link parameter for template")
    send_parser.add_argument("--name", help="Name parameter for template")
    send_parser.add_argument(
        "--experiment-name", "--experiment_name", dest="experiment_name", help="Experiment name"
    )
    send_parser.add_argument("--hostname", help="Hostname parameter for template")
    send_parser.add_argument("--summary", help="Summary parameter for template")

    # announce subcommand
    announce_parser = subparsers.add_parser(
        "announce",
        help="Send announcement to members roster via BCC (defaults to dry-run unless --send is given).",
    )
    announce_parser.add_argument(
        "--account",
        required=True,
        choices=list(ACCOUNTS.keys()),
        help="Account profile (e.g. bachata, leanai)",
    )
    announce_parser.add_argument("--subject", help="Announcement subject line")
    announce_parser.add_argument("--body", help="Announcement body text")
    announce_parser.add_argument("--template", choices=list(TEMPLATES.keys()), help="Optional template name")
    announce_parser.add_argument("--recipients-file", help="Path to members CSV file")
    announce_parser.add_argument(
        "--send", "--execute",
        dest="send",
        action="store_true",
        default=False,
        help="Actually send the announcement. Without this flag, runs in safe dry-run mode.",
    )
    announce_parser.add_argument(
        "--no-bcc-all",
        dest="bcc_all",
        action="store_false",
        default=True,
        help="Put recipients in CC instead of BCC.",
    )
    announce_parser.add_argument("--date", help="Date parameter for template")
    announce_parser.add_argument("--time", help="Time parameter for template")
    announce_parser.add_argument("--location", help="Location parameter for template")
    announce_parser.add_argument("--link", help="Link parameter for template")
    announce_parser.add_argument("--name", help="Name parameter for template")
    announce_parser.add_argument(
        "--experiment-name", "--experiment_name", dest="experiment_name", help="Experiment name"
    )
    announce_parser.add_argument("--hostname", help="Hostname parameter for template")
    announce_parser.add_argument("--summary", help="Summary parameter for template")

    args, unknown = parser.parse_known_args(argv)
    if not args.command:
        parser.print_help()
        return 0

    # Collect template variables from both explicit flags and unknown arguments
    template_kwargs: dict[str, Any] = {}
    idx = 0
    while idx < len(unknown):
        arg = unknown[idx]
        if arg.startswith("--"):
            key = arg[2:].replace("-", "_")
            if idx + 1 < len(unknown) and not unknown[idx + 1].startswith("--"):
                template_kwargs[key] = unknown[idx + 1]
                idx += 2
            else:
                template_kwargs[key] = True
                idx += 1
        else:
            idx += 1

    for field in ["date", "time", "location", "link", "name", "experiment_name", "hostname", "summary"]:
        val = getattr(args, field, None)
        if val is not None:
            template_kwargs[field] = val

    if getattr(args, "template", None):
        subject, body = render_template(args.template, **template_kwargs)
    else:
        subject = getattr(args, "subject", "") or ""
        body = getattr(args, "body", "") or ""

    if args.command == "preview":
        send_account_email(
            account=args.account,
            to=args.to,
            subject=subject,
            body=body,
            dry_run=True,
        )
        return 0

    if args.command == "send":
        if not args.to:
            parser.error("--to is required for send command")
        if not subject:
            parser.error("--subject or --template is required")
        send_account_email(
            account=args.account,
            to=args.to,
            subject=subject,
            body=body,
            cc=args.cc,
            bcc=args.bcc,
            attachments=args.attachments,
            dry_run=not args.send,
        )
        return 0

    if args.command == "announce":
        if not subject:
            parser.error("--subject or --template is required")
        send_announcement(
            account=args.account,
            subject=subject,
            body=body,
            recipients_file=args.recipients_file,
            dry_run=not args.send,
            bcc_all=args.bcc_all,
        )
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
