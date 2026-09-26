"""Offline tests for uutils.club_emailing.

All tests run strictly offline: network calls (smtplib, requests, subprocess)
are monkeypatched to fail if called unexpectedly. Key files under ~/keys are
never touched.
"""
from __future__ import annotations

import csv
import smtplib
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests

import uutils.club_emailing as ce
import uutils.emailing
from uutils.club_emailing import (
    ACCOUNTS,
    TEMPLATES,
    EmailAccount,
    get_account,
    load_members,
    main,
    render_template,
    resolve_account_address,
    resolve_account_app_password_file,
    send_account_email,
    send_announcement,
)


def _block_all_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure any network or process calls raise an error."""

    def forbidden(*args, **kwargs):
        raise AssertionError(f"Forbidden call attempted: args={args!r}, kwargs={kwargs!r}")

    monkeypatch.setattr(smtplib, "SMTP", forbidden)
    monkeypatch.setattr(smtplib, "SMTP_SSL", forbidden)
    monkeypatch.setattr("uutils.emailing.send_email_smtp", forbidden)
    monkeypatch.setattr(requests, "get", forbidden)
    monkeypatch.setattr(requests, "post", forbidden)
    monkeypatch.setattr(requests, "request", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)


class FakeSMTP:
    """Fake smtplib.SMTP recorder for offline testing of live email paths."""

    instances: list[FakeSMTP] = []

    def __init__(self, host: str, port: int, timeout: int = 30):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.starttls_calls: list[bool] = []
        self.login_calls: list[tuple[str, str]] = []
        self.sendmail_calls: list[tuple[str, list[str], str]] = []
        FakeSMTP.instances.append(self)

    def __enter__(self) -> FakeSMTP:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False

    def starttls(self) -> None:
        self.starttls_calls.append(True)

    def login(self, user: str, password: str) -> None:
        self.login_calls.append((user, password))

    def sendmail(self, from_addr: str, to_addrs: list[str], msg: str) -> dict:
        self.sendmail_calls.append((from_addr, to_addrs, msg))
        return {}


# ── 1. Module Docstring Coverage ──────────────────────────────────────


def test_module_docstring_covers_setup():
    """Verify setup instructions, key file paths, and permissions in docstring."""
    doc = ce.__doc__ or ""
    assert "2-Step Verification" in doc
    assert "App Password" in doc
    assert "chmod 600" in doc
    assert "bachata_gmail_address.txt" in doc
    assert "bachata_gmail_app_password.txt" in doc
    assert "bachata_members.csv" in doc
    assert "leanai_gmail_address.txt" in doc
    assert "leanai_gmail_app_password.txt" in doc
    assert "leanai_members.csv" in doc
    assert "dry_run" in doc or "dry-run" in doc.lower()


# ── 2. Account Profiles ────────────────────────────────────────────────


def test_accounts_profiles():
    """Verify default account configurations for bachata and leanai."""
    assert "bachata" in ACCOUNTS
    assert "leanai" in ACCOUNTS

    bachata = ACCOUNTS["bachata"]
    assert bachata.name == "bachata"
    assert "bachata_gmail_address.txt" in str(bachata.address_file)
    assert "bachata_gmail_app_password.txt" in str(bachata.app_password_file)
    assert "bachata_members.csv" in str(bachata.members_file)
    assert bachata.smtp_host == "smtp.gmail.com"
    assert bachata.smtp_port == 587

    leanai = ACCOUNTS["leanai"]
    assert leanai.name == "leanai"
    assert "leanai_gmail_address.txt" in str(leanai.address_file)
    assert "leanai_gmail_app_password.txt" in str(leanai.app_password_file)
    assert "leanai_members.csv" in str(leanai.members_file)
    assert leanai.smtp_host == "smtp.gmail.com"
    assert leanai.smtp_port == 587


def test_get_account():
    """Verify get_account resolution by name and instance."""
    assert get_account("bachata") is ACCOUNTS["bachata"]
    assert get_account("leanai") is ACCOUNTS["leanai"]

    custom = EmailAccount("custom", "addr.txt", "pass.txt", "mem.csv")
    assert get_account(custom) is custom

    with pytest.raises(KeyError, match="Unknown email account"):
        get_account("nonexistent")

    with pytest.raises(TypeError, match="Expected EmailAccount or str"):
        get_account(12345)  # type: ignore[arg-type]


# ── 3. Dry-Run Safety (Hard Rule #1) ──────────────────────────────────


def test_dry_run_default_makes_no_network_or_credential_calls(monkeypatch: pytest.MonkeyPatch, capsys):
    """Calling public APIs with default arguments makes zero network calls and reads no keys."""
    _block_all_network(monkeypatch)

    # Calling send_account_email with default dry_run=True
    preview_email = send_account_email(
        account="bachata",
        to="dancer@example.com",
        subject="Practice Reminder",
        body="\n".join(f"Line {i}" for i in range(1, 16)),
    )
    assert preview_email is not None
    assert preview_email["dry_run"] is True
    assert preview_email["account"] == "bachata"
    assert preview_email["to"] == "dancer@example.com"
    assert preview_email["subject"] == "Practice Reminder"
    assert "Line 1" in preview_email["body"]
    assert "Line 15" not in preview_email["body"]  # Body preview truncates to first 10 lines

    # Calling send_announcement with default dry_run=True (recipients_file=None)
    preview_ann = send_announcement(
        account="leanai",
        subject="Lab Meeting",
        body="Weekly meeting tomorrow at 11am.",
    )
    assert preview_ann is not None
    assert preview_ann["dry_run"] is True
    assert preview_ann["account"] == "leanai"
    assert preview_ann["recipient_count"] == 0
    assert preview_ann["recipients"] == []
    assert preview_ann["bcc_all"] is True

    captured = capsys.readouterr()
    assert "[DRY-RUN]" in captured.out


def test_send_announcement_dry_run_with_explicit_recipients_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """In dry-run mode, recipients_file is read if and only if explicitly passed."""
    _block_all_network(monkeypatch)

    csv_path = tmp_path / "test_roster.csv"
    csv_path.write_text("name,email\nAlice,alice@example.com\nBob,bob@example.com\n", encoding="utf-8")

    preview = send_announcement(
        account="bachata",
        subject="Auditions",
        body="Auditions this weekend.",
        recipients_file=csv_path,
        dry_run=True,
    )
    assert preview is not None
    assert preview["recipient_count"] == 2
    assert preview["recipients"] == ["alice@example.com", "bob@example.com"]


# ── 4. Non-Dry-Run Sending (Offline via Mock) ──────────────────────────


def test_non_dry_run_send_account_email(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Non-dry-run path resolves address/password and calls emailing.send_email_smtp with right args."""
    addr_file = tmp_path / "addr.txt"
    addr_file.write_text("bachata_club@example.com\n", encoding="utf-8")
    pass_file = tmp_path / "pass.txt"
    pass_file.write_text("secret_app_pass\n", encoding="utf-8")

    account = EmailAccount(
        name="bachata_test",
        address_file=addr_file,
        app_password_file=pass_file,
        members_file=tmp_path / "members.csv",
    )

    calls: list[dict] = []

    def mock_send_email_smtp(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("uutils.club_emailing.send_email_smtp", mock_send_email_smtp)

    ret = send_account_email(
        account=account,
        to="recipient@example.com",
        subject="Important Notice",
        body="Body text here.",
        cc="cc@example.com",
        bcc="bcc@example.com",
        dry_run=False,
    )
    assert ret is None
    assert len(calls) == 1
    call = calls[0]
    assert call["to"] == "recipient@example.com"
    assert call["subject"] == "Important Notice"
    assert call["body"] == "Body text here."
    assert call["smtp_user"] == "bachata_club@example.com"
    assert call["from_addr"] == "bachata_club@example.com"
    assert call["smtp_pass_file"] == str(pass_file)
    assert call["cc"] == "cc@example.com"
    assert call["bcc"] == "bcc@example.com"
    assert call["smtp_host"] == "smtp.gmail.com"
    assert call["smtp_port"] == 587


def test_non_dry_run_send_announcement(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """send_announcement sends to account's own address with members in BCC."""
    addr_file = tmp_path / "leanai_addr.txt"
    addr_file.write_text("leanai_lab@example.com\n", encoding="utf-8")
    pass_file = tmp_path / "leanai_pass.txt"
    pass_file.write_text("secret_lab_pass\n", encoding="utf-8")
    members_file = tmp_path / "roster.csv"
    members_file.write_text(
        "email,status\n"
        "researcher1@example.com,active\n"
        "researcher2@example.com,active\n",
        encoding="utf-8",
    )

    account = EmailAccount(
        name="leanai_test",
        address_file=addr_file,
        app_password_file=pass_file,
        members_file=members_file,
    )

    calls: list[dict] = []

    def mock_send_email_smtp(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("uutils.club_emailing.send_email_smtp", mock_send_email_smtp)

    # 1. bcc_all=True (default): recipients go to BCC, To is account's own address
    send_announcement(
        account=account,
        subject="Paper Accepted!",
        body="Congratulations everyone!",
        dry_run=False,
        bcc_all=True,
    )
    assert len(calls) == 1
    call = calls[0]
    assert call["to"] == "leanai_lab@example.com"
    assert call["from_addr"] == "leanai_lab@example.com"
    assert call["smtp_user"] == "leanai_lab@example.com"
    assert call["smtp_pass_file"] == str(pass_file)
    assert call["bcc"] == "researcher1@example.com, researcher2@example.com"
    assert call["cc"] == ""

    # 2. bcc_all=False: recipients go to CC
    calls.clear()
    send_announcement(
        account=account,
        subject="Paper Accepted!",
        body="Congratulations everyone!",
        dry_run=False,
        bcc_all=False,
    )
    assert len(calls) == 1
    call_cc = calls[0]
    assert call_cc["bcc"] == ""
    assert call_cc["cc"] == "researcher1@example.com, researcher2@example.com"


def test_fake_smtp_send_announcement_with_custom_account(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Non-dry-run send_announcement with custom EmailAccount calls sendmail once with
    [own_address, m1, m2, m3], records starttls/login/sendmail, and msg text has no Bcc:
    header and does not contain member addresses."""
    FakeSMTP.instances.clear()
    monkeypatch.setattr(uutils.emailing.smtplib, "SMTP", FakeSMTP)
    monkeypatch.setattr(smtplib, "SMTP_SSL", lambda *a, **k: pytest.fail("Unexpected SSL call"))
    monkeypatch.setattr(requests, "get", lambda *a, **k: pytest.fail("Unexpected network call"))
    monkeypatch.setattr(requests, "post", lambda *a, **k: pytest.fail("Unexpected network call"))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("Unexpected subprocess call"))

    own_address = "bachata_club@example.com"
    app_password = "fake_app_password"
    addr_file = tmp_path / "addr.txt"
    addr_file.write_text(f"{own_address}\n", encoding="utf-8")
    pass_file = tmp_path / "pass.txt"
    pass_file.write_text(f"{app_password}\n", encoding="utf-8")

    members_file = tmp_path / "members.csv"
    m1, m2, m3 = "dancer1@example.com", "dancer2@example.com", "dancer3@example.com"
    members_file.write_text(
        f"email,role\n{m1},lead\n{m2},follow\n{m3},lead\n",
        encoding="utf-8",
    )

    account = EmailAccount(
        name="bachata_custom",
        address_file=addr_file,
        app_password_file=pass_file,
        members_file=members_file,
    )

    send_announcement(
        account=account,
        subject="Practice Tonight!",
        body="Practice is at 7pm in Old Union.",
        dry_run=False,
    )

    assert len(FakeSMTP.instances) == 1
    server = FakeSMTP.instances[0]
    assert len(server.starttls_calls) == 1
    assert server.login_calls == [(own_address, app_password)]
    assert len(server.sendmail_calls) == 1

    from_addr, envelope_recipients, msg_text = server.sendmail_calls[0]
    assert from_addr == own_address
    assert envelope_recipients == [own_address, m1, m2, m3]

    # Verify message text has no Bcc: header
    headers = msg_text.split("\n\n")[0].splitlines()
    assert not any(h.lower().startswith("bcc:") for h in headers)
    assert "Bcc:" not in msg_text

    # Verify message text does not contain member addresses
    assert m1 not in msg_text
    assert m2 not in msg_text
    assert m3 not in msg_text

    # Verify visible headers and content
    assert f"To: {own_address}" in msg_text
    assert f"From: {own_address}" in msg_text
    assert "Practice Tonight!" in msg_text
    assert "Practice is at 7pm in Old Union." in msg_text


def test_fake_smtp_send_announcement_with_env_var_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Non-dry-run send_announcement using env-var overrides calls sendmail once with
    [own_address, m1, m2, m3], and msg text has no Bcc: header and does not contain member addresses."""
    FakeSMTP.instances.clear()
    monkeypatch.setattr(uutils.emailing.smtplib, "SMTP", FakeSMTP)
    monkeypatch.setattr(smtplib, "SMTP_SSL", lambda *a, **k: pytest.fail("Unexpected SSL call"))
    monkeypatch.setattr(requests, "get", lambda *a, **k: pytest.fail("Unexpected network call"))
    monkeypatch.setattr(requests, "post", lambda *a, **k: pytest.fail("Unexpected network call"))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("Unexpected subprocess call"))

    own_address = "leanai_lab@example.com"
    app_password = "env_secret_password"
    pass_file = tmp_path / "env_pass.txt"
    pass_file.write_text(f"{app_password}\n", encoding="utf-8")

    members_file = tmp_path / "roster.csv"
    m1, m2, m3 = "m1@example.com", "m2@example.com", "m3@example.com"
    members_file.write_text(
        f"email,name\n{m1},Alice\n{m2},Bob\n{m3},Charlie\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("UUTILS_LEANAI_GMAIL_ADDRESS", own_address)
    monkeypatch.setenv("UUTILS_LEANAI_GMAIL_APP_PASSWORD_FILE", str(pass_file))

    send_announcement(
        account="leanai",
        subject="Lab Seminar",
        body="Seminar tomorrow at 11am.",
        recipients_file=members_file,
        dry_run=False,
    )

    assert len(FakeSMTP.instances) == 1
    server = FakeSMTP.instances[0]
    assert len(server.starttls_calls) == 1
    assert server.login_calls == [(own_address, app_password)]
    assert len(server.sendmail_calls) == 1

    from_addr, envelope_recipients, msg_text = server.sendmail_calls[0]
    assert from_addr == own_address
    assert envelope_recipients == [own_address, m1, m2, m3]

    headers = msg_text.split("\n\n")[0].splitlines()
    assert not any(h.lower().startswith("bcc:") for h in headers)
    assert "Bcc:" not in msg_text
    assert m1 not in msg_text
    assert m2 not in msg_text
    assert m3 not in msg_text


def test_fake_smtp_single_address_bcc(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Assert a single-address bcc still produces [to, bcc] in envelope recipients, without Bcc: header."""
    FakeSMTP.instances.clear()
    monkeypatch.setattr(uutils.emailing.smtplib, "SMTP", FakeSMTP)
    monkeypatch.setattr(smtplib, "SMTP_SSL", lambda *a, **k: pytest.fail("Unexpected SSL call"))
    monkeypatch.setattr(requests, "get", lambda *a, **k: pytest.fail("Unexpected network call"))
    monkeypatch.setattr(requests, "post", lambda *a, **k: pytest.fail("Unexpected network call"))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("Unexpected subprocess call"))

    own_address = "bachata_club@example.com"
    app_password = "fake_app_password"
    addr_file = tmp_path / "addr.txt"
    addr_file.write_text(f"{own_address}\n", encoding="utf-8")
    pass_file = tmp_path / "pass.txt"
    pass_file.write_text(f"{app_password}\n", encoding="utf-8")

    account = EmailAccount(
        name="bachata_single",
        address_file=addr_file,
        app_password_file=pass_file,
        members_file=tmp_path / "unused.csv",
    )

    to_addr = "recipient@example.com"
    bcc_addr = "supervisor@example.com"

    # Test via send_account_email
    send_account_email(
        account=account,
        to=to_addr,
        subject="Important Notice",
        body="Notice body text.",
        bcc=bcc_addr,
        dry_run=False,
    )

    assert len(FakeSMTP.instances) == 1
    server = FakeSMTP.instances[0]
    assert len(server.starttls_calls) == 1
    assert server.login_calls == [(own_address, app_password)]
    assert len(server.sendmail_calls) == 1

    from_addr, envelope_recipients, msg_text = server.sendmail_calls[0]
    assert from_addr == own_address
    assert envelope_recipients == [to_addr, bcc_addr]

    headers = msg_text.split("\n\n")[0].splitlines()
    assert not any(h.lower().startswith("bcc:") for h in headers)
    assert "Bcc:" not in msg_text
    assert bcc_addr not in msg_text
    assert f"To: {to_addr}" in msg_text

    # Test via direct emailing.send_email_smtp
    FakeSMTP.instances.clear()
    uutils.emailing.send_email_smtp(
        to="alice@example.com",
        subject="Direct SMTP",
        body="Direct call test.",
        smtp_user=own_address,
        smtp_pass=app_password,
        from_addr=own_address,
        bcc="bob@example.com",
    )

    assert len(FakeSMTP.instances) == 1
    server2 = FakeSMTP.instances[0]
    assert len(server2.starttls_calls) == 1
    assert server2.login_calls == [(own_address, app_password)]
    assert len(server2.sendmail_calls) == 1

    from_addr2, envelope_recipients2, msg_text2 = server2.sendmail_calls[0]
    assert from_addr2 == own_address
    assert envelope_recipients2 == ["alice@example.com", "bob@example.com"]
    headers2 = msg_text2.split("\n\n")[0].splitlines()
    assert not any(h.lower().startswith("bcc:") for h in headers2)
    assert "Bcc:" not in msg_text2
    assert "bob@example.com" not in msg_text2


# ── 5. Environment Variable Overrides ─────────────────────────────────


def test_env_var_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Test UUTILS_<NAME>_GMAIL_ADDRESS and UUTILS_<NAME>_GMAIL_APP_PASSWORD_FILE overrides."""
    pass_file = tmp_path / "env_pass.txt"
    pass_file.write_text("env_secret_password\n", encoding="utf-8")

    monkeypatch.setenv("UUTILS_BACHATA_GMAIL_ADDRESS", "env_bachata@example.com")
    monkeypatch.setenv("UUTILS_BACHATA_GMAIL_APP_PASSWORD_FILE", str(pass_file))

    bachata = ACCOUNTS["bachata"]
    assert resolve_account_address(bachata) == "env_bachata@example.com"
    assert resolve_account_app_password_file(bachata) == pass_file


def test_missing_credential_files_raise_outside_dry_run():
    """When files do not exist and dry_run=False, clear errors are raised."""
    dummy = EmailAccount(
        name="dummy",
        address_file="/nonexistent/keys/dummy_addr.txt",
        app_password_file="/nonexistent/keys/dummy_pass.txt",
        members_file="/nonexistent/keys/dummy_mem.csv",
    )
    with pytest.raises(FileNotFoundError, match="Address file for account 'dummy' not found"):
        resolve_account_address(dummy)

    with pytest.raises(FileNotFoundError, match="App password file for account 'dummy' not found"):
        resolve_account_app_password_file(dummy)


# ── 6. Member CSV Loading & Deduplication ─────────────────────────────


def test_load_members_dedupe_and_blanks(tmp_path: Path):
    """Test loading members CSV: skips blanks and performs case-insensitive deduplication."""
    csv_file = tmp_path / "members.csv"
    csv_file.write_text(
        "name,Email,role\n"
        "Alice,alice@example.com,lead\n"
        "Bob,BOB@EXAMPLE.COM,follow\n"
        "Charlie,   ,guest\n"
        "David,,guest\n"
        "Alice Duplicate,ALICE@EXAMPLE.COM,lead\n"
        "Eve,eve@example.com,member\n"
        "Bob Duplicate,bob@example.com,follow\n",
        encoding="utf-8",
    )

    members = load_members(csv_file)
    assert members == [
        "alice@example.com",
        "bob@example.com",
        "eve@example.com",
    ]


def test_load_members_missing_file():
    """Non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Members file not found"):
        load_members("/nonexistent/path/to/roster.csv")


def test_load_members_missing_email_column(tmp_path: Path):
    """CSV missing an email column raises ValueError."""
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("name,phone,role\nAlice,555-1234,lead\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not contain an 'email' column"):
        load_members(bad_csv)


# ── 7. Template Rendering ─────────────────────────────────────────────


def test_templates_all_required_present():
    """Verify the 4 required templates exist in TEMPLATES."""
    required = [
        "bachata_practice_reminder",
        "bachata_welcome",
        "leanai_experiment_finished",
        "leanai_meeting_reminder",
    ]
    for key in required:
        assert key in TEMPLATES


def test_render_template_bachata_practice_reminder():
    subject, body = render_template(
        "bachata_practice_reminder",
        date="Friday, Oct 3",
        time="7:00 PM",
        location="Old Union Ballroom",
    )
    assert "Friday, Oct 3" in subject
    assert "Friday, Oct 3" in body
    assert "7:00 PM" in body
    assert "Old Union Ballroom" in body
    assert "Bachateros" in body


def test_render_template_bachata_welcome():
    subject, body = render_template("bachata_welcome", name="Elena")
    assert "Welcome to Stanford Bachata Club!" in subject
    assert "Hi Elena," in body


def test_render_template_leanai_experiment_finished():
    subject, body = render_template(
        "leanai_experiment_finished",
        experiment_name="lean4_proof_search_v2",
        hostname="gpu-cluster-01",
        summary="Accuracy reached 94.2% on benchmark.",
    )
    assert "lean4_proof_search_v2" in subject
    assert "gpu-cluster-01" in body
    assert "Accuracy reached 94.2%" in body


def test_render_template_leanai_meeting_reminder():
    # Using location parameter
    subj1, body1 = render_template(
        "leanai_meeting_reminder",
        date="Monday, Oct 6",
        time="2:00 PM",
        location="Gates Hall Room 200",
    )
    assert "Monday, Oct 6" in subj1
    assert "Gates Hall Room 200" in body1

    # Using link parameter (location/link alias)
    subj2, body2 = render_template(
        "leanai_meeting_reminder",
        date="Monday, Oct 6",
        time="2:00 PM",
        link="https://stanford.zoom.us/j/123456789",
    )
    assert "https://stanford.zoom.us/j/123456789" in body2


def test_render_template_missing_kwargs_raises():
    """Missing required template arguments raise KeyError."""
    with pytest.raises(KeyError, match="Missing required argument|requires missing argument"):
        render_template("bachata_practice_reminder", date="Friday")  # missing time & location

    with pytest.raises(KeyError, match="Missing required argument|requires missing argument"):
        render_template("bachata_welcome")  # missing name

    with pytest.raises(KeyError, match="Missing required argument|requires missing argument"):
        render_template("leanai_experiment_finished", experiment_name="exp1")  # missing hostname, summary

    with pytest.raises(KeyError, match="location.*or.*link"):
        render_template("leanai_meeting_reminder", date="Monday", time="10am")  # missing location or link

    with pytest.raises(KeyError, match="Unknown template"):
        render_template("nonexistent_template")


# ── 8. CLI Tests ──────────────────────────────────────────────────────


def test_cli_preview(monkeypatch: pytest.MonkeyPatch, capsys):
    """Test CLI preview subcommand is offline and dry-run."""
    _block_all_network(monkeypatch)

    code = main([
        "preview",
        "--account", "bachata",
        "--template", "bachata_practice_reminder",
        "--date", "Saturday, Oct 4",
        "--time", "6:00 PM",
        "--location", "Roble Studio 113",
    ])
    assert code == 0
    captured = capsys.readouterr()
    assert "[DRY-RUN]" in captured.out
    assert "Roble Studio 113" in captured.out


def test_cli_send_defaults_to_dry_run(monkeypatch: pytest.MonkeyPatch, capsys):
    """Test CLI send subcommand defaults to dry-run without --send flag."""
    _block_all_network(monkeypatch)

    code = main([
        "send",
        "--account", "bachata",
        "--to", "dancer@example.com",
        "--subject", "Audition Info",
        "--body", "Audition times are posted.",
    ])
    assert code == 0
    captured = capsys.readouterr()
    assert "[DRY-RUN]" in captured.out
    assert "Audition Info" in captured.out


def test_cli_announce_defaults_to_dry_run(monkeypatch: pytest.MonkeyPatch, capsys):
    """Test CLI announce subcommand defaults to dry-run without --send flag."""
    _block_all_network(monkeypatch)

    code = main([
        "announce",
        "--account", "leanai",
        "--subject", "Lab Seminar",
        "--body", "Guest speaker next Tuesday.",
    ])
    assert code == 0
    captured = capsys.readouterr()
    assert "[DRY-RUN]" in captured.out
    assert "0 recipient(s)" in captured.out
