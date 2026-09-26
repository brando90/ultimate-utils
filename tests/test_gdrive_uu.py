"""Offline unit tests for Google Drive integration (uutils.gdrive_uu).

All tests run strictly offline without network access, without Google libraries
installed, and without requiring ~/keys/ files on the machine.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests

from uutils.gdrive_uu import (
    DEFAULT_CLIENT_SECRETS_FILE,
    DEFAULT_SERVICE_ACCOUNT_FILE,
    DEFAULT_TOKEN_FILE,
    IMAGE_EXTENSIONS,
    MIME_IMAGE,
    GDriveClient,
    build_parser,
    download_images_from_drive,
    filter_image_files,
    get_gdrive_client,
    is_image_file,
    main,
    sync_drive_folder,
    sync_phone_to_local,
)


def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure any network or external subprocess call raises immediately."""

    def forbidden(*args, **kwargs):
        raise AssertionError(f"Forbidden external call: args={args!r}, kwargs={kwargs!r}")

    monkeypatch.setattr(requests, "get", forbidden)
    monkeypatch.setattr(requests, "post", forbidden)
    monkeypatch.setattr(requests, "request", forbidden)
    if hasattr(requests.Session, "request"):
        monkeypatch.setattr(requests.Session, "request", forbidden)


# ── 1. Import and Dependency Tests ────────────────────────────────────

def test_module_imports_without_google_libs():
    """Verify that importing uutils.gdrive_uu does not require Google libraries."""
    assert "uutils.gdrive_uu" in sys.modules
    assert GDriveClient is not None
    assert sync_phone_to_local is not None

    import uutils.gdrive_uu as gmod
    assert not hasattr(gmod, "PlanDict"), "PlanDict must not exist"
    assert not hasattr(gmod, "PlanList"), "PlanList must not exist"


# ── 2. Docstring Requirements Test ────────────────────────────────────

def test_module_docstring_covers_setup_and_cron():
    """Verify the module docstring includes all required setup instructions and data model contract."""
    doc = sys.modules["uutils.gdrive_uu"].__doc__ or ""

    # iOS setup
    assert "iOS" in doc
    assert "Google Drive app" in doc
    assert "Backup" in doc or "Photos" in doc

    # Android setup
    assert "Android" in doc
    assert "folder" in doc.lower()

    # Google Photos and Google Drive no longer auto-sync
    assert "Google Photos" in doc
    assert "no longer" in doc

    # Sharing with service account
    assert "service account" in doc.lower()
    assert "share" in doc.lower()

    # Finding folder ID in URL
    assert "folders/" in doc
    assert "folder ID" in doc or "folder_id" in doc

    # Cron line with --execute and note that machine must be powered on
    assert "cron" in doc.lower()
    assert "--execute" in doc
    assert "powered on" in doc.lower() or "awake" in doc.lower() or "must be on" in doc.lower()

    # Credentials security
    assert "~/keys/" in doc
    assert "dry-run" in doc.lower() or "dry_run" in doc.lower()

    # Data model contract documentation
    assert "GDriveClient" in doc
    assert "describe" in doc
    assert "empty list" in doc.lower() or "[]" in doc


# ── 3. Public Entry Point Offline / Dry-Run Safety ───────────────────

def test_every_public_entry_point_makes_no_network_call_by_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Every public function/method with default args must run offline without auth/network."""
    _block_network(monkeypatch)

    fake_folder = "fake_drive_folder_id_123"
    fake_dest = tmp_path / "downloads"

    # 1. get_gdrive_client
    client = get_gdrive_client()
    assert client.dry_run is True
    assert client.credentials_file == DEFAULT_SERVICE_ACCOUNT_FILE
    assert not isinstance(client, dict)
    desc = client.describe()
    assert type(desc) is dict
    assert desc["dry_run"] is True
    assert desc["credentials_file"] == DEFAULT_SERVICE_ACCOUNT_FILE
    assert desc["scopes"] == client.scopes

    # 2. GDriveClient default constructor
    client_default = GDriveClient()
    assert client_default.dry_run is True
    assert client_default.credentials_file == DEFAULT_SERVICE_ACCOUNT_FILE
    assert not isinstance(client_default, dict)

    # 3. GDriveClient.from_service_account
    client_sa = GDriveClient.from_service_account()
    assert client_sa.dry_run is True
    assert client_sa.credentials_file == DEFAULT_SERVICE_ACCOUNT_FILE
    assert not isinstance(client_sa, dict)

    # 4. GDriveClient.from_oauth2
    client_oauth = GDriveClient.from_oauth2()
    assert client_oauth.dry_run is True
    assert client_oauth.credentials_file == DEFAULT_CLIENT_SECRETS_FILE
    assert not isinstance(client_oauth, dict)

    # 5. client.list_files -> plain empty list
    files = client.list_files()
    assert type(files) is list
    assert len(files) == 0

    # 6. client.list_images -> plain empty list
    images = client.list_images()
    assert type(images) is list
    assert len(images) == 0

    # 7. client.list_folders -> plain empty list
    folders = client.list_folders()
    assert type(folders) is list
    assert len(folders) == 0

    # 8. client.download_file -> plain dict plan
    dl_file = client.download_file("file_id_abc", fake_dest / "test.jpg")
    assert type(dl_file) is dict
    assert dl_file["dry_run"] is True
    assert dl_file["file_id"] == "file_id_abc"
    assert dl_file["dest"] == str(fake_dest / "test.jpg")
    assert "credentials_file" in dl_file

    # 9. client.download_files -> plain empty list
    dl_files = client.download_files([{"id": "1", "name": "pic.jpg"}], fake_dest)
    assert type(dl_files) is list
    assert len(dl_files) == 0

    # 10. client.upload_file -> plain dict plan
    up_file = client.upload_file("/nonexistent/fake_image.png")
    assert type(up_file) is dict
    assert up_file["dry_run"] is True
    assert "credentials_file" in up_file

    # 11. client.upload_files -> plain empty list
    up_files = client.upload_files(["/nonexistent/fake_1.png", "/nonexistent/fake_2.png"])
    assert type(up_files) is list
    assert len(up_files) == 0

    # 12. client.sync_folder -> plain empty list
    sync_res = client.sync_folder(fake_folder, fake_dest)
    assert type(sync_res) is list
    assert len(sync_res) == 0

    # 13. sync_drive_folder convenience function -> plain dict plan
    sync_convenience = sync_drive_folder(fake_folder, fake_dest)
    assert type(sync_convenience) is dict
    assert sync_convenience["dry_run"] is True
    assert sync_convenience["folder_id"] == fake_folder
    assert sync_convenience["dest"] == str(fake_dest)
    assert "credentials_file" in sync_convenience

    # 14. download_images_from_drive convenience function -> plain dict plan
    images_convenience = download_images_from_drive(fake_folder)
    assert type(images_convenience) is dict
    assert images_convenience["dry_run"] is True
    assert images_convenience["folder_id"] == fake_folder
    assert images_convenience["dest"] == "./drive_images"
    assert "credentials_file" in images_convenience

    # 15. sync_phone_to_local pipeline entry point -> plain dict plan
    phone_res = sync_phone_to_local(fake_folder, fake_dest)
    assert type(phone_res) is dict
    assert phone_res["dry_run"] is True
    assert phone_res["folder_id"] == fake_folder
    assert phone_res["dest"] == str(fake_dest)
    assert phone_res["filter"] == "image/"
    assert phone_res["read_only_remote"] is True
    assert "credentials_file" in phone_res


# ── 4. Image Filtering Helper Tests ──────────────────────────────────

def test_is_image_file():
    """Verify is_image_file correctly identifies image files by extension and mimeType."""
    # Standard image extensions
    assert is_image_file("photo.jpg") is True
    assert is_image_file("photo.JPEG") is True
    assert is_image_file("IMAGE.PNG") is True
    assert is_image_file("animation.gif") is True
    assert is_image_file("picture.webp") is True

    # Camera / phone raw and high-efficiency formats
    assert is_image_file("IMG_2026.HEIC") is True
    assert is_image_file("IMG_2026.heif") is True
    assert is_image_file("sample.dng") is True
    assert is_image_file("sample.RAW") is True

    # Non-images
    assert is_image_file("document.pdf") is False
    assert is_image_file("table.csv") is False
    assert is_image_file("script.py") is False
    assert is_image_file("archive.zip") is False

    # MIME type detection
    assert is_image_file("no_extension", mime_type="image/jpeg") is True
    assert is_image_file("no_extension", mime_type="image/heic") is True
    assert is_image_file("no_extension", mime_type="application/pdf") is False


def test_filter_image_files_on_fake_file_listings():
    """Verify filter_image_files extracts only images from simulated Drive listings."""
    fake_files = [
        {"id": "1", "name": "IMG_0001.HEIC", "mimeType": "image/heic"},
        {"id": "2", "name": "IMG_0002.JPG", "mimeType": "image/jpeg"},
        {"id": "3", "name": "notes.txt", "mimeType": "text/plain"},
        {"id": "4", "name": "statement.pdf", "mimeType": "application/pdf"},
        {"id": "5", "name": "screenshot.png", "mimeType": "image/png"},
        {"id": "6", "name": "IMG_0003.dng", "mimeType": "application/octet-stream"},  # extension match
        {"id": "7", "name": "backup.tar.gz", "mimeType": "application/gzip"},
    ]

    filtered = filter_image_files(fake_files)
    filtered_names = [f["name"] for f in filtered]

    assert filtered_names == ["IMG_0001.HEIC", "IMG_0002.JPG", "screenshot.png", "IMG_0003.dng"]


# ── 5. CLI sync-phone Tests ───────────────────────────────────────────

def test_cli_sync_phone_without_execute_prints_plan_and_exits_zero(capsys: pytest.CaptureFixture):
    """CLI sync-phone without --execute must print execution plan and exit 0."""
    argv = [
        "sync-phone",
        "--folder-id", "test_folder_xyz789",
        "--dest", "/tmp/local_phone_images",
    ]
    exit_code = main(argv)
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "[DRY-RUN]" in captured.out
    assert "test_folder_xyz789" in captured.out
    assert "/tmp/local_phone_images" in captured.out
    assert "Images only" in captured.out or "image/" in captured.out


def test_cli_sync_phone_with_credentials_flag(capsys: pytest.CaptureFixture):
    """CLI sync-phone accepts custom --credentials path in dry-run mode."""
    argv = [
        "sync-phone",
        "--folder-id", "folder_abc",
        "--dest", "/tmp/photos",
        "--credentials", "~/keys/custom_service_account.json",
    ]
    exit_code = main(argv)
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "custom_service_account.json" in captured.out


def test_cli_parser_options():
    """Verify the argument parser recognizes expected commands and options."""
    parser = build_parser()

    # sync-phone
    args = parser.parse_args([
        "sync-phone",
        "--folder-id", "fid",
        "--dest", "/dest",
        "--credentials", "cred.json",
        "--max-results", "500",
    ])
    assert args.command == "sync-phone"
    assert args.folder_id == "fid"
    assert args.dest == "/dest"
    assert args.credentials == "cred.json"
    assert args.execute is False
    assert args.max_results == 500

    # With --execute
    args_exec = parser.parse_args(["sync-phone", "-f", "fid", "-d", "/dest", "--execute"])
    assert args_exec.execute is True

    # With --send alias
    args_send = parser.parse_args(["sync-phone", "-f", "fid", "-d", "/dest", "--send"])
    assert args_send.execute is True


def test_cli_bare_invocation_prints_help(capsys: pytest.CaptureFixture):
    """Calling CLI without arguments displays help and returns 0."""
    exit_code = main([])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "sync-phone" in captured.out


def test_cli_imports_subcommand(capsys: pytest.CaptureFixture):
    """'imports' subcommand runs without error."""
    exit_code = main(["imports"])
    assert exit_code == 0


def test_cli_sync_subcommand_dry_run(capsys: pytest.CaptureFixture):
    """'sync' subcommand defaults to dry-run and prints plan."""
    exit_code = main(["sync", "folder_123", "/tmp/sync_dir"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "[DRY-RUN]" in captured.out
    assert "folder_123" in captured.out


def test_cli_list_subcommand_dry_run(capsys: pytest.CaptureFixture):
    """'list' subcommand defaults to dry-run and prints plan."""
    exit_code = main(["list", "folder_123"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "[DRY-RUN]" in captured.out
    assert "folder_123" in captured.out


# ── 6. Client Plain Class & Dry-Run Printed Plan Tests ────────────────

def test_client_is_plain_class_not_dict():
    """Verify GDriveClient does not subclass dict and has describe() method."""
    client = GDriveClient()
    assert not isinstance(client, dict)
    with pytest.raises(TypeError):
        _ = client["dry_run"]

    desc = client.describe()
    assert type(desc) is dict
    assert desc["dry_run"] is True
    assert desc["credentials_file"] == DEFAULT_SERVICE_ACCOUNT_FILE
    assert desc["token_file"] == DEFAULT_TOKEN_FILE
    assert desc["scopes"] == client.scopes
    assert "GDriveClient" in repr(client)


def test_dry_run_list_methods_print_plan_and_return_empty_list(capsys: pytest.CaptureFixture, tmp_path: Path):
    """Methods returning lists in live mode return plain empty list in dry run after printing plan."""
    client = GDriveClient()

    # list_files
    res = client.list_files(folder_id="f1")
    assert type(res) is list
    assert res == []
    captured = capsys.readouterr()
    assert "[DRY-RUN] list_files" in captured.out
    assert "folder_id=f1" in captured.out

    # list_images
    res = client.list_images(folder_id="f2")
    assert type(res) is list
    assert res == []
    captured = capsys.readouterr()
    assert "[DRY-RUN] list_files" in captured.out
    assert "folder_id=f2" in captured.out

    # list_folders
    res = client.list_folders(parent_folder_id="f3")
    assert type(res) is list
    assert res == []
    captured = capsys.readouterr()
    assert "[DRY-RUN] list_files" in captured.out
    assert "folder_id=f3" in captured.out

    # download_files
    res = client.download_files([{"id": "1", "name": "pic.jpg"}], tmp_path)
    assert type(res) is list
    assert res == []
    captured = capsys.readouterr()
    assert "[DRY-RUN] download_files" in captured.out

    # upload_files
    res = client.upload_files(["/fake/path.png"], folder_id="f4")
    assert type(res) is list
    assert res == []
    captured = capsys.readouterr()
    assert "[DRY-RUN] upload_files" in captured.out

    # sync_folder
    res = client.sync_folder("f5", tmp_path)
    assert type(res) is list
    assert res == []
    captured = capsys.readouterr()
    assert "[DRY-RUN] sync_folder" in captured.out


def test_dry_run_action_methods_print_plan_and_return_dict(capsys: pytest.CaptureFixture, tmp_path: Path):
    """Single-object methods and action entry points return plain dict plan and print plan."""
    client = GDriveClient()

    # download_file
    res = client.download_file("f1", tmp_path / "img.jpg")
    assert type(res) is dict
    assert res["dry_run"] is True
    assert res["file_id"] == "f1"
    captured = capsys.readouterr()
    assert "[DRY-RUN] download_file" in captured.out

    # upload_file
    res = client.upload_file("/fake/pic.png")
    assert type(res) is dict
    assert res["dry_run"] is True
    captured = capsys.readouterr()
    assert "[DRY-RUN] upload_file" in captured.out

    # sync_drive_folder
    res = sync_drive_folder("f2", tmp_path)
    assert type(res) is dict
    assert res["dry_run"] is True
    assert res["folder_id"] == "f2"
    captured = capsys.readouterr()
    assert "[DRY-RUN] sync_drive_folder" in captured.out

    # download_images_from_drive
    res = download_images_from_drive("f3")
    assert type(res) is dict
    assert res["dry_run"] is True
    assert res["folder_id"] == "f3"
    captured = capsys.readouterr()
    assert "[DRY-RUN] download_images_from_drive" in captured.out

    # sync_phone_to_local
    res = sync_phone_to_local("f4", tmp_path)
    assert type(res) is dict
    assert res["dry_run"] is True
    assert res["folder_id"] == "f4"
    captured = capsys.readouterr()
    assert "[DRY-RUN] sync-phone" in captured.out
