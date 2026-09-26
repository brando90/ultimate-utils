"""Offline tests for Facebook Page automation (uutils.facebook_uu).

All tests run strictly offline by monkeypatching network calls and using tmp_path.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import requests

from uutils.facebook_uu import (
    DEFAULT_CREDENTIALS_FILE,
    FacebookClient,
    main,
    plan_posts_from_folder,
    post_image,
    post_images_batch,
    post_text,
)


class MockResponse:
    """Mock requests.Response for offline testing."""

    def __init__(self, json_data: dict | None = None, status_code: int = 200, text: str = ""):
        self.status_code = status_code
        self._json_data = json_data if json_data is not None else {}
        self.text = text or json.dumps(self._json_data)

    def json(self) -> dict:
        return self._json_data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}: {self.text}")


# ── 1. Dry-run safety tests (no network calls, no credential file reads) ───


def test_default_dry_run_makes_no_network_call_and_no_credentials_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Prove that default API calls execute purely offline and read no credentials."""

    def forbidden_call(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Forbidden: network call attempted during dry-run!")

    monkeypatch.setattr(requests, "request", forbidden_call)
    monkeypatch.setattr(requests, "get", forbidden_call)
    monkeypatch.setattr(requests, "post", forbidden_call)

    # 1. FacebookClient default initialization has dry_run=True
    client = FacebookClient()
    assert client.dry_run is True

    # 2. from_credentials with dry_run=True must NOT read or require file
    client_creds = FacebookClient.from_credentials("/nonexistent/keys/creds.json", dry_run=True)
    assert client_creds.dry_run is True

    # 3. post_text in dry_run mode
    res_text = client.post_text("Hello from test!")
    assert res_text["dry_run"] is True
    assert res_text["would_post"] == "text"
    assert res_text["message"] == "Hello from test!"
    assert res_text["caption"] == "Hello from test!"
    assert "/feed" in res_text["endpoint"]

    # 4. post_image with nonexistent local file path in dry_run mode
    res_img_local = client.post_image("/nonexistent/file.jpg", caption="A photo")
    assert res_img_local["dry_run"] is True
    assert res_img_local["would_post"] == "photo"
    assert res_img_local["caption"] == "A photo"
    assert res_img_local["source_type"] == "file"
    assert "/photos" in res_img_local["endpoint"]

    # 5. post_image with URL in dry_run mode
    res_img_url = client.post_image("https://example.com/pic.png", caption="Web photo")
    assert res_img_url["dry_run"] is True
    assert res_img_url["would_post"] == "photo"
    assert res_img_url["caption"] == "Web photo"
    assert res_img_url["source_type"] == "url"
    assert "/photos" in res_img_url["endpoint"]

    # 6. post_images_batch in dry_run mode
    batch_res = client.post_images_batch(
        ["/nonexistent/1.jpg", "https://example.com/2.png"],
        captions=["One", "Two"],
    )
    assert len(batch_res) == 2
    assert batch_res[0]["dry_run"] is True
    assert batch_res[0]["caption"] == "One"
    assert batch_res[1]["dry_run"] is True
    assert batch_res[1]["caption"] == "Two"

    # 7. Top-level convenience functions with defaults (dry_run=True)
    top_text = post_text("Top level text")
    assert top_text["dry_run"] is True
    assert top_text["caption"] == "Top level text"

    top_img = post_image("/nonexistent/local.png", caption="Top image")
    assert top_img["dry_run"] is True
    assert top_img["caption"] == "Top image"

    top_batch = post_images_batch(["fake1.jpg", "fake2.png"])
    assert len(top_batch) == 2
    assert top_batch[0]["dry_run"] is True

    # 8. CLI commands without --send default to dry-run
    rc_text = main(["post-text", "--message", "CLI test message"])
    assert rc_text == 0

    rc_post = main(["post", "--image", "/nonexistent/img.jpg", "--caption", "CLI cap"])
    assert rc_post == 0

    rc_preview = main(["preview", "--folder", str(tmp_path)])
    assert rc_preview == 0


# ── 2. Credentials loading tests ──────────────────────────────────────────


def test_from_credentials_file_parsing(tmp_path: Path) -> None:
    """Test loading Facebook credentials from JSON file."""
    creds_file = tmp_path / "facebook_credentials.json"

    # Missing file when dry_run=False raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        FacebookClient.from_credentials(credentials_file=creds_file, dry_run=False)

    # Missing keys raises ValueError
    creds_file.write_text(json.dumps({"page_id": "12345"}), encoding="utf-8")
    with pytest.raises(ValueError, match="page_access_token"):
        FacebookClient.from_credentials(credentials_file=creds_file, dry_run=False)

    creds_file.write_text(json.dumps({"page_access_token": "tok"}), encoding="utf-8")
    with pytest.raises(ValueError, match="page_id"):
        FacebookClient.from_credentials(credentials_file=creds_file, dry_run=False)

    # Valid file loads correctly
    creds_file.write_text(
        json.dumps({
            "page_id": "987654321",
            "page_access_token": "EAAXfakeToken123",
            "api_version": "v21.0",
        }),
        encoding="utf-8",
    )
    client = FacebookClient.from_credentials(credentials_file=creds_file, dry_run=False)
    assert client.page_id == "987654321"
    assert client.page_access_token == "EAAXfakeToken123"
    assert client.api_version == "v21.0"
    assert client.dry_run is False


# ── 3. Mocked HTTP network call tests ─────────────────────────────────────


def test_post_text_mocked_http(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test post_text sends correct HTTP POST request to /<page_id>/feed."""
    captured: dict[str, Any] = {}

    def mock_post(url: str, headers: dict | None = None, data: dict | None = None, **kwargs: Any) -> MockResponse:
        captured["url"] = url
        captured["headers"] = headers
        captured["data"] = data
        captured["kwargs"] = kwargs
        return MockResponse({"id": "987654321_11223344"})

    monkeypatch.setattr(requests, "post", mock_post)

    client = FacebookClient(
        page_id="987654321",
        page_access_token="test-access-token",
        api_version="v21.0",
        dry_run=False,
    )
    res = client.post_text("Exciting announcement!")

    assert captured["url"] == "https://graph.facebook.com/v21.0/987654321/feed"
    assert captured["headers"] == {"Authorization": "Bearer test-access-token"}
    assert captured["data"] == {"message": "Exciting announcement!"}
    assert res == {"id": "987654321_11223344"}


def test_post_image_url_mocked_http(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test post_image with image URL sends url param in data to /<page_id>/photos."""
    captured: dict[str, Any] = {}

    def mock_post(url: str, headers: dict | None = None, data: dict | None = None, **kwargs: Any) -> MockResponse:
        captured["url"] = url
        captured["headers"] = headers
        captured["data"] = data
        captured["kwargs"] = kwargs
        return MockResponse({"id": "photo_1001", "post_id": "987654321_photo_1001"})

    monkeypatch.setattr(requests, "post", mock_post)

    client = FacebookClient(
        page_id="987654321",
        page_access_token="test-access-token",
        api_version="v21.0",
        dry_run=False,
    )
    res = client.post_image("https://example.com/logo.png", caption="Our Company Logo")

    assert captured["url"] == "https://graph.facebook.com/v21.0/987654321/photos"
    assert captured["headers"] == {"Authorization": "Bearer test-access-token"}
    assert captured["data"] == {"url": "https://example.com/logo.png", "caption": "Our Company Logo"}
    assert "files" not in captured["kwargs"]
    assert res["id"] == "photo_1001"


def test_post_image_local_file_mocked_http(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test post_image with local file sends multipart 'source' to /<page_id>/photos."""
    img_file = tmp_path / "diagram.jpg"
    img_file.write_bytes(b"\xff\xd8\xfffakejpegcontent")

    captured: dict[str, Any] = {}

    def mock_post(
        url: str,
        headers: dict | None = None,
        data: dict | None = None,
        files: dict | None = None,
        **kwargs: Any,
    ) -> MockResponse:
        captured["url"] = url
        captured["headers"] = headers
        captured["data"] = data
        captured["files"] = files
        captured["kwargs"] = kwargs
        return MockResponse({"id": "photo_2002", "post_id": "987654321_photo_2002"})

    monkeypatch.setattr(requests, "post", mock_post)

    client = FacebookClient(
        page_id="987654321",
        page_access_token="test-access-token",
        api_version="v21.0",
        dry_run=False,
    )
    res = client.post_image(img_file, caption="Architecture diagram")

    assert captured["url"] == "https://graph.facebook.com/v21.0/987654321/photos"
    assert captured["headers"] == {"Authorization": "Bearer test-access-token"}
    assert captured["data"] == {"caption": "Architecture diagram"}
    assert "source" in captured["files"]
    filename, file_obj = captured["files"]["source"]
    assert filename == "diagram.jpg"
    assert res["id"] == "photo_2002"


def test_post_image_local_file_missing_raises(tmp_path: Path) -> None:
    """Test non-dry-run raises FileNotFoundError if local file does not exist."""
    client = FacebookClient(
        page_id="987654321",
        page_access_token="test-token",
        dry_run=False,
    )
    with pytest.raises(FileNotFoundError):
        client.post_image(tmp_path / "missing.jpg", caption="Missing")


def test_post_images_batch_mocked_http(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test post_images_batch sends sequential requests for each image."""
    img1 = tmp_path / "img1.png"
    img1.write_bytes(b"image1")
    img2 = tmp_path / "img2.jpg"
    img2.write_bytes(b"image2")

    calls: list[dict[str, Any]] = []

    def mock_post(url: str, data: dict | None = None, **kwargs: Any) -> MockResponse:
        calls.append({"url": url, "data": data, **kwargs})
        return MockResponse({"id": f"photo_{len(calls)}"})

    monkeypatch.setattr(requests, "post", mock_post)

    client = FacebookClient(
        page_id="987654321",
        page_access_token="test-token",
        dry_run=False,
    )
    res = client.post_images_batch([img1, img2], captions=["First photo", "Second photo"])

    assert len(res) == 2
    assert len(calls) == 2
    assert calls[0]["data"]["caption"] == "First photo"
    assert calls[1]["data"]["caption"] == "Second photo"


def test_facebook_api_error_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test error handling when Facebook Graph API returns an error structure."""
    def mock_post_err(*args: Any, **kwargs: Any) -> MockResponse:
        return MockResponse(
            {"error": {"message": "Invalid OAuth access token signature.", "type": "OAuthException", "code": 190}},
            status_code=400,
        )

    monkeypatch.setattr(requests, "post", mock_post_err)

    client = FacebookClient(
        page_id="987654321",
        page_access_token="expired-token",
        dry_run=False,
    )
    with pytest.raises(requests.HTTPError):
        client.post_text("Test error")


# ── 4. Folder planning tests ──────────────────────────────────────────────


def test_plan_posts_from_folder(tmp_path: Path) -> None:
    """Test pairing image files (.jpg, .jpeg, .png) with sidecar .txt captions."""
    folder = tmp_path / "drive_images"
    folder.mkdir()

    # Image 1 with sidecar caption
    img1 = folder / "banner.jpg"
    img1.write_bytes(b"banner_bytes")
    cap1 = folder / "banner.txt"
    cap1.write_text("Exciting banner caption\n", encoding="utf-8")

    # Image 2 (.jpeg) with sidecar caption
    img2 = folder / "landscape.JPEG"
    img2.write_bytes(b"landscape_bytes")
    cap2 = folder / "landscape.txt"
    cap2.write_text("Mountain landscape view", encoding="utf-8")

    # Image 3 (.png) without sidecar caption
    img3 = folder / "portrait.png"
    img3.write_bytes(b"portrait_bytes")

    # Standalone .txt without image (should be ignored)
    standalone_txt = folder / "notes.txt"
    standalone_txt.write_text("Some random notes", encoding="utf-8")

    # Unrelated file type (should be ignored)
    pdf = folder / "report.pdf"
    pdf.write_bytes(b"pdf_bytes")

    posts = plan_posts_from_folder(folder)
    assert len(posts) == 3

    # Sorted by filename: banner.jpg, landscape.JPEG, portrait.png
    assert Path(posts[0]["image_path"]).name == "banner.jpg"
    assert posts[0]["caption"] == "Exciting banner caption"
    assert posts[0]["caption_path"] is not None

    assert Path(posts[1]["image_path"]).name == "landscape.JPEG"
    assert posts[1]["caption"] == "Mountain landscape view"
    assert posts[1]["caption_path"] is not None

    assert Path(posts[2]["image_path"]).name == "portrait.png"
    assert posts[2]["caption"] == ""
    assert posts[2]["caption_path"] is None


def test_plan_posts_from_folder_custom_suffix(tmp_path: Path) -> None:
    """Test folder planning with custom caption suffix."""
    folder = tmp_path / "custom_suffix_folder"
    folder.mkdir()

    img = folder / "sample.jpg"
    img.write_bytes(b"sample")
    cap = folder / "sample_caption.txt"
    cap.write_text("Custom suffix caption", encoding="utf-8")

    posts = plan_posts_from_folder(folder, caption_suffix="_caption.txt")
    assert len(posts) == 1
    assert posts[0]["caption"] == "Custom suffix caption"


def test_plan_posts_from_folder_missing_dir_raises(tmp_path: Path) -> None:
    """Test plan_posts_from_folder raises FileNotFoundError for non-existent directory."""
    with pytest.raises(FileNotFoundError):
        plan_posts_from_folder(tmp_path / "nonexistent_dir")


# ── 5. CLI execution tests ────────────────────────────────────────────────


def test_cli_preview(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Test CLI preview subcommand runs offline and prints planned posts."""
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"photo")
    cap = tmp_path / "photo.txt"
    cap.write_text("CLI preview test caption", encoding="utf-8")

    rc = main(["preview", "--folder", str(tmp_path)])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "Planned 1 post(s)" in captured
    assert "CLI preview test caption" in captured


def test_cli_post_dry_run(capsys: pytest.CaptureFixture) -> None:
    """Test CLI post subcommand defaults to dry-run."""
    rc = main(["post", "--image", "https://example.com/cat.jpg", "--caption", "A cat"])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "[DRY-RUN]" in captured


def test_cli_post_text_dry_run(capsys: pytest.CaptureFixture) -> None:
    """Test CLI post-text subcommand defaults to dry-run."""
    rc = main(["post-text", "--message", "Hello from test CLI"])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "[DRY-RUN]" in captured


def test_cli_send_mocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """Test CLI subcommands with --send flag execute HTTP call when mocked."""
    creds_file = tmp_path / "fb_creds.json"
    creds_file.write_text(
        json.dumps({"page_id": "12345", "page_access_token": "token123", "api_version": "v21.0"}),
        encoding="utf-8",
    )

    mock_calls: list[str] = []

    def mock_post(url: str, **kwargs: Any) -> MockResponse:
        mock_calls.append(url)
        return MockResponse({"id": "post_result_id"})

    monkeypatch.setattr(requests, "post", mock_post)

    rc = main([
        "post-text",
        "--message", "Live message",
        "--send",
        "--credentials", str(creds_file),
    ])
    assert rc == 0
    assert len(mock_calls) == 1
    assert "12345/feed" in mock_calls[0]
    out = capsys.readouterr().out
    assert "posted successfully" in out


def test_cli_no_args_shows_help(capsys: pytest.CaptureFixture) -> None:
    """Test CLI with no args prints help and returns 1."""
    rc = main([])
    assert rc == 1
    out = capsys.readouterr().out
    assert "usage:" in out.lower() or "help" in out.lower()


# ── 6. Module docstring requirement checks ────────────────────────────────


def test_module_docstring_covers_setup_and_limitations() -> None:
    """Test module docstring covers setup, permissions, token, and Page limitation."""
    import uutils.facebook_uu as fb_mod

    doc = fb_mod.__doc__ or ""
    # Check permissions
    assert "pages_manage_posts" in doc
    assert "pages_read_engagement" in doc
    # Check setup details
    assert "chmod 600" in doc
    assert DEFAULT_CREDENTIALS_FILE in doc
    # Check personal profile vs Page limitation
    assert "profile" in doc.lower()
    assert "page" in doc.lower()
    # Check dry-run explained
    assert "dry-run" in doc.lower() or "dry_run" in doc
