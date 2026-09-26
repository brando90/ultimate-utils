"""Offline tests for uutils.instagram_uu.

Tests verify:
1. Dry-run default makes no network calls and reads no credential files.
2. The official two-step Graph API container publish flow calls correct URLs in order.
3. Local file paths raise a clear ValueError when dry_run=False, and report warnings in dry-run.
4. Folder post planning pairs image files with sidecar .txt captions.
5. Token refresh helper works offline in dry-run and with mocked requests.
6. CLI preview and post commands behave correctly.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests

from uutils.instagram_uu import (
    DEFAULT_API_VERSION,
    InstagramClient,
    is_public_url,
    main,
    plan_posts_from_folder,
    post_image,
    post_images_batch,
    refresh_long_lived_token,
)


# ── Fixtures & Helpers ──────────────────────────────────────────────────

def _mock_response(status_code: int = 200, json_data: dict | None = None) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_data or {}
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


# ── Test 1: Dry-run defaults make NO network calls and read NO keys ────

def test_dry_run_default_makes_no_network_call_and_reads_no_keys(monkeypatch):
    """Calling the public API with default arguments must make NO network call

    and must not attempt to read ~/keys/ or any credentials file.
    """
    def _fail_network(*args, **kwargs):
        raise RuntimeError("Network call attempted during dry-run!")

    monkeypatch.setattr(requests, "post", _fail_network)
    monkeypatch.setattr(requests, "get", _fail_network)

    # 1. InstagramClient default init
    client = InstagramClient()
    assert client.dry_run is True

    # 2. InstagramClient.from_credentials default init (no credential file read)
    client_from_creds = InstagramClient.from_credentials()
    assert client_from_creds.dry_run is True

    # 3. post_image default call
    res_post = post_image("https://example.com/photo.jpg", caption="Dry-run caption")
    assert res_post["dry_run"] is True
    assert res_post["would_post"] == "https://example.com/photo.jpg"
    assert res_post["caption"] == "Dry-run caption"

    # 4. client.post_image default call
    client_res = client.post_image("https://example.com/photo2.jpg")
    assert client_res["dry_run"] is True
    assert client_res["would_post"] == "https://example.com/photo2.jpg"

    # 5. post_images_batch default call
    res_batch = post_images_batch(
        ["https://example.com/img1.jpg", "https://example.com/img2.jpg"]
    )
    assert len(res_batch) == 2
    assert all(r["dry_run"] is True for r in res_batch)

    # 6. refresh_long_lived_token default call
    res_refresh = refresh_long_lived_token()
    assert res_refresh["dry_run"] is True
    assert res_refresh["would_refresh"] is True


# ── Test 2: Two-step publish flow with mocked requests ──────────────────

def test_two_step_publish_flow_calls_right_urls_in_order(monkeypatch):
    """Test official Meta Graph API two-step publishing flow:

    Step 1: POST https://graph.facebook.com/<ver>/<ig_user_id>/media -> returns creation container id.
    Step 2: POST https://graph.facebook.com/<ver>/<ig_user_id>/media_publish -> returns media id.
    """
    recorded_calls: list[dict] = []

    def _mock_post(url, data=None, timeout=None, **kwargs):
        recorded_calls.append({"url": url, "data": data, "timeout": timeout})
        if url.endswith("/media"):
            return _mock_response(200, {"id": "creation_container_12345"})
        elif url.endswith("/media_publish"):
            return _mock_response(200, {"id": "published_media_67890"})
        raise ValueError(f"Unexpected POST url: {url}")

    monkeypatch.setattr(requests, "post", _mock_post)

    ig_user_id = "17841400012345678"
    token = "fake_access_token_abc"
    client = InstagramClient(
        ig_user_id=ig_user_id,
        access_token=token,
        api_version=DEFAULT_API_VERSION,
        dry_run=False,
    )

    image_url = "https://images.example.com/scenery.jpg"
    caption = "A beautiful sunset #nature"

    result = client.post_image(image_url, caption=caption)

    # Assert two network calls made in order
    assert len(recorded_calls) == 2

    # Verify Step 1: container creation
    call_1 = recorded_calls[0]
    expected_step1_url = f"https://graph.facebook.com/{DEFAULT_API_VERSION}/{ig_user_id}/media"
    assert call_1["url"] == expected_step1_url
    assert call_1["data"]["image_url"] == image_url
    assert call_1["data"]["caption"] == caption
    assert call_1["data"]["access_token"] == token

    # Verify Step 2: container publication
    call_2 = recorded_calls[1]
    expected_step2_url = f"https://graph.facebook.com/{DEFAULT_API_VERSION}/{ig_user_id}/media_publish"
    assert call_2["url"] == expected_step2_url
    assert call_2["data"]["creation_id"] == "creation_container_12345"
    assert call_2["data"]["access_token"] == token

    # Verify return dict
    assert result["status"] == "published"
    assert result["id"] == "published_media_67890"
    assert result["creation_id"] == "creation_container_12345"
    assert result["media_id"] == "published_media_67890"
    assert result["image_url"] == image_url
    assert result["caption"] == caption


# ── Test 3: Local path error outside dry-run ────────────────────────────

@pytest.mark.parametrize(
    "invalid_path",
    [
        "/local/path/to/image.jpg",
        "./relative/path/image.png",
        "relative_file.jpeg",
        "file:///Users/brando/photo.jpg",
        "C:\\Users\\photo.jpg",
    ],
)
def test_local_path_raises_value_error_when_not_dry_run(invalid_path):
    """Instagram Graph API requires a publicly reachable URL, not a local file path.

    When dry_run=False, passing a local path must raise a ValueError.
    """
    client = InstagramClient(
        ig_user_id="178414000",
        access_token="token",
        dry_run=False,
    )
    with pytest.raises(ValueError, match="publicly reachable image URL"):
        client.post_image(invalid_path)

    with pytest.raises(ValueError, match="publicly reachable image URL"):
        post_image(invalid_path, client=client, dry_run=False)


# ── Test 4: Local path reported in dry-run plan ─────────────────────────

def test_local_path_reported_in_dry_run_plan():
    """In dry-run mode, passing a local path does NOT raise, but notes the warning in the plan."""
    client = InstagramClient(dry_run=True)
    plan = client.post_image("/local/path/image.jpg", caption="Test caption")

    assert plan["dry_run"] is True
    assert plan["would_post"] == "/local/path/image.jpg"
    assert plan["caption"] == "Test caption"
    assert plan["is_public_url"] is False
    assert "warning" in plan
    assert "publicly reachable" in plan["warning"]


# ── Test 5: Folder post planning with sidecar captions ──────────────────

def test_plan_posts_from_folder_with_sidecars(tmp_path: Path):
    """plan_posts_from_folder lists images and pairs each with its sidecar .txt caption file."""
    # Create sample image files and sidecar captions
    img1 = tmp_path / "01_beach.jpg"
    img1.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 20)  # dummy JPEG
    cap1 = tmp_path / "01_beach.txt"
    cap1.write_text("Beach day with friends! 🌊", encoding="utf-8")

    img2 = tmp_path / "02_mountains.PNG"
    img2.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)  # dummy PNG
    cap2 = tmp_path / "02_mountains.txt"
    cap2.write_text("Climbed the peak today 🏔️\nSecond line.", encoding="utf-8")

    img3 = tmp_path / "03_food.jpeg"
    img3.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 20)
    # img3 has NO sidecar caption file

    # Non-image files that should be ignored
    (tmp_path / "notes.txt").write_text("Orphan text file", encoding="utf-8")
    (tmp_path / "document.pdf").write_bytes(b"%PDF" + b"\x00" * 10)

    plans = plan_posts_from_folder(tmp_path)

    assert len(plans) == 3

    # Check img1 plan
    p1 = next(p for p in plans if p["filename"] == "01_beach.jpg")
    assert p1["caption"] == "Beach day with friends! 🌊"
    assert p1["caption_file"] == str(cap1)
    assert p1["dry_run"] is True
    assert p1["needs_public_url"] is True

    # Check img2 plan (case-insensitive extension .PNG)
    p2 = next(p for p in plans if p["filename"] == "02_mountains.PNG")
    assert p2["caption"] == "Climbed the peak today 🏔️\nSecond line."
    assert p2["caption_file"] == str(cap2)

    # Check img3 plan (no sidecar)
    p3 = next(p for p in plans if p["filename"] == "03_food.jpeg")
    assert p3["caption"] == ""
    assert p3["caption_file"] is None

    # Check non-existent directory error
    with pytest.raises(FileNotFoundError):
        plan_posts_from_folder(tmp_path / "non_existent_folder")


def test_plan_posts_from_folder_with_double_suffix_sidecar(tmp_path: Path):
    """Supports sidecar captions formatted like photo.jpg.txt as well as photo.txt."""
    img = tmp_path / "sunset.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 10)
    cap = tmp_path / "sunset.jpg.txt"
    cap.write_text("Sunset caption via double suffix", encoding="utf-8")

    plans = plan_posts_from_folder(tmp_path)
    assert len(plans) == 1
    assert plans[0]["caption"] == "Sunset caption via double suffix"


# ── Test 6: Batch image posting ─────────────────────────────────────────

def test_post_images_batch_dry_run_and_mismatched_captions():
    """Test batch posting validation and dry-run execution."""
    client = InstagramClient(dry_run=True)
    urls = [
        "https://example.com/pic1.jpg",
        "https://example.com/pic2.jpg",
        "https://example.com/pic3.jpg",
    ]

    # Mismatched captions length
    with pytest.raises(ValueError, match="captions length"):
        client.post_images_batch(urls, captions=["Only one caption"])

    # Default captions (all empty strings)
    results = client.post_images_batch(urls)
    assert len(results) == 3
    assert [r["caption"] for r in results] == ["", "", ""]
    assert all(r["dry_run"] is True for r in results)

    # Provided captions matching length
    custom_captions = ["First", "Second", "Third"]
    results_custom = client.post_images_batch(urls, captions=custom_captions)
    assert len(results_custom) == 3
    assert [r["caption"] for r in results_custom] == custom_captions


def test_post_images_batch_mocked_execution(monkeypatch):
    """Test batch posting with mocked HTTP calls."""
    call_counts = {"media": 0, "publish": 0}

    def _mock_post(url, data=None, **kwargs):
        if url.endswith("/media"):
            call_counts["media"] += 1
            return _mock_response(200, {"id": f"cont_{call_counts['media']}"})
        elif url.endswith("/media_publish"):
            call_counts["publish"] += 1
            return _mock_response(200, {"id": f"media_{call_counts['publish']}"})
        raise ValueError(f"Unexpected URL: {url}")

    monkeypatch.setattr(requests, "post", _mock_post)

    client = InstagramClient("user123", "token123", dry_run=False)
    urls = ["https://example.com/1.jpg", "https://example.com/2.jpg"]
    caps = ["Cap 1", "Cap 2"]

    results = client.post_images_batch(urls, captions=caps)
    assert len(results) == 2
    assert call_counts["media"] == 2
    assert call_counts["publish"] == 2
    assert results[0]["id"] == "media_1"
    assert results[1]["id"] == "media_2"


# ── Test 7: refresh_long_lived_token ────────────────────────────────────

def test_refresh_long_lived_token_dry_run_and_mocked(monkeypatch):
    """Test refresh_long_lived_token helper in dry-run and with mocked requests."""
    # 1. Dry-run FB exchange token
    fb_dry = refresh_long_lived_token(
        access_token="old_token",
        client_id="app_123",
        client_secret="sec_456",
        dry_run=True,
    )
    assert fb_dry["dry_run"] is True
    assert fb_dry["flow"] == "fb_exchange_token"
    assert "oauth/access_token" in fb_dry["endpoint"]

    # 2. Dry-run IG refresh token
    ig_dry = refresh_long_lived_token(
        access_token="old_token",
        dry_run=True,
    )
    assert ig_dry["dry_run"] is True
    assert ig_dry["flow"] == "ig_refresh_token"
    assert "graph.instagram.com/refresh_access_token" in ig_dry["endpoint"]

    # 3. Mocked FB exchange token call
    def _mock_get_fb(url, params=None, **kwargs):
        assert "oauth/access_token" in url
        assert params["grant_type"] == "fb_exchange_token"
        assert params["client_id"] == "app_123"
        assert params["client_secret"] == "sec_456"
        assert params["fb_exchange_token"] == "old_token"
        return _mock_response(200, {"access_token": "new_long_token", "expires_in": 5184000})

    monkeypatch.setattr(requests, "get", _mock_get_fb)
    fb_res = refresh_long_lived_token(
        access_token="old_token",
        client_id="app_123",
        client_secret="sec_456",
        dry_run=False,
    )
    assert fb_res["access_token"] == "new_long_token"

    # 4. Mocked IG token refresh call
    def _mock_get_ig(url, params=None, **kwargs):
        assert "refresh_access_token" in url
        assert params["grant_type"] == "ig_refresh_token"
        assert params["access_token"] == "old_token"
        return _mock_response(200, {"access_token": "new_ig_token", "expires_in": 5184000})

    monkeypatch.setattr(requests, "get", _mock_get_ig)
    ig_res = refresh_long_lived_token(
        access_token="old_token",
        dry_run=False,
    )
    assert ig_res["access_token"] == "new_ig_token"

    # 5. Missing token in non-dry-run raises ValueError
    with pytest.raises(ValueError, match="access_token is required"):
        refresh_long_lived_token(access_token="", dry_run=False)


# ── Test 8: Credentials file loading ────────────────────────────────────

def test_from_credentials_loading(tmp_path: Path):
    """Test credentials JSON loading: skipped in dry-run, parsed when dry_run=False."""
    # In dry-run, missing file does not raise
    client_dry = InstagramClient.from_credentials(
        credentials_file=tmp_path / "nonexistent.json",
        dry_run=True,
    )
    assert client_dry.dry_run is True

    # Missing file when dry_run=False raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        InstagramClient.from_credentials(
            credentials_file=tmp_path / "nonexistent.json",
            dry_run=False,
        )

    # Missing ig_user_id raises ValueError
    bad_creds_1 = tmp_path / "bad1.json"
    bad_creds_1.write_text(json.dumps({"access_token": "abc"}), encoding="utf-8")
    with pytest.raises(ValueError, match="Missing 'ig_user_id'"):
        InstagramClient.from_credentials(credentials_file=bad_creds_1, dry_run=False)

    # Missing access_token raises ValueError
    bad_creds_2 = tmp_path / "bad2.json"
    bad_creds_2.write_text(json.dumps({"ig_user_id": "123"}), encoding="utf-8")
    with pytest.raises(ValueError, match="Missing 'access_token'"):
        InstagramClient.from_credentials(credentials_file=bad_creds_2, dry_run=False)

    # Valid credentials file
    valid_creds = tmp_path / "valid.json"
    valid_creds.write_text(
        json.dumps({
            "ig_user_id": "1784149999",
            "access_token": "valid_token_xyz",
            "api_version": "v21.0",
        }),
        encoding="utf-8",
    )
    client_valid = InstagramClient.from_credentials(
        credentials_file=valid_creds,
        dry_run=False,
    )
    assert client_valid.dry_run is False
    assert client_valid.ig_user_id == "1784149999"
    assert client_valid.access_token == "valid_token_xyz"
    assert client_valid.api_version == "v21.0"


# ── Test 9: CLI preview command ─────────────────────────────────────────

def test_cli_preview(tmp_path: Path, capsys):
    """Test CLI preview subcommand runs offline and prints planned posts."""
    img = tmp_path / "preview_pic.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 10)
    cap = tmp_path / "preview_pic.txt"
    cap.write_text("Preview caption text", encoding="utf-8")

    code = main(["preview", "--folder", str(tmp_path)])
    assert code == 0

    captured = capsys.readouterr().out
    assert "preview_pic.jpg" in captured
    assert "Preview caption text" in captured


# ── Test 10: CLI post command dry-run and send ──────────────────────────

def test_cli_post_dry_run_by_default(capsys):
    """CLI post subcommand runs in dry-run mode unless --send is passed."""
    code = main([
        "post",
        "--image-url", "https://example.com/cli_test.jpg",
        "--caption", "CLI test caption",
    ])
    assert code == 0

    captured = capsys.readouterr().out
    assert "[DRY-RUN]" in captured
    assert "https://example.com/cli_test.jpg" in captured
    assert "CLI test caption" in captured


def test_cli_post_with_send(monkeypatch, tmp_path: Path):
    """CLI post subcommand with --send executes publish flow using mocked requests."""
    creds_file = tmp_path / "test_creds.json"
    creds_file.write_text(
        json.dumps({"ig_user_id": "17841400", "access_token": "cli_tok"}),
        encoding="utf-8",
    )

    recorded_urls: list[str] = []

    def _mock_post(url, data=None, **kwargs):
        recorded_urls.append(url)
        if url.endswith("/media"):
            return _mock_response(200, {"id": "cli_container_id"})
        elif url.endswith("/media_publish"):
            return _mock_response(200, {"id": "cli_media_id"})
        raise ValueError(f"Unexpected url: {url}")

    monkeypatch.setattr(requests, "post", _mock_post)

    code = main([
        "post",
        "--image-url", "https://example.com/cli_send.jpg",
        "--caption", "Live CLI caption",
        "--credentials", str(creds_file),
        "--send",
    ])
    assert code == 0
    assert len(recorded_urls) == 2


def test_cli_no_args_prints_help(capsys):
    """CLI with no arguments shows help without error."""
    code = main([])
    assert code == 0


# ── Test 11: Error handling during publish flow ─────────────────────────

def test_error_handling_when_meta_api_fails(monkeypatch):
    """Test error handling when Meta Graph API returns an error response."""
    # Container creation error
    def _mock_err_post(url, data=None, **kwargs):
        return _mock_response(200, {"error": {"message": "Invalid OAuth access token", "type": "OAuthException"}})

    monkeypatch.setattr(requests, "post", _mock_err_post)

    client = InstagramClient("user", "token", dry_run=False)
    with pytest.raises(RuntimeError, match="Instagram container creation error"):
        client.post_image("https://example.com/err.jpg")
