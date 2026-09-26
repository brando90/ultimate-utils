"""Instagram Graph API utilities — automate image posting and caption management.

Supports the official Meta Instagram Graph API content-publishing flow:
    Step 1: Create a media container (POST /{ig_user_id}/media)
    Step 2: Publish the container (POST /{ig_user_id}/media_publish)

Quick usage (dry-run by default):
    from uutils.instagram_uu import post_image, plan_posts_from_folder

    # Preview posts from a local folder (e.g. synced from Google Drive)
    posts = plan_posts_from_folder("~/GoogleDrive/InstagramPosts")

    # Dry-run post (makes no network call, reads no credentials)
    post_image("https://example.com/photo.jpg", caption="Hello Instagram!")

Class-based usage:
    from uutils.instagram_uu import InstagramClient

    # Safe dry-run client (default)
    client = InstagramClient.from_credentials(dry_run=True)
    client.post_image("https://example.com/photo.jpg", caption="My Caption")

    # Live posting client (requires credentials file)
    # client = InstagramClient.from_credentials(dry_run=False)
    # client.post_image("https://example.com/photo.jpg", caption="My Caption")

One-time setup (Instagram Professional Account + Meta App):
    1. Professional Account:
       Switch your Instagram account to a Professional Account (Creator or Business)
       in Instagram Settings -> Account type and tools. Personal accounts cannot
       publish via the Graph API.

    2. Link to a Facebook Page:
       In your Facebook Page Settings -> Linked Accounts -> Instagram, connect your
       Instagram Professional account.

    3. Create a Meta Developer App:
       Go to https://developers.facebook.com/apps/ and create a new App (type: "Business").
       Add the "Instagram Graph API" product to the app.

    4. App Permissions:
       Ensure your app has the following permissions:
       - instagram_basic
       - instagram_content_publish
       - pages_show_list
       - pages_read_engagement

    5. Generate a Long-Lived Token:
       In the Meta Graph API Explorer (https://developers.facebook.com/tools/explorer/),
       generate a User Access Token with the permissions above. Exchange it for a
       long-lived (60-day) token via oauth/access_token or refresh_long_lived_token().

    6. Find your Instagram Business Account ID (ig_user_id):
       GET https://graph.facebook.com/v21.0/me/accounts?access_token=<YOUR_TOKEN>
       Locate your Facebook Page ID, then query:
       GET https://graph.facebook.com/v21.0/<PAGE_ID>?fields=instagram_business_account&access_token=<YOUR_TOKEN>
       The returned "instagram_business_account.id" is your ig_user_id.

    7. Save credentials to JSON file:
       cat > ~/keys/instagram_credentials.json << 'JSON'
       {
           "ig_user_id": "YOUR_INSTAGRAM_BUSINESS_ACCOUNT_ID",
           "access_token": "YOUR_LONG_LIVED_ACCESS_TOKEN",
           "api_version": "v21.0"
       }
       JSON
       chmod 600 ~/keys/instagram_credentials.json

Public Image URL Constraint:
    The Instagram Graph API requires a PUBLICLY REACHABLE image URL (http:// or https://),
    NOT a local file path. Meta's servers download the image directly from the provided URL.
    Images from local folders (e.g. synced from Google Drive) must be hosted on a public
    endpoint (e.g. AWS S3, Cloudinary, public Google Drive direct download URL, or a web server)
    before calling post_image() outside dry-run mode. Passing a local file path to post_image()
    when dry_run=False raises a ValueError.

CLI Usage:
    # Preview posts from a local folder with sidecar .txt captions (always offline):
    python -m uutils.instagram_uu preview --folder /path/to/drive_images

    # Dry-run post preview (default: no network request, reads no keys):
    python -m uutils.instagram_uu post --image-url https://example.com/photo.jpg --caption "My post"

    # Live post execution (requires --send flag and credentials file):
    python -m uutils.instagram_uu post --image-url https://example.com/photo.jpg --caption "My post" --send

Refs:
    - Instagram Content Publishing: https://developers.facebook.com/docs/instagram-platform/instagram-graph-api/content-publishing
    - Graph API Container Endpoint: https://developers.facebook.com/docs/graph-api/reference/v21.0/page/media
    - Long-Lived Tokens: https://developers.facebook.com/docs/facebook-login/guides/access-tokens/get-long-lived
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import requests

log = logging.getLogger(__name__)

DEFAULT_CREDENTIALS_FILE = "~/keys/instagram_credentials.json"
DEFAULT_API_VERSION = "v21.0"
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def is_public_url(url: str) -> bool:
    """Check if a given string is a publicly reachable HTTP/HTTPS URL.

    The Instagram Graph API requires images to be accessible via HTTP/HTTPS.
    Local file paths, file:// URIs, or non-HTTP schemes are not valid.
    """
    if not isinstance(url, str):
        return False
    trimmed = url.strip()
    return trimmed.startswith(("http://", "https://"))


def _load_credentials(credentials_file: str | Path = "") -> dict:
    """Load Instagram Graph API credentials from JSON file.

    Called only when dry_run=False.

    Expected JSON structure:
    {
        "ig_user_id": "17841400000000000",
        "access_token": "EAAB...",
        "api_version": "v21.0"  # optional, defaults to v21.0
    }
    """
    fpath = Path(credentials_file or DEFAULT_CREDENTIALS_FILE).expanduser()
    if not fpath.is_file():
        raise FileNotFoundError(
            f"Instagram credentials file not found at {fpath}\n"
            f"Create it with your Instagram Graph API credentials — "
            f"see module docstring for setup instructions."
        )
    try:
        config = json.loads(fpath.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to parse credentials JSON at {fpath}: {exc}") from exc

    ig_user_id = str(config.get("ig_user_id", "")).strip()
    access_token = str(config.get("access_token", "")).strip()
    api_version = str(config.get("api_version", DEFAULT_API_VERSION)).strip() or DEFAULT_API_VERSION

    if not ig_user_id:
        raise ValueError(f"Missing 'ig_user_id' in {fpath}")
    if not access_token:
        raise ValueError(f"Missing 'access_token' in {fpath}")

    return {
        "ig_user_id": ig_user_id,
        "access_token": access_token,
        "api_version": api_version,
    }


class InstagramClient:
    """Client for the official Instagram Graph API content-publishing flow.

    Args:
        ig_user_id: Instagram Business Account ID.
        access_token: Long-lived Meta/Instagram Graph API access token.
        api_version: Graph API version string (default: "v21.0").
        dry_run: If True (default), make no network calls and read no credential files.
    """

    def __init__(
        self,
        ig_user_id: str = "",
        access_token: str = "",
        api_version: str = DEFAULT_API_VERSION,
        dry_run: bool = True,
    ):
        self.ig_user_id = str(ig_user_id).strip()
        self.access_token = str(access_token).strip()
        self.api_version = str(api_version).strip() if api_version else DEFAULT_API_VERSION
        self.dry_run = dry_run
        self.base_url = f"https://graph.facebook.com/{self.api_version}"

    @classmethod
    def from_credentials(
        cls,
        credentials_file: str | Path = DEFAULT_CREDENTIALS_FILE,
        dry_run: bool = True,
    ) -> InstagramClient:
        """Instantiate client from credentials JSON file.

        In dry-run mode (default), credentials are not read from disk.
        """
        if dry_run:
            return cls(
                ig_user_id="dry_run_user",
                access_token="dry_run_token",
                api_version=DEFAULT_API_VERSION,
                dry_run=True,
            )
        creds = _load_credentials(credentials_file)
        return cls(
            ig_user_id=creds["ig_user_id"],
            access_token=creds["access_token"],
            api_version=creds.get("api_version", DEFAULT_API_VERSION),
            dry_run=False,
        )

    def post_image(
        self,
        image_url: str,
        caption: str = "",
        dry_run: bool | None = None,
    ) -> dict:
        """Publish a single image post to Instagram.

        Flow:
            Step 1: POST https://graph.facebook.com/<ver>/<ig_user_id>/media
                    with image_url, caption, access_token -> returns container creation_id.
            Step 2: POST https://graph.facebook.com/<ver>/<ig_user_id>/media_publish
                    with creation_id, access_token -> returns published media id.

        Constraint:
            image_url must be a publicly accessible HTTP/HTTPS URL. Local file paths
            raise a ValueError when dry_run=False. In dry-run mode, the issue is noted
            in the returned plan dict.

        Args:
            image_url: Public HTTP/HTTPS URL of the image.
            caption: Post caption string.
            dry_run: If True, execute offline preview without network calls.
                     Defaults to client's self.dry_run setting.

        Returns:
            Dict containing publish status or dry-run plan.
        """
        is_dry = self.dry_run if dry_run is None else dry_run
        is_url = is_public_url(image_url)

        if not is_url:
            if not is_dry:
                raise ValueError(
                    f"Instagram Graph API requires a publicly reachable image URL "
                    f"(starting with http:// or https://), not a local path: {image_url!r}. "
                    f"Meta's servers must be able to fetch the image from the public internet. "
                    f"Please host the image publicly first."
                )

        if is_dry:
            warning = None
            if not is_url:
                warning = (
                    f"Local path detected ({image_url}). Instagram Graph API requires "
                    f"a publicly reachable image URL (http/https). Host the image publicly "
                    f"before publishing."
                )
            log.info("[DRY-RUN] Instagram post preview: %s, caption=%s", image_url, caption[:100])
            print("[DRY-RUN] Instagram post preview:")
            print(f"  Image URL: {image_url}")
            if warning:
                print(f"  WARNING:   {warning}")
            print(f"  Caption:   {caption if caption else '(none)'}")

            result = {
                "dry_run": True,
                "would_post": image_url,
                "caption": caption,
                "is_public_url": is_url,
                "api_version": self.api_version,
                "ig_user_id": self.ig_user_id,
                "step1_url": f"{self.base_url}/{self.ig_user_id}/media",
                "step2_url": f"{self.base_url}/{self.ig_user_id}/media_publish",
            }
            if warning:
                result["warning"] = warning
            return result

        # ── Step 1: Create media container ──────────────────────────────
        step1_url = f"{self.base_url}/{self.ig_user_id}/media"
        step1_payload: dict[str, str] = {
            "image_url": image_url,
            "access_token": self.access_token,
        }
        if caption:
            step1_payload["caption"] = caption

        log.info("Creating Instagram media container at %s", step1_url)
        resp1 = requests.post(step1_url, data=step1_payload, timeout=30)
        resp1.raise_for_status()
        res1_json = resp1.json()

        if "error" in res1_json:
            raise RuntimeError(f"Instagram container creation error: {res1_json['error']}")

        creation_id = res1_json.get("id")
        if not creation_id:
            raise RuntimeError(f"Instagram Graph API failed to return a container creation id: {res1_json}")

        # ── Step 2: Publish media container ─────────────────────────────
        step2_url = f"{self.base_url}/{self.ig_user_id}/media_publish"
        step2_payload = {
            "creation_id": creation_id,
            "access_token": self.access_token,
        }

        log.info("Publishing Instagram container %s at %s", creation_id, step2_url)
        resp2 = requests.post(step2_url, data=step2_payload, timeout=30)
        resp2.raise_for_status()
        res2_json = resp2.json()

        if "error" in res2_json:
            raise RuntimeError(f"Instagram media publish error: {res2_json['error']}")

        media_id = res2_json.get("id")

        return {
            "id": media_id,
            "creation_id": creation_id,
            "media_id": media_id,
            "status": "published",
            "image_url": image_url,
            "caption": caption,
            "raw_response": res2_json,
        }

    def post_images_batch(
        self,
        image_urls: list[str],
        captions: list[str] | None = None,
        dry_run: bool | None = None,
    ) -> list[dict]:
        """Publish multiple image posts sequentially.

        Args:
            image_urls: List of publicly accessible image URLs.
            captions: List of captions matching image_urls length. Defaults to empty strings.
            dry_run: If True, preview all posts without making network calls.

        Returns:
            List of response dicts (one per image).
        """
        if captions is None:
            captions = ["" for _ in image_urls]
        if len(captions) != len(image_urls):
            raise ValueError(
                f"captions length ({len(captions)}) must match image_urls length ({len(image_urls)})"
            )

        results: list[dict] = []
        for url, caption in zip(image_urls, captions):
            res = self.post_image(url, caption=caption, dry_run=dry_run)
            results.append(res)
        return results

    def refresh_long_lived_token(
        self,
        client_id: str = "",
        client_secret: str = "",
        dry_run: bool | None = None,
    ) -> dict:
        """Refresh long-lived access token using client's credentials."""
        is_dry = self.dry_run if dry_run is None else dry_run
        return refresh_long_lived_token(
            access_token=self.access_token,
            client_id=client_id,
            client_secret=client_secret,
            api_version=self.api_version,
            dry_run=is_dry,
        )


def post_image(
    image_url: str,
    caption: str = "",
    credentials_file: str | Path = "",
    client: InstagramClient | None = None,
    dry_run: bool = True,
) -> dict:
    """Publish an image to Instagram via the official Instagram Graph API.

    Convenience wrapper around InstagramClient.post_image().

    Args:
        image_url: Publicly accessible HTTP/HTTPS URL of the image.
        caption: Caption text for the Instagram post (optional).
        credentials_file: Path to credentials JSON (ignored in dry-run).
        client: Pre-configured InstagramClient instance (optional).
        dry_run: If True (default), make no network requests and read no credentials.

    Returns:
        Dict with publication status and IDs, or dry-run plan.
    """
    if client is None:
        client = InstagramClient.from_credentials(credentials_file=credentials_file, dry_run=dry_run)
    return client.post_image(image_url, caption=caption, dry_run=dry_run)


def post_images_batch(
    image_urls: list[str],
    captions: list[str] | None = None,
    credentials_file: str | Path = "",
    client: InstagramClient | None = None,
    dry_run: bool = True,
) -> list[dict]:
    """Publish a batch of images to Instagram via the official Instagram Graph API.

    Convenience wrapper around InstagramClient.post_images_batch().

    Args:
        image_urls: List of publicly accessible HTTP/HTTPS image URLs.
        captions: Optional list of captions matching image_urls length. Defaults to "".
        credentials_file: Path to credentials JSON (ignored in dry-run).
        client: Pre-configured InstagramClient instance (optional).
        dry_run: If True (default), make no network requests and read no credentials.

    Returns:
        List of dicts with publication status and IDs, or dry-run plans.
    """
    if client is None:
        client = InstagramClient.from_credentials(credentials_file=credentials_file, dry_run=dry_run)
    return client.post_images_batch(image_urls, captions=captions, dry_run=dry_run)


def refresh_long_lived_token(
    access_token: str = "",
    client_id: str = "",
    client_secret: str = "",
    api_version: str = DEFAULT_API_VERSION,
    dry_run: bool = True,
) -> dict:
    """Refresh a long-lived access token.

    Supports two Meta/Instagram token refresh mechanisms:

    1. Facebook App Exchange Token (Meta Graph API):
       When `client_id` and `client_secret` are provided along with `access_token`:
       GET https://graph.facebook.com/<ver>/oauth/access_token
         ?grant_type=fb_exchange_token
         &client_id=<app_id>
         &client_secret=<app_secret>
         &fb_exchange_token=<access_token>

    2. Instagram User Token Refresh (Instagram Basic Display / Login):
       When only `access_token` is provided:
       GET https://graph.instagram.com/refresh_access_token
         ?grant_type=ig_refresh_token
         &access_token=<access_token>

    Args:
        access_token: Existing long-lived access token to refresh.
        client_id: Meta App ID (optional, for fb_exchange_token flow).
        client_secret: Meta App Secret (optional, for fb_exchange_token flow).
        api_version: Graph API version (default: "v21.0").
        dry_run: If True (default), make no network requests and return preview.

    Returns:
        API response dict with new access token and expires_in, or preview dict in dry-run.
    """
    if client_id and client_secret:
        endpoint = f"https://graph.facebook.com/{api_version}/oauth/access_token"
        flow = "fb_exchange_token"
    else:
        endpoint = "https://graph.instagram.com/refresh_access_token"
        flow = "ig_refresh_token"

    if dry_run:
        log.info("[DRY-RUN] Refresh token preview (%s) for %s", flow, endpoint)
        print(f"[DRY-RUN] Refresh token preview ({flow}):")
        print(f"  Endpoint: {endpoint}")
        return {
            "dry_run": True,
            "would_refresh": True,
            "flow": flow,
            "endpoint": endpoint,
            "api_version": api_version,
            "note": "Token refresh preview. No network call made in dry-run.",
        }

    if not access_token:
        raise ValueError("access_token is required to refresh token")

    if flow == "fb_exchange_token":
        params = {
            "grant_type": "fb_exchange_token",
            "client_id": client_id,
            "client_secret": client_secret,
            "fb_exchange_token": access_token,
        }
    else:
        params = {
            "grant_type": "ig_refresh_token",
            "access_token": access_token,
        }

    resp = requests.get(endpoint, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def plan_posts_from_folder(
    folder: str | Path,
    caption_suffix: str = ".txt",
) -> list[dict]:
    """Scan a local folder of images and pair each with a sidecar caption file.

    Useful for preparing posts from a folder synced from Google Drive or local storage.
    Note: Local images cannot be posted directly to Instagram Graph API; they must
    first be uploaded to a publicly accessible HTTP/HTTPS URL. This function only
    reads local files and makes no network requests.

    Args:
        folder: Path to folder containing image files (.jpg, .jpeg, .png).
        caption_suffix: Suffix for sidecar caption files (default: ".txt").
            For example, "photo1.jpg" pairs with "photo1.txt".

    Returns:
        List of dicts describing planned posts:
        [
            {
                "dry_run": True,
                "local_image_path": "/path/to/photo1.jpg",
                "filename": "photo1.jpg",
                "caption": "My caption text...",
                "caption_file": "/path/to/photo1.txt",
                "would_post": "/path/to/photo1.jpg",
                "is_public_url": False,
                "needs_public_url": True,
                "note": "Local image must be hosted at a public HTTP/HTTPS URL before posting.",
            },
            ...
        ]
    """
    fpath = Path(folder).expanduser().resolve()
    if not fpath.is_dir():
        raise FileNotFoundError(f"Folder not found: {fpath}")

    # Find image files, case-insensitively
    image_files = [
        p for p in fpath.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    image_files.sort(key=lambda p: p.name)

    plans: list[dict] = []
    print(f"[DRY-RUN] Planning Instagram posts from folder: {fpath}")
    print(f"Found {len(image_files)} image file(s)")

    for idx, img_path in enumerate(image_files, 1):
        # Support both photo.txt and photo.jpg.txt as sidecar
        candidate1 = img_path.with_suffix(caption_suffix)
        candidate2 = img_path.parent / f"{img_path.name}{caption_suffix}"

        caption = ""
        caption_file_str = None
        if candidate1.is_file():
            caption = candidate1.read_text(encoding="utf-8").strip()
            caption_file_str = str(candidate1)
        elif candidate2.is_file():
            caption = candidate2.read_text(encoding="utf-8").strip()
            caption_file_str = str(candidate2)

        plan = {
            "dry_run": True,
            "local_image_path": str(img_path),
            "filename": img_path.name,
            "caption": caption,
            "caption_file": caption_file_str,
            "would_post": str(img_path),
            "is_public_url": False,
            "needs_public_url": True,
            "note": (
                "Local image from folder must be uploaded/hosted at a public "
                "HTTP/HTTPS URL before posting via Instagram Graph API."
            ),
        }
        plans.append(plan)

        print(f"  [{idx}/{len(image_files)}] {img_path.name}")
        if caption:
            preview_cap = caption.replace("\n", " ")
            if len(preview_cap) > 80:
                preview_cap = preview_cap[:77] + "..."
            print(f"      Caption: {preview_cap}")
        else:
            print("      Caption: (no sidecar caption file)")

    return plans


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for previewing and posting to Instagram."""
    parser = argparse.ArgumentParser(
        prog="python -m uutils.instagram_uu",
        description="Automate Instagram posting via the official Instagram Graph API.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # preview command
    preview_parser = subparsers.add_parser(
        "preview",
        help="Preview posts from a local folder (e.g. synced from Google Drive) with sidecar captions (always offline).",
    )
    preview_parser.add_argument(
        "--folder",
        required=True,
        help="Path to folder containing image files and optional sidecar .txt captions.",
    )
    preview_parser.add_argument(
        "--caption-suffix",
        default=".txt",
        help="Suffix for sidecar caption files (default: .txt).",
    )

    # post command
    post_parser = subparsers.add_parser(
        "post",
        help="Post an image to Instagram (dry-run by default; use --send to publish).",
    )
    post_parser.add_argument(
        "--image-url",
        required=True,
        help="Public HTTP/HTTPS URL of the image to post.",
    )
    post_parser.add_argument(
        "--caption",
        default="",
        help="Caption text for the post.",
    )
    post_parser.add_argument(
        "--credentials",
        default="",
        help="Path to credentials JSON file (default: ~/keys/instagram_credentials.json).",
    )
    post_parser.add_argument(
        "--send",
        action="store_true",
        default=False,
        help="Execute the real post. Without this flag, runs in dry-run mode (no network, no credentials read).",
    )

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "preview":
        plan_posts_from_folder(args.folder, caption_suffix=args.caption_suffix)
        return 0

    if args.command == "post":
        dry_run = not args.send
        client = InstagramClient.from_credentials(
            credentials_file=args.credentials,
            dry_run=dry_run,
        )
        client.post_image(
            image_url=args.image_url,
            caption=args.caption,
            dry_run=dry_run,
        )
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
