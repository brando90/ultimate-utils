"""Facebook Page posting and automation — post text updates, photos, and batches via Meta Graph API.

Quick usage:
    from uutils.facebook_uu import FacebookClient, plan_posts_from_folder, post_image, post_text

    # Dry-run by default (no network calls, no credentials read):
    post_text("Hello from uutils!")
    post_image("path/to/image.jpg", caption="My caption")

    # Folder planning (pairs .jpg/.jpeg/.png with .txt sidecar captions):
    posts = plan_posts_from_folder("path/to/folder")

    # Explicit client usage (requires valid credentials in ~/keys/facebook_credentials.json):
    client = FacebookClient.from_credentials("~/keys/facebook_credentials.json", dry_run=False)
    client.post_text("Announcing our new release!")
    client.post_image("path/to/diagram.png", caption="System architecture")

CLI usage:
    # Preview folder posts offline (makes no network calls, reads no credentials):
    python -m uutils.facebook_uu preview --folder /path/to/folder

    # Dry-run CLI (default behavior without --send):
    python -m uutils.facebook_uu post-text --message "Hello from CLI"
    python -m uutils.facebook_uu post --image /path/to/image.jpg --caption "My caption"

    # Real publishing (requires explicit --send and ~/keys/facebook_credentials.json):
    python -m uutils.facebook_uu post-text --message "Hello from CLI" --send
    python -m uutils.facebook_uu post --image /path/to/image.jpg --caption "My caption" --send

Important API Limitation (Personal Profiles vs Pages):
    Since April 2018 (with the deprecation of the publish_actions permission in Graph API v3.0),
    Meta does NOT permit third-party applications or APIs to publish posts, photos, or updates to a
    personal user profile timeline. Programmatic publishing is exclusively supported for Facebook
    Pages that you manage / administrate. Consequently, this module uses Page Access Tokens and
    targets Page IDs (POST /{page_id}/feed and POST /{page_id}/photos).

One-Time Setup:
    1. Create a Facebook Page:
       If you do not have one, create a Page at https://www.facebook.com/pages/create
       Note your Page ID from Page Settings -> About / Page Transparency.

    2. Create a Meta Developer App:
       - Go to Meta for Developers: https://developers.facebook.com/
       - Create an app (App Type: "Business").
       - In App Dashboard, add the "Facebook Login for Business" or "Graph API" product.

    3. Required Permissions:
       Your token must include the following permissions:
       - `pages_manage_posts` (allows publishing feed posts and photos to the Page)
       - `pages_read_engagement` (allows reading Page data to confirm publications)
       - `pages_show_list` (optional, to enumerate Pages you administer)

    4. Generate a Long-Lived Page Access Token:
       - Open Graph API Explorer: https://developers.facebook.com/tools/explorer/
       - Select your Meta App and User, and add `pages_manage_posts` and `pages_read_engagement`.
       - Generate a short-lived User Access Token.
       - Exchange it for a long-lived User Access Token (valid 60 days):
           GET https://graph.facebook.com/v21.0/oauth/access_token?
               grant_type=fb_exchange_token&
               client_id=YOUR_APP_ID&
               client_secret=YOUR_APP_SECRET&
               fb_exchange_token=SHORT_LIVED_USER_TOKEN
       - Fetch the permanent / long-lived Page Access Token:
           GET https://graph.facebook.com/v21.0/me/accounts?access_token=LONG_LIVED_USER_TOKEN
       - Locate your Page in the returned data array and copy its `id` and `access_token`.

    5. Save Credentials:
       cat > ~/keys/facebook_credentials.json << 'JSON'
       {
           "page_id": "YOUR_PAGE_ID",
           "page_access_token": "YOUR_PAGE_ACCESS_TOKEN",
           "api_version": "v21.0"
       }
       JSON
       chmod 600 ~/keys/facebook_credentials.json

Refs:
    - Meta Graph API Pages: https://developers.facebook.com/docs/pages-api/posts
    - Meta Graph API Photos: https://developers.facebook.com/docs/graph-api/reference/page/photos/#Creating
    - Long-Lived Tokens: https://developers.facebook.com/docs/facebook-login/guides/access-tokens/get-long-lived
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger(__name__)

DEFAULT_CREDENTIALS_FILE = "~/keys/facebook_credentials.json"
DEFAULT_API_VERSION = "v21.0"
GRAPH_API_BASE_URL = "https://graph.facebook.com"
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class FacebookClient:
    """Facebook Graph API client for Page management with dry-run support."""

    def __init__(
        self,
        page_id: str = "",
        page_access_token: str = "",
        api_version: str = DEFAULT_API_VERSION,
        dry_run: bool = True,
    ) -> None:
        self.page_id = str(page_id).strip()
        self.page_access_token = str(page_access_token).strip()
        ver = str(api_version).strip()
        if ver and not ver.startswith("v"):
            ver = f"v{ver}"
        self.api_version = ver or DEFAULT_API_VERSION
        self.dry_run = dry_run

    @classmethod
    def from_credentials(
        cls,
        credentials_file: str | Path = DEFAULT_CREDENTIALS_FILE,
        dry_run: bool = True,
    ) -> FacebookClient:
        """Initialize FacebookClient from credentials file.

        When dry_run=True, no credentials file is read or required.
        """
        if dry_run:
            return cls(page_id="", page_access_token="", api_version=DEFAULT_API_VERSION, dry_run=True)

        fpath = Path(credentials_file).expanduser()
        if not fpath.is_file():
            raise FileNotFoundError(
                f"Facebook credentials file not found at {fpath}\n"
                f"See module docstring for setup and token generation instructions."
            )

        try:
            data = json.loads(fpath.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Failed to parse Facebook credentials JSON at {fpath}: {exc}") from exc

        page_id = str(data.get("page_id", "")).strip()
        page_access_token = str(data.get("page_access_token", "")).strip()
        api_version = str(data.get("api_version", DEFAULT_API_VERSION)).strip()

        if not page_id:
            raise ValueError(f"Missing required key 'page_id' in {fpath}")
        if not page_access_token:
            raise ValueError(f"Missing required key 'page_access_token' in {fpath}")

        return cls(
            page_id=page_id,
            page_access_token=page_access_token,
            api_version=api_version,
            dry_run=False,
        )

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.page_access_token}"}

    def post_text(self, message: str) -> dict[str, Any]:
        """Post a text update to the Page feed (POST /<page_id>/feed).

        Args:
            message: The text content of the post.

        Returns:
            Dict containing API response (or dry-run description).
        """
        target_page = self.page_id or "PAGE_ID"
        if self.dry_run:
            desc = f"[DRY-RUN] Facebook post_text to /{target_page}/feed: {message}"
            print(desc)
            log.info(desc)
            return {
                "dry_run": True,
                "would_post": "text",
                "message": message,
                "caption": message,
                "page_id": target_page,
                "endpoint": f"/{target_page}/feed",
            }

        if not self.page_id:
            raise ValueError("page_id is required to post to Facebook")
        if not self.page_access_token:
            raise ValueError("page_access_token is required to post to Facebook")

        url = f"{GRAPH_API_BASE_URL}/{self.api_version}/{self.page_id}/feed"
        data = {"message": message}
        resp = requests.post(url, headers=self._auth_headers(), data=data, timeout=30)
        resp.raise_for_status()
        res = resp.json()
        if isinstance(res, dict) and "error" in res:
            err = res["error"]
            err_msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            raise RuntimeError(f"Facebook Graph API error: {err_msg}")
        log.info("Facebook post_text succeeded: id=%s", res.get("id", "?"))
        return res

    def post_image(self, image_path_or_url: str | Path, caption: str = "") -> dict[str, Any]:
        """Post an image to the Page photos album (POST /<page_id>/photos).

        If image_path_or_url is a URL, it is sent via the `url` parameter.
        If it is a local file path, it is uploaded as multipart `source`.

        Args:
            image_path_or_url: Local file path or HTTP(S) URL of the image.
            caption: Optional caption text for the photo.

        Returns:
            Dict containing API response (or dry-run description).
        """
        str_img = str(image_path_or_url).strip()
        is_url = str_img.startswith(("http://", "https://"))
        target_page = self.page_id or "PAGE_ID"

        if self.dry_run:
            desc = (
                f"[DRY-RUN] Facebook post_image to /{target_page}/photos: "
                f"image={str_img}, caption={caption}"
            )
            print(desc)
            log.info(desc)
            return {
                "dry_run": True,
                "would_post": "photo",
                "image": str_img,
                "image_path": str_img,
                "caption": caption,
                "source_type": "url" if is_url else "file",
                "page_id": target_page,
                "endpoint": f"/{target_page}/photos",
            }

        if not self.page_id:
            raise ValueError("page_id is required to post to Facebook")
        if not self.page_access_token:
            raise ValueError("page_access_token is required to post to Facebook")

        url = f"{GRAPH_API_BASE_URL}/{self.api_version}/{self.page_id}/photos"
        headers = self._auth_headers()

        if is_url:
            data: dict[str, str] = {"url": str_img}
            if caption:
                data["caption"] = caption
            resp = requests.post(url, headers=headers, data=data, timeout=60)
        else:
            img_path = Path(image_path_or_url).expanduser()
            if not img_path.is_file():
                raise FileNotFoundError(f"Image file not found: {img_path}")
            data = {}
            if caption:
                data["caption"] = caption
            with open(img_path, "rb") as fp:
                files = {"source": (img_path.name, fp)}
                resp = requests.post(url, headers=headers, data=data, files=files, timeout=60)

        resp.raise_for_status()
        res = resp.json()
        if isinstance(res, dict) and "error" in res:
            err = res["error"]
            err_msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            raise RuntimeError(f"Facebook Graph API error: {err_msg}")
        log.info("Facebook post_image succeeded: id=%s", res.get("id", res.get("post_id", "?")))
        return res

    def post_images_batch(
        self,
        images: list[str | Path],
        captions: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Post multiple images sequentially to the Page.

        Args:
            images: List of local image file paths or image URLs.
            captions: Optional list of captions corresponding to each image.

        Returns:
            List of API response dicts (or dry-run description dicts).
        """
        results: list[dict[str, Any]] = []
        for idx, img in enumerate(images):
            cap = captions[idx] if (captions is not None and idx < len(captions)) else ""
            res = self.post_image(img, caption=cap)
            results.append(res)
        return results

    def post_folder(
        self,
        folder: str | Path,
        caption_suffix: str = ".txt",
    ) -> list[dict[str, Any]]:
        """Post all images in a folder with their corresponding sidecar captions.

        Args:
            folder: Local directory containing images and sidecar captions.
            caption_suffix: Suffix for sidecar caption files (default: .txt).

        Returns:
            List of API response dicts (or dry-run description dicts).
        """
        planned = plan_posts_from_folder(folder, caption_suffix=caption_suffix)
        results: list[dict[str, Any]] = []
        for item in planned:
            res = self.post_image(item["image_path"], caption=item["caption"])
            results.append(res)
        return results


def plan_posts_from_folder(
    folder: str | Path,
    caption_suffix: str = ".txt",
) -> list[dict[str, Any]]:
    """Scan a local folder and pair images (.jpg/.jpeg/.png) with sidecar caption files.

    Sidecar captions are matched by image stem + caption_suffix (e.g. `pic.jpg` pairs
    with `pic.txt`). If no sidecar file exists, caption defaults to empty string `""`.

    Args:
        folder: Local directory path to scan.
        caption_suffix: Suffix for sidecar caption files (default: ".txt").

    Returns:
        List of dicts with keys 'image', 'image_path', 'caption', 'caption_path', 'caption_file'.
    """
    folder_path = Path(folder).expanduser()
    if not folder_path.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    images = [
        p for p in folder_path.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    images.sort(key=lambda p: p.name.lower())

    posts: list[dict[str, Any]] = []
    for img in images:
        caption_file = img.parent / f"{img.stem}{caption_suffix}"
        if caption_file.is_file():
            caption = caption_file.read_text(encoding="utf-8").strip()
            cap_str = str(caption_file)
        else:
            caption = ""
            cap_str = None

        posts.append({
            "image": str(img),
            "image_path": str(img),
            "caption": caption,
            "caption_path": cap_str,
            "caption_file": cap_str,
        })
    return posts


def post_text(
    message: str,
    page_id: str = "",
    page_access_token: str = "",
    api_version: str = DEFAULT_API_VERSION,
    credentials_file: str | Path = DEFAULT_CREDENTIALS_FILE,
    dry_run: bool = True,
    client: FacebookClient | None = None,
) -> dict[str, Any]:
    """Post text update to a Facebook Page feed.

    Defaults to dry_run=True (makes no network calls, reads no credentials).
    """
    if client is None:
        if dry_run or (not page_id and not page_access_token):
            client = FacebookClient.from_credentials(credentials_file=credentials_file, dry_run=dry_run)
            if page_id:
                client.page_id = page_id
            if page_access_token:
                client.page_access_token = page_access_token
        else:
            client = FacebookClient(
                page_id=page_id,
                page_access_token=page_access_token,
                api_version=api_version,
                dry_run=dry_run,
            )
    return client.post_text(message)


def post_image(
    image_path_or_url: str | Path,
    caption: str = "",
    page_id: str = "",
    page_access_token: str = "",
    api_version: str = DEFAULT_API_VERSION,
    credentials_file: str | Path = DEFAULT_CREDENTIALS_FILE,
    dry_run: bool = True,
    client: FacebookClient | None = None,
) -> dict[str, Any]:
    """Post an image (local file or URL) to a Facebook Page photos album.

    Defaults to dry_run=True (makes no network calls, reads no credentials).
    """
    if client is None:
        if dry_run or (not page_id and not page_access_token):
            client = FacebookClient.from_credentials(credentials_file=credentials_file, dry_run=dry_run)
            if page_id:
                client.page_id = page_id
            if page_access_token:
                client.page_access_token = page_access_token
        else:
            client = FacebookClient(
                page_id=page_id,
                page_access_token=page_access_token,
                api_version=api_version,
                dry_run=dry_run,
            )
    return client.post_image(image_path_or_url, caption=caption)


def post_images_batch(
    images: list[str | Path],
    captions: list[str] | None = None,
    page_id: str = "",
    page_access_token: str = "",
    api_version: str = DEFAULT_API_VERSION,
    credentials_file: str | Path = DEFAULT_CREDENTIALS_FILE,
    dry_run: bool = True,
    client: FacebookClient | None = None,
) -> list[dict[str, Any]]:
    """Post multiple images to a Facebook Page.

    Defaults to dry_run=True (makes no network calls, reads no credentials).
    """
    if client is None:
        if dry_run or (not page_id and not page_access_token):
            client = FacebookClient.from_credentials(credentials_file=credentials_file, dry_run=dry_run)
            if page_id:
                client.page_id = page_id
            if page_access_token:
                client.page_access_token = page_access_token
        else:
            client = FacebookClient(
                page_id=page_id,
                page_access_token=page_access_token,
                api_version=api_version,
                dry_run=dry_run,
            )
    return client.post_images_batch(images, captions=captions)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for Facebook Graph API automation."""
    parser = argparse.ArgumentParser(
        prog="python -m uutils.facebook_uu",
        description="Facebook Graph API automation CLI for Page posts and photos.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # preview --folder DIR
    preview_parser = subparsers.add_parser(
        "preview", help="Preview posts planned from a local folder (offline)"
    )
    preview_parser.add_argument(
        "--folder", "-f", required=True, help="Folder containing images and sidecar .txt captions"
    )
    preview_parser.add_argument(
        "--caption-suffix", default=".txt", help="Suffix for sidecar caption files (default: .txt)"
    )

    # post --image PATH_OR_URL --caption C [--send]
    post_parser = subparsers.add_parser("post", help="Post an image to Facebook Page")
    post_parser.add_argument("--image", "-i", required=True, help="Path to local image file or image URL")
    post_parser.add_argument("--caption", "-c", default="", help="Photo caption text")
    post_parser.add_argument(
        "--send", action="store_true", default=False, help="Actually publish (defaults to dry-run)"
    )
    post_parser.add_argument(
        "--credentials", default=DEFAULT_CREDENTIALS_FILE, help="Path to credentials JSON"
    )
    post_parser.add_argument("--page-id", default="", help="Facebook Page ID (overrides credentials file)")

    # post-text --message M [--send]
    text_parser = subparsers.add_parser("post-text", help="Post a text update to Facebook Page feed")
    text_parser.add_argument("--message", "-m", required=True, help="Message text to post")
    text_parser.add_argument(
        "--send", action="store_true", default=False, help="Actually publish (defaults to dry-run)"
    )
    text_parser.add_argument(
        "--credentials", default=DEFAULT_CREDENTIALS_FILE, help="Path to credentials JSON"
    )
    text_parser.add_argument("--page-id", default="", help="Facebook Page ID (overrides credentials file)")

    # post-folder --folder DIR [--send]
    folder_parser = subparsers.add_parser("post-folder", help="Post all images from a folder with captions")
    folder_parser.add_argument("--folder", "-f", required=True, help="Folder with images and captions")
    folder_parser.add_argument("--caption-suffix", default=".txt", help="Caption suffix (default: .txt)")
    folder_parser.add_argument(
        "--send", action="store_true", default=False, help="Actually publish (defaults to dry-run)"
    )
    folder_parser.add_argument(
        "--credentials", default=DEFAULT_CREDENTIALS_FILE, help="Path to credentials JSON"
    )
    folder_parser.add_argument("--page-id", default="", help="Facebook Page ID (overrides credentials file)")

    args = parser.parse_args(argv)

    if args.command == "preview":
        planned = plan_posts_from_folder(args.folder, caption_suffix=args.caption_suffix)
        print(f"Planned {len(planned)} post(s) from {args.folder}:")
        for idx, p in enumerate(planned, start=1):
            cap_display = p["caption"] if p["caption"] else "(no caption)"
            print(f"  {idx}. Image: {p['image_path']}")
            print(f"     Caption: {cap_display}")
        return 0

    elif args.command == "post":
        dry_run = not args.send
        client = FacebookClient.from_credentials(credentials_file=args.credentials, dry_run=dry_run)
        if args.page_id:
            client.page_id = args.page_id
        res = client.post_image(args.image, caption=args.caption)
        if dry_run:
            print("[DRY-RUN] Image post simulated. Use --send to actually publish to Facebook.")
        else:
            print(f"Image posted successfully: {res.get('id', res)}")
        return 0

    elif args.command == "post-text":
        dry_run = not args.send
        client = FacebookClient.from_credentials(credentials_file=args.credentials, dry_run=dry_run)
        if args.page_id:
            client.page_id = args.page_id
        res = client.post_text(args.message)
        if dry_run:
            print("[DRY-RUN] Text post simulated. Use --send to actually publish to Facebook.")
        else:
            print(f"Text posted successfully: {res.get('id', res)}")
        return 0

    elif args.command == "post-folder":
        dry_run = not args.send
        client = FacebookClient.from_credentials(credentials_file=args.credentials, dry_run=dry_run)
        if args.page_id:
            client.page_id = args.page_id
        results = client.post_folder(args.folder, caption_suffix=args.caption_suffix)
        if dry_run:
            print(f"[DRY-RUN] Simulated posting {len(results)} images. Use --send to actually publish.")
        else:
            print(f"Successfully posted {len(results)} images.")
        return 0

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv[1:]))
