"""Google Drive utilities — list, search, download, upload files, and phone image sync pipeline.

Quick usage (download images from a folder):
    from uutils.gdrive_uu import GDriveClient
    client = GDriveClient.from_service_account("~/keys/gdrive_service_account.json", dry_run=True)
    images = client.list_files(folder_id="YOUR_FOLDER_ID", mime_filter="image/")
    client.download_files(images, dest_dir="./drive_images")

Convenience functions:
    from uutils.gdrive_uu import sync_drive_folder, sync_phone_to_local

    # Dry-run plan (default, makes no network calls and reads no credentials):
    plan = sync_phone_to_local(folder_id="YOUR_FOLDER_ID", dest_dir="./phone_photos")

    # Live sync (pass dry_run=False):
    # downloaded = sync_phone_to_local(folder_id="YOUR_FOLDER_ID", dest_dir="./phone_photos", dry_run=False)

CLI usage:
    # Dry-run plan (default, safe, makes no network calls and reads no credentials):
    python -m uutils.gdrive_uu sync-phone --folder-id YOUR_FOLDER_ID --dest ./phone_photos

    # Live execution (downloads new images from Drive, read-only on Drive):
    python -m uutils.gdrive_uu sync-phone --folder-id YOUR_FOLDER_ID --dest ./phone_photos --execute

Phone-to-Drive Setup (Phone Images -> Google Drive -> Local Pipeline):
    Note on Google Photos vs Google Drive:
        Since July 2019, Google Photos and Google Drive no longer automatically sync.
        Therefore, you must upload or back up phone photos directly to a dedicated
        Google Drive folder.

    1. iOS Setup:
       - Install the Google Drive app from the App Store.
       - Method A (Backup): In the Google Drive app, tap Menu (hamburger) -> Settings -> Backup ->
         Photos & videos -> turn on "Back up to Google Drive".
       - Method B (Dedicated Folder - Recommended):
         In Google Drive, create a dedicated folder (e.g. "PhonePhotos").
         In the iOS Photos app, select photos -> Share -> Google Drive -> choose the "PhonePhotos"
         folder. Alternatively, configure an iOS Shortcut to upload new photos to this Drive folder.

    2. Android Setup:
       - Install the Google Drive app from Google Play.
       - Create a dedicated folder in Google Drive (e.g. "PhonePhotos").
       - In Google Drive or using automated sync apps (such as FolderSync or Autosync for Google Drive),
         configure synchronization from your phone's camera directory (e.g. /DCIM/Camera) to the
         chosen Google Drive folder.

    3. Share the Drive Folder with the Service Account:
       - Open Google Drive (web or app) and locate your dedicated phone photos folder.
       - Right-click (or tap the three dots) -> Share.
       - Enter the service account email (e.g., name@project.iam.gserviceaccount.com).
       - Select the "Viewer" role (Viewer is sufficient because the pipeline is strictly read-only
         on Google Drive and never uploads or deletes files on Drive).
       - Uncheck "Notify people" and click Send / Share.

    4. Finding the Folder ID from the URL:
       - Open the folder in a web browser: https://drive.google.com/drive/folders/<FOLDER_ID>
       - The folder ID is the alphanumeric string at the end of the URL after "/folders/".
         For example, in:
           https://drive.google.com/drive/folders/1a2b3c4d5e6f7g8h9i_jklmnopqr
         The folder ID is:
           1a2b3c4d5e6f7g8h9i_jklmnopqr

    5. Periodic Sync via Cron:
       - Set up a cron job to automatically download new images periodically.
       - Edit crontab: crontab -e
       - Example cron entry to sync daily at 2:00 AM:
           0 2 * * * /path/to/venv/bin/python -m uutils.gdrive_uu sync-phone --folder-id 1a2b3c4d5e6f7g8h9i_jklmnopqr --dest ~/Pictures/phone_backup --credentials ~/keys/gdrive_service_account.json --execute >> ~/logs/gdrive_sync.log 2>&1
       - NOTE: The host machine must be powered on and awake at the scheduled time for cron to execute.
         If your computer sleeps (e.g. a laptop), consider running the cron job on an always-on server,
         or use a systemd timer / anacron with wake alarms.

Setup: Service Account (headless / automation — recommended)
    1. Go to https://console.cloud.google.com/
    2. Create a project (or select existing)
    3. Enable the Google Drive API: APIs & Services > Enable APIs > search "Google Drive API"
    4. Create a Service Account: APIs & Services > Credentials > Create Credentials > Service Account
    5. Download the JSON key file and save it:
         mv ~/Downloads/your-project-*.json ~/keys/gdrive_service_account.json
         chmod 600 ~/keys/gdrive_service_account.json
    6. Share your Drive folder with the service account email
       (the email looks like: name@project.iam.gserviceaccount.com)

Setup: OAuth2 (interactive / personal Drive access)
    1. Same steps 1-3 above
    2. Create OAuth Client ID: APIs & Services > Credentials > Create Credentials > OAuth Client ID
       - Application type: Desktop app
    3. Download the client secrets JSON:
         mv ~/Downloads/client_secret_*.json ~/keys/gdrive_client_secrets.json
         chmod 600 ~/keys/gdrive_client_secrets.json
    4. On first run, a browser window opens for consent. The token is saved to
       ~/keys/gdrive_token.json for future use (no re-auth needed).

IMPORTANT: Never commit credential files. They live in ~/keys/ which is outside the repo.

Dry-Run Safety and Types Contract:
    By default, all functions and the CLI operate in dry-run mode (dry_run=True).
    In dry-run mode, no network requests are made, no authentication occurs, and no
    credential files are read. The functions print and log a description of the planned action.

    Return types and data contract:
    - GDriveClient: A plain class (not a dict subclass). Retains attributes:
      `dry_run`, `scopes`, `credentials_file`, and `token_file`.
      Call `client.describe() -> dict` to inspect its configuration.
    - List methods: Methods that return a list in live mode (list_files, list_images,
      list_folders, download_files, upload_files, sync_folder) return a plain empty
      list [] in dry-run mode, after printing/logging the plan.
    - Single-object and action entry points: Methods that return one object in live mode
      (download_file, upload_file) and module-level action entry points (sync_drive_folder,
      download_images_from_drive, sync_phone_to_local) return a plain dict plan with
      "dry_run": True in dry-run mode.

Refs:
    - Google Drive API v3: https://developers.google.com/drive/api/v3/reference
    - Python quickstart: https://developers.google.com/drive/api/quickstart/python
    - Service accounts: https://cloud.google.com/iam/docs/service-accounts
"""
from __future__ import annotations

import argparse
import logging
import mimetypes
import os
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Default paths for credentials (all under ~/keys/, never in the repo)
DEFAULT_SERVICE_ACCOUNT_FILE = "~/keys/gdrive_service_account.json"
DEFAULT_CLIENT_SECRETS_FILE = "~/keys/gdrive_client_secrets.json"
DEFAULT_TOKEN_FILE = "~/keys/gdrive_token.json"

# Scopes
SCOPES_READONLY = ["https://www.googleapis.com/auth/drive.readonly"]
SCOPES_FULL = ["https://www.googleapis.com/auth/drive"]

# Common MIME types for filtering
MIME_IMAGE = "image/"
MIME_PDF = "application/pdf"
MIME_FOLDER = "application/vnd.google-apps.folder"
MIME_DOC = "application/vnd.google-apps.document"
MIME_SHEET = "application/vnd.google-apps.spreadsheet"

# Common image file extensions (covers iOS, Android, and camera formats)
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif",
    ".heic", ".heif", ".raw", ".cr2", ".nef", ".arw", ".dng",
}


# ── Image filtering helpers ───────────────────────────────────────────

def is_image_file(filename: str, mime_type: str | None = None) -> bool:
    """Check if a file is an image based on mimeType or file extension.

    Args:
        filename: Name of the file (e.g., "photo.jpg", "IMG_001.HEIC").
        mime_type: Optional MIME type (e.g., "image/jpeg", "image/png").

    Returns:
        True if the file is an image, False otherwise.
    """
    if mime_type and mime_type.startswith("image/"):
        return True
    ext = Path(filename).suffix.lower()
    return ext in IMAGE_EXTENSIONS


def filter_image_files(files: list[dict]) -> list[dict]:
    """Filter a list of Drive file dicts to include only image files.

    Checks both the 'mimeType' field (starting with 'image/') and file extension
    (e.g., .jpg, .png, .heic, .dng) on the 'name' field.

    Args:
        files: List of Google Drive file metadata dicts (must contain 'name', optionally 'mimeType').

    Returns:
        List of file dicts that are identified as images.
    """
    return [
        f for f in files
        if is_image_file(f.get("name", ""), f.get("mimeType"))
    ]


# ── Credential helpers ────────────────────────────────────────────────

def _resolve_path(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()


def _build_service_account_creds(
    credentials_file: str | Path,
    scopes: list[str],
):
    """Build credentials from a service account JSON key file."""
    try:
        from google.oauth2 import service_account
    except ImportError as e:
        raise ImportError(
            f"Google auth libraries required for live execution: {e}\n"
            f"Install with: pip install 'ultimate-utils[gdrive]'"
        ) from e

    creds_path = _resolve_path(credentials_file)
    if not creds_path.is_file():
        raise FileNotFoundError(
            f"Service account key not found: {creds_path}\n"
            f"Download it from Google Cloud Console and save to {creds_path}"
        )
    creds = service_account.Credentials.from_service_account_file(
        str(creds_path), scopes=scopes,
    )
    log.info("Authenticated via service account: %s", creds.service_account_email)
    return creds


def _build_oauth2_creds(
    client_secrets_file: str | Path,
    token_file: str | Path,
    scopes: list[str],
):
    """Build credentials via OAuth2 installed-app flow (interactive on first run)."""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError as e:
        raise ImportError(
            f"Google auth libraries required for live execution: {e}\n"
            f"Install with: pip install 'ultimate-utils[gdrive]'"
        ) from e

    secrets_path = _resolve_path(client_secrets_file)
    token_path = _resolve_path(token_file)

    creds = None
    if token_path.is_file():
        creds = Credentials.from_authorized_user_file(str(token_path), scopes)

    if creds and creds.valid:
        log.info("Using cached OAuth2 token from %s", token_path)
        return creds

    if creds and creds.expired and creds.refresh_token:
        log.info("Refreshing expired OAuth2 token")
        creds.refresh(Request())
    else:
        if not secrets_path.is_file():
            raise FileNotFoundError(
                f"OAuth2 client secrets not found: {secrets_path}\n"
                f"Download from Google Cloud Console and save to {secrets_path}"
            )
        flow = InstalledAppFlow.from_client_secrets_file(str(secrets_path), scopes)
        creds = flow.run_local_server(port=0)
        log.info("OAuth2 authorization completed")

    # Save token for next run
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(creds.to_json())
    os.chmod(str(token_path), 0o600)
    log.info("OAuth2 token saved to %s", token_path)
    return creds


# ── GDriveClient ──────────────────────────────────────────────────────

class GDriveClient:
    """Google Drive API v3 client for listing, downloading, and uploading files.

    Credentials are loaded from files in ~/keys/ (never from the repo).
    Use the class methods `from_service_account()` or `from_oauth2()` to create.
    """

    def __init__(
        self,
        service: Any = None,
        credentials_file: str | Path = DEFAULT_SERVICE_ACCOUNT_FILE,
        dry_run: bool = True,
        scopes: list[str] | None = None,
        token_file: str | Path = DEFAULT_TOKEN_FILE,
    ):
        """Initialize with a Google Drive API service object or in dry-run mode."""
        self._service = service
        self.credentials_file = str(credentials_file)
        self.token_file = str(token_file)
        self.scopes = scopes or SCOPES_READONLY
        self.dry_run = dry_run
        self._dry_run = dry_run

    def describe(self) -> dict:
        """Return a dict describing the client configuration."""
        return {
            "credentials_file": self.credentials_file,
            "token_file": self.token_file,
            "scopes": self.scopes,
            "dry_run": self.dry_run,
        }

    def __repr__(self) -> str:
        return (
            f"GDriveClient(credentials_file={self.credentials_file!r}, "
            f"dry_run={self.dry_run!r}, scopes={self.scopes!r})"
        )

    @classmethod
    def from_service_account(
        cls,
        credentials_file: str | Path = DEFAULT_SERVICE_ACCOUNT_FILE,
        scopes: list[str] | None = None,
        dry_run: bool = True,
    ) -> "GDriveClient":
        """Create a client using a service account key file.

        Args:
            credentials_file: Path to the service account JSON key file.
                              Default: ~/keys/gdrive_service_account.json
            scopes: API scopes. Default: read-only.
            dry_run: If True (default), create client in dry-run mode without auth/network.
        """
        if scopes is None:
            scopes = SCOPES_READONLY

        if dry_run:
            print(f"[DRY-RUN] GDriveClient.from_service_account: credentials_file={credentials_file}, scopes={scopes}")
            log.info("[DRY-RUN] GDriveClient.from_service_account: credentials_file=%s, scopes=%s", credentials_file, scopes)
            return cls(service=None, credentials_file=credentials_file, dry_run=True, scopes=scopes)

        try:
            from googleapiclient.discovery import build
        except ImportError as e:
            raise ImportError(
                f"googleapiclient required for live execution: {e}\n"
                f"Install with: pip install 'ultimate-utils[gdrive]'"
            ) from e

        creds = _build_service_account_creds(credentials_file, scopes)
        service = build("drive", "v3", credentials=creds)
        return cls(service=service, credentials_file=credentials_file, dry_run=False, scopes=scopes)

    @classmethod
    def from_oauth2(
        cls,
        client_secrets_file: str | Path = DEFAULT_CLIENT_SECRETS_FILE,
        token_file: str | Path = DEFAULT_TOKEN_FILE,
        scopes: list[str] | None = None,
        dry_run: bool = True,
    ) -> "GDriveClient":
        """Create a client using OAuth2 (interactive consent on first run).

        Args:
            client_secrets_file: Path to OAuth2 client secrets JSON.
                                 Default: ~/keys/gdrive_client_secrets.json
            token_file: Path to store/load the OAuth2 token.
                        Default: ~/keys/gdrive_token.json
            scopes: API scopes. Default: read-only.
            dry_run: If True (default), create client in dry-run mode without auth/network.
        """
        if scopes is None:
            scopes = SCOPES_READONLY

        if dry_run:
            print(f"[DRY-RUN] GDriveClient.from_oauth2: client_secrets={client_secrets_file}, token={token_file}")
            log.info("[DRY-RUN] GDriveClient.from_oauth2: client_secrets=%s, token=%s", client_secrets_file, token_file)
            return cls(
                service=None,
                credentials_file=client_secrets_file,
                token_file=token_file,
                dry_run=True,
                scopes=scopes,
            )

        try:
            from googleapiclient.discovery import build
        except ImportError as e:
            raise ImportError(
                f"googleapiclient required for live execution: {e}\n"
                f"Install with: pip install 'ultimate-utils[gdrive]'"
            ) from e

        creds = _build_oauth2_creds(client_secrets_file, token_file, scopes)
        service = build("drive", "v3", credentials=creds)
        return cls(
            service=service,
            credentials_file=client_secrets_file,
            token_file=token_file,
            dry_run=False,
            scopes=scopes,
        )

    # ── List / Search ─────────────────────────────────────────────────

    def list_files(
        self,
        folder_id: str | None = None,
        mime_filter: str | None = None,
        query: str | None = None,
        max_results: int = 100,
        order_by: str = "modifiedTime desc",
        dry_run: bool = True,
    ) -> list[dict]:
        """List files in Drive, optionally filtered by folder, MIME type, or query.

        Args:
            folder_id: If set, only list files in this folder.
            mime_filter: MIME type prefix filter (e.g., "image/" for all images,
                         "image/png" for PNGs only).
            query: Raw Drive API query string (overrides folder_id and mime_filter).
                   See: https://developers.google.com/drive/api/v3/search-files
            max_results: Maximum number of files to return.
            order_by: Sort order. Default: newest first.
            dry_run: If True (default), return plan without network calls.

        Returns:
            Empty list if dry_run=True, or list of file metadata dicts if dry_run=False.
        """
        creds_file = getattr(self, "credentials_file", DEFAULT_SERVICE_ACCOUNT_FILE)
        if dry_run or self._dry_run:
            plan = {
                "action": "list_files",
                "folder_id": str(folder_id) if folder_id else None,
                "mime_filter": mime_filter,
                "query": query,
                "max_results": max_results,
                "order_by": order_by,
                "credentials_file": str(creds_file),
                "dry_run": True,
            }
            print(f"[DRY-RUN] list_files: folder_id={folder_id}, mime_filter={mime_filter}, credentials={creds_file}")
            log.info("[DRY-RUN] list_files: %s", plan)
            return []

        if query is None:
            parts = ["trashed = false"]
            if folder_id:
                parts.append(f"'{folder_id}' in parents")
            if mime_filter:
                if "/" in mime_filter and not mime_filter.endswith("/"):
                    parts.append(f"mimeType = '{mime_filter}'")
                else:
                    parts.append(f"mimeType contains '{mime_filter.rstrip('/')}'")
            query = " and ".join(parts)

        all_files = []
        page_token = None
        fields = "nextPageToken, files(id, name, mimeType, modifiedTime, size)"

        while len(all_files) < max_results:
            page_size = min(100, max_results - len(all_files))
            resp = self._service.files().list(
                q=query,
                pageSize=page_size,
                fields=fields,
                orderBy=order_by,
                pageToken=page_token,
            ).execute()

            files = resp.get("files", [])
            all_files.extend(files)

            page_token = resp.get("nextPageToken")
            if not page_token:
                break

        log.info("Listed %d files (query: %s)", len(all_files), query)
        return all_files

    def list_images(
        self,
        folder_id: str | None = None,
        max_results: int = 100,
        dry_run: bool = True,
    ) -> list[dict]:
        """List image files in Drive or a specific folder.

        Convenience wrapper around list_files() with mime_filter="image/".
        """
        return self.list_files(
            folder_id=folder_id,
            mime_filter=MIME_IMAGE,
            max_results=max_results,
            dry_run=dry_run,
        )

    def list_folders(
        self,
        parent_folder_id: str | None = None,
        max_results: int = 100,
        dry_run: bool = True,
    ) -> list[dict]:
        """List sub-folders in Drive or a specific parent folder."""
        return self.list_files(
            folder_id=parent_folder_id,
            mime_filter=MIME_FOLDER,
            max_results=max_results,
            dry_run=dry_run,
        )

    # ── Download ──────────────────────────────────────────────────────

    def download_file(
        self,
        file_id: str,
        dest_path: str | Path,
        overwrite: bool = False,
        dry_run: bool = True,
    ) -> dict | Path:
        """Download a single file from Drive.

        Args:
            file_id: The Google Drive file ID.
            dest_path: Local path to save the file.
            overwrite: If False and dest_path exists, skip download.
            dry_run: If True (default), return plan without network calls.

        Returns:
            Plan dict with 'dry_run': True if dry_run=True, or Path where file was saved.
        """
        creds_file = getattr(self, "credentials_file", DEFAULT_SERVICE_ACCOUNT_FILE)
        if dry_run or self._dry_run:
            plan = {
                "action": "download_file",
                "file_id": str(file_id),
                "dest": str(dest_path),
                "dest_path": str(dest_path),
                "credentials_file": str(creds_file),
                "overwrite": overwrite,
                "dry_run": True,
            }
            print(f"[DRY-RUN] download_file: file_id={file_id}, dest={dest_path}, credentials={creds_file}")
            log.info("[DRY-RUN] download_file: %s", plan)
            return plan

        try:
            from googleapiclient.http import MediaIoBaseDownload
        except ImportError as e:
            raise ImportError(
                f"googleapiclient required for live execution: {e}\n"
                f"Install with: pip install 'ultimate-utils[gdrive]'"
            ) from e

        dest = _resolve_path(dest_path)
        if dest.is_file() and not overwrite:
            log.info("Skipping (already exists): %s", dest)
            return dest

        dest.parent.mkdir(parents=True, exist_ok=True)

        request = self._service.files().get_media(fileId=file_id)
        with open(dest, "wb") as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    log.debug("Download %s: %d%%", dest.name, int(status.progress() * 100))

        log.info("Downloaded: %s (%s)", dest.name, file_id)
        return dest

    def download_files(
        self,
        files: list[dict],
        dest_dir: str | Path,
        overwrite: bool = False,
        dry_run: bool = True,
    ) -> list[Path]:
        """Download multiple files to a local directory.

        Args:
            files: List of file metadata dicts (from list_files / list_images).
            dest_dir: Local directory to save files into.
            overwrite: If False, skip files that already exist locally.
            dry_run: If True (default), return plan without network calls.

        Returns:
            Empty list if dry_run=True, or list of Paths where files were saved.
        """
        creds_file = getattr(self, "credentials_file", DEFAULT_SERVICE_ACCOUNT_FILE)
        if dry_run or self._dry_run:
            plan = {
                "action": "download_files",
                "count": len(files),
                "files": files,
                "dest": str(dest_dir),
                "dest_dir": str(dest_dir),
                "credentials_file": str(creds_file),
                "overwrite": overwrite,
                "dry_run": True,
            }
            print(f"[DRY-RUN] download_files: {len(files)} files to {dest_dir}, credentials={creds_file}")
            log.info("[DRY-RUN] download_files: %s", plan)
            return []

        dest_dir = _resolve_path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        downloaded = []
        for f in files:
            dest_path = dest_dir / f["name"]
            path = self.download_file(f["id"], dest_path, overwrite=overwrite, dry_run=False)
            downloaded.append(path)
        log.info("Downloaded %d/%d files to %s", len(downloaded), len(files), dest_dir)
        return downloaded

    # ── Upload ────────────────────────────────────────────────────────

    def upload_file(
        self,
        local_path: str | Path,
        folder_id: str | None = None,
        name: str | None = None,
        mime_type: str | None = None,
        dry_run: bool = True,
    ) -> dict:
        """Upload a local file to Google Drive.

        Requires SCOPES_FULL (read-write) credentials.

        Args:
            local_path: Path to the local file to upload.
            folder_id: Optional Drive folder ID to upload into.
            name: Name for the file in Drive. Defaults to local filename.
            mime_type: MIME type. Auto-detected if not specified.
            dry_run: If True (default), return plan without network calls.

        Returns:
            Plan dict with 'dry_run': True if dry_run=True, or file metadata dict if dry_run=False.
        """
        creds_file = getattr(self, "credentials_file", DEFAULT_SERVICE_ACCOUNT_FILE)
        if dry_run or self._dry_run:
            plan = {
                "action": "upload_file",
                "local_path": str(local_path),
                "folder_id": str(folder_id) if folder_id else None,
                "name": name or Path(local_path).name,
                "mime_type": mime_type,
                "credentials_file": str(creds_file),
                "dry_run": True,
            }
            print(f"[DRY-RUN] upload_file: local_path={local_path}, folder_id={folder_id}, credentials={creds_file}")
            log.info("[DRY-RUN] upload_file: %s", plan)
            return plan

        try:
            from googleapiclient.http import MediaFileUpload
        except ImportError as e:
            raise ImportError(
                f"googleapiclient required for live execution: {e}\n"
                f"Install with: pip install 'ultimate-utils[gdrive]'"
            ) from e

        src = _resolve_path(local_path)
        if not src.is_file():
            raise FileNotFoundError(f"File not found: {src}")

        if name is None:
            name = src.name
        if mime_type is None:
            mime_type, _ = mimetypes.guess_type(str(src))
            mime_type = mime_type or "application/octet-stream"

        file_metadata: dict = {"name": name}
        if folder_id:
            file_metadata["parents"] = [folder_id]

        media = MediaFileUpload(str(src), mimetype=mime_type, resumable=True)
        result = self._service.files().create(
            body=file_metadata,
            media_body=media,
            fields="id, name, mimeType, modifiedTime, size",
        ).execute()

        log.info("Uploaded: %s -> %s (id: %s)", src.name, name, result["id"])
        return result

    def upload_files(
        self,
        local_paths: list[str | Path],
        folder_id: str | None = None,
        dry_run: bool = True,
    ) -> list[dict]:
        """Upload multiple local files to Google Drive.

        Args:
            local_paths: List of local file paths to upload.
            folder_id: Optional Drive folder ID to upload into.
            dry_run: If True (default), return plan without network calls.

        Returns:
            Empty list if dry_run=True, or list of file metadata dicts if dry_run=False.
        """
        creds_file = getattr(self, "credentials_file", DEFAULT_SERVICE_ACCOUNT_FILE)
        if dry_run or self._dry_run:
            plan = {
                "action": "upload_files",
                "count": len(local_paths),
                "local_paths": [str(p) for p in local_paths],
                "folder_id": str(folder_id) if folder_id else None,
                "credentials_file": str(creds_file),
                "dry_run": True,
            }
            print(f"[DRY-RUN] upload_files: {len(local_paths)} files, folder_id={folder_id}, credentials={creds_file}")
            log.info("[DRY-RUN] upload_files: %s", plan)
            return []

        results = []
        for p in local_paths:
            result = self.upload_file(p, folder_id=folder_id, dry_run=False)
            results.append(result)
        log.info("Uploaded %d files", len(results))
        return results

    # ── Sync ──────────────────────────────────────────────────────────

    def sync_folder(
        self,
        folder_id: str,
        dest_dir: str | Path,
        mime_filter: str | None = None,
        max_results: int = 100,
        dry_run: bool = True,
    ) -> list[Path]:
        """Sync files from a Drive folder to a local directory.

        Only downloads files that don't already exist locally (by name).
        Read-only on Google Drive: never uploads, modifies, or deletes remote files.

        Args:
            folder_id: Google Drive folder ID.
            dest_dir: Local directory to sync into.
            mime_filter: Optional MIME type filter (e.g., "image/" for images only).
            max_results: Maximum number of files to consider.
            dry_run: If True (default), return plan without network calls.

        Returns:
            Empty list if dry_run=True, or list of Paths of newly downloaded files if dry_run=False.
        """
        creds_file = getattr(self, "credentials_file", DEFAULT_SERVICE_ACCOUNT_FILE)
        if dry_run or self._dry_run:
            plan = {
                "action": "sync_folder",
                "folder_id": str(folder_id),
                "dest": str(dest_dir),
                "dest_dir": str(dest_dir),
                "credentials_file": str(creds_file),
                "mime_filter": mime_filter,
                "max_results": max_results,
                "dry_run": True,
            }
            print(f"[DRY-RUN] sync_folder: folder_id={folder_id}, dest={dest_dir}, credentials={creds_file}")
            log.info("[DRY-RUN] sync_folder: %s", plan)
            return []

        dest_dir = _resolve_path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        remote_files = self.list_files(
            folder_id=folder_id,
            mime_filter=mime_filter,
            max_results=max_results,
            dry_run=False,
        )
        if mime_filter == MIME_IMAGE:
            remote_files = filter_image_files(remote_files)

        existing_names = {p.name for p in dest_dir.iterdir() if p.is_file()}
        new_files = [f for f in remote_files if f.get("name") not in existing_names]

        if not new_files:
            log.info("Sync: no new files in folder %s", folder_id)
            return []

        log.info("Sync: %d new files to download (out of %d remote)", len(new_files), len(remote_files))
        return self.download_files(new_files, dest_dir, dry_run=False)


# ── Convenience functions ─────────────────────────────────────────────

def get_gdrive_client(
    credentials_file: str | Path = DEFAULT_SERVICE_ACCOUNT_FILE,
    token_file: str | Path = DEFAULT_TOKEN_FILE,
    scopes: list[str] | None = None,
    use_oauth2: bool = False,
    dry_run: bool = True,
) -> GDriveClient:
    """Factory: create a GDriveClient with sensible defaults.

    Tries service account first (for automation). Falls back to OAuth2 if
    use_oauth2=True (for personal Drive access with interactive consent).

    Args:
        credentials_file: Path to service account key or OAuth2 client secrets.
        token_file: Path to store OAuth2 token (only used with use_oauth2=True).
        scopes: API scopes. Default: read-only.
        use_oauth2: If True, use OAuth2 flow instead of service account.
        dry_run: If True (default), create client in dry-run mode without auth/network.

    Returns:
        GDriveClient instance (in dry-run mode if dry_run=True).
    """
    if dry_run:
        print(f"[DRY-RUN] get_gdrive_client: credentials_file={credentials_file}, use_oauth2={use_oauth2}")
        log.info("[DRY-RUN] get_gdrive_client: credentials_file=%s, use_oauth2=%s", credentials_file, use_oauth2)
        if use_oauth2:
            return GDriveClient.from_oauth2(
                client_secrets_file=credentials_file,
                token_file=token_file,
                scopes=scopes,
                dry_run=True,
            )
        return GDriveClient.from_service_account(
            credentials_file=credentials_file,
            scopes=scopes,
            dry_run=True,
        )

    if use_oauth2:
        return GDriveClient.from_oauth2(
            client_secrets_file=credentials_file,
            token_file=token_file,
            scopes=scopes,
            dry_run=False,
        )
    return GDriveClient.from_service_account(
        credentials_file=credentials_file,
        scopes=scopes,
        dry_run=False,
    )


def sync_drive_folder(
    folder_id: str,
    dest_dir: str | Path,
    credentials_file: str | Path = DEFAULT_SERVICE_ACCOUNT_FILE,
    mime_filter: str | None = None,
    max_results: int = 100,
    use_oauth2: bool = False,
    token_file: str | Path = DEFAULT_TOKEN_FILE,
    dry_run: bool = True,
) -> dict | list[Path]:
    """One-shot sync: download new files from a Drive folder to a local directory.

    Args:
        folder_id: Google Drive folder ID (from the folder URL).
        dest_dir: Local directory to sync files into.
        credentials_file: Path to credentials JSON. Default: ~/keys/gdrive_service_account.json
        mime_filter: Optional MIME filter (e.g., "image/" for images only).
        max_results: Max files to consider.
        use_oauth2: Use OAuth2 instead of service account.
        token_file: OAuth2 token cache path.
        dry_run: If True (default), return plan without network calls.

    Returns:
        Plan dict with 'dry_run': True if dry_run=True, or list of Paths of newly downloaded files if dry_run=False.
    """
    if dry_run:
        plan = {
            "action": "sync_drive_folder",
            "folder_id": str(folder_id),
            "dest": str(dest_dir),
            "dest_dir": str(dest_dir),
            "credentials_file": str(credentials_file),
            "mime_filter": mime_filter,
            "max_results": max_results,
            "use_oauth2": use_oauth2,
            "dry_run": True,
        }
        print(f"[DRY-RUN] sync_drive_folder: folder_id={folder_id}, dest={dest_dir}, credentials={credentials_file}")
        log.info("[DRY-RUN] sync_drive_folder: %s", plan)
        return plan

    client = get_gdrive_client(
        credentials_file=credentials_file,
        token_file=token_file,
        use_oauth2=use_oauth2,
        dry_run=False,
    )
    return client.sync_folder(
        folder_id=folder_id,
        dest_dir=dest_dir,
        mime_filter=mime_filter,
        max_results=max_results,
        dry_run=False,
    )


def download_images_from_drive(
    folder_id: str,
    dest_dir: str | Path = "./drive_images",
    credentials_file: str | Path = DEFAULT_SERVICE_ACCOUNT_FILE,
    max_results: int = 100,
    use_oauth2: bool = False,
    token_file: str | Path = DEFAULT_TOKEN_FILE,
    dry_run: bool = True,
) -> dict | list[Path]:
    """Download all images from a Google Drive folder.

    Convenience function that combines authentication, listing, and downloading.

    Args:
        folder_id: Google Drive folder ID.
        dest_dir: Local directory to save images. Default: ./drive_images
        credentials_file: Path to credentials JSON.
        max_results: Max images to download.
        use_oauth2: Use OAuth2 instead of service account.
        token_file: OAuth2 token cache path.
        dry_run: If True (default), return plan without network calls.

    Returns:
        Plan dict with 'dry_run': True if dry_run=True, or list of Paths of downloaded image files if dry_run=False.
    """
    if dry_run:
        plan = {
            "action": "download_images_from_drive",
            "folder_id": str(folder_id),
            "dest": str(dest_dir),
            "dest_dir": str(dest_dir),
            "credentials_file": str(credentials_file),
            "mime_filter": MIME_IMAGE,
            "max_results": max_results,
            "use_oauth2": use_oauth2,
            "dry_run": True,
        }
        print(f"[DRY-RUN] download_images_from_drive: folder_id={folder_id}, dest={dest_dir}, credentials={credentials_file}")
        log.info("[DRY-RUN] download_images_from_drive: %s", plan)
        return plan

    return sync_drive_folder(
        folder_id=folder_id,
        dest_dir=dest_dir,
        credentials_file=credentials_file,
        mime_filter=MIME_IMAGE,
        max_results=max_results,
        use_oauth2=use_oauth2,
        token_file=token_file,
        dry_run=False,
    )


def sync_phone_to_local(
    folder_id: str,
    dest_dir: str | Path,
    credentials_file: str | Path = DEFAULT_SERVICE_ACCOUNT_FILE,
    max_results: int = 1000,
    use_oauth2: bool = False,
    token_file: str | Path = DEFAULT_TOKEN_FILE,
    dry_run: bool = True,
) -> dict | list[Path]:
    """Download new images from a phone-synced Google Drive folder to a local directory.

    Read-only on Google Drive: never uploads, modifies, or deletes files on Drive.

    Args:
        folder_id: Google Drive folder ID containing phone images.
        dest_dir: Local destination directory to save images.
        credentials_file: Path to service account JSON (or OAuth2 secrets).
        max_results: Maximum files to consider.
        use_oauth2: Use OAuth2 credentials instead of service account.
        token_file: Path to OAuth2 token cache.
        dry_run: If True (default), print and return the sync plan without network calls.

    Returns:
        Plan dict with 'dry_run': True if dry_run=True, or list of Paths of newly downloaded files if dry_run=False.
    """
    if dry_run:
        plan = {
            "action": "sync-phone",
            "folder_id": str(folder_id),
            "dest": str(dest_dir),
            "dest_dir": str(dest_dir),
            "credentials_file": str(credentials_file),
            "filter": "image/",
            "read_only_remote": True,
            "dry_run": True,
        }
        print(f"[DRY-RUN] sync-phone: folder_id={folder_id}, dest={dest_dir}, credentials={credentials_file}")
        log.info("[DRY-RUN] sync-phone: %s", plan)
        return plan

    client = get_gdrive_client(
        credentials_file=credentials_file,
        token_file=token_file,
        use_oauth2=use_oauth2,
        dry_run=False,
    )
    return client.sync_folder(
        folder_id=folder_id,
        dest_dir=dest_dir,
        mime_filter=MIME_IMAGE,
        max_results=max_results,
        dry_run=False,
    )


# ── CLI & Testing ─────────────────────────────────────────────────────

def _test_imports():
    """Verify that Google API libraries are importable."""
    try:
        from googleapiclient.discovery import build
        from google.oauth2 import service_account
        print("Google API client libraries: OK")
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install 'ultimate-utils[gdrive]'")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m uutils.gdrive_uu",
        description="Google Drive utilities — list, sync, and download phone images.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # sync-phone subcommand (Requirement 4)
    phone_parser = subparsers.add_parser(
        "sync-phone",
        help="Sync phone images from a Google Drive folder to a local directory (read-only on Drive)",
    )
    phone_parser.add_argument("--folder-id", "-f", required=True, help="Google Drive folder ID")
    phone_parser.add_argument("--dest", "-d", required=True, help="Local destination directory")
    phone_parser.add_argument(
        "--credentials", "-c",
        default=DEFAULT_SERVICE_ACCOUNT_FILE,
        help="Path to credentials JSON (default: ~/keys/gdrive_service_account.json)",
    )
    phone_parser.add_argument(
        "--execute", "--send",
        action="store_true",
        default=False,
        dest="execute",
        help="Execute actual download (default is dry-run)",
    )
    phone_parser.add_argument(
        "--max-results",
        type=int,
        default=1000,
        help="Maximum files to consider (default: 1000)",
    )

    # sync subcommand (generic folder sync)
    sync_parser = subparsers.add_parser("sync", help="Sync files from a Drive folder")
    sync_parser.add_argument("folder_id", nargs="?", default=None, help="Google Drive folder ID")
    sync_parser.add_argument("dest_dir", nargs="?", default=None, help="Local destination directory")
    sync_parser.add_argument("--folder-id", "-f", dest="flag_folder_id", default=None, help="Google Drive folder ID")
    sync_parser.add_argument("--dest", "-d", dest="flag_dest", default=None, help="Local destination directory")
    sync_parser.add_argument("--credentials", "-c", default=DEFAULT_SERVICE_ACCOUNT_FILE, help="Path to credentials JSON")
    sync_parser.add_argument("--execute", "--send", action="store_true", default=False, dest="execute", help="Execute sync")

    # list subcommand
    list_parser = subparsers.add_parser("list", help="List files in a Drive folder")
    list_parser.add_argument("folder_id", nargs="?", default=None, help="Google Drive folder ID")
    list_parser.add_argument("--folder-id", "-f", dest="flag_folder_id", default=None, help="Google Drive folder ID")
    list_parser.add_argument("--credentials", "-c", default=DEFAULT_SERVICE_ACCOUNT_FILE, help="Path to credentials JSON")
    list_parser.add_argument("--execute", "--send", action="store_true", default=False, dest="execute", help="Execute listing")

    # imports subcommand
    subparsers.add_parser("imports", help="Verify Google API client dependencies")

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for uutils.gdrive_uu."""
    if argv is None:
        argv = sys.argv[1:]

    parser = build_parser()
    if not argv:
        parser.print_help()
        return 0

    args = parser.parse_args(argv)

    if args.command == "sync-phone":
        if not args.execute:
            print("[DRY-RUN] sync-phone plan:")
            print(f"  Drive Folder ID  : {args.folder_id}")
            print(f"  Local Destination: {args.dest}")
            print(f"  Credentials File : {args.credentials}")
            print("  Filter           : Images only")
            print("  Action           : Download new images from Drive to local directory (read-only on Drive)")
            print("  Mode             : DRY-RUN (pass --execute to perform actual download)")
            sync_phone_to_local(
                folder_id=args.folder_id,
                dest_dir=args.dest,
                credentials_file=args.credentials,
                max_results=args.max_results,
                dry_run=True,
            )
            return 0
        else:
            downloaded = sync_phone_to_local(
                folder_id=args.folder_id,
                dest_dir=args.dest,
                credentials_file=args.credentials,
                max_results=args.max_results,
                dry_run=False,
            )
            print(f"Synced {len(downloaded)} new images to {args.dest}")
            return 0

    elif args.command == "sync":
        folder_id = args.flag_folder_id or args.folder_id
        dest_dir = args.flag_dest or args.dest_dir or "./drive_sync"
        if not folder_id:
            print("Error: folder_id required for sync")
            return 1
        if not args.execute:
            print(f"[DRY-RUN] sync plan: folder_id={folder_id}, dest={dest_dir}, credentials={args.credentials}")
            sync_drive_folder(
                folder_id=folder_id,
                dest_dir=dest_dir,
                credentials_file=args.credentials,
                dry_run=True,
            )
            return 0
        new_files = sync_drive_folder(
            folder_id=folder_id,
            dest_dir=dest_dir,
            credentials_file=args.credentials,
            dry_run=False,
        )
        print(f"Synced {len(new_files)} new files to {dest_dir}")
        return 0

    elif args.command == "list":
        folder_id = args.flag_folder_id or args.folder_id
        if not folder_id:
            print("Error: folder_id required for list")
            return 1
        client = get_gdrive_client(credentials_file=args.credentials, dry_run=not args.execute)
        if not args.execute:
            print(f"[DRY-RUN] list plan: folder_id={folder_id}, credentials={args.credentials}")
            client.list_files(folder_id=folder_id, dry_run=True)
            return 0
        files = client.list_files(folder_id=folder_id, dry_run=False)
        print(f"Found {len(files)} files:")
        for f in files:
            size = f.get("size", "N/A")
            print(f"  {f.get('name')} ({f.get('mimeType')}, {size} bytes, id={f.get('id')})")
        return 0

    elif args.command == "imports":
        _test_imports()
        return 0

    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
