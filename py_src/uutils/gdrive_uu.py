"""Google Drive utilities — list, search, download, and upload files.

Quick usage (download images from a folder):
    from uutils.gdrive_uu import GDriveClient
    client = GDriveClient.from_service_account("~/keys/gdrive_service_account.json")
    images = client.list_files(folder_id="YOUR_FOLDER_ID", mime_filter="image/")
    client.download_files(images, dest_dir="./drive_images")

Convenience function:
    from uutils.gdrive_uu import sync_drive_folder
    new_files = sync_drive_folder(
        folder_id="YOUR_FOLDER_ID",
        dest_dir="./drive_images",
        credentials_file="~/keys/gdrive_service_account.json",
        mime_filter="image/",
    )

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

Refs:
    - Google Drive API v3: https://developers.google.com/drive/api/v3/reference
    - Python quickstart: https://developers.google.com/drive/api/quickstart/python
    - Service accounts: https://cloud.google.com/iam/docs/service-accounts
"""
from __future__ import annotations

import io
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

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


# ── Credential helpers ────────────────────────────────────────────────

def _resolve_path(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()


def _build_service_account_creds(
    credentials_file: str | Path,
    scopes: list[str],
):
    """Build credentials from a service account JSON key file."""
    from google.oauth2 import service_account
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
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

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

    def __init__(self, service):
        """Initialize with a Google Drive API service object.

        Prefer using GDriveClient.from_service_account() or
        GDriveClient.from_oauth2() instead of calling this directly.
        """
        self._service = service

    @classmethod
    def from_service_account(
        cls,
        credentials_file: str | Path = DEFAULT_SERVICE_ACCOUNT_FILE,
        scopes: list[str] | None = None,
    ) -> "GDriveClient":
        """Create a client using a service account key file.

        Args:
            credentials_file: Path to the service account JSON key file.
                              Default: ~/keys/gdrive_service_account.json
            scopes: API scopes. Default: read-only.
        """
        from googleapiclient.discovery import build
        if scopes is None:
            scopes = SCOPES_READONLY
        creds = _build_service_account_creds(credentials_file, scopes)
        service = build("drive", "v3", credentials=creds)
        return cls(service)

    @classmethod
    def from_oauth2(
        cls,
        client_secrets_file: str | Path = DEFAULT_CLIENT_SECRETS_FILE,
        token_file: str | Path = DEFAULT_TOKEN_FILE,
        scopes: list[str] | None = None,
    ) -> "GDriveClient":
        """Create a client using OAuth2 (interactive consent on first run).

        Args:
            client_secrets_file: Path to OAuth2 client secrets JSON.
                                 Default: ~/keys/gdrive_client_secrets.json
            token_file: Path to store/load the OAuth2 token.
                        Default: ~/keys/gdrive_token.json
            scopes: API scopes. Default: read-only.
        """
        from googleapiclient.discovery import build
        if scopes is None:
            scopes = SCOPES_READONLY
        creds = _build_oauth2_creds(client_secrets_file, token_file, scopes)
        service = build("drive", "v3", credentials=creds)
        return cls(service)

    # ── List / Search ─────────────────────────────────────────────────

    def list_files(
        self,
        folder_id: str | None = None,
        mime_filter: str | None = None,
        query: str | None = None,
        max_results: int = 100,
        order_by: str = "modifiedTime desc",
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

        Returns:
            List of file metadata dicts with keys: id, name, mimeType,
            modifiedTime, size (when available).
        """
        if query is None:
            parts = ["trashed = false"]
            if folder_id:
                parts.append(f"'{folder_id}' in parents")
            if mime_filter:
                if "/" in mime_filter and not mime_filter.endswith("/"):
                    # Exact MIME type
                    parts.append(f"mimeType = '{mime_filter}'")
                else:
                    # Prefix match (e.g., "image/" matches image/png, image/jpeg, etc.)
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
    ) -> list[dict]:
        """List image files in Drive or a specific folder.

        Convenience wrapper around list_files() with mime_filter="image/".
        """
        return self.list_files(
            folder_id=folder_id,
            mime_filter=MIME_IMAGE,
            max_results=max_results,
        )

    def list_folders(
        self,
        parent_folder_id: str | None = None,
        max_results: int = 100,
    ) -> list[dict]:
        """List sub-folders in Drive or a specific parent folder."""
        return self.list_files(
            folder_id=parent_folder_id,
            mime_filter=MIME_FOLDER,
            max_results=max_results,
        )

    # ── Download ──────────────────────────────────────────────────────

    def download_file(
        self,
        file_id: str,
        dest_path: str | Path,
        overwrite: bool = False,
    ) -> Path:
        """Download a single file from Drive.

        Args:
            file_id: The Google Drive file ID.
            dest_path: Local path to save the file.
            overwrite: If False and dest_path exists, skip download.

        Returns:
            The Path where the file was saved.
        """
        from googleapiclient.http import MediaIoBaseDownload

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
    ) -> list[Path]:
        """Download multiple files to a local directory.

        Args:
            files: List of file metadata dicts (from list_files / list_images).
            dest_dir: Local directory to save files into.
            overwrite: If False, skip files that already exist locally.

        Returns:
            List of Paths where files were saved.
        """
        dest_dir = _resolve_path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        downloaded = []
        for f in files:
            dest_path = dest_dir / f["name"]
            path = self.download_file(f["id"], dest_path, overwrite=overwrite)
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
    ) -> dict:
        """Upload a local file to Google Drive.

        Requires SCOPES_FULL (read-write) credentials.

        Args:
            local_path: Path to the local file to upload.
            folder_id: Optional Drive folder ID to upload into.
            name: Name for the file in Drive. Defaults to local filename.
            mime_type: MIME type. Auto-detected if not specified.

        Returns:
            File metadata dict (id, name, mimeType, etc.) of the uploaded file.
        """
        from googleapiclient.http import MediaFileUpload
        import mimetypes

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
    ) -> list[dict]:
        """Upload multiple local files to Google Drive.

        Args:
            local_paths: List of local file paths to upload.
            folder_id: Optional Drive folder ID to upload into.

        Returns:
            List of file metadata dicts for uploaded files.
        """
        results = []
        for p in local_paths:
            result = self.upload_file(p, folder_id=folder_id)
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
    ) -> list[Path]:
        """Sync files from a Drive folder to a local directory.

        Only downloads files that don't already exist locally (by name).

        Args:
            folder_id: Google Drive folder ID.
            dest_dir: Local directory to sync into.
            mime_filter: Optional MIME type filter (e.g., "image/" for images only).
            max_results: Maximum number of files to consider.

        Returns:
            List of Paths of newly downloaded files (skipped files not included).
        """
        dest_dir = _resolve_path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        # List remote files
        remote_files = self.list_files(
            folder_id=folder_id,
            mime_filter=mime_filter,
            max_results=max_results,
        )

        # Figure out which are new
        existing_names = {p.name for p in dest_dir.iterdir() if p.is_file()}
        new_files = [f for f in remote_files if f["name"] not in existing_names]

        if not new_files:
            log.info("Sync: no new files in folder %s", folder_id)
            return []

        log.info("Sync: %d new files to download (out of %d remote)",
                 len(new_files), len(remote_files))
        return self.download_files(new_files, dest_dir)


# ── Convenience functions ─────────────────────────────────────────────

def get_gdrive_client(
    credentials_file: str | Path = DEFAULT_SERVICE_ACCOUNT_FILE,
    token_file: str | Path = DEFAULT_TOKEN_FILE,
    scopes: list[str] | None = None,
    use_oauth2: bool = False,
) -> GDriveClient:
    """Factory: create a GDriveClient with sensible defaults.

    Tries service account first (for automation). Falls back to OAuth2 if
    use_oauth2=True (for personal Drive access with interactive consent).

    Args:
        credentials_file: Path to service account key or OAuth2 client secrets.
        token_file: Path to store OAuth2 token (only used with use_oauth2=True).
        scopes: API scopes. Default: read-only.
        use_oauth2: If True, use OAuth2 flow instead of service account.
    """
    if use_oauth2:
        return GDriveClient.from_oauth2(credentials_file, token_file, scopes)
    return GDriveClient.from_service_account(credentials_file, scopes)


def sync_drive_folder(
    folder_id: str,
    dest_dir: str | Path,
    credentials_file: str | Path = DEFAULT_SERVICE_ACCOUNT_FILE,
    mime_filter: str | None = None,
    max_results: int = 100,
    use_oauth2: bool = False,
    token_file: str | Path = DEFAULT_TOKEN_FILE,
) -> list[Path]:
    """One-shot sync: download new files from a Drive folder to a local directory.

    Args:
        folder_id: Google Drive folder ID (from the folder URL).
        dest_dir: Local directory to sync files into.
        credentials_file: Path to credentials JSON. Default: ~/keys/gdrive_service_account.json
        mime_filter: Optional MIME filter (e.g., "image/" for images only).
        max_results: Max files to consider.
        use_oauth2: Use OAuth2 instead of service account.
        token_file: OAuth2 token cache path.

    Returns:
        List of Paths of newly downloaded files.

    Example:
        # Sync images from a shared Drive folder
        new_images = sync_drive_folder(
            folder_id="1ABCxyz...",
            dest_dir="./experiment_images",
            credentials_file="~/keys/gdrive_service_account.json",
            mime_filter="image/",
        )
        print(f"Downloaded {len(new_images)} new images")
    """
    if use_oauth2:
        client = GDriveClient.from_oauth2(credentials_file, token_file)
    else:
        client = GDriveClient.from_service_account(credentials_file)
    return client.sync_folder(folder_id, dest_dir, mime_filter, max_results)


def download_images_from_drive(
    folder_id: str,
    dest_dir: str | Path = "./drive_images",
    credentials_file: str | Path = DEFAULT_SERVICE_ACCOUNT_FILE,
    max_results: int = 100,
    use_oauth2: bool = False,
    token_file: str | Path = DEFAULT_TOKEN_FILE,
) -> list[Path]:
    """Download all images from a Google Drive folder.

    Convenience function that combines authentication, listing, and downloading.

    Args:
        folder_id: Google Drive folder ID.
        dest_dir: Local directory to save images. Default: ./drive_images
        credentials_file: Path to credentials JSON.
        max_results: Max images to download.
        use_oauth2: Use OAuth2 instead of service account.
        token_file: OAuth2 token cache path.

    Returns:
        List of Paths of downloaded image files.
    """
    return sync_drive_folder(
        folder_id=folder_id,
        dest_dir=dest_dir,
        credentials_file=credentials_file,
        mime_filter=MIME_IMAGE,
        max_results=max_results,
        use_oauth2=use_oauth2,
        token_file=token_file,
    )


# ── Tests ─────────────────────────────────────────────────────────────

def _test_imports():
    """Verify that Google API libraries are importable."""
    try:
        from googleapiclient.discovery import build
        from google.oauth2 import service_account
        print("Google API client libraries: OK")
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")


def _test_list_files(folder_id: str, credentials_file: str = DEFAULT_SERVICE_ACCOUNT_FILE):
    """Test listing files from a Drive folder (requires valid credentials)."""
    client = GDriveClient.from_service_account(credentials_file)
    files = client.list_files(folder_id=folder_id)
    print(f"Found {len(files)} files:")
    for f in files:
        size = f.get("size", "N/A")
        print(f"  {f['name']} ({f['mimeType']}, {size} bytes, id={f['id']})")
    return files


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m uutils.gdrive_uu imports          # test imports")
        print("  python -m uutils.gdrive_uu list FOLDER_ID   # list files in folder")
        print("  python -m uutils.gdrive_uu sync FOLDER_ID DEST_DIR  # sync folder")
        print()
        _test_imports()
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd == "imports":
        _test_imports()
    elif cmd == "list":
        folder_id = sys.argv[2]
        creds = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_SERVICE_ACCOUNT_FILE
        _test_list_files(folder_id, creds)
    elif cmd == "sync":
        folder_id = sys.argv[2]
        dest_dir = sys.argv[3] if len(sys.argv) > 3 else "./drive_sync"
        new_files = sync_drive_folder(folder_id=folder_id, dest_dir=dest_dir)
        print(f"Synced {len(new_files)} new files to {dest_dir}")
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
