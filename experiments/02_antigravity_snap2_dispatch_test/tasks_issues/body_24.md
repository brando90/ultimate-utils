GitHub issue #24 "Automate Facebook posting with captions from Google Drive images".

Create `py_src/uutils/facebook_uu.py` using the **official Facebook Graph API** via plain `requests` (do not add `facebook-sdk`; it is unmaintained).
- Important limitation to document in the module docstring: since 2018 the Graph API cannot publish to a **personal profile timeline**; programmatic posting works for **Pages** you admin (Page access token). The module therefore targets a Page id.
- Credentials JSON at `~/keys/facebook_credentials.json` with keys `page_id`, `page_access_token`, optional `api_version` (default `"v21.0"`); loaded only when `dry_run=False`.
- `FacebookClient(page_id="", page_access_token="", api_version="v21.0", dry_run=True)` and `FacebookClient.from_credentials(credentials_file=..., dry_run=True)`.
- `post_text(message) -> dict` (POST `/<page_id>/feed`), `post_image(image_path_or_url, caption="") -> dict` (POST `/<page_id>/photos`; local file -> multipart `source`, URL -> `url` param), `post_images_batch(images, captions=None) -> list[dict]`.
- Local-folder pipeline point (e.g. a folder synced from Google Drive), no new deps: `plan_posts_from_folder(folder, caption_suffix=".txt") -> list[dict]` pairs `.jpg/.jpeg/.png` images with sidecar caption `<name>.txt` files. Reads local files only.
- Dry-run returns dicts like `{"dry_run": True, "would_post": ..., "caption": ...}` and prints a preview.
- CLI: `python -m uutils.facebook_uu preview --folder DIR` (offline) and `post --image PATH_OR_URL --caption C [--send]`, `post-text --message M [--send]`.
- Module docstring: one-time setup (Meta developer app, Page, `pages_manage_posts` + `pages_read_engagement` permissions, long-lived Page token, save JSON with `chmod 600`) and usage.
- Tests: `tests/test_facebook_uu.py` (dry-run default no HTTP; correct endpoint/params for text, URL image and local-file image with mocked `requests`; folder planning in `tmp_path`).
