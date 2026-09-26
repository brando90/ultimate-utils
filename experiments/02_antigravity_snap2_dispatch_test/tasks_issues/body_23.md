GitHub issue #23 "Automate Instagram posting with captions from Google Drive images".

Create `py_src/uutils/instagram_uu.py` using the **official Instagram Graph API** content-publishing flow via plain `requests` (no instagrapi, no unofficial APIs).
- Credentials JSON at `~/keys/instagram_credentials.json` with keys `ig_user_id`, `access_token`, optional `api_version` (default `"v21.0"`); loaded only when `dry_run=False`.
- `InstagramClient(ig_user_id="", access_token="", api_version="v21.0", dry_run=True)` and `InstagramClient.from_credentials(credentials_file=..., dry_run=True)`.
- `post_image(image_url, caption="") -> dict`: step 1 POST `https://graph.facebook.com/<ver>/<ig_user_id>/media` with `image_url`, `caption`, `access_token` -> creation id; step 2 POST `.../<ig_user_id>/media_publish` with `creation_id`. Important API constraint: the Graph API needs a **publicly reachable image URL**, not a local path; if a local path is passed, raise a clear `ValueError` explaining this (in dry-run just report it in the plan).
- `post_images_batch(image_urls, captions=None) -> list[dict]` (captions default to "" and must match length).
- `refresh_long_lived_token()` helper documented (GET `.../refresh_access_token` style or `oauth/access_token` with `fb_exchange_token`), dry-run by default.
- Drive/pipeline integration point without new deps: `plan_posts_from_folder(folder, caption_suffix=".txt") -> list[dict]` that lists images (`.jpg/.jpeg/.png`) in a local folder (e.g. a folder synced from Google Drive) and pairs each with a sidecar caption file `<name>.txt` if present. This only reads local files.
- Dry-run returns dicts like `{"dry_run": True, "would_post": ..., "caption": ...}` and prints a preview so captions can be reviewed before posting.
- CLI: `python -m uutils.instagram_uu preview --folder DIR` (always offline) and `post --image-url URL --caption C [--send]`.
- Module docstring: one-time setup (Instagram Professional account linked to a Facebook Page, Meta app, long-lived token, save JSON to `~/keys/instagram_credentials.json` with `chmod 600`), the public-URL constraint, and usage.
- Tests: `tests/test_instagram_uu.py` (dry-run default no HTTP; two-step publish flow calls the right URLs in order with mocked `requests`; local-path error; folder planning with sidecar captions in `tmp_path`).
