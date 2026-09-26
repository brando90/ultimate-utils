GitHub issue #28 "Zulip automation: messaging, stream monitoring, and notifications".

Create `py_src/uutils/zulip_uu.py` with a `ZulipClient` that talks to the official Zulip REST API (`<site>/api/v1/...`, HTTP basic auth with bot email + API key) using plain `requests` (the `zulip` pip package is not required).
- `ZulipClient(site: str = "", email: str = "", api_key: str = "", dry_run: bool = True)` and `ZulipClient.from_zuliprc(zuliprc_file: str = "~/keys/zuliprc", dry_run: bool = True)` which parses the standard `[api]` section (`email`, `key`, `site`) with `configparser`, only when `dry_run=False`.
- Methods: `send_message(stream, topic, content)` (POST `/messages`, type `stream`), `send_dm(user_email, content)` (type `direct`), `get_messages(stream=None, topic=None, limit=50)` (GET `/messages` with a JSON `narrow`, `anchor="newest"`, `num_before=limit`, `num_after=0`), `get_streams()` (GET `/streams`), `upload_file(file_path) -> str` (POST `/user_uploads`, returns the URL), `get_unread_count()` (summarise unread messages per stream/topic, e.g. from GET `/messages` with narrow `is:unread`).
- All calls go through one private `_request(method, path, **kwargs)`; in dry-run it returns `{"result": "success", "dry_run": True, ...}` and prints a one-line description; otherwise it sends the request and raises a clear error when `result != "success"`.
- A convenience `notify(content, stream, topic, dry_run=True)`.
- CLI: `python -m uutils.zulip_uu send --stream S --topic T --content C [--send]` (dry-run unless `--send`).
- Module docstring: one-time setup (Zulip Settings -> Personal -> Bots -> Add bot, download its zuliprc, save to `~/keys/zuliprc`, `chmod 600`) and usage examples.
- Tests: `tests/test_zulip_uu.py` (dry-run default makes no HTTP call; zuliprc parsing from a `tmp_path` fake file; correct URL/auth/payload for send_message and get_messages with mocked `requests`; error on non-success).
