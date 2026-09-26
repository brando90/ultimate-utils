GitHub issue #32 "Automate SMS/texting via Google Messages".

Create `py_src/uutils/sms_uu.py` with two backends, both via plain `requests`, both dry-run by default. (The Google Messages Web browser-automation option needs a paired phone and is out of scope here; mention it in the docstring as future work.)
- Twilio backend (reliable, sends from a separate Twilio number): credentials JSON `~/keys/twilio_credentials.json` with `account_sid`, `auth_token`, `from_number`, optional `self_number`. `send_sms(to, message)` = POST `https://api.twilio.com/2010-04-01/Accounts/<sid>/Messages.json` with basic auth `(sid, token)` and form fields `To`, `From`, `Body`.
- Tasker + AutoRemote backend (Android, sends from Brando's real number via the phone): key file `~/keys/tasker_autoremote_key.txt`; `send_sms(to, message)` = GET `https://autoremotejoaomgcd.appspot.com/sendmessage` with params `key` and `message` formatted as `sms=:=<to>=:=<message>` (document that a matching Tasker profile must exist on the phone).
- A `SMSClient` with `SMSClient.from_twilio(credentials_file=..., dry_run=True)`, `SMSClient.from_autoremote(key_file=..., self_number="", dry_run=True)`, methods `send_sms(to, message) -> dict` and `send_self_reminder(message) -> dict` (to `self_number`; clear error if missing). Credential files are read only when `dry_run=False`.
- Phone number validation/normalisation to E.164 (`+` and digits; strip spaces/dashes/parentheses; reject empty or too-short numbers) with a clear `ValueError`.
- Dry-run returns `{"dry_run": True, "backend": ..., "to": ..., "message": ...}` and prints a preview.
- CLI: `python -m uutils.sms_uu send --backend twilio|autoremote --to NUMBER --message M [--send]`.
- Module docstring: one-time setup for each backend (Twilio account + number, costs about $1/month plus per-message fees; Tasker + AutoRemote on Android), where credentials go (`chmod 600`), usage.
- Tests: `tests/test_sms_uu.py` (dry-run default no HTTP; Twilio request URL/auth/form with mocked `requests`; AutoRemote URL/params; normalisation cases; `send_self_reminder` without `self_number` errors).
