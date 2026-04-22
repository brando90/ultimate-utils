"""Twitter / X posting — post tweets (with optional media) from uutils.

Designed so downstream repos (e.g. Brando's personal-website repo) can
``pip install ultimate-utils`` and drop a line at the end of a blog-post
publish hook to auto-tweet:

    from uutils.twitter_uu import post_tweet
    post_tweet("New post: Embracing the AI Agent Era — https://brando90.github.io/post/...")

Auth (OAuth 1.0a user context, required to POST tweets)
-------------------------------------------------------
1. Apply for a developer account at https://developer.x.com/
2. Create a Project + App, enable "Read and Write" permissions.
3. Generate four credentials in the Keys and Tokens tab:
     - API Key            (a.k.a. consumer key)
     - API Key Secret     (a.k.a. consumer secret)
     - Access Token
     - Access Token Secret
4. Save them to a JSON file (chmod 600) — default path is
   ``~/keys/twitter_api_config.json``:

       {
           "api_key":             "...",
           "api_secret":          "...",
           "access_token":        "...",
           "access_token_secret": "..."
       }

   OR export the matching env vars (``TWITTER_API_KEY``,
   ``TWITTER_API_SECRET``, ``TWITTER_ACCESS_TOKEN``,
   ``TWITTER_ACCESS_TOKEN_SECRET``).

Dependencies
------------
Uses ``tweepy`` under the hood (the de-facto Python X client). It is an
optional runtime dep — install with ``pip install tweepy`` when you first
call a posting function. This keeps ``pip install ultimate-utils`` light
for users who don't tweet.

Refs
----
- X API v2 POST /2/tweets: https://docs.x.com/x-api/posts/creation-of-a-post
- Media upload v1.1:       https://developer.x.com/en/docs/x-api/v1/media/upload-media/overview
- tweepy:                  https://docs.tweepy.org/en/stable/
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_CONFIG_FILE = "~/keys/twitter_api_config.json"
MAX_TWEET_LEN = 280  # plain-text tweet limit for standard accounts
_REQUIRED_KEYS = ("api_key", "api_secret", "access_token", "access_token_secret")
_ENV_VAR_MAP = {
    "api_key":             "TWITTER_API_KEY",
    "api_secret":          "TWITTER_API_SECRET",
    "access_token":        "TWITTER_ACCESS_TOKEN",
    "access_token_secret": "TWITTER_ACCESS_TOKEN_SECRET",
}


def _load_config(config_file: str = "") -> dict[str, str]:
    """Load X credentials from env vars first, then a JSON file.

    Env vars always win if all four are set, so CI/cron jobs don't need a
    config file on disk.
    """
    from_env = {k: os.environ.get(v, "") for k, v in _ENV_VAR_MAP.items()}
    if all(from_env.values()):
        return from_env

    fpath = Path(config_file or DEFAULT_CONFIG_FILE).expanduser()
    if not fpath.is_file():
        missing_env = [v for v in _ENV_VAR_MAP.values() if not os.environ.get(v)]
        raise FileNotFoundError(
            f"Twitter config not found at {fpath} and env vars missing: "
            f"{', '.join(missing_env)}. See module docstring for setup."
        )
    config = json.loads(fpath.read_text())
    missing = [k for k in _REQUIRED_KEYS if not config.get(k)]
    if missing:
        raise ValueError(
            f"Twitter config {fpath} missing keys: {', '.join(missing)}. "
            f"See module docstring for required schema."
        )
    return {k: str(config[k]) for k in _REQUIRED_KEYS}


def _import_tweepy():
    try:
        import tweepy  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "tweepy is required for Twitter posting. Install with: pip install tweepy"
        ) from exc
    return tweepy


def _build_clients(config: dict[str, str]):
    """Build a v2 Client (for POST /2/tweets) and a v1.1 API (for media uploads)."""
    tweepy = _import_tweepy()
    client_v2 = tweepy.Client(
        consumer_key=config["api_key"],
        consumer_secret=config["api_secret"],
        access_token=config["access_token"],
        access_token_secret=config["access_token_secret"],
    )
    auth_v1 = tweepy.OAuth1UserHandler(
        config["api_key"], config["api_secret"],
        config["access_token"], config["access_token_secret"],
    )
    api_v1 = tweepy.API(auth_v1)
    return client_v2, api_v1


def _check_tweet_length(text: str) -> None:
    if len(text) > MAX_TWEET_LEN:
        raise ValueError(
            f"Tweet is {len(text)} chars; max is {MAX_TWEET_LEN}. "
            f"Shorten the text or use thread_tweets() for a thread."
        )


def post_tweet(
    text: str,
    config_file: str = "",
    dry_run: bool = False,
    reply_to_tweet_id: str | int | None = None,
) -> dict[str, Any]:
    """Post a plain-text tweet. Returns {'id': str, 'text': str}.

    Args:
        text: Tweet body (max 280 chars for standard accounts).
        config_file: Path to JSON credentials; defaults to
            ``~/keys/twitter_api_config.json`` or env vars.
        dry_run: If True, log what would be posted and return a stub.
        reply_to_tweet_id: If set, post as a reply to that tweet (used by
            ``thread_tweets`` to chain tweets together).
    """
    _check_tweet_length(text)
    if dry_run:
        log.info("[DRY-RUN] tweet (%d chars): %s", len(text), text)
        return {"id": "dry-run", "text": text}

    config = _load_config(config_file)
    client_v2, _ = _build_clients(config)
    kwargs: dict[str, Any] = {"text": text}
    if reply_to_tweet_id is not None:
        kwargs["in_reply_to_tweet_id"] = reply_to_tweet_id
    resp = client_v2.create_tweet(**kwargs)
    data = getattr(resp, "data", None) or {}
    tweet_id = str(data.get("id", ""))
    log.info("Posted tweet %s: %s", tweet_id, text)
    return {"id": tweet_id, "text": data.get("text", text)}


def post_tweet_with_media(
    text: str,
    media_paths: list[Path | str],
    config_file: str = "",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Post a tweet with up to 4 images / 1 video / 1 gif attached.

    Media is uploaded via the v1.1 media endpoint, then referenced by id in
    the v2 POST /2/tweets call (X's current supported flow).
    """
    _check_tweet_length(text)
    paths = [Path(p).expanduser() for p in media_paths]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Media files not found: {missing}")
    if len(paths) > 4:
        raise ValueError(f"X allows at most 4 images per tweet; got {len(paths)}")

    if dry_run:
        log.info("[DRY-RUN] tweet (%d chars) with %d media: %s",
                 len(text), len(paths), text)
        return {"id": "dry-run", "text": text, "media": [str(p) for p in paths]}

    config = _load_config(config_file)
    client_v2, api_v1 = _build_clients(config)
    media_ids = [str(api_v1.media_upload(filename=str(p)).media_id) for p in paths]
    resp = client_v2.create_tweet(text=text, media_ids=media_ids)
    data = getattr(resp, "data", None) or {}
    tweet_id = str(data.get("id", ""))
    log.info("Posted tweet %s with %d media: %s", tweet_id, len(media_ids), text)
    return {"id": tweet_id, "text": data.get("text", text), "media_ids": media_ids}


def thread_tweets(
    texts: list[str],
    config_file: str = "",
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Post a thread: each text becomes a reply to the previous tweet.

    Useful when a blog-post announcement exceeds 280 chars.
    """
    if not texts:
        raise ValueError("thread_tweets requires a non-empty list of texts")
    results: list[dict[str, Any]] = []
    reply_to: str | None = None
    for text in texts:
        result = post_tweet(
            text, config_file=config_file, dry_run=dry_run, reply_to_tweet_id=reply_to,
        )
        results.append(result)
        reply_to = result["id"] if result["id"] != "dry-run" else None
    return results


def post_blog_announcement(
    title: str,
    url: str,
    summary: str = "",
    hashtags: list[str] | None = None,
    config_file: str = "",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Convenience helper: format a blog-post announcement tweet.

    Shape:
        "<title>\\n\\n<summary>\\n<url>\\n#tag1 #tag2"

    Truncates ``summary`` to keep the full tweet under 280 chars.
    """
    tags = " ".join(f"#{t.lstrip('#')}" for t in (hashtags or []))
    header = f"{title}\n\n"
    footer_parts = [p for p in (url, tags) if p]
    footer = ("\n" + "\n".join(footer_parts)) if footer_parts else ""
    budget = MAX_TWEET_LEN - len(header) - len(footer)
    if budget < 0:
        raise ValueError(
            f"Title + url + hashtags alone are {len(header) + len(footer)} chars; "
            f"shorten the title or drop hashtags."
        )
    trimmed_summary = summary if len(summary) <= budget else summary[: max(budget - 1, 0)].rstrip() + "…"
    text = f"{header}{trimmed_summary}{footer}".strip()
    return post_tweet(text, config_file=config_file, dry_run=dry_run)


if __name__ == "__main__":
    # Dry-run smoke test — no credentials needed.
    post_tweet("Hello world from uutils.twitter_uu (dry-run)", dry_run=True)
    post_blog_announcement(
        title="Embracing the AI Agent Era",
        url="https://brando90.github.io/post/ai-agent-era",
        summary="Why building agent harnesses matters more than ever.",
        hashtags=["ai", "agents"],
        dry_run=True,
    )
    thread_tweets(
        ["part 1 of the thread", "part 2 continues here", "part 3 wraps up"],
        dry_run=True,
    )
    print("twitter_uu dry-run tests passed.")
