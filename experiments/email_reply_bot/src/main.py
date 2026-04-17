"""Daemon entry point.

Usage:
    python -m experiments.email_reply_bot.src.main --config config.yaml
    python -m experiments.email_reply_bot.src.main --config config.yaml --dry-run
    python -m experiments.email_reply_bot.src.main --config config.yaml --once
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import yaml

from .dispatcher import LLMClient
from .pipeline import Pipeline, PipelineConfig
from .store import Store

log = logging.getLogger("email_reply_bot")


def _load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _make_gmail_client(cfg: dict):
    from .real_gmail import RealGmailClient

    return RealGmailClient(
        user=cfg["gmail"]["user"],
        bot_from_addr=cfg["gmail"]["bot_from_addr"],
        credentials_path=Path(cfg["gmail"]["credentials_path"]).expanduser(),
        token_path=Path(cfg["gmail"]["token_path"]).expanduser(),
    )


def _make_llm_client(cfg: dict) -> LLMClient:
    from .real_llm import AnthropicClient

    return AnthropicClient(
        model=cfg["llm"].get("model", "claude-opus-4-7"),
        max_tokens=int(cfg["llm"].get("max_tokens", 4096)),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--once", action="store_true", help="Process one batch and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Don't send outbound mail.")
    parser.add_argument("--interval", type=int, default=30, help="Poll interval (seconds).")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = _load_config(args.config)
    pause_file = Path(cfg.get("pause_file", "/tmp/email_bot_pause"))
    pipeline = Pipeline(
        gmail=_make_gmail_client(cfg),
        llm=_make_llm_client(cfg),
        store=Store(Path(cfg["store_path"]).expanduser()),
        config=PipelineConfig(
            bot_from_addr=cfg["gmail"]["bot_from_addr"],
            workdir=cfg.get("workdir"),
            rate_limit_per_hour=int(cfg.get("rate_limit_per_hour", 10)),
            require_auth_headers=bool(cfg.get("require_auth_headers", True)),
            dry_run=args.dry_run,
        ),
    )
    label = cfg["gmail"].get("label", "INBOX")

    while True:
        if pause_file.exists():
            log.info("paused via %s; sleeping", pause_file)
        else:
            try:
                results = pipeline.run_once(label)
                for r in results:
                    log.info("result: accepted=%s reason=%s", r.accepted, r.reason)
            except Exception:
                log.exception("pipeline.run_once failed")
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
