"""Daemon entry point and offline demo.

Usage:
    # Safe offline demo against bundled .eml fixtures (default, no network/clauded/send):
    python -m src.main
    python -m src.main --fixtures tests/fixtures

    # Live watcher (requires --live and config; acts only with --send):
    python -m src.main --live --config config.yaml           # dry-run watcher (no mail sent)
    python -m src.main --live --send --config config.yaml    # live outbound mail
    python -m src.main --live --once --config config.yaml    # single batch
"""

from __future__ import annotations

import argparse
import logging
import tempfile
import time
from pathlib import Path

import yaml

try:
    from .dispatcher import LLMClient
    from .gmail_client import FetchedMessage
    from .pipeline import Pipeline, PipelineConfig
    from .store import Store
except ImportError:
    from src.dispatcher import LLMClient
    from src.gmail_client import FetchedMessage
    from src.pipeline import Pipeline, PipelineConfig
    from src.store import Store

log = logging.getLogger("email_reply_bot")


def _load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _make_gmail_client(cfg: dict):
    try:
        from .real_gmail import RealGmailClient
    except ImportError:
        from src.real_gmail import RealGmailClient

    return RealGmailClient(
        user=cfg["gmail"]["user"],
        bot_from_addr=cfg["gmail"]["bot_from_addr"],
        credentials_path=Path(cfg["gmail"]["credentials_path"]).expanduser(),
        token_path=Path(cfg["gmail"]["token_path"]).expanduser(),
    )


def _make_llm_client(cfg: dict) -> LLMClient:
    try:
        from .real_llm import ClaudeCLIClient
    except ImportError:
        from src.real_llm import ClaudeCLIClient

    llm_cfg = cfg.get("llm", {}) if isinstance(cfg, dict) else {}
    timeout_s = llm_cfg.get("timeout_s", 600.0)
    if timeout_s is not None:
        timeout_s = float(timeout_s)
    clauded_bin = str(llm_cfg.get("clauded_bin", "clauded"))
    return ClaudeCLIClient(
        timeout_s=timeout_s,
        clauded_bin=clauded_bin,
    )


class _DemoGmailClient:
    """In-memory fake Gmail client for safe offline demo."""

    def __init__(self, fixtures: list[tuple[str, bytes]]) -> None:
        self.messages = [
            FetchedMessage(
                message_id=f"demo-{idx}-{name}",
                thread_id=f"demo-thr-{idx}",
                raw=raw_bytes,
            )
            for idx, (name, raw_bytes) in enumerate(fixtures)
        ]
        self._fetched = False

    def fetch_unseen(self, label: str) -> list[FetchedMessage]:
        if not self._fetched:
            self._fetched = True
            return list(self.messages)
        return []

    def send_threaded_reply(
        self,
        *,
        thread_id: str,
        in_reply_to: str,
        references: str,
        to: str,
        subject: str,
        body_text: str,
    ) -> str:
        raise RuntimeError("send_threaded_reply cannot be called in demo mode")


class _EchoLLMClient:
    """Safe echo LLM client for offline demo mode — makes no external calls."""

    def run(self, *, prompt: str, system: str, workdir: str | None = None) -> str:
        first_line = prompt.strip().splitlines()[0] if prompt.strip() else ""
        return f"[Echo response to: {first_line}]"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Email reply bot daemon and security filter demo.")
    parser.add_argument("--config", type=Path, default=None, help="Path to config YAML (used with --live).")
    parser.add_argument("--live", action="store_true", help="Run live Gmail watcher and clauded CLI.")
    parser.add_argument(
        "--send",
        action="store_true",
        help="Enable outbound email sending (requires --live; otherwise has no effect).",
    )
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=None,
        help="Directory of .eml files to test in offline demo mode (default: tests/fixtures).",
    )
    parser.add_argument("--once", action="store_true", help="Process one batch and exit.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Explicitly disable outbound sending (default is True unless both --live and --send are given).",
    )
    parser.add_argument("--interval", type=int, default=30, help="Poll interval in seconds.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.live:
        if args.send:
            log.warning("--send passed without --live; --send has no effect in offline demo mode.")

        fixtures_dir = args.fixtures
        if fixtures_dir is None:
            fixtures_dir = Path(__file__).resolve().parent.parent / "tests" / "fixtures"

        eml_files = sorted(fixtures_dir.glob("*.eml")) if fixtures_dir.is_dir() else []
        fixtures = [(f.name, f.read_bytes()) for f in eml_files]

        with tempfile.TemporaryDirectory() as tmp_dir:
            store = Store(Path(tmp_dir) / "demo_state.sqlite")
            demo_gmail = _DemoGmailClient(fixtures)
            demo_llm = _EchoLLMClient()
            pipeline = Pipeline(
                gmail=demo_gmail,
                llm=demo_llm,
                store=store,
                config=PipelineConfig(
                    bot_from_addr="brandojazz@gmail.com",
                    workdir=None,
                    rate_limit_per_hour=10,
                    require_auth_headers=True,
                    dry_run=True,
                ),
            )
            results = pipeline.run_once("INBOX")
            for f, r in zip(eml_files, results):
                print(f"[{f.name}] accepted={r.accepted} reason={r.reason}")
        return 0

    # Live mode requires a valid config file
    if args.config is None:
        raise ValueError("--live mode requires --config <path>")
    cfg = _load_config(args.config)

    # Dry-run is the default everywhere: sending requires both --live and --send, and not --dry-run
    dry_run = not (args.live and args.send and not args.dry_run)

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
            dry_run=dry_run,
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
