"""Tests for main entry point safe defaults and CLI flags."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import src.main as main_mod
from src.main import main


def test_main_default_touches_no_network_or_llm(monkeypatch: pytest.MonkeyPatch, capsys):
    def fail_subprocess(*args, **kwargs):
        raise AssertionError("subprocess.run was called in default demo mode!")

    def fail_gmail_builder(*args, **kwargs):
        raise AssertionError("_make_gmail_client was called in default demo mode!")

    monkeypatch.setattr(subprocess, "run", fail_subprocess)
    monkeypatch.setattr(main_mod, "_make_gmail_client", fail_gmail_builder)

    exit_code = main([])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "[alias_tag.eml] accepted=True" in captured.out
    assert "[legit_brando_science.eml] accepted=True" in captured.out
    assert "[stranger.eml] accepted=False" in captured.out


def test_send_without_live_has_no_effect(monkeypatch: pytest.MonkeyPatch, capsys):
    def fail_subprocess(*args, **kwargs):
        raise AssertionError("subprocess.run was called without --live!")

    def fail_gmail_builder(*args, **kwargs):
        raise AssertionError("_make_gmail_client was called without --live!")

    monkeypatch.setattr(subprocess, "run", fail_subprocess)
    monkeypatch.setattr(main_mod, "_make_gmail_client", fail_gmail_builder)

    exit_code = main(["--send"])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "[legit_brandojazz.eml] accepted=True" in captured.out


def test_custom_fixtures_dir(tmp_path: Path, capsys):
    # Put a single fixture in tmp_path
    fixture_src = Path(__file__).parent / "fixtures" / "legit_brando_science.eml"
    target_fixture = tmp_path / "test_one.eml"
    target_fixture.write_bytes(fixture_src.read_bytes())

    exit_code = main(["--fixtures", str(tmp_path)])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "[test_one.eml] accepted=True" in captured.out


def test_live_without_config_raises():
    with pytest.raises(ValueError, match="--live mode requires --config"):
        main(["--live"])


def test_live_mode_wiring(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        """
gmail:
  user: "me"
  bot_from_addr: "brandojazz@gmail.com"
  credentials_path: "~/keys/creds.json"
  token_path: "~/keys/token.json"
llm:
  timeout_s: 30
store_path: "{tmp_path}/state.sqlite"
""".replace("{tmp_path}", str(tmp_path))
    )

    mock_gmail = MagicMock()
    mock_gmail.fetch_unseen.return_value = []
    mock_llm = MagicMock()

    monkeypatch.setattr(main_mod, "_make_gmail_client", lambda cfg: mock_gmail)
    monkeypatch.setattr(main_mod, "_make_llm_client", lambda cfg: mock_llm)

    # Test live without --send: pipeline config dry_run must be True
    captured_pipelines = []
    orig_pipeline_init = main_mod.Pipeline.__init__

    def track_pipeline_init(self, *args, **kwargs):
        orig_pipeline_init(self, *args, **kwargs)
        captured_pipelines.append(self)

    monkeypatch.setattr(main_mod.Pipeline, "__init__", track_pipeline_init)

    ret = main(["--live", "--config", str(cfg_file), "--once"])
    assert ret == 0
    assert len(captured_pipelines) == 1
    assert captured_pipelines[0].cfg.dry_run is True

    # Test live with --send: pipeline config dry_run must be False
    captured_pipelines.clear()
    ret = main(["--live", "--send", "--config", str(cfg_file), "--once"])
    assert ret == 0
    assert len(captured_pipelines) == 1
    assert captured_pipelines[0].cfg.dry_run is False
