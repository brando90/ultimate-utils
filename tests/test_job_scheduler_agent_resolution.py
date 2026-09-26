import os
import stat
import subprocess
from pathlib import Path
import pytest

from uutils.job_scheduler_uu import scheduler


@pytest.fixture
def clean_path(monkeypatch):
    """Ensure PATH is empty so we don't accidentally pick up system binaries."""
    monkeypatch.setenv("PATH", "")


def create_executable(path: Path, content: str, make_executable: bool = True):
    path.write_text(content)
    if make_executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def test_agent_resolution_skips_broken(tmp_path, clean_path, monkeypatch):
    bin_dir1 = tmp_path / "bin1"
    bin_dir2 = tmp_path / "bin2"
    bin_dir1.mkdir()
    bin_dir2.mkdir()

    # Broken clauded in first dir (no valid shebang)
    clauded_broken = bin_dir1 / "clauded"
    create_executable(clauded_broken, "#\\!/bin/bash\necho broken")

    # Valid clauded in second dir
    clauded_valid = bin_dir2 / "clauded"
    create_executable(clauded_valid, "#!/bin/sh\necho clauded --version\nexit 0")

    monkeypatch.setenv("PATH", f"{bin_dir1}{os.pathsep}{bin_dir2}")

    # Should pick the valid one
    agent = scheduler._find_agent_binary()
    assert agent is not None
    name, cmd = agent
    assert name == "clauded"
    assert cmd[0] == str(clauded_valid)

    # (e) The resolved path actually executes without OSError
    result = subprocess.run([cmd[0], "--version"], capture_output=True, text=True, check=True)
    assert "clauded --version" in result.stdout


def test_agent_resolution_fallback_to_next(tmp_path, clean_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    # Broken clauded
    clauded_broken = bin_dir / "clauded"
    create_executable(clauded_broken, "#\\!/bin/bash\necho broken")

    # Valid codex
    codex_valid = bin_dir / "codex"
    create_executable(codex_valid, "#!/bin/sh\necho codex")

    monkeypatch.setenv("PATH", str(bin_dir))

    # Should pick codex because clauded is broken
    agent = scheduler._find_agent_binary()
    assert agent is not None
    name, cmd = agent
    assert name == "codex"
    assert cmd[0] == str(codex_valid)


def test_agent_resolution_none_ready(tmp_path, clean_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    # Broken clauded
    clauded_broken = bin_dir / "clauded"
    create_executable(clauded_broken, "just text")

    # Broken codex
    codex_broken = bin_dir / "codex"
    create_executable(codex_broken, "#! missing path", make_executable=False)

    monkeypatch.setenv("PATH", str(bin_dir))

    agent = scheduler._find_agent_binary()
    assert agent is None


def test_lifecycle_email_dry_run(tmp_path, clean_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    # Valid clauded
    clauded_valid = bin_dir / "clauded"
    create_executable(clauded_valid, "#!/bin/sh\necho ok")

    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setenv("UUTILS_WATCHER_NOTIFY_DRY_RUN", "1")

    # Monkeypatch Popen to raise an exception if it gets called
    def mock_popen(*args, **kwargs):
        raise RuntimeError("Popen should not be called during dry run")
    monkeypatch.setattr(subprocess, "Popen", mock_popen)

    # Should not raise RuntimeError
    scheduler._send_daemon_lifecycle_email("TEST", "test body")
