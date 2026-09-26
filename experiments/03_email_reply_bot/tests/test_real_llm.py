"""Tests for ClaudeCLIClient."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock

import pytest

from src.real_llm import ClaudeCLIClient


def test_command_construction(monkeypatch: pytest.MonkeyPatch):
    mock_run = MagicMock()
    mock_run.return_value = subprocess.CompletedProcess(
        args=["clauded", "-p", "sys\n\nprompt"],
        returncode=0,
        stdout="  Claude answer here   \n",
        stderr="",
    )
    monkeypatch.setattr(subprocess, "run", mock_run)

    client = ClaudeCLIClient(timeout_s=120.0, clauded_bin="custom-clauded")
    out = client.run(prompt="prompt", system="sys", workdir="/tmp/work")

    assert out == "Claude answer here"
    mock_run.assert_called_once_with(
        ["custom-clauded", "-p", "sys\n\nprompt"],
        cwd="/tmp/work",
        capture_output=True,
        text=True,
        timeout=120.0,
    )


def test_prompt_without_system(monkeypatch: pytest.MonkeyPatch):
    mock_run = MagicMock()
    mock_run.return_value = subprocess.CompletedProcess(
        args=["clauded", "-p", "just prompt"],
        returncode=0,
        stdout="answer",
        stderr="",
    )
    monkeypatch.setattr(subprocess, "run", mock_run)

    client = ClaudeCLIClient()
    out = client.run(prompt="just prompt", system="")
    assert out == "answer"
    mock_run.assert_called_once_with(
        ["clauded", "-p", "just prompt"],
        cwd=None,
        capture_output=True,
        text=True,
        timeout=600.0,
    )


def test_nonzero_exit_raises_error(monkeypatch: pytest.MonkeyPatch):
    mock_run = MagicMock()
    mock_run.return_value = subprocess.CompletedProcess(
        args=["clauded", "-p", "prompt"],
        returncode=2,
        stdout="",
        stderr="Authentication failed",
    )
    monkeypatch.setattr(subprocess, "run", mock_run)

    client = ClaudeCLIClient()
    with pytest.raises(RuntimeError, match="clauded failed with exit code 2: Authentication failed"):
        client.run(prompt="prompt", system="sys")


def test_timeout_raises_error(monkeypatch: pytest.MonkeyPatch):
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=10.0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    client = ClaudeCLIClient(timeout_s=10.0)
    with pytest.raises(RuntimeError, match="clauded command timed out after 10.0s"):
        client.run(prompt="prompt", system="sys")
