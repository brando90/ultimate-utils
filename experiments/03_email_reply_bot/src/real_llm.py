"""Claude CLI-backed implementation of :class:`LLMClient`.

Invokes the local ``clauded -p`` command via subprocess instead of direct
API calls or third-party SDKs.
"""

from __future__ import annotations

import subprocess


class ClaudeCLIClient:
    """Run a prompt through the local ``clauded -p`` CLI."""

    def __init__(
        self,
        *,
        timeout_s: float | None = 600.0,
        clauded_bin: str = "clauded",
    ) -> None:
        self.timeout_s = timeout_s
        self.clauded_bin = clauded_bin

    def run(self, *, prompt: str, system: str, workdir: str | None = None) -> str:
        combined = f"{system}\n\n{prompt}" if system else prompt
        cmd = [self.clauded_bin, "-p", combined]
        try:
            res = subprocess.run(
                cmd,
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"clauded command timed out after {self.timeout_s}s"
            ) from exc

        if res.returncode != 0:
            err = res.stderr.strip() if res.stderr else ""
            raise RuntimeError(
                f"clauded failed with exit code {res.returncode}: {err}"
            )

        return res.stdout.strip()
