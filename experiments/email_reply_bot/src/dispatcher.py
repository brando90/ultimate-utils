"""Run a Claude Agent SDK session against a prompt and return the final answer.

The :class:`LLMClient` protocol lets us swap in a fake during tests. The real
implementation (``ClaudeSDKClient``) is instantiated lazily and only when used.
"""

from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    def run(self, *, prompt: str, system: str, workdir: str | None = None) -> str: ...


SYSTEM_PROMPT = (
    "You are Claude responding to an email from Brando. "
    "The email body is the instruction — answer it directly and concisely. "
    "If the instruction asks you to investigate a codebase, do so and report findings. "
    "Keep the reply focused and email-friendly: short paragraphs, no unnecessary preamble. "
    "Do not include a greeting or signature — those are added by the mailer."
)


def build_prompt(*, subject: str, body: str) -> str:
    parts: list[str] = []
    if subject:
        parts.append(f"Subject: {subject}")
    parts.append("")
    parts.append(body.strip())
    return "\n".join(parts).strip()
