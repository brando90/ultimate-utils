"""Claude-API-backed implementation of :class:`LLMClient`.

Separated from ``dispatcher`` so tests don't need the Anthropic SDK.
"""

from __future__ import annotations

import os


class AnthropicClient:
    """Direct Anthropic Messages API call — simplest possible v1.

    For the full Claude Agent SDK (with tool use, file I/O, etc.), swap this
    for a ``claude-agent-sdk`` invocation. This class intentionally stays small
    so the read-only v1 stays easy to reason about.
    """

    def __init__(self, *, model: str = "claude-opus-4-7", max_tokens: int = 4096) -> None:
        from anthropic import Anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        self._client = Anthropic(api_key=api_key) if api_key else Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def run(self, *, prompt: str, system: str, workdir: str | None = None) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        parts: list[str] = []
        for block in response.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
