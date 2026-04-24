"""Anthropic Claude provider.

Uses Claude's native *tool-use* feature to force a JSON response that matches
`output_schema`. The tool is declared inline per-call with the schema; we
consume the first `tool_use` block the model emits.

Imports the ``anthropic`` SDK lazily so the base server install doesn't
depend on it. Install via ``pycaret-server[llm-anthropic]`` or
``pycaret-server[llm]``.
"""

from __future__ import annotations

from typing import Any


class AnthropicLLMProvider:
    """Claude-backed `LLMProvider`."""

    def __init__(
        self,
        *,
        api_key: str,
        model_name: str = "claude-sonnet-4-5",
        base_url: str | None = None,
    ) -> None:
        self.name = "anthropic"
        self.model_name = model_name
        self._api_key = api_key
        self._base_url = base_url

    def _client(self):
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise RuntimeError(
                "anthropic SDK not installed. `pip install pycaret-server[llm-anthropic]`."
            ) from exc
        kwargs: dict[str, Any] = {"api_key": self._api_key}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        return Anthropic(**kwargs)

    def complete(
        self,
        *,
        system: str,
        user: str,
        output_schema: dict,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> dict:
        client = self._client()
        # Declare an inline tool that wraps the output schema. Claude will
        # emit a `tool_use` content block with the structured JSON input.
        tool = {
            "name": "return_advice",
            "description": "Return the advisory payload as structured JSON.",
            "input_schema": output_schema,
        }
        resp = client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            tools=[tool],
            tool_choice={"type": "tool", "name": "return_advice"},
            messages=[{"role": "user", "content": user}],
        )
        for block in resp.content:
            # anthropic.types.ToolUseBlock
            if getattr(block, "type", None) == "tool_use":
                return dict(getattr(block, "input", {}) or {})
        raise RuntimeError("Claude response had no tool_use block; cannot parse advice.")
