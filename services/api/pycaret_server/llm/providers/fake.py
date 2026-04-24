"""Deterministic fake provider for tests.

Emits a stable shape matching `LLMAdvice`. Any test that wants to drive a
specific output can override the `canned_response` attribute before calling
`complete()`.
"""

from __future__ import annotations

from typing import Any


class FakeLLMProvider:
    """In-memory stand-in for an LLM. No network, no cost, deterministic."""

    def __init__(
        self,
        *,
        model_name: str = "fake-model",
        canned_response: dict | None = None,
    ) -> None:
        self.name = "fake"
        self.model_name = model_name
        # Last prompt seen — useful for assertions.
        self.last_system: str | None = None
        self.last_user: str | None = None
        self.canned_response: dict | None = canned_response
        # Simulated latency for realism.
        self.latency_ms = 42.0

    def complete(
        self,
        *,
        system: str,
        user: str,
        output_schema: dict,
        max_tokens: int = 1024,  # noqa: ARG002
        temperature: float = 0.2,  # noqa: ARG002
    ) -> dict[str, Any]:
        self.last_system = system
        self.last_user = user
        if self.canned_response is not None:
            return self.canned_response
        # Generic shape that passes `LLMAdvice.model_validate`.
        return {
            "suggested_config_json": {
                "note": "fake provider — wire a real provider in workspace LLM settings",
                "schema_keys": list((output_schema or {}).get("properties", {}).keys()),
            },
            "suggested_action": "Review the suggested config, then submit a run.",
            "reasoning_summary": (
                "Fake LLM provider: deterministic output for tests + dev. "
                "Replace with Anthropic or OpenAI in workspace settings."
            ),
            "risk_flags": [],
        }
