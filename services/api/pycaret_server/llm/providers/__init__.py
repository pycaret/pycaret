"""Provider registry + factory.

`get_provider(row)` is the seam every consultation path goes through. It's
keyed on `LLMProviderSetting.provider`. Adding Google / Azure / Ollama later
means implementing one class + adding one registry entry here.
"""

from __future__ import annotations

from pycaret_server.llm.providers.base import LLMProvider
from pycaret_server.llm.providers.fake import FakeLLMProvider

__all__ = ["LLMProvider", "get_provider", "register_provider", "register_fake_for_tests"]

# Keyed on provider name — matches `LLMProviderSetting.provider`.
_FACTORIES: dict[str, object] = {}


def _anthropic_factory(*, api_key: str, model_name: str, base_url: str | None) -> LLMProvider:
    from pycaret_server.llm.providers.anthropic_provider import AnthropicLLMProvider

    return AnthropicLLMProvider(api_key=api_key, model_name=model_name, base_url=base_url)


def _openai_factory(*, api_key: str, model_name: str, base_url: str | None) -> LLMProvider:
    from pycaret_server.llm.providers.openai_provider import OpenAILLMProvider

    return OpenAILLMProvider(api_key=api_key, model_name=model_name, base_url=base_url)


_FACTORIES["anthropic"] = _anthropic_factory
_FACTORIES["openai"] = _openai_factory


def register_provider(name: str, factory) -> None:
    """Add / override a provider factory. Primarily for tests + plugins."""
    _FACTORIES[name] = factory


def register_fake_for_tests(canned_response: dict | None = None) -> None:
    """Install `FakeLLMProvider` under every provider name so tests can exercise
    the full router path without hitting real APIs. Idempotent.
    """

    def _fake(*, api_key: str, model_name: str, base_url: str | None) -> LLMProvider:  # noqa: ARG001
        return FakeLLMProvider(
            model_name=model_name or "fake-model",
            canned_response=canned_response,
        )

    for name in (
        "anthropic",
        "openai",
        "google",
        "azure_openai",
        "ollama",
        "custom_openai_compatible",
    ):
        _FACTORIES[name] = _fake


def get_provider(
    *,
    provider: str,
    api_key: str,
    model_name: str,
    base_url: str | None = None,
) -> LLMProvider:
    """Return a fresh provider instance for a configured `LLMProviderSetting` row."""
    factory = _FACTORIES.get(provider)
    if factory is None:
        raise ValueError(f"unknown LLM provider {provider!r}")
    return factory(api_key=api_key, model_name=model_name, base_url=base_url)  # type: ignore[operator]
