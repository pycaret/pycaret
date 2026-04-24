"""LLM advisory layer for the PyCaret Control Plane.

Implements the router pattern from `docs/revamp/DECISIONS.md § session 13 · 3`:
provider-agnostic from day one, with Anthropic (Claude) and OpenAI as two
first-class backends. The ``FakeLLMProvider`` backs tests so CI never hits a
real API.

Every consultation returns the same envelope — `LLMAdvice` — consumed
uniformly by the UI regardless of which provider produced it.

**Safety contract** (CONTROL_PLANE_SPEC § 12.3): LLM output is advisory. Every
consultation yields ``suggested_config_json`` + ``suggested_action`` +
``reasoning_summary`` + ``risk_flags``. The deterministic engine executes what
the user approves; the LLM never triggers a side effect directly.
"""

from pycaret_server.llm.providers import LLMProvider, get_provider
from pycaret_server.llm.router import LLMRouter, get_router, reset_router
from pycaret_server.llm.schemas import LLMAdvice

__all__ = [
    "LLMAdvice",
    "LLMProvider",
    "LLMRouter",
    "get_provider",
    "get_router",
    "reset_router",
]
