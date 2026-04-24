"""OpenAI provider.

Uses OpenAI's structured-output mode (``response_format={"type": "json_schema", ...}``)
to force a JSON response matching `output_schema`. Works against the native
OpenAI API + Azure OpenAI + any OpenAI-compatible endpoint (Ollama, LM Studio,
vLLM) via ``base_url``.

Imports the ``openai`` SDK lazily. Install via ``pycaret-server[llm-openai]``
or ``pycaret-server[llm]``.
"""

from __future__ import annotations

import json
from typing import Any


class OpenAILLMProvider:
    """OpenAI-chat-completions-backed `LLMProvider`."""

    def __init__(
        self,
        *,
        api_key: str,
        model_name: str = "gpt-4o-mini",
        base_url: str | None = None,
    ) -> None:
        self.name = "openai"
        self.model_name = model_name
        self._api_key = api_key
        self._base_url = base_url

    def _client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai SDK not installed. `pip install pycaret-server[llm-openai]`."
            ) from exc
        kwargs: dict[str, Any] = {"api_key": self._api_key}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        return OpenAI(**kwargs)

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
        # Wrap the schema in OpenAI's json_schema envelope.
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "llm_advice",
                "schema": output_schema,
                "strict": True,
            },
        }
        resp = client.chat.completions.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        if not resp.choices:
            raise RuntimeError("OpenAI response had no choices.")
        raw = resp.choices[0].message.content
        if not raw:
            raise RuntimeError("OpenAI response had empty content.")
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"OpenAI returned malformed JSON: {exc}") from exc
