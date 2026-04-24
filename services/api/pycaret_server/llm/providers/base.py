"""`LLMProvider` Protocol — the one interface every LLM backend implements.

The router dispatches on `provider` name; every backend normalises its native
response into a plain ``dict`` that matches `LLMAdvice.model_validate(...)`.
Prompt + output-schema shape come from `consultations/*.py`; the provider
is only responsible for the I/O mechanics.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """Minimum surface every provider exposes.

    We keep it tiny on purpose: one method that asks for a structured JSON
    completion against a given schema. Tool-use (Anthropic) and JSON mode
    (OpenAI) both land on the same signature.
    """

    name: str
    model_name: str

    def complete(
        self,
        *,
        system: str,
        user: str,
        output_schema: dict,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> dict:
        """Send a prompt, return a dict matching `output_schema`.

        Parameters
        ----------
        system
            System prompt. Usually the persona + constraints ("You are a data
            scientist consultant. Always output valid JSON matching the
            schema. Never execute code.").
        user
            The message (prompt). Consultation files render this with the
            user's dataset profile, task type, etc.
        output_schema
            A JSON-schema dict the provider is asked to conform the response
            to. Providers that don't support structured output natively (few
            these days) re-ask with "please output only valid JSON matching …".
        max_tokens
            Upper bound; consultations rarely exceed a few hundred tokens.
        temperature
            Default low — we want reproducible advice, not novelty.

        Returns
        -------
        dict
            The parsed JSON the model returned. The router then validates
            this into `LLMAdvice`.

        Raises
        ------
        RuntimeError
            On transport / auth / parse failure. The router wraps into an
            `LLMConsultation` row with ``error`` set + raises upstream.
        """
        ...
