"""`LLMRouter` — single entry point every consultation goes through.

Responsibilities:
  1. Load the active `LLMProviderSetting` row for a workspace.
  2. Build a provider instance via `get_provider(...)`.
  3. Call `provider.complete(...)` with the consultation's system/user/schema.
  4. Normalise the response into `LLMAdvice`.
  5. Persist an `LLMConsultation` audit row (even on failure, so errors are
     discoverable in the UI).
  6. Return `(advice, consultation_row_id)`.

Why a router class rather than a bare function: the router is the right
place to own cross-cutting concerns (rate limits, caching, cost accounting,
usage quotas per workspace) when they arrive. Today it's thin.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from pycaret_server.crypto import decrypt as _decrypt_secret
from pycaret_server.db import LLMConsultation, LLMProviderSetting
from pycaret_server.llm.providers import LLMProvider, get_provider
from pycaret_server.llm.schemas import LLMAdvice

_log = logging.getLogger(__name__)


@dataclass
class ConsultationContext:
    """Everything a router call needs, shaped for reuse across consultation types."""

    workspace_id: str
    user_id: str
    consultation_type: str
    system: str
    user: str
    output_schema: dict
    # Optional FKs — these correlate the consultation to the domain object.
    project_id: str | None = None
    experiment_id: str | None = None
    run_id: str | None = None


class NoLLMConfigured(RuntimeError):
    """Raised when a workspace has no enabled LLM provider."""


class LLMRouter:
    """One instance per process. Dispatches every consultation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- config

    def get_active_setting(self, session: Session, workspace_id: str) -> LLMProviderSetting | None:
        """Return the first enabled provider setting for the workspace.

        v1 semantics: workspaces have AT MOST one enabled provider at a time.
        Multi-provider routing (cost-aware, capability-aware) is a V2 concern.
        """
        return session.scalar(
            select(LLMProviderSetting).where(
                LLMProviderSetting.workspace_id == workspace_id,
                LLMProviderSetting.enabled.is_(True),
            )
        )

    def build_provider(self, setting: LLMProviderSetting) -> LLMProvider:
        if not setting.api_key_encrypted:
            raise NoLLMConfigured(f"LLM provider {setting.provider!r} has no API key set.")
        api_key = _decrypt_secret(setting.api_key_encrypted)
        return get_provider(
            provider=setting.provider,
            api_key=api_key,
            model_name=setting.model_name,
            base_url=setting.base_url,
        )

    # ---------------------------------------------------------- consultations

    def consult(
        self,
        session: Session,
        ctx: ConsultationContext,
    ) -> tuple[LLMAdvice, LLMConsultation]:
        """Run one consultation end-to-end. Always persists an audit row.

        Raises
        ------
        NoLLMConfigured
            When the workspace has no enabled provider.
        RuntimeError
            When the provider SDK errors. The audit row is written before the
            raise so the UI can surface the failure from `/llm/consultations`.
        """
        setting = self.get_active_setting(session, ctx.workspace_id)
        if setting is None:
            raise NoLLMConfigured(
                "No LLM provider configured + enabled for this workspace. "
                "Configure one under workspace LLM settings."
            )

        provider = self.build_provider(setting)
        started_at = time.perf_counter()
        raw: dict[str, Any] | None = None
        error: str | None = None
        try:
            raw = provider.complete(
                system=ctx.system,
                user=ctx.user,
                output_schema=ctx.output_schema,
            )
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            _log.exception("LLM consultation failed")
        latency_ms = (time.perf_counter() - started_at) * 1000

        advice: LLMAdvice
        if raw is not None:
            try:
                advice = LLMAdvice.model_validate(raw)
            except Exception as exc:  # noqa: BLE001
                error = error or f"response_validation_error: {exc}"
                advice = LLMAdvice(
                    suggested_config_json=dict(raw) if isinstance(raw, dict) else {},
                    reasoning_summary="(malformed LLM response)",
                    risk_flags=["malformed_response"],
                )
        else:
            advice = LLMAdvice(risk_flags=["provider_error"])

        row = LLMConsultation(
            id=str(uuid.uuid4()),
            workspace_id=ctx.workspace_id,
            project_id=ctx.project_id,
            experiment_id=ctx.experiment_id,
            run_id=ctx.run_id,
            type=ctx.consultation_type,
            provider=setting.provider,
            model_name=setting.model_name,
            prompt=_truncate(ctx.user, 20_000),
            response_json=advice.model_dump(),
            generated_config_json=advice.suggested_config_json or None,
            latency_ms=latency_ms,
            error=error,
            created_by=ctx.user_id,
        )
        session.add(row)
        session.commit()
        session.refresh(row)

        if error:
            # Persist first, then surface to the caller.
            raise RuntimeError(error)
        return advice, row

    def test_connection(self, setting: LLMProviderSetting) -> tuple[bool, str | None, float | None]:
        """Lightweight round-trip to verify the provider + API key work.

        Returns ``(ok, error_message, latency_ms)``.
        """
        provider = self.build_provider(setting)
        started = time.perf_counter()
        try:
            provider.complete(
                system="You are a health-check probe. Respond with a minimal JSON object.",
                user="ping",
                output_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["ok"],
                    "properties": {"ok": {"type": "boolean"}},
                },
                max_tokens=32,
            )
            latency = (time.perf_counter() - started) * 1000
            return True, None, latency
        except Exception as exc:  # noqa: BLE001
            latency = (time.perf_counter() - started) * 1000
            return False, f"{type(exc).__name__}: {exc}", latency


# --------------------------------------------------------------- singleton


_router: LLMRouter | None = None
_lock = threading.Lock()


def get_router() -> LLMRouter:
    global _router
    with _lock:
        if _router is None:
            _router = LLMRouter()
        return _router


def reset_router() -> None:
    """For test fixtures + reload mode."""
    global _router
    with _lock:
        _router = None


def _truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[:n] + f"\n… (truncated, original length {len(s)})"
