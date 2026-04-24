"""Failure debugger — reads a failed Run's error + event tail and suggests fixes.

UX: inline card on `/runs/:id` when `run.status == 'failed'`. The card's
payload is the exact error string + the final slice of the event stream
(where the failure is), so the LLM can tell the user whether it's a data
problem, a config problem, or a bona fide bug.

Only runs `failed` — we don't debug successes (use `run_explanation`) or
in-flight runs (nothing to debug yet).
"""

from __future__ import annotations

import json
from typing import Any

from pycaret_server.llm.schemas import LLMAdvice

SYSTEM = (
    "You are a senior ML engineer debugging a failed PyCaret run. You have "
    "the error message, the run's config snapshot, and the tail of the engine "
    "event stream.\n"
    "\n"
    "Diagnose the failure in plain prose. Distinguish three categories:\n"
    "  - DATA (schema mismatch, missing target, all-nan column, class-"
    "imbalance 0 positives in a fold, non-numeric where numeric expected)\n"
    "  - CONFIG (wrong task type for target, incompatible model for dtype, "
    "train_size too small for fold count)\n"
    "  - ENGINE (upstream library error; version skew; rare race condition)\n"
    "\n"
    "In suggested_config_json, propose a minimal config change that would "
    "likely unblock the next attempt (e.g. "
    '`{"next_action": "retry_with_different_model", "model_id": "rf"}` '
    'or `{"next_action": "fix_target", "target": "y"}`). Never invent '
    "columns. Never claim certainty you don't have; flag uncertainty in "
    "risk_flags.\n"
)

OUTPUT_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "suggested_config_json",
        "suggested_action",
        "reasoning_summary",
        "risk_flags",
    ],
    "properties": {
        "suggested_config_json": {
            "type": "object",
            "description": "Minimal config change that likely unblocks the next attempt.",
            "additionalProperties": True,
        },
        "suggested_action": {
            "type": "string",
            "description": "One-line user-facing fix (e.g. 'Rename target column from y to target').",
        },
        "reasoning_summary": {
            "type": "string",
            "description": (
                "Diagnosis, 3-5 sentences. Start with the category "
                "(DATA / CONFIG / ENGINE), then the specific signal that led there."
            ),
        },
        "risk_flags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Uncertainty markers (e.g. 'multiple_candidate_causes', 'needs_dataset_inspection').",
        },
    },
}


def _truncate_events(events: list[dict[str, Any]], max_events: int = 40) -> list[dict[str, Any]]:
    """Keep the run-start + the failure tail so the LLM sees both ends."""
    if len(events) <= max_events:
        return events
    head = events[:5]
    tail = events[-(max_events - 5) :]
    return (
        head
        + [{"kind": "__truncated__", "message": f"{len(events) - max_events} events elided"}]
        + tail
    )


def build_prompt(
    *,
    run_snapshot: dict[str, Any] | None,
    error: str | None,
    events: list[dict[str, Any]],
) -> tuple[str, str]:
    """Return (system, user) prompt — user dict emphasises the error + event tail."""
    user = json.dumps(
        {
            "error": error or "(no error message captured)",
            "snapshot": run_snapshot or {},
            "events_tail": _truncate_events(events, 40),
        },
        indent=2,
        default=str,
    )
    return SYSTEM, user


def parse_response(raw: dict) -> LLMAdvice:
    try:
        return LLMAdvice.model_validate(raw)
    except Exception:
        return LLMAdvice(
            suggested_config_json=dict(raw) if isinstance(raw, dict) else {},
            reasoning_summary="(malformed LLM response — raw dict preserved)",
            risk_flags=["malformed_response"],
        )
