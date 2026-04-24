"""Run explainer — reads a completed Run's snapshot + leaderboard + events
and produces a plain-prose explanation of what happened + what to try next.

Intended UX: a collapsible card on `/runs/:id` that the user opens after a
run succeeds. Low-stakes, reads-only.

Prompt input: the Run row itself (status, duration, snapshot), the
leaderboard JSON (already stored on the Run), and the tail of the event
stream (the last ~50 engine events — enough to see what plan ran + any
warnings without overwhelming the context window).
"""

from __future__ import annotations

import json
from typing import Any

from pycaret_server.llm.schemas import LLMAdvice

SYSTEM = (
    "You are a senior ML engineer reviewing a completed PyCaret run. You see "
    "the run's config snapshot, the full leaderboard, and the engine event "
    "stream. Explain, in plain prose:\n"
    "  1. What ran (task, plan, dataset, models tried).\n"
    "  2. Which model won and WHY — look at the leaderboard metrics, not the model class alone.\n"
    "  3. Suspicious signals (e.g. tiny margin between top-2 models, "
    "CV-std larger than mean-diff, AUC=1.0 on a real-world dataset).\n"
    "  4. Concrete next experiments, prioritised.\n"
    "\n"
    "Ground every claim in the data provided. Don't invent models or metrics. "
    "Keep reasoning_summary to 3-6 sentences. In suggested_config_json, shape "
    "hints like {'next_actions': ['tune_top_model', 'add_interaction_features', "
    "'stratified_cv', ...]}.\n"
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
            "description": "Next-steps hints. Common key: `next_actions` (array of strings).",
            "additionalProperties": True,
        },
        "suggested_action": {
            "type": "string",
            "description": "One-line recommended next step.",
        },
        "reasoning_summary": {
            "type": "string",
            "description": "Plain-prose explanation (3-6 sentences).",
        },
        "risk_flags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Flags like 'overfit_suspected', 'tiny_margin', 'unbalanced_evaluation'.",
        },
    },
}


def _truncate_events(events: list[dict[str, Any]], max_events: int = 50) -> list[dict[str, Any]]:
    """Keep only the most recent N events so the prompt stays bounded."""
    if len(events) <= max_events:
        return events
    # Keep the first few (setup / experiment.started) and the last (finish + errors)
    # so the LLM sees both ends of the timeline.
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
    status: str,
    duration_ms: float | None,
    leaderboard: list[dict[str, Any]] | None,
    events: list[dict[str, Any]],
    error: str | None,
) -> tuple[str, str]:
    """Return (system, user) prompt for the run explainer."""
    user_payload = {
        "status": status,
        "duration_ms": duration_ms,
        "error": error,
        "snapshot": run_snapshot or {},
        "leaderboard": leaderboard or [],
        "events_tail": _truncate_events(events, 50),
    }
    user = json.dumps(user_payload, indent=2, default=str)
    return SYSTEM, user


def parse_response(raw: dict) -> LLMAdvice:
    try:
        return LLMAdvice.model_validate(raw)
    except Exception:
        return LLMAdvice(
            suggested_config_json=dict(raw) if isinstance(raw, dict) else {},
            suggested_action="",
            reasoning_summary="(malformed LLM response — raw dict preserved in suggested_config_json)",
            risk_flags=["malformed_response"],
        )
