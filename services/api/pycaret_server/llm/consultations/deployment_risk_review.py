"""Deployment risk reviewer — checks a Pipeline before it goes live.

UX: modal on `/pipelines/:id` that opens when the user clicks "Review
before deploy" in the deploy sidebar. Takes no user input — the payload
is derived from the Pipeline + its origin Run.

Flags: missing preprocessing steps (imputer, encoder) in the fitted
pipeline, suspicious leaderboard (AUC=1.0, tiny sample size), no
holdout evaluation, target-column assumptions, version-skew between
training-time sklearn and today's sklearn.
"""

from __future__ import annotations

import json
from typing import Any

from pycaret_server.llm.schemas import LLMAdvice

SYSTEM = (
    "You are a senior ML platform engineer reviewing a trained pipeline "
    "before it ships to production. You have the pipeline metadata "
    "(name, model_id, origin run snapshot, sha256, tags), the origin run's "
    "leaderboard, and the run's terminal metrics.\n"
    "\n"
    "Review whether this pipeline is safe to deploy. Specifically check:\n"
    "  - SAMPLE SIZE — is the training data too small for production generalisation?\n"
    "  - OVERFIT — is the leaderboard's top metric suspiciously high (AUC≈1.0, acc≈1.0)?\n"
    "  - TINY MARGIN — is the top model barely ahead of a simpler one?\n"
    "  - PREPROCESSING — does the config include imputation + encoding? Missing "
    "one means the deployed endpoint will crash on real data with nulls or strings.\n"
    "  - METRIC CHOICE — is the primary metric appropriate for the task/business goal?\n"
    "  - VERSION SKEW — if training sklearn version ≠ serving sklearn version, flag it.\n"
    "\n"
    "Output a verdict in suggested_action ('APPROVE', 'APPROVE WITH CAVEATS', "
    "'DO NOT DEPLOY') followed by the reasoning. suggested_config_json "
    "can hint at deployment-time config (e.g. `{'monitor': ['feature_drift'], "
    "'auth_mode': 'workspace'}`). Be direct; overestimate risks.\n"
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
            "description": "Optional deployment-time hints (monitor, auth_mode, replicas).",
            "additionalProperties": True,
        },
        "suggested_action": {
            "type": "string",
            "description": (
                "Verdict: 'APPROVE' / 'APPROVE WITH CAVEATS: …' / "
                "'DO NOT DEPLOY: …'. Start with the literal verdict word."
            ),
        },
        "reasoning_summary": {
            "type": "string",
            "description": "4-6 sentences walking through each risk checked.",
        },
        "risk_flags": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Named risks: 'overfit_suspected', 'tiny_margin', "
                "'small_training_sample', 'missing_imputer', 'missing_encoder', "
                "'version_skew', 'unmonitored_drift'."
            ),
        },
    },
}


def build_prompt(
    *,
    pipeline: dict[str, Any],
    origin_run_snapshot: dict[str, Any] | None,
    leaderboard: list[dict[str, Any]] | None,
    origin_run_status: str | None,
) -> tuple[str, str]:
    """Return (system, user) prompt for the reviewer."""
    user = json.dumps(
        {
            "pipeline": pipeline,
            "origin_run_status": origin_run_status,
            "origin_run_snapshot": origin_run_snapshot or {},
            "leaderboard": leaderboard or [],
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
