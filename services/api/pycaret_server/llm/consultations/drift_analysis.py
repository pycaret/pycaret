"""Drift analyst — interprets a DriftReport and recommends an action.

6th / final copilot in SPEC § 12.2. Takes a ``DriftReport`` row (feature
drift scores + prediction drift + window + sample size) + the parent
``Deployment`` + the origin ``Pipeline`` snapshot, and produces a verdict
in the same spirit as the deployment reviewer:

  - ``RETRAIN NOW``       — drift is severe, model is likely stale
  - ``INVESTIGATE``       — drift is meaningful but root cause unclear
  - ``MONITOR``           — some drift but not yet actionable; tighten alerts
  - ``NO ACTION``         — drift is noise / expected / below thresholds

The literal verdict prefix lets the UI tone-code with `.startsWith()`
the same way the deployment reviewer does (APPROVE / APPROVE WITH CAVEATS
/ DO NOT DEPLOY).
"""

from __future__ import annotations

import json
from typing import Any

from pycaret_server.llm.schemas import LLMAdvice

SYSTEM = (
    "You are a senior MLOps engineer interpreting a drift report for a "
    "production ML deployment. You have:\n"
    "  - the drift report summary (score, bucketed status, window, sample size)\n"
    "  - per-feature drift scores (PSI / KS / chi2 / missing-rate) keyed by "
    "feature name\n"
    "  - the prediction distribution shift vs baseline (JS divergence)\n"
    "  - the deployment metadata (endpoint slug, inference count, last "
    "inference time, p95 latency)\n"
    "  - the pipeline metadata (name, model class, training tags)\n"
    "\n"
    "Decide whether the owner should act. Specifically:\n"
    "  - LOOK FOR CONCENTRATION — is drift coming from one or two features, "
    "or spread across many? One dominant feature often = a data-source change "
    "(upstream ETL, unit change, schema drift). Spread = true concept drift.\n"
    "  - LOOK AT SAMPLE SIZE — if sample_size < 200 treat drift scores "
    "skeptically; recommend MONITOR not RETRAIN.\n"
    "  - PREDICTION DRIFT vs FEATURE DRIFT — prediction drift without feature "
    "drift is suspicious (label definition change? model-side bug?). "
    "Feature drift without prediction drift means the model is robust — "
    "MONITOR, not retrain.\n"
    "  - MISSING-RATE SPIKES — a feature with large missing-rate drift is "
    "often pipeline breakage upstream; recommend INVESTIGATE over RETRAIN.\n"
    "\n"
    "Output a verdict in suggested_action starting with the literal word:\n"
    "  'RETRAIN NOW: <why>'\n"
    "  'INVESTIGATE: <what to look at first>'\n"
    "  'MONITOR: <threshold / window to tighten>'\n"
    "  'NO ACTION: <reason drift is not actionable>'\n"
    "suggested_config_json can carry retraining hints "
    "(e.g. `{'retrain_window_days': 30, 'refresh_baseline': true}`). "
    "Be concrete. Name specific features when they dominate.\n"
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
            "description": (
                "Optional retraining / monitoring hints "
                "(retrain_window_days, refresh_baseline, alert_channels)."
            ),
            "additionalProperties": True,
        },
        "suggested_action": {
            "type": "string",
            "description": (
                "Verdict: 'RETRAIN NOW: …' / 'INVESTIGATE: …' / "
                "'MONITOR: …' / 'NO ACTION: …'. Start with the literal verdict."
            ),
        },
        "reasoning_summary": {
            "type": "string",
            "description": (
                "4-6 sentences: which features drove drift, whether "
                "prediction drift followed, sample-size caveat, verdict."
            ),
        },
        "risk_flags": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Named risks: 'concentrated_drift', 'diffuse_drift', "
                "'prediction_drift_without_feature_drift', "
                "'missing_rate_spike', 'small_sample', 'stale_baseline', "
                "'possible_data_source_change'."
            ),
        },
    },
}


def build_prompt(
    *,
    drift_report: dict[str, Any],
    deployment: dict[str, Any],
    pipeline: dict[str, Any] | None,
) -> tuple[str, str]:
    """Return (system, user) prompt for the analyst."""
    user = json.dumps(
        {
            "drift_report": drift_report,
            "deployment": deployment,
            "pipeline": pipeline or {},
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
