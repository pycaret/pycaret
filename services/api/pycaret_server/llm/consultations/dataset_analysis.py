"""Dataset consultant — reads a CSV's profile and suggests a task + target +
preprocessing strategy + risk flags.

The prompt includes the column sample + row count; the LLM doesn't get the
raw file (too large, and it's not needed — column types + cardinality +
cardinality-of-target are what matters for task inference).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pycaret_server.llm.schemas import LLMAdvice

# System prompt — keeps the LLM inside its lane.
SYSTEM = (
    "You are a senior data-science consultant reviewing a new dataset for a "
    "PyCaret experiment. Your job is to suggest the task type, target column, "
    "preprocessing strategy, and primary metric. You must:\n"
    "- Base every recommendation on the profile provided; do not invent columns.\n"
    "- Flag target-leakage risks, class-imbalance, small samples, or high-cardinality "
    "categorical features that could break downstream AutoML.\n"
    "- Return ONLY structured JSON matching the tool / output schema — no prose outside it.\n"
    "- Be conservative: recommend the simplest preprocessing that makes sense; "
    "the user will override if needed.\n"
)

# JSON-schema mirror of `LLMAdvice`. Every provider is called with this.
# `additionalProperties` stays false so the model can't invent top-level fields.
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
                "Partial RunConfig hint. Common keys: task_type, target, "
                "primary_metric, preprocessing.{normalize,encoding,imputation}, "
                "class_imbalance_strategy."
            ),
            "additionalProperties": True,
        },
        "suggested_action": {
            "type": "string",
            "description": "One-line user-facing action (e.g. 'Start a classification run with fold=5').",
        },
        "reasoning_summary": {
            "type": "string",
            "description": "Plain-prose explanation (2-5 sentences).",
        },
        "risk_flags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Named signals (e.g. 'target_leakage_suspected', 'small_sample', 'high_cardinality_features').",
        },
    },
}


def build_prompt(csv_path: str, task_type_hint: str | None = None) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the LLM.

    Reads the CSV header + a 200-row sample to compute type hints + cardinality.
    Everything the LLM sees is captured in the `prompt` column of the
    `llm_consultations` row for auditability.
    """
    p = Path(csv_path)
    if not p.is_file():
        raise FileNotFoundError(f"CSV not found at {csv_path!r}")

    # Tiny sample — enough to compute types + cardinality without shipping PII-heavy data.
    head = pd.read_csv(p, nrows=200)
    total_rows = sum(1 for _ in p.open(encoding="utf-8")) - 1

    columns_profile = []
    for c in head.columns:
        series = head[c]
        nunique = int(series.nunique(dropna=True))
        sample_values = series.dropna().head(3).astype(str).tolist()
        columns_profile.append(
            {
                "name": str(c),
                "dtype": str(series.dtype),
                "n_unique_in_sample": nunique,
                "null_fraction_in_sample": round(float(series.isna().mean()), 4),
                "sample_values": sample_values,
            }
        )

    user = json.dumps(
        {
            "total_rows": total_rows,
            "sample_rows_inspected": len(head),
            "columns": columns_profile,
            "task_type_hint": task_type_hint,
        },
        indent=2,
        default=str,
    )
    return SYSTEM, user


def parse_response(raw: dict) -> LLMAdvice:
    """Normalise a provider's raw dict response into `LLMAdvice`.

    Providers that produce the right shape pass straight through; malformed
    outputs are coerced into a best-effort `LLMAdvice` so the audit row
    still persists something useful.
    """
    try:
        return LLMAdvice.model_validate(raw)
    except Exception:
        # Defensive: don't let a malformed response lose the audit trail.
        return LLMAdvice(
            suggested_config_json=dict(raw) if isinstance(raw, dict) else {},
            suggested_action="",
            reasoning_summary="(malformed LLM response — raw dict preserved in suggested_config_json)",
            risk_flags=["malformed_response"],
        )
