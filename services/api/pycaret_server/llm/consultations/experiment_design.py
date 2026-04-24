"""Experiment designer — takes a dataset + a user goal and returns a full
RunConfig-shaped proposal.

Input contract:
  - CSV-upload DataSource (we read its column profile — same helper shape as
    dataset_analysis so the prompt stays consistent across consultations).
  - A free-text "goal" from the user ("predict churn", "build a
    high-recall model for fraud screening", "baseline regression on prices").

Output contract: the standard `LLMAdvice` envelope. `suggested_config_json`
is a partial RunConfig hint — keys that align with the canonical
`CONTROL_PLANE_SPEC § 6.1` schema. We don't ship a one-click-apply button
in the UI for v1; the user reads + copies into the New Experiment wizard.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pycaret_server.llm.schemas import LLMAdvice

SYSTEM = (
    "You are a senior ML consultant designing a PyCaret experiment end-to-end. "
    "Given a dataset profile + a user goal in plain English, propose a full "
    "RunConfig-shaped setup. Ground every choice in the profile provided.\n"
    "\n"
    "Your JSON output must include, under suggested_config_json:\n"
    "  - task_type:         'classification' | 'regression' | 'clustering' | 'anomaly'\n"
    "  - target:            target column name (null for unsupervised)\n"
    "  - train_size:        fraction (0.5 – 0.95)\n"
    "  - fold:              int (3 – 10)\n"
    "  - primary_metric:    metric name appropriate for the task\n"
    "  - preprocessing:     { normalize: bool, transformation: bool, "
    "remove_outliers: bool, feature_selection: bool }\n"
    "  - model_shortlist:   ['lr', 'rf', 'xgb', …] — 3-6 task-appropriate models\n"
    "  - class_imbalance_strategy: 'none' | 'smote' | 'undersample' (classification only)\n"
    "\n"
    "Be conservative: prefer interpretable simple models first, turn off any "
    "preprocessing that the profile doesn't need. Flag risks explicitly in "
    "risk_flags (e.g. 'target_leakage_suspected', 'class_imbalance', "
    "'small_sample', 'high_cardinality_categorical'). Never invent columns.\n"
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
            "description": "Partial RunConfig. See CONTROL_PLANE_SPEC § 6.1 for the canonical shape.",
            "additionalProperties": True,
        },
        "suggested_action": {
            "type": "string",
            "description": "One-line instruction the user can follow (e.g. 'Start a compare run with fold=5 targeting auc').",
        },
        "reasoning_summary": {
            "type": "string",
            "description": "Plain-prose rationale (2-5 sentences) — why this task, why these models, why this metric.",
        },
        "risk_flags": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}


def build_prompt(csv_path: str, goal: str) -> tuple[str, str]:
    """Serialise the dataset profile + user goal into (system, user) prompts."""
    p = Path(csv_path)
    if not p.is_file():
        raise FileNotFoundError(f"CSV not found at {csv_path!r}")

    head = pd.read_csv(p, nrows=200)
    total_rows = sum(1 for _ in p.open(encoding="utf-8")) - 1

    columns = []
    for c in head.columns:
        s = head[c]
        columns.append(
            {
                "name": str(c),
                "dtype": str(s.dtype),
                "n_unique_in_sample": int(s.nunique(dropna=True)),
                "null_fraction_in_sample": round(float(s.isna().mean()), 4),
                "sample_values": s.dropna().head(3).astype(str).tolist(),
            }
        )

    user = json.dumps(
        {
            "user_goal": goal.strip(),
            "total_rows": total_rows,
            "sample_rows_inspected": len(head),
            "columns": columns,
        },
        indent=2,
        default=str,
    )
    return SYSTEM, user


def parse_response(raw: dict) -> LLMAdvice:
    """Defensive parse: fall back to a best-effort `LLMAdvice` if the response
    is malformed so the audit row is still meaningful."""
    try:
        return LLMAdvice.model_validate(raw)
    except Exception:
        return LLMAdvice(
            suggested_config_json=dict(raw) if isinstance(raw, dict) else {},
            suggested_action="",
            reasoning_summary="(malformed LLM response — raw dict preserved in suggested_config_json)",
            risk_flags=["malformed_response"],
        )
