"""Pydantic models for the LLM advisory surface.

All provider output is normalised to `LLMAdvice` before it hits the DB — this
is the envelope the UI consumes regardless of which provider ran.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

# Allowlisted providers — matches CONTROL_PLANE_SPEC § 4.14.
PROVIDERS = ("anthropic", "openai", "google", "azure_openai", "ollama", "custom_openai_compatible")

# Consultation types — matches CONTROL_PLANE_SPEC § 4.15.
CONSULTATION_TYPES = (
    "dataset_analysis",
    "experiment_design",
    "metric_selection",
    "pipeline_recommendation",
    "run_summary",
    "failure_debugging",
    "deployment_risk_review",
    "drift_analysis",
)


class LLMAdvice(BaseModel):
    """The canonical output envelope for every consultation.

    Per CONTROL_PLANE_SPEC § 12.3, the LLM proposes — the user approves — the
    engine executes. This envelope is what the user approves: a machine-readable
    config suggestion + a human-readable rationale + explicit risk flags.
    """

    suggested_config_json: dict = Field(
        default_factory=dict,
        description="Machine-readable config the LLM recommends (RunConfig-shaped when applicable).",
    )
    suggested_action: str = Field(
        default="",
        description="One-line action the user can take (e.g. 'Run a classification experiment with fold=5').",
    )
    reasoning_summary: str = Field(
        default="",
        description="Plain-prose explanation of why.",
    )
    risk_flags: list[str] = Field(
        default_factory=list,
        description="Named risk signals (e.g. 'target_leakage_suspected', 'small_sample').",
    )


# ───────────────────────────────────────────── provider settings (CRUD)


class LLMProviderSettingWrite(BaseModel):
    provider: str = Field(description="One of: " + " | ".join(PROVIDERS))
    api_key: str | None = Field(default=None, description="Plaintext API key; stored as-is for v1.")
    base_url: str | None = None
    model_name: str
    enabled: bool = True
    config: dict | None = None


class LLMProviderSettingRead(BaseModel):
    id: str
    workspace_id: str
    provider: str
    base_url: str | None
    model_name: str
    enabled: bool
    config: dict | None
    has_api_key: bool  # never leak the key; UI just shows "configured / not"
    created_at: datetime
    created_by: str


# ───────────────────────────────────────────── consultation records


class LLMConsultationRead(BaseModel):
    id: str
    workspace_id: str
    project_id: str | None
    experiment_id: str | None
    run_id: str | None
    type: str
    provider: str
    model_name: str
    prompt: str
    response_json: dict
    generated_config_json: dict | None
    latency_ms: float | None
    error: str | None
    created_at: datetime
    created_by: str


# ───────────────────────────────────────────── request bodies


class AnalyzeDatasetRequest(BaseModel):
    data_source_id: str
    workspace_id: str
    task_type_hint: str | None = Field(
        default=None,
        description="Optional hint (classification / regression / …). LLM may override.",
    )


class DesignExperimentRequest(BaseModel):
    data_source_id: str
    workspace_id: str
    goal: str = Field(
        min_length=1,
        max_length=2000,
        description="Plain-English goal — 'predict churn', 'baseline regression on prices'.",
    )


class ExplainRunRequest(BaseModel):
    run_id: str


class TestConnectionResponse(BaseModel):
    ok: bool
    provider: str
    model_name: str
    error: str | None = None
    latency_ms: float | None = None
