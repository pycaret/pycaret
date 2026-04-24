"""Pydantic request/response models shared across routes."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field

# ---------------------------------------------------------------- auth


class BootstrapRequest(BaseModel):
    """First-run admin + workspace bootstrap."""

    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    display_name: str | None = None
    workspace_name: str = Field(min_length=1, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenPairResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds until access_token expiry


class UserResponse(BaseModel):
    id: str
    email: EmailStr
    display_name: str | None
    is_active: bool
    is_superuser: bool
    created_at: datetime


# ---------------------------------------------------------------- workspaces


class WorkspaceCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    description: str | None = None


class WorkspaceResponse(BaseModel):
    id: str
    name: str
    description: str | None
    created_at: datetime
    created_by: str


# ---------------------------------------------------------------- projects


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    description: str | None = None
    tags: list[str] = Field(default_factory=list)


class ProjectResponse(BaseModel):
    id: str
    workspace_id: str
    name: str
    description: str | None
    tags: list[str]
    created_at: datetime
    created_by: str


# ---------------------------------------------------------------- experiments


class ExperimentCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    task: str = Field(
        description="classification | regression | clustering | anomaly | time_series"
    )
    target: str | None = None
    setup_params: dict = Field(default_factory=dict)
    data_source_id: str | None = None


class ExperimentResponse(BaseModel):
    id: str
    project_id: str
    name: str
    task: str
    target: str | None
    setup_params: dict
    data_source_id: str | None
    created_at: datetime
    created_by: str


# ---------------------------------------------------------------- runs


class RunCreate(BaseModel):
    """Request body for ``POST /api/v1/experiments/{id}/runs``.

    Exactly one input source must be provided — ``sklearn_dataset`` (a tiny
    built-in sklearn frame, handy for tutorials and tests) or ``data_inline``
    (a list of row dicts). S3 / CSV uploads will be added alongside the data
    sources router.
    """

    plan: str = Field(
        default="setup",
        description="One of: setup | create | compare.",
    )
    model_id: str | None = Field(
        default=None, description="Required when plan='create'; PyCaret model id (e.g. 'lr')."
    )
    plan_params: dict = Field(default_factory=dict)

    sklearn_dataset: str | None = None
    data_inline: list[dict] | None = None


class RunResponse(BaseModel):
    id: str
    experiment_id: str
    status: str
    started_at: datetime | None
    finished_at: datetime | None
    duration_ms: float | None
    error: str | None
    leaderboard: list[dict] | dict | None
    metrics_summary: dict | None
    snapshot: dict | None
    created_at: datetime
    created_by: str


class EventResponse(BaseModel):
    id: str
    run_id: str
    kind: str
    message: str | None
    payload: dict | None
    duration_ms: float | None
    emitted_at: datetime


# ---------------------------------------------------------------- setup/bootstrap


class SetupStatusResponse(BaseModel):
    is_bootstrapped: bool
    user_count: int
    workspace_count: int
