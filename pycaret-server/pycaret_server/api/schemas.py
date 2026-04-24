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


# ---------------------------------------------------------------- setup/bootstrap


class SetupStatusResponse(BaseModel):
    is_bootstrapped: bool
    user_count: int
    workspace_count: int
