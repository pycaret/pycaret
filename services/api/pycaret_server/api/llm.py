"""LLM advisory routes.

Six endpoints under ``/api/v1/``:

  GET   /workspaces/{id}/llm/settings          current active setting
  PUT   /workspaces/{id}/llm/settings          upsert the enabled provider
  POST  /workspaces/{id}/llm/test-connection   round-trip probe
  POST  /llm/analyze-dataset                   dataset consultant
  GET   /workspaces/{id}/llm/consultations     audit history
  GET   /llm/consultations/{id}                single consultation

Any workspace member can read settings / run consultations. Writing settings
requires admin (gated by ``_require_admin``).
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from pycaret_server.api.schemas import SetupStatusResponse  # noqa: F401 — keeps import graph stable
from pycaret_server.api.workspaces import _require_access, _require_admin
from pycaret_server.auth import CurrentUser
from pycaret_server.db import DataSource, LLMConsultation, LLMProviderSetting, Workspace, get_db
from pycaret_server.llm.consultations import dataset_analysis
from pycaret_server.llm.router import ConsultationContext, NoLLMConfigured, get_router
from pycaret_server.llm.schemas import (
    PROVIDERS,
    AnalyzeDatasetRequest,
    LLMConsultationRead,
    LLMProviderSettingRead,
    LLMProviderSettingWrite,
    TestConnectionResponse,
)

router = APIRouter(tags=["llm"])


# ---------------------------------------------------------------- helpers


def _serialise_setting(row: LLMProviderSetting) -> LLMProviderSettingRead:
    return LLMProviderSettingRead(
        id=row.id,
        workspace_id=row.workspace_id,
        provider=row.provider,
        base_url=row.base_url,
        model_name=row.model_name,
        enabled=row.enabled,
        config=dict(row.config) if row.config else None,
        has_api_key=bool(row.api_key_encrypted),
        created_at=row.created_at,
        created_by=row.created_by,
    )


def _serialise_consultation(row: LLMConsultation) -> LLMConsultationRead:
    return LLMConsultationRead(
        id=row.id,
        workspace_id=row.workspace_id,
        project_id=row.project_id,
        experiment_id=row.experiment_id,
        run_id=row.run_id,
        type=row.type,
        provider=row.provider,
        model_name=row.model_name,
        prompt=row.prompt,
        response_json=dict(row.response_json or {}),
        generated_config_json=dict(row.generated_config_json)
        if row.generated_config_json
        else None,
        latency_ms=row.latency_ms,
        error=row.error,
        created_at=row.created_at,
        created_by=row.created_by,
    )


# ------------------------------------------------------------ settings CRUD


@router.get(
    "/workspaces/{workspace_id}/llm/settings",
    response_model=LLMProviderSettingRead | None,
)
def get_settings(
    workspace_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> LLMProviderSettingRead | None:
    """Return the currently-enabled LLM setting for the workspace, or null."""
    if db.get(Workspace, workspace_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "workspace not found")
    _require_access(user, db, workspace_id)
    row = get_router().get_active_setting(db, workspace_id)
    return _serialise_setting(row) if row else None


@router.put(
    "/workspaces/{workspace_id}/llm/settings",
    response_model=LLMProviderSettingRead,
)
def upsert_settings(
    workspace_id: str,
    payload: LLMProviderSettingWrite,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> LLMProviderSettingRead:
    """Create or update the workspace's provider setting. Admin-gated.

    Uniqueness is on (workspace_id, provider) — switching providers creates a
    new row; the previous provider's row is flipped to `enabled=False` so we
    retain the audit trail.
    """
    if db.get(Workspace, workspace_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "workspace not found")
    _require_admin(user, db, workspace_id)

    if payload.provider not in PROVIDERS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"provider must be one of {list(PROVIDERS)}",
        )

    # Disable any other enabled rows in this workspace.
    for other in db.scalars(
        select(LLMProviderSetting).where(
            LLMProviderSetting.workspace_id == workspace_id,
            LLMProviderSetting.enabled.is_(True),
        )
    ).all():
        if other.provider != payload.provider:
            other.enabled = False

    row = db.scalar(
        select(LLMProviderSetting).where(
            LLMProviderSetting.workspace_id == workspace_id,
            LLMProviderSetting.provider == payload.provider,
        )
    )
    if row is None:
        row = LLMProviderSetting(
            workspace_id=workspace_id,
            provider=payload.provider,
            api_key_encrypted=payload.api_key,
            base_url=payload.base_url,
            model_name=payload.model_name,
            enabled=payload.enabled,
            config=dict(payload.config) if payload.config else None,
            created_by=user.id,
        )
        db.add(row)
    else:
        # Preserve existing key if caller passed no new one (PUT-merge).
        if payload.api_key is not None:
            row.api_key_encrypted = payload.api_key
        row.base_url = payload.base_url
        row.model_name = payload.model_name
        row.enabled = payload.enabled
        row.config = dict(payload.config) if payload.config else None
    db.commit()
    db.refresh(row)
    return _serialise_setting(row)


@router.post(
    "/workspaces/{workspace_id}/llm/test-connection",
    response_model=TestConnectionResponse,
)
def test_connection(
    workspace_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> TestConnectionResponse:
    """Verify the workspace's configured provider + API key actually work."""
    if db.get(Workspace, workspace_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "workspace not found")
    _require_access(user, db, workspace_id)

    row = get_router().get_active_setting(db, workspace_id)
    if row is None:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "no LLM provider configured + enabled for this workspace",
        )
    ok, err, latency = get_router().test_connection(row)
    return TestConnectionResponse(
        ok=ok, provider=row.provider, model_name=row.model_name, error=err, latency_ms=latency
    )


# ------------------------------------------------------------ consultations


@router.post(
    "/llm/analyze-dataset",
    response_model=LLMConsultationRead,
)
def analyze_dataset(
    payload: AnalyzeDatasetRequest,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> LLMConsultationRead:
    """Run the dataset-consultant advisory for one ``csv_upload`` DataSource."""
    if db.get(Workspace, payload.workspace_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "workspace not found")
    _require_access(user, db, payload.workspace_id)

    ds = db.get(DataSource, payload.data_source_id)
    if ds is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "data source not found")
    if ds.workspace_id != payload.workspace_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "workspace_id mismatch")
    if ds.kind != "csv_upload":
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"only csv_upload data sources are supported (got {ds.kind!r}).",
        )
    csv_path = (ds.config or {}).get("path")
    if not csv_path:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "data source missing 'path' in config",
        )

    try:
        system, user_prompt = dataset_analysis.build_prompt(csv_path, payload.task_type_hint)
    except FileNotFoundError as exc:
        raise HTTPException(status.HTTP_410_GONE, str(exc)) from exc

    ctx = ConsultationContext(
        workspace_id=payload.workspace_id,
        user_id=user.id,
        consultation_type="dataset_analysis",
        system=system,
        user=user_prompt,
        output_schema=dataset_analysis.OUTPUT_SCHEMA,
    )
    try:
        _advice, row = get_router().consult(db, ctx)
    except NoLLMConfigured as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except RuntimeError as exc:
        # Audit row was already persisted — surface the error to the caller but
        # don't 500, since the failure is in an *advisory* subsystem.
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
    return _serialise_consultation(row)


@router.get(
    "/workspaces/{workspace_id}/llm/consultations",
    response_model=list[LLMConsultationRead],
)
def list_consultations(
    workspace_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
    limit: int = 50,
) -> list[LLMConsultationRead]:
    """Recent consultations for a workspace (newest first)."""
    if db.get(Workspace, workspace_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "workspace not found")
    _require_access(user, db, workspace_id)
    rows = db.scalars(
        select(LLMConsultation)
        .where(LLMConsultation.workspace_id == workspace_id)
        .order_by(LLMConsultation.created_at.desc())
        .limit(max(1, min(limit, 500)))
    ).all()
    return [_serialise_consultation(r) for r in rows]


@router.get("/llm/consultations/{consultation_id}", response_model=LLMConsultationRead)
def get_consultation(
    consultation_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> LLMConsultationRead:
    row = db.get(LLMConsultation, consultation_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "consultation not found")
    _require_access(user, db, row.workspace_id)
    return _serialise_consultation(row)
