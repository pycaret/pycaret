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
from pycaret_server.crypto import encrypt as _encrypt_secret
from pycaret_server.db import (
    DataSource,
    Deployment,
    DriftReport,
    Event,
    Experiment,
    LLMConsultation,
    LLMProviderSetting,
    Pipeline,
    Project,
    Run,
    Workspace,
    get_db,
)
from pycaret_server.llm.consultations import (
    dataset_analysis,
    deployment_risk_review,
    drift_analysis,
    experiment_design,
    failure_debugging,
    run_explanation,
)
from pycaret_server.llm.router import ConsultationContext, NoLLMConfigured, get_router
from pycaret_server.llm.schemas import (
    PROVIDERS,
    AnalyzeDatasetRequest,
    AnalyzeDriftRequest,
    DebugRunRequest,
    DesignExperimentRequest,
    ExplainRunRequest,
    LLMConsultationRead,
    LLMProviderSettingRead,
    LLMProviderSettingWrite,
    ReviewDeploymentRequest,
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
            api_key_encrypted=_encrypt_secret(payload.api_key) if payload.api_key else None,
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
            row.api_key_encrypted = _encrypt_secret(payload.api_key)
        row.base_url = payload.base_url
        row.model_name = payload.model_name
        row.enabled = payload.enabled
        row.config = dict(payload.config) if payload.config else None
    db.commit()
    db.refresh(row)
    return _serialise_setting(row)


@router.delete(
    "/workspaces/{workspace_id}/llm/settings",
    status_code=status.HTTP_204_NO_CONTENT,
)
def clear_settings(
    workspace_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> None:
    """Remove the workspace's stored LLM provider row (key + config + all).

    Idempotent: 204 even if nothing was stored. Used by the UI's "Clear" button
    so users can fully remove an API key without entering a replacement first.
    """
    if db.get(Workspace, workspace_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "workspace not found")
    _require_admin(user, db, workspace_id)
    rows = db.scalars(
        select(LLMProviderSetting).where(
            LLMProviderSetting.workspace_id == workspace_id
        )
    ).all()
    for row in rows:
        db.delete(row)
    db.commit()


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


@router.post(
    "/llm/design-experiment",
    response_model=LLMConsultationRead,
)
def design_experiment(
    payload: DesignExperimentRequest,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> LLMConsultationRead:
    """Suggest a full RunConfig from a dataset + a plain-English goal.

    Reads the CSV profile (same columns + sample-values shape as
    `analyze-dataset`) and asks the LLM to propose task type, target,
    metric, preprocessing, and a model shortlist. User approves + copies
    into the New Experiment wizard — no one-click apply in v1 (waits on
    the canonical `RunConfig` Pydantic model, MVP-1 exit criterion).
    """
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
            status.HTTP_500_INTERNAL_SERVER_ERROR, "data source missing 'path' in config"
        )

    biz_ctx = payload.business_context if payload.include_business_context else None
    try:
        system, user_prompt = experiment_design.build_prompt(
            csv_path, payload.goal, business_context=biz_ctx
        )
    except FileNotFoundError as exc:
        raise HTTPException(status.HTTP_410_GONE, str(exc)) from exc

    ctx = ConsultationContext(
        workspace_id=payload.workspace_id,
        user_id=user.id,
        consultation_type="experiment_design",
        system=system,
        user=user_prompt,
        output_schema=experiment_design.OUTPUT_SCHEMA,
    )
    try:
        _advice, row = get_router().consult(db, ctx)
    except NoLLMConfigured as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
    return _serialise_consultation(row)


@router.post(
    "/llm/explain-run",
    response_model=LLMConsultationRead,
)
def explain_run(
    payload: ExplainRunRequest,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> LLMConsultationRead:
    """Explain a completed run + suggest next experiments.

    Reads the Run's snapshot + leaderboard + the tail of its event stream,
    asks the LLM for a plain-prose explanation + concrete next actions.
    Guarded: only runs in terminal states can be explained.
    """
    run = db.get(Run, payload.run_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "run not found")

    # Access control traverses run → experiment → project → workspace.
    exp = db.get(Experiment, run.experiment_id)
    if exp is None:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "run has no experiment — data integrity issue",
        )
    proj = db.get(Project, exp.project_id)
    workspace_id = proj.workspace_id if proj else None
    if workspace_id is None:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "run has no workspace — data integrity issue",
        )
    _require_access(user, db, workspace_id)

    if run.status not in ("succeeded", "failed", "cancelled"):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"run is in state {run.status!r}; wait for a terminal state before explaining.",
        )

    # Pull the run's event stream for context. Bounded to the first + last 50
    # in the consultation module; we send everything here.
    events = [
        {
            "kind": e.kind,
            "message": e.message,
            "payload": dict(e.payload) if e.payload else {},
            "duration_ms": e.duration_ms,
            "emitted_at": e.emitted_at.isoformat() if e.emitted_at else None,
        }
        for e in db.scalars(
            select(Event).where(Event.run_id == run.id).order_by(Event.emitted_at.asc())
        ).all()
    ]
    leaderboard = run.leaderboard if isinstance(run.leaderboard, list) else None

    system, user_prompt = run_explanation.build_prompt(
        run_snapshot=run.snapshot,
        status=run.status,
        duration_ms=run.duration_ms,
        leaderboard=leaderboard,
        events=events,
        error=run.error,
    )

    ctx = ConsultationContext(
        workspace_id=workspace_id,
        user_id=user.id,
        consultation_type="run_summary",
        system=system,
        user=user_prompt,
        output_schema=run_explanation.OUTPUT_SCHEMA,
        project_id=exp.project_id,
        experiment_id=exp.id,
        run_id=run.id,
    )
    try:
        _advice, row = get_router().consult(db, ctx)
    except NoLLMConfigured as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
    return _serialise_consultation(row)


@router.post(
    "/llm/debug-run",
    response_model=LLMConsultationRead,
)
def debug_run(
    payload: DebugRunRequest,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> LLMConsultationRead:
    """Diagnose a failed run's error + event tail.

    Only runs in ``status='failed'`` can be debugged; succeeded runs get
    `explain-run`, in-flight runs have nothing to debug yet.
    """
    run = db.get(Run, payload.run_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "run not found")
    exp = db.get(Experiment, run.experiment_id)
    proj = db.get(Project, exp.project_id) if exp else None
    workspace_id = proj.workspace_id if proj else None
    if workspace_id is None:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "run has no workspace — data integrity issue",
        )
    _require_access(user, db, workspace_id)

    if run.status != "failed":
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"debug is for failed runs only; run is {run.status!r}. "
            "Use /llm/explain-run for succeeded runs.",
        )

    events = [
        {
            "kind": e.kind,
            "message": e.message,
            "payload": dict(e.payload) if e.payload else {},
            "duration_ms": e.duration_ms,
            "emitted_at": e.emitted_at.isoformat() if e.emitted_at else None,
        }
        for e in db.scalars(
            select(Event).where(Event.run_id == run.id).order_by(Event.emitted_at.asc())
        ).all()
    ]

    system, user_prompt = failure_debugging.build_prompt(
        run_snapshot=run.snapshot,
        error=run.error,
        events=events,
    )
    ctx = ConsultationContext(
        workspace_id=workspace_id,
        user_id=user.id,
        consultation_type="failure_debugging",
        system=system,
        user=user_prompt,
        output_schema=failure_debugging.OUTPUT_SCHEMA,
        project_id=exp.project_id if exp else None,
        experiment_id=exp.id if exp else None,
        run_id=run.id,
    )
    try:
        _advice, row = get_router().consult(db, ctx)
    except NoLLMConfigured as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
    return _serialise_consultation(row)


@router.post(
    "/llm/review-deployment",
    response_model=LLMConsultationRead,
)
def review_deployment(
    payload: ReviewDeploymentRequest,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> LLMConsultationRead:
    """Pre-deploy safety review for a Pipeline.

    Reads the Pipeline + its origin Run snapshot + leaderboard, asks the
    LLM to flag overfit / tiny-margin / missing-preprocessing risks, and
    deliver a verdict (APPROVE / APPROVE WITH CAVEATS / DO NOT DEPLOY).
    Advisory: the user still decides.
    """
    pipeline = db.get(Pipeline, payload.pipeline_id)
    if pipeline is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "pipeline not found")
    _require_access(user, db, pipeline.workspace_id)

    origin_run = db.get(Run, pipeline.origin_run_id) if pipeline.origin_run_id else None
    pipeline_payload = {
        "id": pipeline.id,
        "name": pipeline.name,
        "description": pipeline.description,
        "tags": list(pipeline.tags or []),
        "model_id": pipeline.model_id,
        "sha256": pipeline.sha256,
        "origin_run_id": pipeline.origin_run_id,
    }
    leaderboard = (
        origin_run.leaderboard if origin_run and isinstance(origin_run.leaderboard, list) else None
    )

    system, user_prompt = deployment_risk_review.build_prompt(
        pipeline=pipeline_payload,
        origin_run_snapshot=origin_run.snapshot if origin_run else None,
        leaderboard=leaderboard,
        origin_run_status=origin_run.status if origin_run else None,
    )
    ctx = ConsultationContext(
        workspace_id=pipeline.workspace_id,
        user_id=user.id,
        consultation_type="deployment_risk_review",
        system=system,
        user=user_prompt,
        output_schema=deployment_risk_review.OUTPUT_SCHEMA,
        run_id=pipeline.origin_run_id,
    )
    try:
        _advice, row = get_router().consult(db, ctx)
    except NoLLMConfigured as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
    return _serialise_consultation(row)


@router.post(
    "/llm/analyze-drift",
    response_model=LLMConsultationRead,
)
def analyze_drift(
    payload: AnalyzeDriftRequest,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> LLMConsultationRead:
    """Interpret a drift report + recommend an action.

    Reads the DriftReport + its owning Deployment + (if linked) the Pipeline
    snapshot, asks the LLM to classify drift severity and name specific
    features driving it. Verdict: RETRAIN NOW / INVESTIGATE / MONITOR /
    NO ACTION. Advisory — the user still decides.
    """
    report = db.get(DriftReport, payload.drift_report_id)
    if report is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "drift report not found")
    dep = db.get(Deployment, report.deployment_id)
    if dep is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "owning deployment missing")
    _require_access(user, db, dep.workspace_id)

    pipeline = db.get(Pipeline, dep.pipeline_id) if dep.pipeline_id else None

    report_payload = {
        "id": report.id,
        "window_start": report.window_start.isoformat(),
        "window_end": report.window_end.isoformat(),
        "drift_score": report.drift_score,
        "drift_status": report.drift_status,
        "feature_drift": dict(report.feature_drift_json or {}),
        "prediction_drift": dict(report.prediction_drift_json)
        if report.prediction_drift_json
        else None,
        "sample_size": report.sample_size,
    }
    deployment_payload = {
        "id": dep.id,
        "endpoint_slug": dep.endpoint_slug,
        "status": dep.status,
        "inference_count": dep.inference_count,
        "last_inference_at": dep.last_inference_at.isoformat() if dep.last_inference_at else None,
        "p50_latency_ms": dep.p50_latency_ms,
        "p95_latency_ms": dep.p95_latency_ms,
        "error_count": dep.error_count,
    }
    pipeline_payload: dict | None = None
    if pipeline is not None:
        pipeline_payload = {
            "id": pipeline.id,
            "name": pipeline.name,
            "model_id": pipeline.model_id,
            "tags": list(pipeline.tags or []),
        }

    system, user_prompt = drift_analysis.build_prompt(
        drift_report=report_payload,
        deployment=deployment_payload,
        pipeline=pipeline_payload,
    )
    ctx = ConsultationContext(
        workspace_id=dep.workspace_id,
        user_id=user.id,
        consultation_type="drift_analysis",
        system=system,
        user=user_prompt,
        output_schema=drift_analysis.OUTPUT_SCHEMA,
    )
    try:
        _advice, row = get_router().consult(db, ctx)
    except NoLLMConfigured as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except RuntimeError as exc:
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
