"""Reusable run-dispatch helper.

Extracted from ``api/runs.py:submit_run`` so the scheduler (and any future
programmatic enqueue path) can submit a Run without going through HTTP.

Single entry point:

    from pycaret_server.runs.dispatch import dispatch_run
    run = dispatch_run(db, experiment, payload, user_id=...)

Same validations, same orchestrator handoff, same Run row shape.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, status
from pycaret.core.tasks import TaskType
from sqlalchemy.orm import Session

from pycaret_server.api.schemas import RunCreate
from pycaret_server.db import DataSource, Experiment, Project, Run
from pycaret_server.runs.orchestrator import RunSpec, get_orchestrator


def dispatch_run(
    db: Session,
    experiment: Experiment,
    payload: RunCreate,
    *,
    user_id: str,
) -> Run:
    """Validate ``payload``, persist a queued ``Run``, and submit to the orchestrator.

    Raises ``HTTPException`` on validation failure (so route handlers can
    bubble up unchanged). Schedule-driven callers can catch and convert.
    """
    if payload.plan not in ("setup", "create", "compare", "search"):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"invalid plan {payload.plan!r}; must be one of setup|create|compare|search",
        )
    if payload.plan == "create" and not payload.model_id:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "plan='create' requires model_id",
        )
    if not (payload.sklearn_dataset or payload.data_inline or payload.data_source_id):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "must supply sklearn_dataset, data_inline, or data_source_id",
        )

    data_source_path: str | None = None
    if payload.data_source_id:
        ds = db.get(DataSource, payload.data_source_id)
        if ds is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "data source not found")
        project_ws = db.get(Project, experiment.project_id).workspace_id
        if ds.workspace_id != project_ws:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "data source belongs to a different workspace than the experiment",
            )
        if ds.kind != "csv_upload":
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"data source kind {ds.kind!r} not yet supported for runs",
            )
        data_source_path = (ds.config or {}).get("path")
        if not data_source_path:
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "data source missing 'path' in config",
            )

    effective_target = payload.target or experiment.target

    snapshot: dict[str, Any] = {
        "task": experiment.task,
        "target": effective_target,
        "setup_params": dict(experiment.setup_params or {}),
        "plan": payload.plan,
        "model_id": payload.model_id,
        "plan_params": dict(payload.plan_params or {}),
        "sklearn_dataset": payload.sklearn_dataset,
        "data_inline_rows": len(payload.data_inline) if payload.data_inline else 0,
        "data_source_id": payload.data_source_id,
    }

    r = Run(
        experiment_id=experiment.id,
        status="queued",
        created_by=user_id,
        snapshot=snapshot,
    )
    db.add(r)
    db.commit()
    db.refresh(r)

    spec = RunSpec(
        run_id=r.id,
        experiment_id=experiment.id,
        task=TaskType(experiment.task),
        target=effective_target,
        setup_params=dict(experiment.setup_params or {}),
        plan=payload.plan,  # type: ignore[arg-type]
        model_id=payload.model_id,
        plan_params=dict(payload.plan_params or {}),
        sklearn_dataset=payload.sklearn_dataset,
        data_inline=list(payload.data_inline) if payload.data_inline else None,
        data_source_path=data_source_path,
        target_override=payload.target,
    )
    get_orchestrator().submit(spec)
    return r
