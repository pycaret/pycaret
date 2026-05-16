"""Phase 0-v2 run dispatcher.

Replaces the legacy ``Trial-owns-Run`` dispatcher. Now:

- Compare runs are *decomposed* into N Trial rows + N Trial-Jobs (one
  per algorithm). Workers run each ``create_model(model_id)``
  independently and stamp their Trial row. The Run's status is
  derived from its Trials' statuses.
- ``create`` runs spawn a single Trial + a single Trial-Job.
- ``search`` (hyperparameter search) runs spawn a single Trial of
  kind=``tuned`` + a single Trial-Job (uses the engine's search-plan
  worker handler).
- ``setup`` runs don't produce Trials; they're a single sanity-check
  Job that runs setup() and exits.

The orchestrator no longer dispatches the Run wholesale — that legacy
path stays only as the *fallback* for ``setup`` plans (the only kind
that doesn't decompose into Trials).
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import HTTPException, status
from pycaret.core.tasks import TaskType
from sqlalchemy.orm import Session

from pycaret_server.api.schemas import RunCreate
from pycaret_server.config import get_settings
from pycaret_server.db import DataSource, Experiment, Job, Project, Run, Trial
from pycaret_server.runs.decompose import model_ids_for_compare

_log = logging.getLogger(__name__)


def dispatch_run(
    db: Session,
    experiment: Experiment,
    payload: RunCreate,
    *,
    user_id: str,
    triggered_by: str | None = None,
    triggered_by_id: str | None = None,
) -> Run:
    """Validate ``payload``, persist a queued ``Run`` + its Trials +
    their Jobs, and submit. Returns the Run row.

    Raises ``HTTPException`` on validation failure (route handlers
    bubble it up; the scheduler converts).
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
        "data_inline": list(payload.data_inline) if payload.data_inline else None,
        "data_inline_rows": len(payload.data_inline) if payload.data_inline else 0,
        "data_source_id": payload.data_source_id,
        "data_source_path": data_source_path,
        "triggered_by": triggered_by or "user",
        "triggered_by_id": triggered_by_id,
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

    # Lineage: experiment → run, and data_source → run when one's
    # specified. Best-effort — failing to write an edge doesn't fail
    # the dispatch.
    try:
        from pycaret_server.api.connections import record_lineage

        proj = db.get(Project, experiment.project_id)
        ws_id = proj.workspace_id if proj else None
        record_lineage(
            db,
            workspace_id=ws_id,
            source_kind="experiment",
            source_id=experiment.id,
            target_kind="run",
            target_id=r.id,
            relation="dispatched",
            metadata={"plan": payload.plan, "triggered_by": triggered_by or "user"},
        )
        if payload.data_source_id:
            record_lineage(
                db,
                workspace_id=ws_id,
                source_kind="data_source",
                source_id=payload.data_source_id,
                target_kind="run",
                target_id=r.id,
                relation="trained_on",
                metadata={"plan": payload.plan},
            )
    except Exception:  # noqa: BLE001
        pass

    # ──────────────── decompose into Trial rows + Trial-Jobs ────────
    if payload.plan == "compare":
        plan_params = dict(payload.plan_params or {})
        include = plan_params.get("include_models") or plan_params.get("include")
        exclude = plan_params.get("exclude_models") or plan_params.get("exclude")
        model_ids = model_ids_for_compare(
            experiment.task, include=include, exclude=exclude
        )
        if not model_ids:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"no compare model_ids resolved for task={experiment.task!r}",
            )
        _spawn_trials(db, r, experiment, model_ids, snapshot, user_id=user_id, kind="compare")
    elif payload.plan == "create":
        _spawn_trials(
            db, r, experiment, [str(payload.model_id)], snapshot, user_id=user_id, kind="manual"
        )
    elif payload.plan == "search":
        # Search decomposes into a variant × model grid. Each cell
        # gets its own Trial-Job (kind=tuned) that re-fits the
        # experiment with the variant's setup overrides and calls
        # create_model(model_id).
        plan_params = dict(payload.plan_params or {})
        variants = list(plan_params.get("variants") or [{}])
        compare_params = dict(plan_params.get("compare_params") or {})
        include = (
            compare_params.get("include")
            or plan_params.get("include_models")
            or model_ids_for_compare(experiment.task)
        )
        _spawn_search_trials(
            db,
            r,
            experiment,
            variants,
            list(include),
            snapshot,
            user_id=user_id,
        )
    elif payload.plan == "setup":
        # Setup only — no Trials produced; the legacy in-process
        # orchestrator runs it directly to confirm the experiment
        # config validates against the data.
        _submit_legacy_setup(r, experiment, payload, data_source_path, snapshot)

    return r


def _spawn_trials(
    db: Session,
    run: Run,
    experiment: Experiment,
    model_ids: list[str],
    snapshot: dict[str, Any],
    *,
    user_id: str,
    kind: str,
) -> None:
    """Persist one Trial + one ``kind="trial"`` Job per model id."""
    project = db.get(Project, experiment.project_id)
    if project is None:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "experiment's project row missing",
        )
    workspace_id = project.workspace_id
    settings = get_settings()
    action_id = str(uuid.uuid4())

    plan_params = snapshot.get("plan_params") or {}
    setup_params = experiment.setup_params or {}
    # Auto-route to the gpu queue when the experiment opted into GPU
    # mode and the caller didn't pin a queue. The worker pool decides
    # whether a given worker can actually claim a "gpu" Job; this is
    # just a hint so the dispatch path doesn't need to know.
    explicit_queue = plan_params.get("queue") or setup_params.get("queue")
    use_gpu = bool(
        snapshot.get("use_gpu")
        or setup_params.get("use_gpu")
        or plan_params.get("use_gpu")
    )
    queue_hint = explicit_queue or ("gpu" if use_gpu else "default")
    queue_name = (
        str(queue_hint)
        if queue_hint in ("default", "cpu-heavy", "gpu", "inference")
        else "default"
    )
    requested_resources = {"gpu": 1} if queue_name == "gpu" else None

    trials: list[Trial] = []
    for model_id in model_ids:
        t = Trial(
            run_id=run.id,
            experiment_id=experiment.id,
            workspace_id=workspace_id,
            name=model_id,
            model_id=model_id,
            kind=kind,
            status="queued",
            is_best=False,
            created_by_action_id=action_id,
        )
        db.add(t)
        trials.append(t)
    db.commit()
    for t in trials:
        db.refresh(t)

    # Lineage: run → trial for each candidate.
    try:
        from pycaret_server.api.connections import record_lineage

        for t in trials:
            record_lineage(
                db,
                workspace_id=workspace_id,
                source_kind="run",
                source_id=run.id,
                target_kind="trial",
                target_id=t.id,
                relation="contains",
                metadata={"model_id": t.model_id, "kind": kind},
            )
    except Exception:  # noqa: BLE001
        pass

    # Strip compare-only knobs from plan_params before each Trial-Job
    # inherits them — ``include_models`` / ``exclude_models`` /
    # ``n_select`` belong on the dispatcher's decompose step, not the
    # engine's ``create_model`` call.
    trial_plan_params = {
        k: v
        for k, v in (plan_params or {}).items()
        if k
        not in {
            "include_models",
            "exclude_models",
            "include",
            "exclude",
            "n_select",
            "queue",
        }
    }
    trial_snapshot = {**snapshot, "plan_params": trial_plan_params}

    # One Job per Trial. Payload mirrors the snapshot but stamps the
    # specific model_id this Trial is responsible for.
    jobs: list[Job] = []
    for t in trials:
        job_payload = {
            **trial_snapshot,
            "run_id": run.id,
            "trial_id": t.id,
            "experiment_id": experiment.id,
            "model_id": t.model_id,
            "trial_kind": kind,
        }
        job = Job(
            kind="trial",
            status="queued",
            run_id=run.id,
            payload=job_payload,
            queue=queue_name,
            correlation_id=t.id,
            requested_resources=requested_resources,
        )
        db.add(job)
        jobs.append(job)
    db.commit()
    for j in jobs:
        db.refresh(j)

    if settings.runs_backend == "redis":
        # Try to enqueue. On failure, fall back to the in-process
        # executor — dev workflows keep working without Redis.
        try:
            from pycaret_server.runs.queue_redis import enqueue_job

            for j in jobs:
                enqueue_job(j.id, queue=j.queue, redis_url=settings.redis_url)
            return
        except Exception:  # noqa: BLE001
            _log.warning("redis backend selected but enqueue failed; running in-process")

    # In-process fallback: submit each Trial-Job to the orchestrator's
    # ThreadPoolExecutor. They run in parallel up to max_workers.
    from pycaret_server.runs.orchestrator import get_orchestrator

    for j in jobs:
        get_orchestrator().submit_trial_job(j.id)


def _spawn_search_trials(
    db: Session,
    run: Run,
    experiment: Experiment,
    variants: list[dict[str, Any]],
    include: list[str],
    snapshot: dict[str, Any],
    *,
    user_id: str,
) -> None:
    """Decompose a ``search`` Run into a variant × model grid.

    Each (variant, model) combination becomes one Trial of kind
    ``tuned``. The variant's setup-param override rides on the Job
    payload so the worker can merge it with the experiment's
    baseline setup before calling ``create_model``.
    """
    project = db.get(Project, experiment.project_id)
    if project is None:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "experiment's project row missing",
        )
    workspace_id = project.workspace_id
    settings = get_settings()
    action_id = str(uuid.uuid4())

    # Same GPU routing rule as the compare path: any variant that flips
    # ``use_gpu`` on in the override pushes the whole search onto the
    # ``gpu`` queue. Mixed search across GPU + CPU variants isn't
    # supported in this cut — pin a queue via plan_params.queue if you
    # need it.
    base_setup = experiment.setup_params or {}
    plan_params = snapshot.get("plan_params") or {}
    explicit_queue = plan_params.get("queue") or base_setup.get("queue")
    use_gpu = (
        bool(base_setup.get("use_gpu"))
        or bool(plan_params.get("use_gpu"))
        or any(bool((v or {}).get("use_gpu")) for v in variants)
    )
    queue_hint = explicit_queue or ("gpu" if use_gpu else "default")
    search_queue = (
        str(queue_hint)
        if queue_hint in ("default", "cpu-heavy", "gpu", "inference")
        else "default"
    )
    search_requested_resources = {"gpu": 1} if search_queue == "gpu" else None

    trials: list[Trial] = []
    job_payloads: list[dict[str, Any]] = []
    for variant_idx, override in enumerate(variants):
        for model_id in include:
            name = f"variant-{variant_idx}-{model_id}"
            t = Trial(
                run_id=run.id,
                experiment_id=experiment.id,
                workspace_id=workspace_id,
                name=name,
                model_id=str(model_id),
                kind="tuned",
                status="queued",
                is_best=False,
                created_by_action_id=action_id,
            )
            db.add(t)
            trials.append(t)
            payload = {
                **snapshot,
                "run_id": run.id,
                "experiment_id": experiment.id,
                "model_id": str(model_id),
                "trial_kind": "tuned",
                "variant_index": variant_idx,
                "variant_override": dict(override or {}),
            }
            job_payloads.append(payload)
    db.commit()
    for t in trials:
        db.refresh(t)

    # Lineage: run → trial for each (variant, model) candidate.
    try:
        from pycaret_server.api.connections import record_lineage

        for t in trials:
            record_lineage(
                db,
                workspace_id=workspace_id,
                source_kind="run",
                source_id=run.id,
                target_kind="trial",
                target_id=t.id,
                relation="contains",
                metadata={"model_id": t.model_id, "kind": "tuned"},
            )
    except Exception:  # noqa: BLE001
        pass

    jobs: list[Job] = []
    for t, p in zip(trials, job_payloads, strict=False):
        p_stamped = {**p, "trial_id": t.id}
        # Strip compare-only knobs from the per-Trial plan_params.
        if "plan_params" in p_stamped:
            pp = dict(p_stamped["plan_params"])
            for k in ("variants", "compare_params", "include_models", "exclude_models", "include", "exclude"):
                pp.pop(k, None)
            p_stamped["plan_params"] = pp
        job = Job(
            kind="trial",
            status="queued",
            run_id=run.id,
            payload=p_stamped,
            queue=search_queue,
            correlation_id=t.id,
            requested_resources=search_requested_resources,
        )
        db.add(job)
        jobs.append(job)
    db.commit()
    for j in jobs:
        db.refresh(j)

    if settings.runs_backend == "redis":
        try:
            from pycaret_server.runs.queue_redis import enqueue_job

            for j in jobs:
                enqueue_job(j.id, queue=j.queue, redis_url=settings.redis_url)
            return
        except Exception:  # noqa: BLE001
            _log.warning("redis backend selected but enqueue failed; running in-process")

    from pycaret_server.runs.orchestrator import get_orchestrator

    for j in jobs:
        get_orchestrator().submit_trial_job(j.id)


def _submit_legacy_setup(
    run: Run,
    experiment: Experiment,
    payload: RunCreate,
    data_source_path: str | None,
    snapshot: dict[str, Any],
) -> None:
    """Setup-only plan — no Trials. Goes through the legacy
    in-process orchestrator for validation (inprocess mode) or a
    ``kind="run"`` Job (redis mode)."""
    from pycaret_server.runs.orchestrator import RunSpec, get_orchestrator

    spec = RunSpec(
        run_id=run.id,
        experiment_id=experiment.id,
        task=TaskType(experiment.task),
        target=payload.target or experiment.target,
        setup_params=dict(experiment.setup_params or {}),
        plan="setup",
        model_id=payload.model_id,
        plan_params=dict(payload.plan_params or {}),
        sklearn_dataset=payload.sklearn_dataset,
        data_inline=list(payload.data_inline) if payload.data_inline else None,
        data_source_path=data_source_path,
        target_override=payload.target,
    )
    get_orchestrator().submit(spec)
    # Best-effort: drop a Job row so the audit trail in inprocess mode
    # mirrors the redis-mode shape.
    _ = snapshot  # accepted for symmetry with the trial path

