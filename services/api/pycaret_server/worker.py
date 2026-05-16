"""Phase 1 worker entrypoint.

Long-running process. Polls Redis for queued Job ids; for each one,
looks the Job up in the DB, dispatches it to the matching handler,
heartbeats while it runs, then marks it terminal.

Launch with:

    pycaret-server worker --queues default --worker-id my-host-1

Or run multiple workers on different hosts pointed at the same Redis +
Postgres. Each worker claims Jobs atomically via the ``locked_by`` /
``locked_at`` columns so two workers never run the same Job.

The handler registry is open: any future Job kind (drift_check, batch_predict)
adds itself to ``_HANDLERS`` without touching this loop.
"""

from __future__ import annotations

import logging
import signal
import threading
import time
import uuid
from datetime import UTC, datetime
from typing import Callable

from sqlalchemy import select

from pycaret_server.config import get_settings
from pycaret_server.db import Job, Run, get_session

_log = logging.getLogger("pycaret.worker")

# Job kind → handler function. Each handler receives the Job row +
# a DB session and is responsible for updating the Job's status.
JobHandler = Callable[[Job, "object"], None]
_HANDLERS: dict[str, JobHandler] = {}


def register_handler(kind: str, fn: JobHandler) -> None:
    """Register a handler for a Job kind. Idempotent — last registration wins."""
    _HANDLERS[kind] = fn


# -------------------------------------------------------- built-in handlers


def _handle_run(job: Job, session) -> None:
    """Run-kind Job: build the RunSpec from the snapshot, hand to the orchestrator,
    block until it finishes, then mark the Job terminal.

    The orchestrator already writes Trial/Run rows and emits events; the
    worker's job is just to wait + reflect status on the Job row.
    """
    from pycaret.core.tasks import TaskType
    from pycaret_server.runs.orchestrator import RunSpec, get_orchestrator

    if not job.run_id:
        raise ValueError(f"job {job.id} kind=run has no run_id")

    run = session.get(Run, job.run_id)
    if run is None:
        raise ValueError(f"run {job.run_id} not found for job {job.id}")

    p = job.payload or {}
    spec = RunSpec(
        run_id=run.id,
        experiment_id=p.get("experiment_id") or run.experiment_id,
        task=TaskType(p.get("task") or (run.snapshot or {}).get("task")),
        target=p.get("target_override") or (run.snapshot or {}).get("target"),
        setup_params=dict(p.get("setup_params") or {}),
        plan=p.get("plan") or "compare",
        model_id=p.get("model_id"),
        plan_params=dict(p.get("plan_params") or {}),
        sklearn_dataset=p.get("sklearn_dataset"),
        data_source_path=p.get("data_source_path"),
        target_override=p.get("target_override"),
    )
    orch = get_orchestrator()
    orch.submit(spec)
    orch.wait_for(run.id)


register_handler("run", _handle_run)


def _handle_trial(job: Job, session) -> None:
    """Phase 0-v2: execute one ``kind="trial"`` Job.

    Delegates to ``runs.orchestrator.execute_trial_job`` so the
    in-process executor and the redis worker share one code path.
    """
    from pycaret_server.runs.orchestrator import execute_trial_job

    execute_trial_job(job.id)


register_handler("trial", _handle_trial)


def _handle_drift_check(job: Job, session) -> None:
    """Phase 9 + 10: evaluate alert rules for a workspace and deliver
    every fired one to its destination.

    ``payload`` carries ``{workspace_id}``. Optionally
    ``{auto_retrain_trial_ids: [...]}`` to chain into Phase 9 retrains.
    """
    from sqlalchemy import select
    from pycaret_server.api.monitoring import evaluate_rules_for_workspace
    from pycaret_server.db import AlertRule
    from pycaret_server.monitoring import deliver_fired_rule

    p = job.payload or {}
    workspace_id = p.get("workspace_id")
    if not workspace_id:
        raise ValueError("drift_check payload missing workspace_id")
    fired = evaluate_rules_for_workspace(session, workspace_id)
    delivery_log: list[dict] = []
    for f in fired:
        ok, err = deliver_fired_rule(f)
        delivery_log.append({"rule_id": f["rule_id"], "ok": ok, "error": err})
        rule = session.get(AlertRule, f["rule_id"])
        if rule is not None:
            rule.last_status = "delivered" if ok else "delivery_failed"
            rule.last_error = err
    if fired:
        session.commit()
    job.payload = {**p, "fired": fired, "deliveries": delivery_log}


register_handler("drift_check", _handle_drift_check)


def _handle_retrain(job: Job, session) -> None:
    """Phase 9 / Phase 0-v2: scheduled retrain — fires a new Run that
    contains one Trial of the same ``model_id`` as the source.

    ``payload`` carries ``{trial_id, triggered_by, triggered_by_id?}``.
    The new Run inherits the source Run's snapshot (data source,
    target, etc.); the resulting Trial back-links via
    ``parent_trial_ids`` so lineage is preserved.
    """
    from pycaret_server.api.schemas import RunCreate
    from pycaret_server.db import Experiment, Run, Trial
    from pycaret_server.runs.dispatch import dispatch_run

    p = job.payload or {}
    trial_id = p.get("trial_id")
    if not trial_id:
        raise ValueError("retrain payload missing trial_id")
    trial = session.get(Trial, trial_id)
    if trial is None or trial.experiment_id is None:
        raise ValueError(f"trial {trial_id} not found or unbound")
    exp = session.get(Experiment, trial.experiment_id)
    if exp is None:
        raise ValueError(f"experiment for trial {trial_id} missing")
    src_run = session.get(Run, trial.run_id) if trial.run_id else None
    snap = dict((src_run.snapshot if src_run else None) or {})
    create_payload = RunCreate(
        plan="create",
        model_id=trial.model_id,
        sklearn_dataset=snap.get("sklearn_dataset"),
        data_source_id=snap.get("data_source_id"),
        target=snap.get("target"),
    )
    # Resolve the actual user to credit on the audit trail. Order of
    # preference: schedule's creator → source Run's creator → source
    # Trial's experiment creator. The Run row's ``triggered_by`` /
    # ``triggered_by_id`` still records that this was an automated fire,
    # so the audit log can render "compare run by alice @ 2026-05-12 —
    # triggered by drift rule X".
    from pycaret_server.db import ScheduledJob

    actor_user_id: str | None = None
    tb_id = p.get("triggered_by_id")
    if tb_id:
        sched = session.get(ScheduledJob, tb_id)
        if sched is not None and sched.created_by:
            actor_user_id = sched.created_by
    if actor_user_id is None and src_run is not None and src_run.created_by:
        actor_user_id = src_run.created_by
    if actor_user_id is None:
        actor_user_id = exp.created_by
    dispatch_run(
        session,
        exp,
        create_payload,
        user_id=actor_user_id,
        triggered_by=str(p.get("triggered_by") or "drift"),
        triggered_by_id=tb_id,
    )


register_handler("retrain", _handle_retrain)


def _handle_batch_predict(job: Job, session) -> None:
    """Phase 9: batch-predict.

    ``payload`` carries ``{deployment_id, input_data_source_id}``.
    Steps:

    1. Resolve the deployment + its referenced Pipeline (or
       RegisteredModelVersion).
    2. Load the input dataset via its driver's ``read_full``.
    3. Run ``pipeline.predict`` on the frame.
    4. Stream the predictions as a CSV to the ObjectStore under
       ``runs/batch_predict/<job_id>.csv``.
    5. Write a lineage edge ``data_source → deployment`` plus a
       follow-on ``deployment → predictions_uri`` annotation.
    """
    import csv
    import io

    from pycaret_server.api.connections import record_lineage
    from pycaret_server.datasources import get_driver
    from pycaret_server.db import (
        DataSource,
        Deployment,
        Pipeline,
        RegisteredModelVersion,
    )
    from pycaret_server.storage import get_object_store

    p = job.payload or {}
    deployment_id = p.get("deployment_id")
    input_ds_id = p.get("input_dataset_id") or p.get("input_data_source_id")
    if not deployment_id or not input_ds_id:
        raise ValueError(
            "batch_predict payload requires deployment_id + input_data_source_id"
        )
    d = session.get(Deployment, deployment_id)
    if d is None:
        raise ValueError(f"deployment {deployment_id} not found")
    ds = session.get(DataSource, input_ds_id)
    if ds is None:
        raise ValueError(f"input data source {input_ds_id} not found")

    # Resolve the artifact URI to load. Phase 7 RegisteredModelVersion
    # wins when set; otherwise fall back to the legacy Pipeline.
    artifact_uri: str | None = None
    if d.registered_model_version_id:
        v = session.get(RegisteredModelVersion, d.registered_model_version_id)
        artifact_uri = v.stored_path if v else None
    if artifact_uri is None and d.pipeline_id:
        pipe = session.get(Pipeline, d.pipeline_id)
        artifact_uri = pipe.stored_path if pipe else None
    if artifact_uri is None:
        raise ValueError(
            f"deployment {deployment_id} has no resolvable artifact"
        )

    # Load the model into memory.
    store = get_object_store()
    try:
        blob = store.get_bytes(artifact_uri)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"could not fetch artifact: {exc}") from exc
    try:
        import cloudpickle as _pkl
    except ImportError:
        import pickle as _pkl  # type: ignore[no-redef]
    model = _pkl.loads(blob)

    # Read the input data. Resolve secrets via the back-linked
    # Connection (if any) so credentialed sources work in the worker.
    from pycaret_server.api.connections import resolve_datasource_secret

    driver = get_driver(ds.kind)
    df = driver.read_full(
        config=dict(ds.config or {}),
        secret_value=resolve_datasource_secret(session, ds),
    )
    if df is None or getattr(df, "empty", False):
        raise ValueError("input data source returned no rows")

    preds = model.predict(df)

    # Write predictions as CSV to object storage.
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["row_index", "prediction"])
    for i, p_value in enumerate(preds):
        writer.writerow([i, p_value])
    out_uri = store.put_bytes(
        f"batch_predict/{job.id}.csv",
        buf.getvalue().encode("utf-8"),
    )

    # Lineage: data_source → deployment, deployment → predictions URI.
    record_lineage(
        session,
        workspace_id=d.workspace_id,
        source_kind="data_source",
        source_id=input_ds_id,
        target_kind="deployment",
        target_id=deployment_id,
        relation="batch_predict_input",
        metadata={"job_id": job.id, "n_rows": int(len(df))},
    )
    record_lineage(
        session,
        workspace_id=d.workspace_id,
        source_kind="deployment",
        source_id=deployment_id,
        target_kind="batch_prediction",
        target_id=job.id,
        relation="batch_predict_output",
        metadata={"output_uri": out_uri, "n_predictions": int(len(preds))},
    )

    # Stamp the Job payload so the UI can render a download link.
    job.payload = {
        **p,
        "output_uri": out_uri,
        "n_rows": int(len(df)),
        "n_predictions": int(len(preds)),
    }


register_handler("batch_predict", _handle_batch_predict)


def _handle_dataset_refresh(job: Job, session) -> None:
    """Phase 9: dataset refresh Job — re-runs the DataSource driver's
    introspect + writes a new Dataset version row.

    Mirrors ``POST /data-sources/{id}/refresh`` so a scheduled refresh
    yields the same artifacts the manual button does.
    """
    from pycaret_server.datasources import get_driver
    from pycaret_server.db import DataSource, Dataset
    from sqlalchemy import select

    p = job.payload or {}
    ds_id = p.get("data_source_id")
    if not ds_id:
        raise ValueError("dataset_refresh payload missing data_source_id")
    ds = session.get(DataSource, ds_id)
    if ds is None:
        raise ValueError(f"data source {ds_id} not found")
    from pycaret_server.api.connections import resolve_datasource_secret

    driver = get_driver(ds.kind)
    schema = driver.introspect(
        config=dict(ds.config or {}),
        secret_value=resolve_datasource_secret(session, ds),
        sample_rows=20,
    )
    prior = session.scalar(
        select(Dataset)
        .where(Dataset.data_source_id == ds_id)
        .order_by(Dataset.version.desc())
        .limit(1)
    )
    version = (prior.version + 1) if prior else 1
    session.add(
        Dataset(
            workspace_id=ds.workspace_id,
            data_source_id=ds_id,
            version=version,
            name=ds.name,
            schema_json={
                "columns": schema.columns,
                "sample_rows": schema.sample_rows,
            },
            row_count=schema.row_count,
            byte_count=schema.byte_count,
            created_by=ds.created_by,
        )
    )


register_handler("dataset_refresh", _handle_dataset_refresh)


def _handle_git_publish(job: Job, session) -> None:
    """Phase 5: publish a project's experiments/trials/runs to a Git repo.

    ``payload`` carries ``{git_repository_id, project_id, commit_message?}``.
    Reuses ``git.service.publish_project`` so the manual button and the
    auto-publish-on-run-complete path go through identical code.
    """
    from pycaret_server.db import GitRepository
    from pycaret_server.git.service import publish_project

    p = job.payload or {}
    repo_id = p.get("git_repository_id")
    project_id = p.get("project_id")
    if not repo_id or not project_id:
        raise ValueError(
            "git_publish payload requires git_repository_id + project_id"
        )
    repo = session.get(GitRepository, repo_id)
    if repo is None:
        raise ValueError(f"git repository {repo_id!r} not found")
    if not repo.enabled:
        # Soft-skip: row was disabled between enqueue and execute.
        job.payload = {**p, "skipped": "repo disabled"}
        return
    result = publish_project(
        session,
        repo,
        project_id,
        commit_message=p.get("commit_message"),
    )
    job.payload = {**p, **result}
    if not result.get("ok"):
        raise RuntimeError(result.get("error") or "git publish failed")


register_handler("git_publish", _handle_git_publish)


# -------------------------------------------------------- main loop


class _Stopped(Exception):
    """Raised when SIGINT/SIGTERM arrives."""


def serve(
    *,
    queues: list[str] | None = None,
    worker_id: str | None = None,
    redis_url: str | None = None,
    poll_interval: float = 5.0,
) -> int:
    """Run the worker forever (until SIGINT / SIGTERM).

    Returns the exit code. ``0`` for clean shutdown; non-zero only on
    catastrophic startup failure (e.g. Redis unreachable + no fallback).
    """
    settings = get_settings()
    queues = queues or settings.worker_queues.split(",")
    queues = [q.strip() for q in queues if q.strip()]
    worker_id = worker_id or settings.worker_id or f"worker-{uuid.uuid4().hex[:8]}"
    redis_url = redis_url or settings.redis_url

    stop_event = threading.Event()

    def _handle_signal(signum, _frame):  # noqa: ANN001
        _log.info("worker %s received signal %s, shutting down", worker_id, signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    _log.info(
        "worker %s starting; queues=%s redis=%s", worker_id, queues, redis_url
    )

    from pycaret_server.runs import queue_redis

    if not queue_redis.is_healthy(redis_url):
        _log.error("Redis at %s is not reachable — refusing to start", redis_url)
        return 2

    while not stop_event.is_set():
        try:
            job_id = queue_redis.dequeue_job(
                queues, redis_url=redis_url, timeout=poll_interval
            )
            if job_id is None:
                # Idle tick — surface a heartbeat in the DB so an admin
                # page can render "worker X last seen N seconds ago".
                continue
            _claim_and_run(job_id, worker_id=worker_id, redis_url=redis_url)
        except KeyboardInterrupt:
            break
        except Exception:  # noqa: BLE001
            _log.exception("worker loop iteration failed")
            time.sleep(1.0)

    _log.info("worker %s stopped", worker_id)
    return 0


def _can_run_job(job: Job) -> tuple[bool, str | None]:
    """Phase 14 GPU-routing gate.

    Refuses a Job whose ``requested_resources.gpu >= 1`` (or queue is
    ``gpu``) when the worker can't actually see a CUDA device. Avoids
    a CPU-only worker draining a GPU queue into permanent failures.
    """
    from pycaret_server.runtime import detect_gpus

    needs_gpu = (
        (job.requested_resources or {}).get("gpu", 0) or 0
    ) >= 1 or job.queue == "gpu"
    if not needs_gpu:
        return True, None
    inv = detect_gpus()
    if inv.available:
        return True, None
    return False, (
        "worker has no GPU available "
        f"(probe={inv.source}, error={inv.error or 'no device'})"
    )


def _claim_and_run(job_id: str, *, worker_id: str, redis_url: str) -> None:
    """Mark a Job as running with our ``locked_by`` stamp, then execute.

    Atomically takes the lock: ``UPDATE jobs SET status='running',
    locked_by=:w WHERE id=:id AND status='queued'``. If 0 rows updated,
    another worker beat us — return silently.
    """
    from sqlalchemy import update
    from pycaret_server.runs import queue_redis as _q

    session = get_session()
    try:
        now = datetime.now(UTC)
        res = session.execute(
            update(Job)
            .where(Job.id == job_id, Job.status == "queued")
            .values(
                status="running",
                locked_by=worker_id,
                locked_at=now,
                started_at=now,
                attempts=Job.attempts + 1,
            )
        )
        session.commit()
        if res.rowcount == 0:
            return

        job = session.get(Job, job_id)
        if job is None:
            return

        # GPU routing gate — if we can't satisfy the Job's hardware
        # ask, requeue it for another worker rather than failing it.
        can_run, reason = _can_run_job(job)
        if not can_run:
            job.status = "queued"
            job.locked_by = None
            job.locked_at = None
            job.started_at = None
            job.attempts = max(0, job.attempts - 1)
            _log.info(
                "worker %s released job %s (%s)", worker_id, job.id, reason
            )
            session.commit()
            # Best-effort: put it back on the queue so a different
            # worker picks it up. Inprocess mode skips this — no queue.
            try:
                from pycaret_server.config import get_settings

                if get_settings().runs_backend == "redis":
                    _q.enqueue_job(job.id, queue=job.queue, redis_url=redis_url)
            except Exception:  # noqa: BLE001
                pass
            return

        handler = _HANDLERS.get(job.kind)
        if handler is None:
            _fail_job(session, job, f"no handler registered for kind={job.kind!r}")
            return

        try:
            _q.heartbeat(job.id, redis_url=redis_url)
            handler(job, session)
            _finish_job(session, job)
        except Exception as exc:  # noqa: BLE001
            _fail_job(session, job, f"{type(exc).__name__}: {exc}")
    finally:
        session.close()


def _finish_job(session, job: Job) -> None:
    job.status = "succeeded"
    job.finished_at = datetime.now(UTC)
    job.locked_by = None
    session.commit()


def _fail_job(session, job: Job, error: str) -> None:
    job.status = "failed" if job.attempts >= job.max_attempts else "queued"
    job.error = error
    job.finished_at = datetime.now(UTC)
    job.locked_by = None
    session.commit()
