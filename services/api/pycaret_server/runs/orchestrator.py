"""In-process run orchestrator.

Turns a persisted `Run` row (status=queued) into an actual engine invocation:

- Marks the row `running`, then executes the requested plan in a thread-pool
  worker.
- Attaches a `DBEventLogger` so every engine `Event` becomes an `events` row
  and also fans out to any WebSocket subscribers.
- On success: saves the fitted pipeline to `ARTIFACT_DIR/runs/<run_id>/pipeline.pkl`,
  writes an `Artifact` row, stores the leaderboard JSON on the Run, marks
  status=succeeded.
- On failure: captures the exception repr in `error`, marks status=failed.

No Celery / Redis — MVP runs inside the FastAPI process. Swap `_executor` for
a process pool or out-of-process queue without touching call sites.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pycaret.core.tasks import TaskType

from pycaret_server.config import get_settings
from pycaret_server.db import Artifact, Experiment, Project, Run, Trial, get_session
from pycaret_server.runs.broker import event_broker
from pycaret_server.runs.logger_bridge import DBEventLogger
from pycaret_server.runs.plans import (
    PlanName,
    execute_plan,
    execute_search_plan,
    load_csv,
    load_inline,
    load_sklearn_dataset,
)

_log = logging.getLogger(__name__)


class _CancelledError(RuntimeError):
    """Raised inside a worker when its cancel_event is set. Mapped to
    ``Run.status = "cancelled"`` by the outer catch block."""


# ----------------------------------------------------------------------- spec


@dataclass
class RunSpec:
    """Everything the worker needs to execute a run. Plain-Python; built by the
    API route from DB + request payload."""

    run_id: str
    experiment_id: str
    task: TaskType
    target: str | None
    setup_params: dict[str, Any]

    plan: PlanName = "setup"
    model_id: str | None = None
    plan_params: dict[str, Any] | None = None

    # Exactly one of these must be set.
    sklearn_dataset: str | None = None
    data_inline: list[dict] | None = None
    data_source_path: str | None = None  # resolved CSV path from DataSource.config
    # Override target column (e.g. data source didn't embed one).
    target_override: str | None = None
    # Cooperative cancellation: submit() stamps this; the worker checks it
    # between stages. The API route's cancel endpoint sets it.
    cancel_event: Any | None = None


# -------------------------------------------------------------- engine factory


def _build_experiment(task: TaskType, target: str | None, setup_params: dict[str, Any]):
    """Instantiate the right `Experiment` subclass and attach a quiet logger."""
    from pycaret.tasks import (
        AnomalyExperiment,
        ClassificationExperiment,
        ClusteringExperiment,
        RegressionExperiment,
        TimeSeriesExperiment,
    )

    cls = {
        TaskType.CLASSIFICATION: ClassificationExperiment,
        TaskType.REGRESSION: RegressionExperiment,
        TaskType.CLUSTERING: ClusteringExperiment,
        TaskType.ANOMALY: AnomalyExperiment,
        TaskType.TIME_SERIES: TimeSeriesExperiment,
    }[task]

    kwargs: dict[str, Any] = dict(setup_params)
    # Pop any server-only keys we might sneak in later (none today, but reserved).
    kwargs.pop("_server", None)
    if target is not None and cls in (ClassificationExperiment, RegressionExperiment):
        kwargs.setdefault("target", target)
    return cls(**kwargs)


# ----------------------------------------------------------------- orchestrator


class RunOrchestrator:
    """Run submit + execute. One per app; fetch via `get_orchestrator()`."""

    def __init__(self, max_workers: int = 2) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="pyc-run")
        # Tracks in-flight Futures so shutdown can wait on them.
        self._futures: dict[str, Future] = {}
        # Per-run cancellation flags the worker polls between stages.
        self._cancel_events: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    # --------------------------------------------------- public submit / status

    def submit(self, spec: RunSpec) -> Future:
        """Queue `spec` for background execution. Returns a Future for tests."""
        # Install a cancellation event every submit so the worker can poll it.
        if spec.cancel_event is None:
            spec.cancel_event = threading.Event()
        with self._lock:
            self._cancel_events[spec.run_id] = spec.cancel_event
        fut = self._executor.submit(self._execute, spec)
        with self._lock:
            self._futures[spec.run_id] = fut

        def _cleanup(_f: Future) -> None:
            with self._lock:
                self._futures.pop(spec.run_id, None)
                self._cancel_events.pop(spec.run_id, None)

        fut.add_done_callback(_cleanup)
        return fut

    def cancel(self, run_id: str) -> bool:
        """Signal cooperative cancellation. Returns True if a worker was notified.

        The worker checks the event between plan stages and raises a
        `_CancelledError` if set, which the `_execute` catch block maps to
        ``Run.status = "cancelled"``.
        """
        with self._lock:
            ev = self._cancel_events.get(run_id)
        if ev is None:
            return False
        ev.set()
        return True

    def wait_for(self, run_id: str, timeout: float | None = None) -> None:
        """Block until a given run finishes (tests + synchronous clients)."""
        with self._lock:
            fut = self._futures.get(run_id)
        if fut is not None:
            fut.result(timeout=timeout)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)

    # ---------------------------------------------------------------- worker

    def _execute(self, spec: RunSpec) -> None:
        """Run the plan. Catches every exception and writes it back to the row."""

        def _checkpoint() -> None:
            """Raise if cancellation was signalled. Called at stage boundaries."""
            if spec.cancel_event is not None and spec.cancel_event.is_set():
                raise _CancelledError("run cancelled by user")

        self._transition(spec.run_id, status="running", started_at=datetime.now(UTC))
        t0 = datetime.now(UTC)
        try:
            _checkpoint()
            df, target = self._load_data(spec)
            _checkpoint()

            if spec.plan == "search":
                # Search plan re-fits per variant; orchestrator owns the
                # build/fit loop instead of pre-fitting one experiment.
                def _build(setup_params: dict[str, Any], tgt: str | None):
                    exp_inner = _build_experiment(spec.task, tgt, setup_params)
                    exp_inner.logger = DBEventLogger(
                        run_id=spec.run_id, experiment_id=spec.experiment_id
                    )
                    return exp_inner

                outcome = execute_search_plan(
                    build_exp=_build,
                    df=df,
                    target=target or spec.target,
                    base_setup_params=spec.setup_params,
                    plan_params=spec.plan_params,
                    checkpoint=_checkpoint,
                )
            else:
                exp = _build_experiment(
                    spec.task, target or spec.target, spec.setup_params
                )
                exp.logger = DBEventLogger(
                    run_id=spec.run_id, experiment_id=spec.experiment_id
                )
                exp.fit(df)
                _checkpoint()
                outcome = execute_plan(
                    exp,
                    spec.plan,
                    model_id=spec.model_id,
                    plan_params=spec.plan_params,
                )
            _checkpoint()

            artifact_row = None
            if outcome.best_model is not None:
                artifact_row = self._save_pipeline(spec.run_id, outcome.best_model)

            leaderboard = _leaderboard_to_json(outcome.leaderboard)
            self._transition(
                spec.run_id,
                status="succeeded",
                finished_at=datetime.now(UTC),
                duration_ms=(datetime.now(UTC) - t0).total_seconds() * 1000,
                leaderboard=leaderboard,
                metrics_summary=_summarise_metrics(leaderboard),
                extra_artifact=artifact_row,
            )
            _fire_run_event(spec.run_id, "run.succeeded")
        except _CancelledError:
            _log.info("run %s cancelled", spec.run_id)
            self._transition(
                spec.run_id,
                status="cancelled",
                finished_at=datetime.now(UTC),
                duration_ms=(datetime.now(UTC) - t0).total_seconds() * 1000,
                error="cancelled by user",
            )
            _fire_run_event(spec.run_id, "run.cancelled")
        except Exception as exc:  # noqa: BLE001 — persist *everything* the worker throws
            _log.exception("run %s failed", spec.run_id)
            self._transition(
                spec.run_id,
                status="failed",
                finished_at=datetime.now(UTC),
                duration_ms=(datetime.now(UTC) - t0).total_seconds() * 1000,
                error=f"{type(exc).__name__}: {exc}",
            )
            _fire_run_event(spec.run_id, "run.failed")
        finally:
            # Close the WebSocket stream cleanly.
            event_broker.close_run(spec.run_id)

    # ------------------------------------------------------ supporting methods

    def _load_data(self, spec: RunSpec) -> tuple[pd.DataFrame, str | None]:
        if spec.sklearn_dataset:
            df, tgt = load_sklearn_dataset(spec.sklearn_dataset)
            # An explicit override wins (useful if the frame has the target
            # under a different column name).
            return df, spec.target_override or tgt
        if spec.data_inline:
            return load_inline(spec.data_inline), spec.target_override
        if spec.data_source_path:
            return load_csv(spec.data_source_path), spec.target_override
        raise ValueError("run spec must supply sklearn_dataset, data_inline, or data_source_path")

    def _save_pipeline(self, run_id: str, model: Any) -> dict[str, Any] | None:
        """Pickle `model` under the artifact dir and return a dict describing the Artifact row."""
        try:
            import cloudpickle
        except ImportError:  # cloudpickle is a PyCaret core dep; safety net only
            import pickle as cloudpickle  # type: ignore[no-redef]

        settings = get_settings()
        run_dir: Path = settings.artifact_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "pipeline.pkl"

        blob = cloudpickle.dumps(model)
        path.write_bytes(blob)
        sha = hashlib.sha256(blob).hexdigest()

        return {
            "kind": "pipeline_pickle",
            "path": str(path),
            "sha256": sha,
            "size_bytes": len(blob),
            "content_type": "application/octet-stream",
        }

    def _transition(
        self,
        run_id: str,
        *,
        status: str | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
        duration_ms: float | None = None,
        leaderboard: dict | list | None = None,
        metrics_summary: dict | None = None,
        error: str | None = None,
        extra_artifact: dict[str, Any] | None = None,
    ) -> None:
        session = get_session()
        try:
            run: Run | None = session.get(Run, run_id)
            if run is None:  # deleted while running — nothing to do
                return
            if status is not None:
                run.status = status
            if started_at is not None:
                run.started_at = started_at
            if finished_at is not None:
                run.finished_at = finished_at
            if duration_ms is not None:
                run.duration_ms = duration_ms
            if leaderboard is not None:
                run.leaderboard = leaderboard
                _persist_trials(session, run, leaderboard)
            if metrics_summary is not None:
                run.metrics_summary = metrics_summary
            if error is not None:
                run.error = error
            if extra_artifact is not None:
                session.add(
                    Artifact(
                        id=str(uuid.uuid4()),
                        run_id=run_id,
                        **extra_artifact,
                    )
                )
            session.commit()
        finally:
            session.close()


# --------------------------------------------------------------- helpers


def _leaderboard_to_json(lb: pd.DataFrame | None) -> list[dict[str, Any]] | None:
    if lb is None:
        return None
    try:
        return lb.reset_index().to_dict(orient="records")
    except Exception:
        return None


def _summarise_metrics(
    leaderboard: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    """Very small summary: number of rows + best row (first entry post-rank)."""
    if not leaderboard:
        return None
    return {
        "rows": len(leaderboard),
        "best": leaderboard[0] if leaderboard else None,
    }


def _fire_run_event(run_id: str, event_type: str) -> None:
    """Best-effort webhook fan-out for run lifecycle transitions.

    Loads the workspace_id off the run's experiment → project chain and
    asynchronously dispatches. Never raises so a webhook target outage
    can't escalate into a run-status corruption.
    """
    try:
        from pycaret_server.webhooks import fire_event_async

        with get_session() as s:
            run: Run | None = s.get(Run, run_id)
            if run is None:
                return
            workspace_id = (
                s.query(Project.workspace_id)
                .join(Experiment, Experiment.project_id == Project.id)
                .filter(Experiment.id == run.experiment_id)
                .scalar()
            )
            if not workspace_id:
                return
            payload = {
                "workspace_id": workspace_id,
                "run_id": run.id,
                "experiment_id": run.experiment_id,
                "status": run.status,
                "duration_ms": run.duration_ms,
                "error": run.error,
            }
        fire_event_async(event_type, payload)
    except Exception:  # noqa: BLE001
        _log.exception("failed to fire webhook %s for run %s", event_type, run_id)


def _persist_trials(
    session,
    run: Run,
    leaderboard: dict | list | None,
) -> None:
    """Promote each leaderboard row to a queryable ``Trial``.

    Idempotent: clears any existing trials for this run before inserting,
    so re-running a leaderboard write (e.g. retry) doesn't duplicate.
    """
    rows = leaderboard if isinstance(leaderboard, list) else None
    if not rows:
        return

    workspace_id = (
        session.query(Project.workspace_id)
        .join(Experiment, Experiment.project_id == Project.id)
        .filter(Experiment.id == run.experiment_id)
        .scalar()
    )
    if workspace_id is None:
        return

    # Idempotent re-runs: drop existing trials for this run first.
    session.query(Trial).filter(Trial.run_id == run.id).delete(synchronize_session=False)

    for rank_idx, lb_row in enumerate(rows):
        if not isinstance(lb_row, dict):
            continue
        model_id = str(lb_row.get("Model") or lb_row.get("index") or f"trial_{rank_idx}")
        # Strip the model id off the metrics map so it isn't doubly persisted.
        metrics = {k: v for k, v in lb_row.items() if k not in ("Model", "index")}
        session.add(
            Trial(
                id=str(uuid.uuid4()),
                run_id=run.id,
                workspace_id=workspace_id,
                model_id=model_id,
                rank=rank_idx + 1,
                metrics=metrics,
                is_best=(rank_idx == 0),
                fitted_pipeline_id=None,
            )
        )


# ---------------------------------------------------------------- singleton

_orchestrator: RunOrchestrator | None = None
_orchestrator_lock = threading.Lock()


def get_orchestrator() -> RunOrchestrator:
    """Return the app-level orchestrator, lazily instantiated."""
    global _orchestrator
    with _orchestrator_lock:
        if _orchestrator is None:
            _orchestrator = RunOrchestrator()
        return _orchestrator


def reset_orchestrator() -> None:
    """Tear down the singleton. Tests use this between fixtures."""
    global _orchestrator
    with _orchestrator_lock:
        if _orchestrator is not None:
            _orchestrator.shutdown()
            _orchestrator = None
