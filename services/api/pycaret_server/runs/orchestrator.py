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
from sqlalchemy import select

from pycaret_server.config import get_settings
from pycaret_server.db import Artifact, Experiment, Job, Project, Run, Trial, get_session
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


@dataclass
class TrialActionSpec:
    """A follow-on action on existing trial(s) of a succeeded run.

    Tune / Ensemble work on one source trial; Blend / Stack work on N.
    The action runs in a worker and persists its result as a *new trial*
    inside the same run (no new Run row) with ``kind`` set accordingly and
    ``parent_trial_ids`` back-linking the sources. Events stream into the
    run's existing event stream so the user's open log drawer keeps
    updating without any extra wiring.
    """

    run_id: str
    experiment_id: str
    workspace_id: str
    task: TaskType
    target: str | None
    setup_params: dict[str, Any]

    # Data resolution — copied from the source run's snapshot so we
    # reproduce its split.
    sklearn_dataset: str | None = None
    data_inline: list[dict] | None = None
    data_source_path: str | None = None

    # tune | ensemble | blend | stack — drives both the engine call and
    # the Trial.kind on the resulting row.
    action: str = "tune"
    action_params: dict[str, Any] | None = None

    # Source trials (pickle paths + ids for the back-link).
    source_stored_paths: list[str] | None = None
    source_trial_ids: list[str] | None = None
    # Label of the first source, used to suggest a model_id on the result.
    source_model_id: str | None = None


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

    def submit_trial_action(self, spec: TrialActionSpec) -> Future:
        """Queue a follow-on tune / ensemble / blend / stack action.

        Resolution is by run_id — the action's events stream into the
        parent run and the resulting Trial lands in that run's trials
        table. No new Run row is created.
        """
        return self._executor.submit(self._execute_action, spec)

    def submit_trial_job(self, job_id: str) -> Future:
        """Phase 0-v2: submit one ``kind="trial"`` Job to the executor.

        Each Trial-Job runs independently of every other one — that's
        what makes ``compare`` parallel. The executor pool's
        ``max_workers`` caps concurrency in inprocess mode; redis mode
        scales out via the worker container instead.
        """
        fut = self._executor.submit(execute_trial_job, job_id)
        with self._lock:
            self._futures.setdefault(job_id, fut)
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
        """Block until a given Run reaches a terminal state.

        Phase 0-v2: a Run owns N Trial-Jobs; the Run is "done" when
        every Trial-Job future has completed (the reconciler runs as
        part of each Trial-Job's tail, so the Run status flips before
        we return).

        Walks every in-flight Future associated with this Run's Jobs
        and awaits them. Legacy setup runs still register under the
        ``run_id`` key directly; we wait on that future too.
        """
        # 1. Direct ``run_id`` future (legacy setup path).
        with self._lock:
            direct = self._futures.get(run_id)
        if direct is not None:
            try:
                direct.result(timeout=timeout)
            except Exception:  # noqa: BLE001 — Run row carries the error
                pass
        # 2. Trial-Job futures associated with this Run.
        from sqlalchemy import select

        from pycaret_server.db import Job, get_session

        with get_session() as s:
            job_ids = list(
                s.scalars(
                    select(Job.id).where(Job.run_id == run_id, Job.kind == "trial")
                ).all()
            )
        for jid in job_ids:
            with self._lock:
                fut = self._futures.get(jid)
            if fut is not None:
                try:
                    fut.result(timeout=timeout)
                except Exception:  # noqa: BLE001
                    pass

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

            # Session 25: keep every fitted candidate so the UI can offer
            # model-detail / download / per-trial promote.
            trial_artifacts: dict[str, dict[str, Any]] = {}
            ranked_ids = list(outcome.extra.get("ranked_ids") or [])
            for idx, model in enumerate(outcome.models or []):
                if model is None:
                    continue
                mid = ranked_ids[idx] if idx < len(ranked_ids) else f"trial_{idx}"
                try:
                    trial_artifacts[str(mid)] = self._save_trial_artifact(
                        spec.run_id, str(mid), model
                    )
                except Exception:  # noqa: BLE001 — one bad pickle shouldn't kill the run
                    _log.exception("failed to persist trial %s", mid)

            leaderboard = _leaderboard_to_json(outcome.leaderboard)
            self._transition(
                spec.run_id,
                status="succeeded",
                finished_at=datetime.now(UTC),
                duration_ms=(datetime.now(UTC) - t0).total_seconds() * 1000,
                leaderboard=leaderboard,
                metrics_summary=_summarise_metrics(leaderboard),
                extra_artifact=artifact_row,
                trial_artifacts=trial_artifacts,
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

    def _execute_action(self, spec: TrialActionSpec) -> None:
        """Run a follow-on tune / ensemble / blend / stack action.

        Walks the same path the main worker does for compare: rebuild the
        engine `Experiment` from the source run's setup, re-fit on the
        original dataset (so the split is reproducible), then call the
        matching engine verb against the source trial pickle(s). Persists
        the result as a new ``Trial`` row inside the source run, labelled
        with ``kind`` + ``parent_trial_ids``.
        """
        run_id = spec.run_id
        try:
            # Re-load data + re-fit experiment so the verbs have a live
            # ``self._fit_state`` to draw splits from.
            data_spec = RunSpec(
                run_id=run_id,
                experiment_id=spec.experiment_id,
                task=spec.task,
                target=spec.target,
                setup_params=spec.setup_params,
                sklearn_dataset=spec.sklearn_dataset,
                data_inline=spec.data_inline,
                data_source_path=spec.data_source_path,
            )
            df, resolved_target = self._load_data(data_spec)
            exp = _build_experiment(
                spec.task, resolved_target or spec.target, spec.setup_params
            )
            exp.logger = DBEventLogger(
                run_id=run_id, experiment_id=spec.experiment_id
            )
            exp.fit(df)

            # Load the source pickle(s) and hand them to the engine verb.
            # Phase 2: source URIs may be ``file://`` or ``s3://`` — route
            # through the ObjectStore so cloud backends work without a
            # second code path.
            try:
                import cloudpickle
            except ImportError:  # noqa: BLE001
                import pickle as cloudpickle  # type: ignore[no-redef]
            from pycaret_server.storage import get_object_store

            store = get_object_store()
            source_paths = spec.source_stored_paths or []
            sources = []
            for p in source_paths:
                if "://" in p:
                    sources.append(cloudpickle.loads(store.get_bytes(p)))
                else:
                    # Legacy bare path — read directly.
                    sources.append(cloudpickle.loads(Path(p).read_bytes()))
            if not sources:
                raise ValueError("no source trial pickles provided")

            params = dict(spec.action_params or {})
            action = spec.action

            if action == "tune":
                result = exp.tune_model(sources[0], **params)
                output_kind = "tuned"
                output_model_id = (
                    spec.source_model_id or "tuned"
                )
            elif action == "ensemble":
                result = exp.ensemble_model(sources[0], **params)
                output_kind = "ensembled"
                method = params.get("method", "Bagging")
                output_model_id = (
                    f"{spec.source_model_id or 'model'}_{str(method).lower()}"
                )
            elif action == "blend":
                result = exp.blend_models(sources, **params)
                output_kind = "blended"
                output_model_id = "blended"
            elif action == "stack":
                result = exp.stack_models(sources, **params)
                output_kind = "stacked"
                output_model_id = "stacked"
            else:
                raise ValueError(f"unknown action {action!r}")

            pipeline = getattr(result, "pipeline", None)
            metrics_df = getattr(result, "metrics", None)
            if pipeline is None:
                raise RuntimeError(
                    f"{action} returned no pipeline — engine misbehaved"
                )

            # Pickle + insert Trial row in the source run.
            artifact = self._save_trial_artifact(
                run_id, output_model_id, pipeline
            )

            metrics_payload: dict[str, Any] = {}
            if metrics_df is not None and hasattr(metrics_df, "loc"):
                try:
                    if "Mean" in metrics_df.index:
                        metrics_payload = {
                            k: (float(v) if isinstance(v, (int, float)) else v)
                            for k, v in metrics_df.loc["Mean"].to_dict().items()
                        }
                except Exception:  # noqa: BLE001
                    metrics_payload = {}

            # Tune carries the search history on the result; stash it on the
            # new trial's params so the UI can render an optimization-history
            # chart without a separate endpoint.
            extra_params: dict[str, Any] = dict(artifact.get("params") or {})
            if action == "tune":
                best_params = getattr(result, "best_params", None)
                cv_results = getattr(result, "cv_results", None)
                if best_params is not None:
                    extra_params["_best_params"] = _jsonable_dict(best_params)
                if cv_results is not None:
                    extra_params["_cv_history"] = _shrink_cv_history(cv_results)

            self._insert_followon_trial(
                run_id=run_id,
                workspace_id=spec.workspace_id,
                model_id=output_model_id,
                kind=output_kind,
                parent_trial_ids=spec.source_trial_ids or [],
                metrics=metrics_payload,
                stored_path=artifact["stored_path"],
                sha256=artifact["sha256"],
                size_bytes=artifact["size_bytes"],
                params=extra_params,
            )
            _fire_run_event(run_id, f"trial.{action}.succeeded")
        except Exception as exc:  # noqa: BLE001
            # Stack to stderr so dev users actually see why the action failed
            # even if uvicorn's reloader is muting the logging tree.
            import traceback as _tb

            print(
                f"[orchestrator] trial action {spec.action!r} on run "
                f"{run_id} failed: {type(exc).__name__}: {exc}",
                flush=True,
            )
            _tb.print_exc()
            _log.exception(
                "trial action %s on run %s failed", spec.action, run_id
            )
            # Surface the failure as a single event on the run's stream so
            # the user sees it in their drawer; no Run.status flip.
            try:
                logger = DBEventLogger(
                    run_id=run_id, experiment_id=spec.experiment_id
                )
                from pycaret.logging.events import EventKind  # late import

                logger.log(
                    EventKind.ERROR,
                    message=(
                        f"{spec.action} failed: {type(exc).__name__}: {exc}"
                    ),
                    payload={"action": spec.action, "error": str(exc)},
                )
            except Exception as inner:  # noqa: BLE001
                print(
                    f"[orchestrator] also failed to emit error event: {inner}",
                    flush=True,
                )

    def _insert_followon_trial(
        self,
        *,
        run_id: str,
        workspace_id: str,
        model_id: str,
        kind: str,
        parent_trial_ids: list[str],
        metrics: dict[str, Any],
        stored_path: str,
        sha256: str,
        size_bytes: int,
        params: dict[str, Any],
    ) -> None:
        """Append a follow-on trial spawned by a tune/ensemble/blend/stack action.

        Phase 0-v2: a follow-on Trial lands in the *same Run* as its
        source (parent_trial_ids[0] resolves the parent Trial → its
        run_id). The new Trial is created with status=succeeded since
        we already have the artifact + metrics at this point — no
        separate Job scheduling needed.
        """
        session = get_session()
        try:
            umbrella: Run | None = session.get(Run, run_id)
            experiment_id = (
                umbrella.experiment_id if umbrella is not None else None
            )
            # Find the source trial's run (the new follow-on Trial
            # lands in that same Run so the leaderboard groups them).
            source_run_id: str | None = run_id  # default to the umbrella
            if parent_trial_ids:
                src = session.get(Trial, parent_trial_ids[0])
                if src is not None and src.run_id:
                    source_run_id = src.run_id

            trial_id = str(uuid.uuid4())
            now = datetime.now(UTC)
            session.add(
                Trial(
                    id=trial_id,
                    run_id=source_run_id,
                    experiment_id=experiment_id,
                    workspace_id=workspace_id,
                    name=model_id,
                    model_id=model_id,
                    kind=kind,
                    status="succeeded",
                    started_at=now,
                    finished_at=now,
                    duration_ms=0.0,
                    metrics=metrics,
                    stored_path=stored_path,
                    sha256=sha256,
                    size_bytes=size_bytes,
                    params=params,
                    parent_trial_ids=list(parent_trial_ids),
                    created_by_action_id=run_id,
                    fitted_pipeline_id=None,
                )
            )
            session.commit()
        finally:
            session.close()

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
        """Pickle ``model`` via the ObjectStore. Returns Artifact-row fields.

        Phase 2: routes through the ObjectStore singleton so the same code
        works against local-fs, S3, or MinIO. ``path`` is now a URI
        (``file:///...`` or ``s3://bucket/...``); downstream code reads
        it back via the same ObjectStore.
        """
        try:
            import cloudpickle
        except ImportError:  # cloudpickle is a PyCaret core dep; safety net only
            import pickle as cloudpickle  # type: ignore[no-redef]
        from pycaret_server.storage import get_object_store

        blob = cloudpickle.dumps(model)
        sha = hashlib.sha256(blob).hexdigest()
        uri = get_object_store().put_bytes(f"runs/{run_id}/pipeline.pkl", blob)

        return {
            "kind": "pipeline_pickle",
            "path": uri,
            "sha256": sha,
            "size_bytes": len(blob),
            "content_type": "application/octet-stream",
        }

    def _save_trial_artifact(
        self, run_id: str, model_id: str, model: Any
    ) -> dict[str, Any]:
        """Pickle one trial's fitted pipeline via the ObjectStore.

        Each trial's bytes live at ``runs/<run_id>/trials/<safe_id>.pkl``
        on whichever backend ``get_object_store()`` resolves to.
        """
        try:
            import cloudpickle
        except ImportError:
            import pickle as cloudpickle  # type: ignore[no-redef]
        from pycaret_server.storage import get_object_store

        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in model_id) or "trial"
        blob = cloudpickle.dumps(model)
        sha = hashlib.sha256(blob).hexdigest()
        uri = get_object_store().put_bytes(
            f"runs/{run_id}/trials/{safe}.pkl", blob
        )
        return {
            "stored_path": uri,
            "sha256": sha,
            "size_bytes": len(blob),
            "params": _extract_params(model),
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
        trial_artifacts: dict[str, dict[str, Any]] | None = None,
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
                # Phase 0-v2: the leaderboard JSON stays on the Run for
                # legacy UI surfaces, but the *authoritative* per-trial
                # rows are now written at dispatch time and stamped by
                # the Trial-Job workers themselves. No more lazy
                # post-completion persistence.
                run.leaderboard = leaderboard
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


def _duration_ms(started_at: datetime | None, finished_at: datetime) -> float:
    """Compute a duration in ms, treating naive datetimes as UTC.

    SQLite drops ``tzinfo`` on round-trip; rows loaded back from the DB
    are tz-naive even though we stamp them tz-aware. Coerce both sides
    to UTC before subtracting so the arithmetic never trips on a
    naive/aware mismatch.
    """
    if started_at is None:
        return 0.0
    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=UTC)
    if finished_at.tzinfo is None:
        finished_at = finished_at.replace(tzinfo=UTC)
    return (finished_at - started_at).total_seconds() * 1000


def execute_trial_job(job_id: str) -> None:
    """Phase 0-v2: execute a single ``kind="trial"`` Job.

    This is the new per-Trial worker entrypoint — runs
    ``create_model(model_id)`` for ONE algorithm, stamps the matching
    Trial row, then calls :func:`reconcile_run_status` so the parent
    Run flips state when its last Trial finishes.

    Called from both the in-process executor (via
    ``orchestrator.submit_trial_job``) and the redis worker (via the
    ``trial`` handler registered in ``worker.py``).
    """
    from pycaret.core.tasks import TaskType

    from pycaret_server.runs.logger_bridge import DBEventLogger

    session = get_session()
    try:
        job = session.get(Job, job_id)
        if job is None:
            return
        payload = job.payload or {}
        trial_id = payload.get("trial_id")
        run_id = payload.get("run_id")
        if not trial_id or not run_id:
            return
        trial = session.get(Trial, trial_id)
        run = session.get(Run, run_id)
        if trial is None or run is None:
            return

        # Mark Trial running + bump Run to running on first Trial.
        now = datetime.now(UTC)
        trial.status = "running"
        trial.started_at = trial.started_at or now
        if run.status == "queued":
            run.status = "running"
            run.started_at = run.started_at or now
        session.commit()
        _fire_run_event(run_id, "trial.running")

        # Build the engine + fit + create_model for this trial's id.
        # Search-plan trials carry a ``variant_override`` that gets
        # merged on top of the base setup_params before fit().
        base_setup = dict(payload.get("setup_params") or {})
        variant_override = dict(payload.get("variant_override") or {})
        merged_setup = {**base_setup, **variant_override}
        spec = RunSpec(
            run_id=run_id,
            experiment_id=payload.get("experiment_id", run.experiment_id),
            task=TaskType(payload.get("task") or (run.snapshot or {}).get("task")),
            target=payload.get("target") or (run.snapshot or {}).get("target"),
            setup_params=merged_setup,
            plan="create",
            model_id=str(payload.get("model_id") or trial.model_id),
            plan_params=dict(payload.get("plan_params") or {}),
            sklearn_dataset=payload.get("sklearn_dataset"),
            data_inline=payload.get("data_inline"),
            data_source_path=payload.get("data_source_path"),
            target_override=payload.get("target_override"),
        )
        orchestrator = get_orchestrator()
        try:
            df, target = orchestrator._load_data(spec)  # noqa: SLF001
            exp = _build_experiment(
                spec.task, target or spec.target, spec.setup_params
            )
            exp.logger = DBEventLogger(
                run_id=run_id, experiment_id=spec.experiment_id
            )
            exp.fit(df)
            outcome = execute_plan(
                exp,
                "create",
                model_id=spec.model_id,
                plan_params=spec.plan_params,
            )
            # ``create`` always returns one model in outcome.models[0].
            model = (outcome.models or [None])[0]
            if model is None and outcome.best_model is not None:
                model = outcome.best_model
            if model is None:
                raise RuntimeError("engine returned no fitted model")

            artifact = orchestrator._save_trial_artifact(  # noqa: SLF001
                run_id, trial.model_id, model
            )

            # Pluck this trial's row out of the engine's leaderboard
            # so we capture the per-fold metrics it computed during
            # `pull()`. Falls back to outcome.extra if leaderboard is
            # missing for some reason.
            metrics_dict: dict[str, Any] = {}
            lb_records = _leaderboard_to_json(outcome.leaderboard) or []
            if lb_records:
                first = lb_records[0]
                if isinstance(first, dict):
                    metrics_dict = {
                        k: v for k, v in first.items() if k not in ("Model", "index")
                    }

            finished = datetime.now(UTC)
            trial.metrics = metrics_dict
            trial.stored_path = artifact["stored_path"]
            trial.sha256 = artifact["sha256"]
            trial.size_bytes = artifact["size_bytes"]
            trial.params = artifact["params"]
            trial.status = "succeeded"
            trial.finished_at = finished
            trial.duration_ms = _duration_ms(trial.started_at, finished)
            session.commit()
            _fire_run_event(run_id, "trial.succeeded")
        except Exception as exc:  # noqa: BLE001
            _log.exception("trial job %s failed", job_id)
            finished = datetime.now(UTC)
            trial.status = "failed"
            trial.error = f"{type(exc).__name__}: {exc}"
            trial.finished_at = finished
            trial.duration_ms = _duration_ms(trial.started_at, finished)
            session.commit()
            _fire_run_event(run_id, "trial.failed")

        # Last Trial standing? Flip the Run.
        reconcile_run_status(session, run_id)
    finally:
        session.close()


def reconcile_run_status(session, run_id: str) -> None:
    """Recompute a Run's status from its Trials' statuses.

    Rules:

    - Any Trial still ``queued`` or ``running`` → Run is ``running``.
    - Otherwise (all terminal), Run is ``succeeded`` if every Trial
      succeeded; ``failed`` if any failed; ``cancelled`` if any
      cancelled (and none failed); otherwise ``succeeded``.
    - When the Run flips terminal, we cache an aggregate leaderboard
      JSON on it for legacy UI surfaces.

    Force-refreshes the session's identity map first so concurrent
    workers' commits become visible — without this, a worker who
    finished after the others might still see their Trial rows in
    their pre-commit state and never flip the Run.
    """
    # Drop cached state so we see the latest committed Trial rows
    # written by sibling workers.
    session.expire_all()
    run: Run | None = session.get(Run, run_id)
    if run is None:
        return
    if run.status in ("succeeded", "failed", "cancelled"):
        return
    trials = session.scalars(
        select(Trial).where(Trial.run_id == run_id)
    ).all()
    if not trials:
        return
    statuses = {t.status for t in trials}
    if statuses & {"queued", "running"}:
        # Some still in flight — keep Run as running.
        run.status = "running"
        session.commit()
        return
    # All terminal.
    if "failed" in statuses and "succeeded" not in statuses:
        run.status = "failed"
    elif "cancelled" in statuses and "succeeded" not in statuses:
        run.status = "cancelled"
    elif "failed" in statuses:
        # Mixed: some succeeded + some failed. Keep the Run as
        # ``succeeded`` so the user can still promote winners, but the
        # error column carries the failed-trial count for visibility.
        run.status = "succeeded"
        run.error = f"{sum(1 for t in trials if t.status == 'failed')} of {len(trials)} trials failed"
    else:
        run.status = "succeeded"
    run.finished_at = datetime.now(UTC)
    run.duration_ms = _duration_ms(run.started_at, run.finished_at)

    # Rank + leaderboard cache from succeeded Trials' metrics.
    # Search-plan trials are named ``variant-<idx>-<model>``; surface
    # the variant index as a column so legacy callers can keep
    # grouping by Variant the way ``execute_search_plan`` used to.
    succeeded = [t for t in trials if t.status == "succeeded" and t.metrics]
    leaderboard: list[dict[str, Any]] = []
    for t in succeeded:
        row: dict[str, Any] = {"Model": t.model_id}
        if t.name and t.name.startswith("variant-"):
            parts = t.name.split("-", 2)
            if len(parts) >= 2 and parts[1].isdigit():
                row["Variant"] = int(parts[1])
        row.update(t.metrics or {})
        leaderboard.append(row)
    # Sort by Accuracy / AUC / R2 / Score if present (higher = better),
    # else by MAE / RMSE (lower = better). This is the same heuristic
    # the API's primary-metric picker uses.
    def _sort_key(row: dict[str, Any]) -> float:
        for k in ("Accuracy", "AUC", "F1", "R2"):
            v = row.get(k)
            if isinstance(v, (int, float)):
                return -float(v)
        for k in ("MAE", "RMSE", "MSE"):
            v = row.get(k)
            if isinstance(v, (int, float)):
                return float(v)
        return 0.0
    leaderboard.sort(key=_sort_key)
    run.leaderboard = leaderboard
    # Stamp ranks back on the Trial rows.
    for rank, row in enumerate(leaderboard, start=1):
        for t in trials:
            if t.model_id == row.get("Model") and t.status == "succeeded":
                t.rank = rank
                t.is_best = rank == 1
                break
    session.commit()
    _fire_run_event(run_id, f"run.{run.status}")
    # Phase 5 auto-publish: when a Run lands in ``succeeded``, enqueue
    # one ``git_publish`` Job per linked repo with ``auto_publish=True``.
    # Failures are swallowed so a flaky Git remote can't corrupt run
    # status — the per-repo error lives on the Job + GitRepository row.
    if run.status == "succeeded":
        try:
            _enqueue_auto_publish(session, run)
        except Exception:  # noqa: BLE001
            _log.exception("auto-publish enqueue failed for run %s", run_id)
    # Best-effort: close the WebSocket fan-out stream.
    try:
        from pycaret_server.runs.broker import event_broker

        event_broker.close_run(run_id)
    except Exception:  # noqa: BLE001
        pass


def _enqueue_auto_publish(session, run: "Run") -> None:
    """Look up every enabled, auto-publish-on GitRepository linked to
    this Run's project and enqueue a ``git_publish`` Job for each.

    Same code path as the manual publish button — the worker handler
    calls ``publish_project`` against the same row. No new code in the
    publish service itself.
    """
    from pycaret_server.config import get_settings
    from pycaret_server.db import Experiment, GitRepository, Job, Project

    exp = session.get(Experiment, run.experiment_id)
    if exp is None or exp.project_id is None:
        return
    proj = session.get(Project, exp.project_id)
    if proj is None:
        return
    repos = session.scalars(
        select(GitRepository).where(
            GitRepository.project_id == exp.project_id,
            GitRepository.workspace_id == proj.workspace_id,
            GitRepository.enabled.is_(True),
            GitRepository.auto_publish.is_(True),
        )
    ).all()
    if not repos:
        return
    settings = get_settings()
    for repo in repos:
        job = Job(
            kind="git_publish",
            status="queued",
            queue="default",
            payload={
                "git_repository_id": repo.id,
                "project_id": exp.project_id,
                "commit_message": f"pycaret: auto-publish run {run.id}",
            },
            correlation_id=run.id,
        )
        session.add(job)
        session.flush()
        # Best-effort enqueue on redis backend; inprocess relies on the
        # worker poller seeing the row directly.
        if settings.runs_backend == "redis":
            try:
                from pycaret_server.runs.queue_redis import enqueue_job

                enqueue_job(job.id, queue=job.queue, redis_url=settings.redis_url)
            except Exception:  # noqa: BLE001
                _log.exception(
                    "git_publish: redis enqueue failed for job %s", job.id
                )
        else:
            # Inprocess: run synchronously via the worker handler
            # registry so the user sees the result without standing up
            # a separate worker container.
            try:
                from pycaret_server.worker import _HANDLERS  # noqa: PLC2701

                handler = _HANDLERS.get("git_publish")
                if handler is not None:
                    handler(job, session)
                    job.status = "succeeded"
                    job.finished_at = datetime.now(UTC)
            except Exception as exc:  # noqa: BLE001
                _log.exception("git_publish inprocess run failed")
                job.status = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
                job.finished_at = datetime.now(UTC)
    session.commit()


def _extract_params(model: Any) -> dict[str, Any]:
    """Pull JSON-safe hyperparams off the final estimator of a Pipeline.

    Falls back to ``repr(value)`` for anything that isn't JSON-serializable
    (e.g. nested estimators, numpy types) so the model-detail UI can still
    render the field rather than 500ing on the column read.
    """
    estimator = model
    if hasattr(model, "steps") and model.steps:
        estimator = model.steps[-1][1]
    try:
        raw = estimator.get_params(deep=False)
    except Exception:  # noqa: BLE001
        return {}
    import json

    out: dict[str, Any] = {}
    for k, v in raw.items():
        try:
            json.dumps(v)
            out[k] = v
        except Exception:  # noqa: BLE001
            out[k] = repr(v)
    return out


def _jsonable_dict(d: Any) -> dict[str, Any]:
    """Coerce a mapping to a JSON-safe dict (str / number / bool / None;
    everything else becomes repr). Used to stash tune ``best_params`` on a
    new trial without worrying about numpy scalars or non-serialisable
    estimator objects."""
    import json

    if not hasattr(d, "items"):
        return {}
    out: dict[str, Any] = {}
    for k, v in d.items():
        try:
            json.dumps(v)
            out[str(k)] = v
        except Exception:  # noqa: BLE001
            out[str(k)] = repr(v)
    return out


def _shrink_cv_history(cv_results: Any) -> dict[str, list]:
    """Reduce sklearn's ``cv_results_`` dict to plot-friendly columns.

    Tuning's search engine (RandomizedSearchCV / GridSearchCV) returns a
    rich dict keyed by the score / param / fold. We keep just the columns
    the UI needs: per-iteration mean score, std, and the params dict —
    bounded to 200 rows so a wild grid doesn't blow up the trial row.
    """
    if not hasattr(cv_results, "keys") and not isinstance(cv_results, dict):
        return {}
    try:
        mean = list(cv_results.get("mean_test_score", []))[:200]
        std = list(cv_results.get("std_test_score", []))[:200]
        # params is a list of dicts in sklearn's contract.
        raw_params = list(cv_results.get("params", []))[:200]
        params = [_jsonable_dict(p) for p in raw_params]
    except Exception:  # noqa: BLE001
        return {}
    return {
        "mean_test_score": [
            float(x) if isinstance(x, (int, float)) else None for x in mean
        ],
        "std_test_score": [
            float(x) if isinstance(x, (int, float)) else None for x in std
        ],
        "params": params,
    }


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
