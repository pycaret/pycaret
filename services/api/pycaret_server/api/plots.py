"""Plot rendering endpoints.

The engine's ``pycaret.plots.*`` modules return ``plotly.graph_objects.Figure``
objects. These endpoints expose them over HTTP as JSON the React UI's
``react-plotly.js`` component can consume directly.

Two render styles:

- **Run-bound** (``GET /runs/{run_id}/plots/{kind}``): for screens that
  show diagnostics for a fitted model from a recorded run. The route
  loads the most recently promoted Pipeline for the run + the holdout
  data referenced in the run snapshot, then generates the plot
  on-demand. (Disk caching is post-4.0 polish.)

- **Dataset-only** (``POST /datasets/{ds_id}/plots/eda``): for the EDA
  screen that doesn't need a fitted model — just the raw frame.

Both render styles return an envelope::

    {
        "kind": "<plot kind>",
        "task": "<classification|regression|...>",
        "figure": <plotly figure dict — pass directly to react-plotly>,
        "generated_at": <iso timestamp>,
    }
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from pycaret_server.auth import CurrentUser
from pycaret_server.db import DataSource, Pipeline, Run, get_db

# Plot kinds the API knows how to render. Maps task → list of kinds.
_PLOT_REGISTRY: dict[str, dict[str, dict[str, Any]]] = {
    "classification": {
        "confusion_matrix": {"requires": ["pipeline", "X_test", "y_test"]},
        "roc_curve": {"requires": ["pipeline", "X_test", "y_test"]},
        "pr_curve": {"requires": ["pipeline", "X_test", "y_test"]},
        "calibration_curve": {"requires": ["pipeline", "X_test", "y_test"], "binary_only": True},
        "threshold_curve": {"requires": ["pipeline", "X_test", "y_test"], "binary_only": True},
        "lift_curve": {"requires": ["pipeline", "X_test", "y_test"], "binary_only": True},
        "gain_curve": {"requires": ["pipeline", "X_test", "y_test"], "binary_only": True},
        "class_distribution": {"requires": ["y_train"]},
        "feature_importance": {"requires": ["pipeline"]},
        "permutation_importance": {"requires": ["pipeline", "X_test", "y_test"]},
    },
    "regression": {
        "residuals": {"requires": ["pipeline", "X_test", "y_test"]},
        "residuals_distribution": {"requires": ["pipeline", "X_test", "y_test"]},
        "prediction_error": {"requires": ["pipeline", "X_test", "y_test"]},
        "learning_curve": {"requires": ["pipeline", "X_train", "y_train"]},
        "feature_importance": {"requires": ["pipeline"]},
        "permutation_importance": {"requires": ["pipeline", "X_test", "y_test"]},
    },
    "clustering": {
        "elbow_curve": {"requires": ["pipeline", "X"]},
        "silhouette_curve": {"requires": ["pipeline", "X"]},
        "silhouette_plot": {"requires": ["pipeline", "X"]},
        "cluster_distribution": {"requires": ["pipeline"]},
        "embedding_2d": {"requires": ["pipeline", "X"]},
    },
    "anomaly": {
        "score_distribution": {"requires": ["pipeline", "X"]},
        "anomaly_map": {"requires": ["pipeline", "X"]},
    },
    "time_series": {
        "forecast": {"requires": ["y_true", "y_pred", "history"]},
        "decomposition": {"requires": ["y_train"]},
        "acf": {"requires": ["y_train"]},
        "pacf": {"requires": ["y_train"]},
        "residual_diagnostics": {"requires": ["y_true", "y_pred"]},
    },
    "eda": {
        "column_distribution": {"requires": ["data", "column"]},
        "correlation_heatmap": {"requires": ["data"]},
        "missingness_map": {"requires": ["data"]},
        "target_vs_feature": {"requires": ["data", "feature", "target"]},
        "profile_summary": {"requires": ["data"]},
    },
}


router = APIRouter(tags=["plots"])


# ---------------------------------------------------------------------------
# Discovery — what plots can I ask for?
# ---------------------------------------------------------------------------


@router.get("/plots/registry")
def plot_registry() -> dict[str, Any]:
    """Return the catalog of plot kinds per task.

    The UI uses this to populate the "Add chart" picker on the
    Model Card / Run Detail screens. Cheap; no auth needed.
    """
    return {
        "tasks": {task: list(kinds.keys()) for task, kinds in _PLOT_REGISTRY.items()},
        "details": _PLOT_REGISTRY,
    }


# ---------------------------------------------------------------------------
# Run-bound plots.
# ---------------------------------------------------------------------------


def _resolve_run_pipeline(run_id: str, db: Session) -> Pipeline:
    """Return the most recently promoted Pipeline for the run.

    Raises 404 if the run never produced a saved pipeline (i.e. the
    user hasn't called ``/promote`` yet).
    """
    pipe = db.scalar(
        select(Pipeline)
        .where(Pipeline.origin_run_id == run_id)
        .order_by(Pipeline.created_at.desc())
        .limit(1)
    )
    if pipe is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "No saved pipeline for this run. Call POST /runs/{id}/promote "
                "to register the fitted model first."
            ),
        )
    return pipe


def _load_pipeline_artifact(stored_path: str) -> Any:
    """Load the joblib-pickled pipeline from ``stored_path``."""
    import joblib

    return joblib.load(stored_path)


def _resolve_run_task(run: Run) -> str:
    """Best-effort task derivation from the run snapshot."""
    snap = run.snapshot or {}
    task = (snap.get("task") or snap.get("task_type") or "").lower()
    if task in _PLOT_REGISTRY:
        return task
    # Fallback: experiment.task_type (some runs only carry that one).
    if run.experiment is not None and getattr(run.experiment, "task_type", None):
        return str(run.experiment.task_type).lower()
    raise HTTPException(status_code=400, detail="Could not determine run task type.")


def _load_dataset_frame(snap: dict[str, Any], db: Session) -> Any:
    """Load the run's holdout source frame.

    Tries (in order):
      1. ``snap['data_source_id']`` → re-read the CSV upload from disk
      2. ``snap['sklearn_dataset']`` → re-load via the same helper the
         orchestrator uses for sklearn datasets (iris/wine/etc.)
      3. Otherwise returns None — callers raise 404

    ``data_inline`` runs are not re-loadable: the snapshot only stores
    the row count, not the rows themselves. Plots for those runs need
    a different strategy (store inline rows as a synthetic DataSource
    at run-submit time, TODO).
    """
    import pandas as pd

    # Path 1: CSV upload via DataSource row.
    data_source_id = snap.get("data_source_id")
    if data_source_id:
        ds = db.get(DataSource, data_source_id)
        if ds is not None:
            cfg = ds.config or {}
            path = cfg.get("path") or cfg.get("stored_path")
            if path:
                try:
                    path_str = str(path)
                    if path_str.endswith(".parquet"):
                        return pd.read_parquet(path_str)
                    return pd.read_csv(path_str)
                except Exception:
                    pass

    # Path 2: bundled sklearn dataset (iris / wine / breast_cancer / diabetes).
    sklearn_name = snap.get("sklearn_dataset")
    if sklearn_name:
        try:
            from pycaret_server.runs.plans import load_sklearn_dataset

            df, _target = load_sklearn_dataset(sklearn_name)
            return df
        except Exception:
            pass

    return None


@router.get("/runs/{run_id}/plots/{kind}")
def render_run_plot(
    run_id: str,
    kind: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict[str, Any]:
    """Generate a plot for a recorded run.

    Reloads the promoted pipeline + holdout data, calls the matching
    function in ``pycaret.plots.<task>``, returns the Plotly figure as
    a dict. Heavy on the first request — caching is post-4.0.0 polish.
    """
    from pycaret_server.api.runs import _run_access  # late import to avoid cycle

    run = _run_access(run_id, user, db)
    task = _resolve_run_task(run)

    if kind not in _PLOT_REGISTRY.get(task, {}):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown plot kind {kind!r} for task {task!r}.",
        )

    pipe_row = _resolve_run_pipeline(run_id, db)
    pipeline = _load_pipeline_artifact(pipe_row.stored_path)

    # Recover the holdout frame from the run snapshot. Falls back across
    # data_source_id → sklearn_dataset (data_inline isn't re-loadable yet).
    snap = run.snapshot or {}
    target = snap.get("target")
    df = _load_dataset_frame(snap, db)
    if df is None:
        # Be specific so the frontend can surface a useful empty state.
        if snap.get("data_inline_rows"):
            detail = (
                "This run was created with inline rows (data_inline). "
                "Plot rendering for inline-data runs is not yet supported — "
                "re-run with a CSV upload or sklearn dataset to render diagnostics."
            )
        else:
            detail = (
                "Could not load the holdout dataset referenced by the run "
                "snapshot. The original CSV may have been deleted from disk."
            )
        raise HTTPException(status_code=404, detail=detail)

    # If the target wasn't pinned in the snapshot but we loaded a
    # sklearn dataset, the helper exposes the conventional target name
    # (e.g., "target" for iris). Re-derive only when missing.
    if not target and snap.get("sklearn_dataset"):
        try:
            from pycaret_server.runs.plans import load_sklearn_dataset

            _, target = load_sklearn_dataset(snap["sklearn_dataset"])
        except Exception:
            pass

    # Recompute a holdout split with the run's session_id for stability.
    figure = _compute_run_plot(task, kind, pipeline, df, target, snap)

    return {
        "kind": kind,
        "task": task,
        "figure": figure.to_dict() if hasattr(figure, "to_dict") else figure,
        "generated_at": datetime.now(UTC).isoformat(),
    }


def _compute_run_plot(
    task: str, kind: str, pipeline: Any, df: Any, target: str | None, snap: dict[str, Any]
) -> Any:
    """Dispatch to the right ``pycaret.plots.<task>.<kind>`` function.

    Splits the dataframe into holdout train/test using the snapshot's
    ``train_size`` + ``session_id`` so we re-derive what the engine had.
    """
    from pycaret import plots as p
    from sklearn.model_selection import train_test_split

    train_size = float(snap.get("train_size", 0.7))
    seed = int(snap.get("session_id") or 0)

    if task == "classification":
        if target is None or target not in df.columns:
            raise HTTPException(
                status_code=400, detail="Run snapshot is missing the target column."
            )
        y = df[target]
        X = df.drop(columns=[target])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, random_state=seed, stratify=y
        )
        if kind == "confusion_matrix":
            return p.classification.confusion_matrix(pipeline, X_test, y_test)
        if kind == "roc_curve":
            return p.classification.roc_curve(pipeline, X_test, y_test)
        if kind == "pr_curve":
            return p.classification.pr_curve(pipeline, X_test, y_test)
        if kind == "calibration_curve":
            return p.classification.calibration_curve(pipeline, X_test, y_test)
        if kind == "threshold_curve":
            return p.classification.threshold_curve(pipeline, X_test, y_test)
        if kind == "lift_curve":
            return p.classification.lift_curve(pipeline, X_test, y_test)
        if kind == "gain_curve":
            return p.classification.gain_curve(pipeline, X_test, y_test)
        if kind == "class_distribution":
            return p.classification.class_distribution(y_train)
        if kind == "feature_importance":
            return p.regression.feature_importance(pipeline, feature_names=list(X_train.columns))
        if kind == "permutation_importance":
            return p.feature.permutation_importance(pipeline, X_test, y_test, n_repeats=5)

    if task == "regression":
        if target is None or target not in df.columns:
            raise HTTPException(
                status_code=400, detail="Run snapshot is missing the target column."
            )
        y = df[target]
        X = df.drop(columns=[target])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, random_state=seed
        )
        if kind == "residuals":
            return p.regression.residuals(pipeline, X_test, y_test)
        if kind == "residuals_distribution":
            return p.regression.residuals_distribution(pipeline, X_test, y_test)
        if kind == "prediction_error":
            return p.regression.prediction_error(pipeline, X_test, y_test)
        if kind == "learning_curve":
            return p.regression.learning_curve(pipeline, X_train, y_train, cv=3)
        if kind == "feature_importance":
            return p.regression.feature_importance(pipeline, feature_names=list(X_train.columns))
        if kind == "permutation_importance":
            return p.feature.permutation_importance(pipeline, X_test, y_test, n_repeats=5)

    if task == "clustering":
        if kind == "elbow_curve":
            return p.clustering.elbow_curve(pipeline, df)
        if kind == "silhouette_curve":
            return p.clustering.silhouette_curve(pipeline, df)
        if kind == "silhouette_plot":
            return p.clustering.silhouette_plot(pipeline, df)
        if kind == "cluster_distribution":
            return p.clustering.cluster_distribution(pipeline)
        if kind == "embedding_2d":
            return p.clustering.embedding_2d(pipeline, df)

    if task == "anomaly":
        if kind == "score_distribution":
            return p.anomaly.score_distribution(pipeline, df)
        if kind == "anomaly_map":
            return p.anomaly.anomaly_map(pipeline, df)

    raise HTTPException(
        status_code=501,
        detail=f"plot {task!r}.{kind!r} is registered but its computation isn't wired yet.",
    )


# ---------------------------------------------------------------------------
# Dataset-only plots (EDA).
# ---------------------------------------------------------------------------


@router.get("/datasets/{data_source_id}/plots/eda/{kind}")
def render_eda_plot(
    data_source_id: str,
    kind: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
    column: str | None = None,
    feature: str | None = None,
    target: str | None = None,
) -> dict[str, Any]:
    """Generate an EDA plot from a stored dataset.

    Used by the EDA screen — no fitted model required; just the raw
    frame. Per-plot params (``column`` / ``feature`` / ``target``) come
    from query string for stateless caching.
    """
    if kind not in _PLOT_REGISTRY["eda"]:
        raise HTTPException(status_code=400, detail=f"Unknown EDA plot kind {kind!r}.")
    df = _load_dataset_frame({"data_source_id": data_source_id}, db)
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not found or unreadable.")

    from pycaret import plots as p

    if kind == "column_distribution":
        if not column:
            raise HTTPException(status_code=400, detail="?column=<name> required.")
        figure = p.eda.column_distribution(df, column)
    elif kind == "correlation_heatmap":
        figure = p.eda.correlation_heatmap(df)
    elif kind == "missingness_map":
        figure = p.eda.missingness_map(df)
    elif kind == "target_vs_feature":
        if not feature or not target:
            raise HTTPException(status_code=400, detail="?feature=<a>&target=<b> both required.")
        figure = p.eda.target_vs_feature(df, feature, target)
    elif kind == "profile_summary":
        figure = p.eda.profile_summary(df)
    else:
        raise HTTPException(status_code=501, detail=f"EDA kind {kind!r} not yet wired.")

    return {
        "kind": kind,
        "task": "eda",
        "figure": figure.to_dict(),
        "generated_at": datetime.now(UTC).isoformat(),
    }
