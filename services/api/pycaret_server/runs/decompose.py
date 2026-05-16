"""Decompose a Run into per-Trial work units.

Phase 0-v2: a ``compare`` Run isn't one engine call anymore. The
orchestrator pre-creates N Trial rows (one per algorithm) + N Jobs
(``kind="trial"``); workers pull them in parallel and run
``create_model`` independently. This module owns the "which model ids
should we fan out to" decision.
"""

from __future__ import annotations

from typing import Any


# Default model lists per task. Mirrors the engine's compare_models
# defaults, trimmed to algorithms that are guaranteed to work without
# optional packages (no xgboost / lightgbm / catboost unless the user
# opts in via ``plan_params.include_models``).
_DEFAULTS: dict[str, list[str]] = {
    "classification": [
        "lr",
        "dt",
        "rf",
        "et",
        "ada",
        "gbc",
        "lda",
        "qda",
        "nb",
        "knn",
        "ridge",
        "svm",
    ],
    "regression": [
        "lr",
        "lasso",
        "ridge",
        "en",
        "lar",
        "llar",
        "omp",
        "br",
        "ard",
        "par",
        "ransac",
        "tr",
        "huber",
        "kr",
        "svm",
        "knn",
        "dt",
        "rf",
        "et",
        "ada",
        "gbr",
    ],
    "clustering": ["kmeans", "ap", "meanshift", "sc", "hclust", "dbscan", "optics", "birch"],
    "anomaly": [
        "abod",
        "iforest",
        "cluster",
        "cof",
        "histogram",
        "knn",
        "lof",
        "svm",
        "pca",
        "mcd",
        "sod",
        "sos",
    ],
    "time_series": [
        "naive",
        "snaive",
        "polytrend",
        "arima",
        "auto_arima",
        "ets",
        "exp_smooth",
        "theta",
        "stlf",
        "croston",
    ],
}


def model_ids_for_compare(
    task: str,
    *,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> list[str]:
    """Pick the model ids a compare Run should fan out to.

    Priority:
      1. ``include`` (caller's explicit list) — takes precedence.
      2. Default list for the task, filtered by ``exclude``.

    Returns a stable order so each compare Run's Trial-list reads
    consistently across runs.
    """
    if include:
        return [str(m) for m in include]
    base = list(_DEFAULTS.get(task, []))
    if exclude:
        ex = {str(m) for m in exclude}
        base = [m for m in base if m not in ex]
    return base


def default_model_ids(task: str) -> list[str]:
    """Public view of the per-task default list — for the UI's
    "show me what compare will run" preview."""
    return list(_DEFAULTS.get(task, []))
