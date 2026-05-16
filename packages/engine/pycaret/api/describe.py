"""Static + runtime introspection functions for the UI / agent.

The bulk of model and metric metadata in 3.x lived inside container class
docstrings and `__init__` bodies. For 4.0 we surface just enough *static* info
— from a hand-curated registry seeded from the legacy containers — to drive a
React form without actually constructing every container (which requires a
live experiment and seeds NumPy state, etc.).

Runtime-aware variants (`list_available_models(experiment)`) go through a live
Experiment so the returned cards reflect what's actually installed/active.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pycaret.api.cards import MetricCard, ModelCard, ParameterCard, ParameterKind
from pycaret.api.schemas import SetupParamSchema
from pycaret.core.errors import ConfigurationError, UnknownModelError
from pycaret.core.tasks import TaskType

if TYPE_CHECKING:
    from pycaret.core.experiment import Experiment


# -----------------------------------------------------------------------------
# Static registries.
#
# These are hand-curated for the common case (not exhaustive). The
# `list_available_models(experiment)` path goes through the legacy containers
# to return the *complete, runtime-accurate* picture.
# -----------------------------------------------------------------------------

_CLASSIFICATION_MODELS: list[ModelCard] = [
    ModelCard(
        id="lr",
        name="Logistic Regression",
        task="classification",
        library="sklearn",
        description="Linear model for binary/multiclass classification.",
        tags=["linear", "probabilistic"],
    ),
    ModelCard(
        id="knn",
        name="K-Neighbors Classifier",
        task="classification",
        library="sklearn",
        tags=["instance-based"],
    ),
    ModelCard(
        id="nb",
        name="Naive Bayes",
        task="classification",
        library="sklearn",
        tags=["probabilistic", "linear"],
    ),
    ModelCard(
        id="dt",
        name="Decision Tree Classifier",
        task="classification",
        library="sklearn",
        tags=["tree"],
    ),
    ModelCard(
        id="svm",
        name="SVM - Linear Kernel",
        task="classification",
        library="sklearn",
        tags=["linear", "margin-based"],
    ),
    ModelCard(
        id="rbfsvm",
        name="SVM - Radial Kernel",
        task="classification",
        library="sklearn",
        is_turbo=False,
        tags=["margin-based"],
    ),
    ModelCard(
        id="gpc",
        name="Gaussian Process Classifier",
        task="classification",
        library="sklearn",
        is_turbo=False,
        tags=["probabilistic"],
    ),
    ModelCard(
        id="mlp",
        name="MLP Classifier",
        task="classification",
        library="sklearn",
        is_turbo=False,
        tags=["neural"],
    ),
    ModelCard(
        id="ridge",
        name="Ridge Classifier",
        task="classification",
        library="sklearn",
        tags=["linear"],
    ),
    ModelCard(
        id="rf",
        name="Random Forest Classifier",
        task="classification",
        library="sklearn",
        tags=["tree", "ensemble"],
    ),
    ModelCard(
        id="qda",
        name="Quadratic Discriminant Analysis",
        task="classification",
        library="sklearn",
        tags=["probabilistic"],
    ),
    ModelCard(
        id="ada",
        name="Ada Boost Classifier",
        task="classification",
        library="sklearn",
        tags=["boosted"],
    ),
    ModelCard(
        id="gbc",
        name="Gradient Boosting Classifier",
        task="classification",
        library="sklearn",
        tags=["boosted"],
    ),
    ModelCard(
        id="lda",
        name="Linear Discriminant Analysis",
        task="classification",
        library="sklearn",
        tags=["linear", "probabilistic"],
    ),
    ModelCard(
        id="et",
        name="Extra Trees Classifier",
        task="classification",
        library="sklearn",
        tags=["tree", "ensemble"],
    ),
    ModelCard(
        id="xgboost",
        name="Extreme Gradient Boosting",
        task="classification",
        library="xgboost",
        gpu_enabled=True,
        tags=["boosted", "ensemble"],
    ),
    ModelCard(
        id="lightgbm",
        name="Light Gradient Boosting Machine",
        task="classification",
        library="lightgbm",
        gpu_enabled=True,
        tags=["boosted", "ensemble"],
    ),
    ModelCard(
        id="catboost",
        name="CatBoost Classifier",
        task="classification",
        library="catboost",
        gpu_enabled=True,
        tags=["boosted", "ensemble"],
    ),
    ModelCard(
        id="dummy",
        name="Dummy Classifier",
        task="classification",
        library="sklearn",
        is_turbo=False,
        tags=["baseline"],
    ),
]

_REGRESSION_MODELS: list[ModelCard] = [
    ModelCard(
        id="lr", name="Linear Regression", task="regression", library="sklearn", tags=["linear"]
    ),
    ModelCard(
        id="lasso",
        name="Lasso Regression",
        task="regression",
        library="sklearn",
        tags=["linear", "regularized"],
    ),
    ModelCard(
        id="ridge",
        name="Ridge Regression",
        task="regression",
        library="sklearn",
        tags=["linear", "regularized"],
    ),
    ModelCard(
        id="en",
        name="Elastic Net",
        task="regression",
        library="sklearn",
        tags=["linear", "regularized"],
    ),
    ModelCard(
        id="lar",
        name="Least Angle Regression",
        task="regression",
        library="sklearn",
        tags=["linear"],
    ),
    ModelCard(
        id="llar",
        name="Lasso Least Angle Regression",
        task="regression",
        library="sklearn",
        tags=["linear", "regularized"],
    ),
    ModelCard(
        id="omp",
        name="Orthogonal Matching Pursuit",
        task="regression",
        library="sklearn",
        tags=["linear", "sparse"],
    ),
    ModelCard(
        id="br",
        name="Bayesian Ridge",
        task="regression",
        library="sklearn",
        tags=["linear", "probabilistic"],
    ),
    ModelCard(
        id="ard",
        name="Automatic Relevance Determination",
        task="regression",
        library="sklearn",
        is_turbo=False,
        tags=["linear", "probabilistic"],
    ),
    ModelCard(
        id="par",
        name="Passive Aggressive Regressor",
        task="regression",
        library="sklearn",
        tags=["linear", "online"],
    ),
    ModelCard(
        id="ransac",
        name="Random Sample Consensus",
        task="regression",
        library="sklearn",
        is_turbo=False,
        tags=["linear", "robust"],
    ),
    ModelCard(
        id="tr",
        name="TheilSen Regressor",
        task="regression",
        library="sklearn",
        is_turbo=False,
        tags=["linear", "robust"],
    ),
    ModelCard(
        id="huber",
        name="Huber Regressor",
        task="regression",
        library="sklearn",
        tags=["linear", "robust"],
    ),
    ModelCard(
        id="kr",
        name="Kernel Ridge",
        task="regression",
        library="sklearn",
        is_turbo=False,
        tags=["kernel"],
    ),
    ModelCard(
        id="svm",
        name="Support Vector Regression",
        task="regression",
        library="sklearn",
        is_turbo=False,
        tags=["kernel"],
    ),
    ModelCard(
        id="knn",
        name="K-Neighbors Regressor",
        task="regression",
        library="sklearn",
        tags=["instance-based"],
    ),
    ModelCard(
        id="dt", name="Decision Tree Regressor", task="regression", library="sklearn", tags=["tree"]
    ),
    ModelCard(
        id="rf",
        name="Random Forest Regressor",
        task="regression",
        library="sklearn",
        tags=["tree", "ensemble"],
    ),
    ModelCard(
        id="et",
        name="Extra Trees Regressor",
        task="regression",
        library="sklearn",
        tags=["tree", "ensemble"],
    ),
    ModelCard(
        id="ada", name="AdaBoost Regressor", task="regression", library="sklearn", tags=["boosted"]
    ),
    ModelCard(
        id="gbr",
        name="Gradient Boosting Regressor",
        task="regression",
        library="sklearn",
        tags=["boosted"],
    ),
    ModelCard(
        id="mlp",
        name="MLP Regressor",
        task="regression",
        library="sklearn",
        is_turbo=False,
        tags=["neural"],
    ),
    ModelCard(
        id="xgboost",
        name="Extreme Gradient Boosting",
        task="regression",
        library="xgboost",
        gpu_enabled=True,
        tags=["boosted", "ensemble"],
    ),
    ModelCard(
        id="lightgbm",
        name="Light Gradient Boosting Machine",
        task="regression",
        library="lightgbm",
        gpu_enabled=True,
        tags=["boosted", "ensemble"],
    ),
    ModelCard(
        id="catboost",
        name="CatBoost Regressor",
        task="regression",
        library="catboost",
        gpu_enabled=True,
        tags=["boosted", "ensemble"],
    ),
    ModelCard(
        id="dummy",
        name="Dummy Regressor",
        task="regression",
        library="sklearn",
        is_turbo=False,
        tags=["baseline"],
    ),
]

_MODELS_BY_TASK: dict[TaskType, list[ModelCard]] = {
    TaskType.CLASSIFICATION: _CLASSIFICATION_MODELS,
    TaskType.REGRESSION: _REGRESSION_MODELS,
    # clustering / anomaly / time_series fall through to runtime-only listing below
}


_CLASSIFICATION_METRICS: list[MetricCard] = [
    MetricCard(
        id="Accuracy",
        name="Accuracy",
        task="classification",
        greater_is_better=True,
        is_default=True,
        description="Classification accuracy (fraction correct).",
    ),
    MetricCard(
        id="AUC",
        name="Area Under the Curve",
        task="classification",
        greater_is_better=True,
        description="ROC AUC (binary); weighted OvR for multiclass.",
    ),
    MetricCard(
        id="Recall",
        name="Recall",
        task="classification",
        greater_is_better=True,
        description="True-positive rate.",
    ),
    MetricCard(id="Prec.", name="Precision", task="classification", greater_is_better=True),
    MetricCard(id="F1", name="F1 Score", task="classification", greater_is_better=True),
    MetricCard(id="Kappa", name="Cohen's Kappa", task="classification", greater_is_better=True),
    MetricCard(
        id="MCC",
        name="Matthews Correlation Coefficient",
        task="classification",
        greater_is_better=True,
    ),
]

_REGRESSION_METRICS: list[MetricCard] = [
    MetricCard(id="MAE", name="Mean Absolute Error", task="regression", greater_is_better=False),
    MetricCard(id="MSE", name="Mean Squared Error", task="regression", greater_is_better=False),
    MetricCard(
        id="RMSE",
        name="Root Mean Squared Error",
        task="regression",
        greater_is_better=False,
        is_default=True,
    ),
    MetricCard(id="R2", name="R-Squared", task="regression", greater_is_better=True),
    MetricCard(
        id="RMSLE", name="Root Mean Squared Log Error", task="regression", greater_is_better=False
    ),
    MetricCard(
        id="MAPE", name="Mean Absolute Percentage Error", task="regression", greater_is_better=False
    ),
]

_METRICS_BY_TASK: dict[TaskType, list[MetricCard]] = {
    TaskType.CLASSIFICATION: _CLASSIFICATION_METRICS,
    TaskType.REGRESSION: _REGRESSION_METRICS,
}


# -----------------------------------------------------------------------------
# Setup-parameter schema (the common kwargs accepted by every Experiment.fit).
#
# This is the same cross-task core — task subclasses may extend it. A React
# form can render directly from the returned `SetupParamSchema`.
# -----------------------------------------------------------------------------


def _common_setup_params(task: TaskType) -> list[ParameterCard]:
    """Return the schema of every ``setup(...)`` knob exposed for ``task``.

    Hand-curated so each task only sees the params it actually accepts —
    e.g. clustering has no ``train_size``, time-series has no ``normalize``.
    Adding a new param? Add it here once, set the right group + help, and
    the React form picks it up automatically (no client work needed).
    """
    is_supervised = task in (TaskType.CLASSIFICATION, TaskType.REGRESSION)
    is_unsupervised = task in (TaskType.CLUSTERING, TaskType.ANOMALY)
    is_time_series = task == TaskType.TIME_SERIES

    if task == TaskType.CLASSIFICATION:
        fold_default = "stratifiedkfold"
        fold_choices = ["kfold", "stratifiedkfold", "groupkfold"]
    elif task == TaskType.REGRESSION:
        fold_default = "kfold"
        fold_choices = ["kfold", "groupkfold"]
    elif task == TaskType.TIME_SERIES:
        fold_default = "expanding"
        fold_choices = ["expanding", "sliding"]
    else:
        fold_default = "kfold"
        fold_choices = ["kfold"]

    params: list[ParameterCard] = []

    # ─── Data ──────────────────────────────────────────────────────
    if is_supervised:
        params.append(
            ParameterCard(
                name="target",
                kind=ParameterKind.COLUMN,
                required=True,
                group="Data",
                description="Column the model will learn to predict.",
            )
        )
        params.append(
            ParameterCard(
                name="train_size",
                kind=ParameterKind.FLOAT,
                default=0.7,
                minimum=0.1,
                maximum=0.99,
                group="Data",
                description=(
                    "Fraction of rows used for training; the remainder becomes "
                    "the holdout used to evaluate the final model."
                ),
            )
        )
    if is_time_series:
        params.append(
            ParameterCard(
                name="fh",
                kind=ParameterKind.INT,
                default=1,
                minimum=1,
                group="Data",
                description="Forecast horizon — how many steps ahead to predict.",
            )
        )
        params.append(
            ParameterCard(
                name="seasonal_period",
                kind=ParameterKind.INT,
                default=None,
                minimum=1,
                group="Data",
                description=(
                    "Seasonality length in steps (e.g. 12 for monthly with "
                    "yearly cycle). None lets the engine auto-detect."
                ),
            )
        )

    # ─── Preprocessing ────────────────────────────────────────────
    params.append(
        ParameterCard(
            name="preprocess",
            kind=ParameterKind.BOOL,
            default=True,
            group="Preprocessing",
            description=(
                "Master switch — when off, the dataset is fed to the model "
                "untouched. Useful when your data is already pipelined."
            ),
        )
    )
    if not is_time_series:
        params.append(
            ParameterCard(
                name="normalize",
                kind=ParameterKind.BOOL,
                default=False,
                group="Preprocessing",
                description=(
                    "Scale numeric features to zero-mean / unit-variance. "
                    "Helps distance-based models (KNN, SVM, neural nets)."
                ),
            )
        )
        params.append(
            ParameterCard(
                name="transformation",
                kind=ParameterKind.BOOL,
                default=False,
                group="Preprocessing",
                description=(
                    "Apply a power transform (Yeo-Johnson) to push numeric "
                    "features toward a Gaussian distribution."
                ),
            )
        )
    if is_supervised:
        params.append(
            ParameterCard(
                name="remove_outliers",
                kind=ParameterKind.BOOL,
                default=False,
                group="Preprocessing",
                description=(
                    "Drop rows flagged as outliers by Isolation Forest before "
                    "training. Off by default — outliers are sometimes signal."
                ),
            )
        )

    # ─── Feature engineering ──────────────────────────────────────
    if not is_time_series:
        params.append(
            ParameterCard(
                name="feature_selection",
                kind=ParameterKind.BOOL,
                default=False,
                group="Feature engineering",
                description=(
                    "Run a quick importance pass and drop low-signal features "
                    "before training. Speeds up high-cardinality datasets."
                ),
            )
        )

    # ─── Cross-validation ─────────────────────────────────────────
    if not is_unsupervised:
        params.append(
            ParameterCard(
                name="fold",
                kind=ParameterKind.INT,
                default=3 if is_time_series else 10,
                minimum=2,
                maximum=100,
                group="Cross-validation",
                description=(
                    "Number of CV folds. Higher = more stable metrics, slower runs."
                ),
            )
        )
        params.append(
            ParameterCard(
                name="fold_strategy",
                kind=ParameterKind.ENUM,
                default=fold_default,
                choices=fold_choices,
                group="Cross-validation",
                description=(
                    "How to split each fold. Stratified preserves class balance "
                    "for classification; expanding/sliding are time-series safe."
                ),
            )
        )

    # ─── Reproducibility ──────────────────────────────────────────
    params.append(
        ParameterCard(
            name="session_id",
            kind=ParameterKind.INT,
            default=None,
            group="Reproducibility",
            description=(
                "Random seed. Pin this to make a run reproducible row-for-row; "
                "leave None to let the engine pick a fresh seed."
            ),
        )
    )

    # ─── Compute ──────────────────────────────────────────────────
    params.append(
        ParameterCard(
            name="n_jobs",
            kind=ParameterKind.INT,
            default=-1,
            group="Compute",
            description="Parallel worker count. -1 uses every CPU core.",
        )
    )
    params.append(
        ParameterCard(
            name="use_gpu",
            kind=ParameterKind.BOOL,
            default=False,
            group="Compute",
            description=(
                "Route GPU-capable models (XGBoost / LightGBM / CatBoost) to "
                "their GPU backends. Quietly ignored when no GPU is visible."
            ),
        )
    )

    # ─── Logging ──────────────────────────────────────────────────
    params.append(
        ParameterCard(
            name="log_experiment",
            kind=ParameterKind.BOOL,
            default=False,
            group="Logging",
            description=(
                "Capture every engine event in the run's structured log. "
                "Always on for the platform — toggle this off only when "
                "running the engine standalone."
            ),
        )
    )
    params.append(
        ParameterCard(
            name="verbose",
            kind=ParameterKind.BOOL,
            default=False,
            group="Logging",
            description="Print engine progress to stdout (notebook-style).",
        )
    )
    return params


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def _coerce_task(task: TaskType | str) -> TaskType:
    if isinstance(task, TaskType):
        return task
    try:
        return TaskType(task)
    except ValueError as e:
        raise ConfigurationError(f"Unknown task: {task!r}") from e


def list_models(task: TaskType | str) -> list[ModelCard]:
    """Return every model registered for a task (static view — all variants,
    regardless of whether their package is installed)."""
    task = _coerce_task(task)
    models = _MODELS_BY_TASK.get(task)
    if models is None:
        return []
    return list(models)


def describe_model(task: TaskType | str, model_id: str) -> ModelCard:
    """Return the `ModelCard` for a specific model id (static view)."""
    for card in list_models(task):
        if card.id == model_id:
            return card
    raise UnknownModelError(f"{model_id!r} is not a registered model for task {task!r}.")


def list_metrics(task: TaskType | str) -> list[MetricCard]:
    """Return every metric registered for a task."""
    task = _coerce_task(task)
    return list(_METRICS_BY_TASK.get(task, []))


def describe_setup_params(task: TaskType | str) -> SetupParamSchema:
    """Return the schema describing every parameter `setup(...)` accepts.

    Schema is task-aware — params that don't apply to the chosen task
    (e.g. ``train_size`` for clustering) are filtered out. Groups are
    emitted in display order: Data → Preprocessing → Feature engineering
    → Cross-validation → Reproducibility → Compute → Logging.
    """
    task = _coerce_task(task)
    params = _common_setup_params(task)
    groups: list[str] = []
    seen: set[str] = set()
    for p in params:
        if p.group and p.group not in seen:
            groups.append(p.group)
            seen.add(p.group)
    return SetupParamSchema(task=task.value, parameters=params, groups=groups)


def list_available_models(experiment: Experiment) -> list[ModelCard]:
    """Runtime-aware model list for a specific live experiment.

    Flags `is_available=False` on cards whose library is missing. Uses the
    static registry as the source of truth for card metadata — availability
    is the only runtime-derived field.
    """
    from dataclasses import replace

    cards = list_models(experiment.task)
    available = _available_libraries()
    return [
        replace(card, is_available=(card.library in available or card.library == "sklearn"))
        for card in cards
    ]


def _available_libraries() -> set[str]:
    """Return the set of optional-dep libraries importable in this env."""
    from importlib.util import find_spec

    libs = {"sklearn"}  # always present as a core dep
    for name in ("xgboost", "catboost", "lightgbm", "sktime", "pyod"):
        if find_spec(name) is not None:
            libs.add(name)
    return libs
