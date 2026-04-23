"""The PyCaret 4.0 `Experiment` base class — a real sklearn-compatible object.

Design rationale in `docs/revamp/ARCHITECTURE.md`.

Two concrete guarantees this class delivers:

1. **It's a `BaseEstimator`.** `get_params`, `set_params`, `__sklearn_tags__`,
   `__sklearn_is_fitted__`, HTML repr, `clone` — all work.
2. **`fit(X, y=None)` is the real entry point.** PyCaret 4.0 is OOP-only; the
   3.x module-level `setup()` functional API was removed.

Class hierarchy
---------------

    Experiment
        ├── SupervisedExperiment
        │       ├── ClassificationExperiment
        │       ├── RegressionExperiment
        │       └── TimeSeriesExperiment
        └── UnsupervisedExperiment
                ├── ClusteringExperiment
                └── AnomalyExperiment

- `Experiment` (this module) hosts task-agnostic verbs: `fit`, `create_model`,
  `predict_model`, `plot_model`, `evaluate_model`, persistence, introspection,
  and data-access properties.
- `SupervisedExperiment` (`pycaret.core.supervised`) adds comparison/tuning
  verbs: `compare_models`, `tune_model`, `ensemble_model`, `blend_models`,
  `stack_models`, `calibrate_model`, `finalize_model`, `interpret_model`,
  `automl`, `get_leaderboard`.
- `UnsupervisedExperiment` (`pycaret.core.unsupervised`) adds `assign_model`.

Transition note
---------------

During the multi-session revamp every verb delegates to a legacy
`_SupervisedExperiment` / `_UnsupervisedExperiment` held as `self._legacy`.
Each verb is rewritten natively on top of `sklearn.pipeline.Pipeline` in
subsequent sessions, replacing the delegation call in place.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any

from sklearn.base import BaseEstimator

from pycaret.core.errors import ConfigurationError, NotFittedError
from pycaret.core.results import CreateResult, PredictResult
from pycaret.core.tasks import TaskType
from pycaret.logging.base import BaseLogger, NullLogger
from pycaret.logging.events import EventKind
from pycaret.logging.memory import MemoryLogger

if TYPE_CHECKING:
    import pandas as pd
    from sklearn.model_selection import BaseCrossValidator
    from sklearn.pipeline import Pipeline


class Experiment(BaseEstimator):
    """Task-agnostic base for every PyCaret 4.0 experiment.

    Parameters
    ----------
    task : TaskType
        Fixed by task subclasses; not intended for direct user configuration.
    target : str, optional
        Column name of the target / label. Required for supervised tasks only.
    session_id : int, optional
    train_size : float, default=0.7
    fold : int, default=10
    fold_strategy : str | BaseCrossValidator, default="stratifiedkfold"
    preprocess : bool, default=True
    normalize : bool, default=False
    transformation : bool, default=False
    remove_outliers : bool, default=False
    feature_selection : bool, default=False
    n_jobs : int, default=-1
    use_gpu : bool, default=False
    logger : BaseLogger, optional
        Event-stream logger. Default `NullLogger()` (silent).
    log_experiment : bool, default=False
        If True *and* `logger is None`, a `MemoryLogger` is installed.
    verbose : bool, default=False
        Compatibility flag passed to the legacy engine for progress-bar control.

    Notes
    -----
    - Parameters are stored verbatim per sklearn convention; no work in __init__.
    - Subclasses should pre-configure `task` and (optionally) override
      `_build_legacy_experiment()`.
    """

    # --------------------------------------------------------------------- init

    def __init__(
        self,
        *,
        task: TaskType = TaskType.CLASSIFICATION,
        target: str | None = None,
        session_id: int | None = None,
        train_size: float = 0.7,
        fold: int = 10,
        fold_strategy: str | BaseCrossValidator = "stratifiedkfold",
        preprocess: bool = True,
        normalize: bool = False,
        transformation: bool = False,
        remove_outliers: bool = False,
        feature_selection: bool = False,
        n_jobs: int = -1,
        use_gpu: bool = False,
        logger: BaseLogger | None = None,
        log_experiment: bool = False,
        verbose: bool = False,
    ) -> None:
        self.task = task
        self.target = target
        self.session_id = session_id
        self.train_size = train_size
        self.fold = fold
        self.fold_strategy = fold_strategy
        self.preprocess = preprocess
        self.normalize = normalize
        self.transformation = transformation
        self.remove_outliers = remove_outliers
        self.feature_selection = feature_selection
        self.n_jobs = n_jobs
        self.use_gpu = use_gpu
        self.logger = logger
        self.log_experiment = log_experiment
        self.verbose = verbose

    # --------------------------------------------------------------- sklearn API

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.requires_fit = True
        return tags

    def __sklearn_is_fitted__(self) -> bool:
        return getattr(self, "_fitted", False)

    # ------------------------------------------------------------- fit / setup

    def fit(
        self,
        X: pd.DataFrame,
        y: Any | None = None,
        **setup_kwargs: Any,
    ) -> Experiment:
        """Run preprocessing, splits, and state setup. Returns self.

        Supervised tasks:
            - Pass `X` containing the target column *or* pass `(X, y)`.
            - `self.target` must be set, or `y` must carry a name.

        Unsupervised tasks:
            - Pass `X` only. `self.target` is ignored.
        """
        import pandas as pd

        if self._is_supervised():
            data = self._coerce_supervised_fit_inputs(X, y)
        else:
            if isinstance(X, pd.Series):
                X = X.to_frame()
            data = X

        self._experiment_id = str(uuid.uuid4())
        self._legacy = self._build_legacy_experiment()

        if self.logger is None:
            self.logger = (
                MemoryLogger(experiment_id=self._experiment_id)
                if self.log_experiment
                else NullLogger()
            )

        t0 = time.perf_counter()
        self.logger.log(
            EventKind.EXPERIMENT_STARTED,
            message=f"Starting {self.task.value} experiment",
            payload={"target": self.target, "rows": len(data)},
        )

        self._legacy.setup(
            **self._build_legacy_setup_kwargs(data, setup_kwargs),
        )
        self._fitted = True

        self.logger.log(
            EventKind.EXPERIMENT_FITTED,
            message="Experiment fitted and ready",
            duration_ms=(time.perf_counter() - t0) * 1000,
        )
        return self

    # --- hooks for subclasses -----------------------------------------------

    def _is_supervised(self) -> bool:
        return self.task.is_supervised

    def _coerce_supervised_fit_inputs(self, X, y):
        """Normalize (X, y) vs (data) call shapes for supervised tasks."""
        import pandas as pd

        if y is None:
            if self.target is None:
                raise ConfigurationError(
                    "`target` must be set on the Experiment, or pass y= explicitly."
                )
            if self.target not in X.columns:
                raise ConfigurationError(
                    f"target column {self.target!r} not found in the DataFrame."
                )
            return X
        if isinstance(y, pd.Series) and y.name:
            target_name = str(y.name)
        else:
            target_name = self.target or "target"
        data = X.copy()
        data[target_name] = y
        if self.target is None:
            self.target = target_name
        return data

    def _build_legacy_experiment(self):
        """Construct the legacy-engine instance to delegate to.

        Task subclasses override this to choose the right god-class subclass
        from `pycaret.internal.pycaret_experiment`.
        """
        raise NotImplementedError("Subclasses must implement _build_legacy_experiment()")

    def _build_legacy_setup_kwargs(
        self, data: pd.DataFrame, extra: dict[str, Any]
    ) -> dict[str, Any]:
        """Map the Experiment's stored config onto the legacy `setup()` kwargs.

        Subclasses may override for task-specific setup parameters (e.g. time
        series uses `fh`, `seasonal_period` rather than `target`).
        """
        kwargs: dict[str, Any] = {
            "data": data,
            "target": self.target,
            "session_id": self.session_id,
            "train_size": self.train_size,
            "fold": self.fold,
            "fold_strategy": self.fold_strategy,
            "preprocess": self.preprocess,
            "normalize": self.normalize,
            "transformation": self.transformation,
            "remove_outliers": self.remove_outliers,
            "feature_selection": self.feature_selection,
            "n_jobs": self.n_jobs,
            "use_gpu": self.use_gpu,
            "verbose": self.verbose,
            "html": False,
            "log_experiment": False,
        }
        kwargs.update(extra)
        # Unsupervised legacy setup() has a narrower kwarg surface — drop
        # supervised-specific AND transformation-chain kwargs its signature
        # does not accept. (Clustering/anomaly preprocessing is controlled
        # via constructor args like `normalize` / `preprocess` only.)
        if not self._is_supervised():
            for k in (
                "target",
                "train_size",
                "fold",
                "fold_strategy",
                "transformation",
                "remove_outliers",
                "feature_selection",
            ):
                kwargs.pop(k, None)
        return kwargs

    # ------------------------------------------------------- task-agnostic verbs

    def _require_fitted(self) -> None:
        if not self.__sklearn_is_fitted__():
            raise NotFittedError("Experiment is not fitted. Call `.fit(data)` first.")

    def create_model(self, estimator: Any, *args: Any, **kwargs: Any) -> CreateResult:
        """Train a single model and return a typed `CreateResult`."""
        self._require_fitted()
        t0 = time.perf_counter()
        self.logger.log(
            EventKind.MODEL_CREATE_STARTED,
            payload={"estimator": self._describe_estimator(estimator)},
        )
        model = self._legacy.create_model(estimator, *args, **kwargs)
        metrics = self._safe_pull()
        self.logger.log(
            EventKind.MODEL_CREATED,
            duration_ms=(time.perf_counter() - t0) * 1000,
            payload={"estimator": self._describe_estimator(estimator)},
        )
        return CreateResult(
            pipeline=model,
            model_id=str(estimator) if isinstance(estimator, str) else type(model).__name__,
            metrics=metrics,
            params=self._safe_params(model),
        )

    def predict_model(self, estimator: Any, *args: Any, **kwargs: Any) -> PredictResult:
        """Run prediction and return a typed `PredictResult`."""
        self._require_fitted()
        predictions = self._legacy.predict_model(estimator, *args, **kwargs)
        metrics = self._safe_pull()
        self.logger.log(
            EventKind.MODEL_PREDICTED,
            payload={"n_rows": int(len(predictions)) if predictions is not None else 0},
        )
        return PredictResult(predictions=predictions, metrics=metrics)

    def plot_model(self, estimator: Any, *args: Any, **kwargs: Any) -> Any:
        """Delegates to the legacy plot dispatcher. Phase 3 of the roadmap
        replaces this with a Plotly-native registry."""
        self._require_fitted()
        return self._legacy.plot_model(estimator, *args, **kwargs)

    def evaluate_model(self, estimator: Any, *args: Any, **kwargs: Any) -> Any:
        self._require_fitted()
        return self._legacy.evaluate_model(estimator, *args, **kwargs)

    def pull(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Return the most recent metrics DataFrame emitted by the engine."""
        return self._legacy.pull(*args, **kwargs)

    def models(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return self._legacy.models(*args, **kwargs)

    def get_metrics(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return self._legacy.get_metrics(*args, **kwargs)

    def add_metric(self, *args: Any, **kwargs: Any) -> Any:
        return self._legacy.add_metric(*args, **kwargs)

    def remove_metric(self, *args: Any, **kwargs: Any) -> None:
        return self._legacy.remove_metric(*args, **kwargs)

    def get_config(self, *args: Any, **kwargs: Any) -> Any:
        return self._legacy.get_config(*args, **kwargs)

    def set_config(self, *args: Any, **kwargs: Any) -> Any:
        return self._legacy.set_config(*args, **kwargs)

    def save_model(self, model: Pipeline, path: Any, *args: Any, **kwargs: Any) -> Any:
        self._require_fitted()
        out = self._legacy.save_model(model, path, *args, **kwargs)
        self.logger.log(EventKind.MODEL_SAVED, payload={"path": str(path)})
        return out

    def load_model(self, *args: Any, **kwargs: Any) -> Any:
        return self._legacy.load_model(*args, **kwargs)

    def save_experiment(self, *args: Any, **kwargs: Any) -> Any:
        return self._legacy.save_experiment(*args, **kwargs)

    @staticmethod
    def load_experiment(*args: Any, **kwargs: Any) -> Any:
        from pycaret.internal.pycaret_experiment.supervised_experiment import _SupervisedExperiment

        return _SupervisedExperiment.load_experiment(*args, **kwargs)

    # ---------------------------------------------------------- introspection

    def list_models(self):
        """Typed model list for this experiment; runtime-aware."""
        from pycaret.api.describe import list_available_models

        return list_available_models(self)

    def list_metrics_cards(self):
        from pycaret.api.describe import list_metrics

        return list_metrics(self.task)

    def describe_setup_params(self):
        from pycaret.api.describe import describe_setup_params

        return describe_setup_params(self.task)

    # ----------------------------------------------- convenience properties

    @property
    def X(self) -> pd.DataFrame:
        self._require_fitted()
        return self._legacy.X

    @property
    def X_train(self) -> pd.DataFrame:
        self._require_fitted()
        return self._legacy.X_train

    @property
    def X_test(self) -> pd.DataFrame:
        self._require_fitted()
        return self._legacy.X_test

    @property
    def y(self):
        self._require_fitted()
        return self._legacy.y

    @property
    def y_train(self):
        self._require_fitted()
        return self._legacy.y_train

    @property
    def y_test(self):
        self._require_fitted()
        return self._legacy.y_test

    @property
    def preprocess_pipeline(self) -> Pipeline:
        self._require_fitted()
        return self._legacy.pipeline

    @property
    def events(self) -> list:
        """Replay of everything the logger saw (empty if not using MemoryLogger)."""
        return list(getattr(self.logger, "events", []))

    # ------------------------------------------------------- internal helpers

    def _safe_pull(self):
        try:
            return self._legacy.pull()
        except Exception:
            return None

    @staticmethod
    def _safe_params(model: Any) -> dict[str, Any]:
        try:
            return dict(model.get_params(deep=False).items())
        except Exception:
            return {}

    @staticmethod
    def _describe_estimator(estimator: Any) -> str:
        if isinstance(estimator, str):
            return estimator
        return type(estimator).__name__

    @staticmethod
    def _ranked_ids_from_leaderboard(leaderboard: Any) -> list[str]:
        try:
            if leaderboard is None:
                return []
            if hasattr(leaderboard, "columns") and "Model" in leaderboard.columns:
                return list(leaderboard["Model"].astype(str))
            return [str(i) for i in leaderboard.index]
        except Exception:
            return []
