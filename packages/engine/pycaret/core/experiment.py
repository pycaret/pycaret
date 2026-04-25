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
        # Session-29 drain: snapshot the user-facing data accessors off the
        # legacy state once at fit time. The properties (`self.X`, `X_train`
        # etc.) now read these snapshots instead of going through
        # ``self._legacy.<attr>`` on every access. Live mutation semantics
        # are preserved because we hold references, not copies.
        self._snapshot_fit_state()
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

    def create_model(
        self,
        estimator: Any,
        *,
        fold: Any | None = None,
        cross_validation: bool = True,
        fit_kwargs: dict | None = None,
        round: int = 4,
        verbose: bool = False,
        **estimator_kwargs: Any,
    ) -> CreateResult:
        """Train a single model and return a typed ``CreateResult``.

        Session-24 drain (supervised path): for classification + regression
        experiments, this verb no longer delegates to
        ``self._legacy.create_model``. It resolves the estimator, runs
        cross-validation with the task's metric registry, refits on the
        full training set, and returns a **real sklearn Pipeline**
        (preprocessor + trained model glued together). Downstream verbs
        like ``predict_model`` can call ``.predict`` on the returned
        pipeline directly — no more transitional bare-estimator branch.

        Clustering / anomaly / time-series still delegate to the legacy
        engine (their create_model code paths have different shapes;
        separate drain sessions).

        Parameters
        ----------
        estimator : str or sklearn-compatible object
            Either a model ID from the engine's registry (e.g. ``"lr"``,
            ``"rf"``) or a pre-constructed estimator with ``.fit`` /
            ``.predict``.
        fold : int or cross-validator, optional
            If ``None``, uses the experiment's configured CV generator
            (``self.fold`` + ``self.fold_strategy``).
        cross_validation : bool, default=True
            If False, skips CV + just fits on the training set.
        fit_kwargs : dict, optional
            Extra kwargs forwarded to ``model.fit``.
        round : int, default=4
            Decimal places for the metrics DataFrame.
        verbose : bool, default=False
            Reserved; currently ignored (legacy progress-bar hook).
        **estimator_kwargs
            When ``estimator`` is a registry ID, forwarded to its
            constructor (merged over the registry's defaults).

        Returns
        -------
        CreateResult
            ``pipeline`` is a real sklearn Pipeline (preprocessing +
            trained model); ``metrics`` is the per-fold score DataFrame
            (index: ``Fold 0..N-1``, ``Mean``, ``Std``) when CV ran, else
            ``None``; ``model_id`` is the registry key or the estimator
            class name; ``params`` is the estimator's `get_params(deep=False)`.
        """
        self._require_fitted()

        from pycaret.core.tasks import TaskType

        # Non-supervised / time-series paths still on the legacy engine.
        if self.task not in (TaskType.CLASSIFICATION, TaskType.REGRESSION):
            # Forward whatever the caller passed as estimator kwargs
            # (num_clusters= / fraction= for clustering, fh= for TS, etc.)
            # but DROP `fold` which only applies to supervised CV.
            return self._create_model_legacy(
                estimator,
                verbose=verbose,
                **estimator_kwargs,
            )

        return self._create_model_supervised_native(
            estimator,
            fold=fold,
            cross_validation=cross_validation,
            fit_kwargs=fit_kwargs or {},
            round=round,
            verbose=verbose,
            estimator_kwargs=estimator_kwargs,
        )

    # ---------------------- create_model — native supervised path

    def _create_model_supervised_native(
        self,
        estimator: Any,
        *,
        fold: Any | None,
        cross_validation: bool,
        fit_kwargs: dict,
        round: int,
        verbose: bool,
        estimator_kwargs: dict,
    ) -> CreateResult:
        """Native supervised create_model (classification + regression)."""
        from copy import deepcopy

        t0 = time.perf_counter()

        # ---- resolve estimator: str → registry → instance; else use as-is
        model, model_id = self._resolve_supervised_estimator(estimator, estimator_kwargs)

        self.logger.log(
            EventKind.MODEL_CREATE_STARTED,
            payload={"estimator": model_id},
        )

        # ---- pull transformed training data from the fit-time snapshot
        X_train = self._fit_state["X_train_transformed"]
        y_train = self._fit_state["y_train_transformed"]

        # ---- CV (optional) + collect the metrics DataFrame
        metrics_df = None
        if cross_validation:
            cv = fold if fold is not None else self._fit_state["fold_generator"]
            metrics_df = self._cross_validate_supervised(
                model=model, X=X_train, y=y_train, cv=cv, round_=round
            )

        # ---- final fit on the full training set
        model.fit(X_train, y_train, **fit_kwargs)

        # ---- assemble a full Pipeline: preprocessor + trained model
        pipeline = deepcopy(self.preprocess_pipeline)
        pipeline.steps.append((model_id, model))

        # ---- emit event + build result
        self.logger.log(
            EventKind.MODEL_CREATED,
            duration_ms=(time.perf_counter() - t0) * 1000,
            payload={"estimator": model_id},
        )
        self._set_last_metrics(metrics_df)
        return CreateResult(
            pipeline=pipeline,
            model_id=model_id,
            metrics=metrics_df,
            params=self._safe_params(model),
        )

    def _resolve_supervised_estimator(
        self, estimator: Any, estimator_kwargs: dict
    ) -> tuple[Any, str]:
        """Return (instantiated model, model_id) from a string or object."""
        if isinstance(estimator, str):
            registry = self._fit_state.get("model_registry", {})
            if not registry or estimator not in registry:
                raise ConfigurationError(
                    f"Unknown model id {estimator!r}. Call "
                    "Experiment.list_models() for available IDs."
                )
            container = registry[estimator]
            merged = {**container.args, **estimator_kwargs}
            return container.class_def(**merged), estimator
        if not hasattr(estimator, "fit") or not hasattr(estimator, "predict"):
            raise TypeError(
                "`estimator` must be a registry ID string or an object with "
                ".fit / .predict methods."
            )
        return estimator, type(estimator).__name__

    def _cross_validate_supervised(
        self,
        *,
        model: Any,
        X: pd.DataFrame,
        y: Any,
        cv: Any,
        round_: int = 4,
    ) -> pd.DataFrame | None:
        """Run k-fold CV + build a per-fold DataFrame of metrics.

        Uses ``deepcopy(model)`` per fold so fits don't leak across folds.
        Metric columns come from the task's metric registry (same one
        used by ``predict_model``); index is ``Fold 0..N-1`` then
        ``Mean`` and ``Std`` rows.
        """
        from copy import deepcopy

        import pandas as _pd

        from pycaret.core.tasks import TaskType
        from pycaret.utils.generic import calculate_metrics

        try:
            if self.task == TaskType.CLASSIFICATION:
                from pycaret.containers.metrics.classification import (
                    get_all_metric_containers,
                )

                metrics_registry = get_all_metric_containers({}, raise_errors=False)
            elif self.task == TaskType.REGRESSION:
                from pycaret.containers.metrics.regression import (
                    get_all_metric_containers,
                )

                metrics_registry = get_all_metric_containers({}, raise_errors=False)
            else:
                return None
        except Exception:
            return None

        per_fold: list[dict] = []
        try:
            split_iter = cv.split(X, y)
        except TypeError:
            split_iter = cv.split(X)
        for train_idx, val_idx in split_iter:
            X_tr = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
            X_val = X.iloc[val_idx] if hasattr(X, "iloc") else X[val_idx]
            y_tr = y.iloc[train_idx] if hasattr(y, "iloc") else y[train_idx]
            y_val = y.iloc[val_idx] if hasattr(y, "iloc") else y[val_idx]

            m = deepcopy(model)
            try:
                m.fit(X_tr, y_tr)
            except Exception:
                # A single-fold fit failure shouldn't sink the whole CV;
                # record zeros + continue.
                per_fold.append({})
                continue
            try:
                preds = m.predict(X_val)
            except Exception:
                per_fold.append({})
                continue
            try:
                proba = m.predict_proba(X_val)
                pred_proba = proba[:, 1] if proba.shape[1] == 2 else proba
            except Exception:
                pred_proba = None
            try:
                scores = calculate_metrics(
                    metrics=metrics_registry,
                    y_test=y_val,
                    pred=preds,
                    pred_proba=pred_proba,
                )
            except Exception:
                scores = {}
            per_fold.append(scores)

        if not per_fold or not any(per_fold):
            return None

        df = _pd.DataFrame(per_fold)
        df.index = [f"Fold {i}" for i in range(len(df))]
        # Mean + Std aggregate rows. Guard numeric-only so string columns
        # (if any ever sneak in) don't break the mean call.
        df.loc["Mean"] = df.mean(numeric_only=True)
        df.loc["Std"] = df.std(numeric_only=True)
        return df.round(round_)

    def _create_model_legacy(
        self, estimator: Any, *, verbose: bool = False, **kwargs: Any
    ) -> CreateResult:
        """Fallback for tasks whose create_model hasn't been drained yet
        (time-series, clustering, anomaly). Delegates to the legacy engine.

        ``**kwargs`` carries task-specific parameters (``num_clusters=``,
        ``fraction=``, ``fh=``, …). Caller is responsible for not passing
        supervised-only kwargs like ``fold=``.
        """
        t0 = time.perf_counter()
        self.logger.log(
            EventKind.MODEL_CREATE_STARTED,
            payload={"estimator": self._describe_estimator(estimator)},
        )
        model = self._legacy.create_model(estimator, verbose=verbose, **kwargs)
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

    def predict_model(
        self,
        estimator: Any,
        data: pd.DataFrame | None = None,
        *,
        raw_score: bool = False,
        round: int = 4,
        verbose: bool = False,
    ) -> PredictResult:
        """Run prediction and return a typed ``PredictResult``.

        Session-23 drain: this verb no longer delegates to
        ``self._legacy.predict_model``. It calls ``estimator.predict`` /
        ``estimator.predict_proba`` directly, with a transitional accommodation
        for bare estimators.

        Parameters
        ----------
        estimator : sklearn.pipeline.Pipeline or fitted estimator
            Preferred: a fitted sklearn Pipeline with preprocessing baked in
            (that's what ``create_model`` / ``compare_models`` / ``tune_model``
            will return once their drains land in sessions 24+). For now,
            a bare fitted estimator is also accepted — we wrap it on-the-fly
            with ``self.preprocess_pipeline`` to transform new data. Must
            implement ``.predict``; raises TypeError otherwise.
        data : pandas.DataFrame, optional
            New input. If ``None``, the holdout set is used (``self.X_test``
            / ``self.y_test`` for supervised tasks, ``self.X`` for
            unsupervised). If ``data`` contains the target column, it's
            split off automatically + used to compute metrics.
        raw_score : bool, default=False
            Classification only. True → per-class probability columns
            (``prediction_score_<class>``). False (default) → single
            ``prediction_score`` column with winning-class probability.
        round : int, default=4
            Decimal places for probability / metric columns.
        verbose : bool, default=False
            Reserved for future notebook-progress hooks; currently ignored.

        Returns
        -------
        PredictResult
            ``predictions`` is a DataFrame with the original X + (if known)
            the target column + ``prediction_label`` (+ optional
            ``prediction_score`` for classification). Clustering / anomaly
            use task-specific columns (``Cluster`` / ``Anomaly`` +
            ``Anomaly_Score``).
            ``metrics`` is a one-row DataFrame for supervised tasks when
            ``y`` is known; ``None`` otherwise.
        """
        self._require_fitted()

        import numpy as np
        import pandas as _pd
        from sklearn.pipeline import Pipeline as _SkPipeline

        from pycaret.core.tasks import TaskType
        from pycaret.utils.constants import LABEL_COLUMN, SCORE_COLUMN

        if not hasattr(estimator, "predict"):
            raise TypeError(
                "predict_model expects a fitted estimator with a `.predict` "
                "method (ideally a sklearn Pipeline with preprocessing "
                f"baked in). Got {type(estimator).__name__!r}."
            )

        t0 = time.perf_counter()

        # -------- decide whether we need to transform data first.
        # Post session 24 + session 28: supervised + unsupervised create_model
        # both return real Pipelines, so the transitional bare-estimator
        # branch is dead for the supported task types. We keep it as a
        # belt-and-braces fallback for callers passing in their own bare
        # estimators (uncommon but legal — `predict_model` accepts any
        # object with `.predict`).
        estimator_is_pipeline = isinstance(estimator, _SkPipeline)
        preprocessor: Any | None = None
        if not estimator_is_pipeline:
            try:
                preprocessor = self.preprocess_pipeline
            except Exception:
                preprocessor = None

        # -------- source X + optional y
        is_supervised = self._is_supervised()
        if data is None:
            if is_supervised:
                X, y = self.X_test, self.y_test
            else:
                X, y = self.X, None
        else:
            if is_supervised and self.target is not None and self.target in data.columns:
                y = data[self.target]
                X = data.drop(columns=[self.target])
            else:
                X, y = data, None

        # -------- predict
        if preprocessor is not None and not estimator_is_pipeline:
            # Transform X through the legacy preprocessing chain first.
            X_for_pred = preprocessor.transform(X)
            if isinstance(X_for_pred, tuple):
                # InternalPipeline.transform can return (X, y) when y is
                # passed in; we only want X.
                X_for_pred = X_for_pred[0]
        else:
            X_for_pred = X

        preds = np.asarray(estimator.predict(X_for_pred))

        out = X.copy()
        if is_supervised and y is not None:
            # Re-attach target column so ground truth is visible next to
            # prediction_label. Index alignment is implicit — pandas handles it.
            out[self.target] = y

        # -------- task-specific label + score columns
        if self.task == TaskType.CLUSTERING:
            out["Cluster"] = [f"Cluster {i}" for i in preds]
        elif self.task == TaskType.ANOMALY:
            out["Anomaly"] = preds
            if hasattr(estimator, "decision_function"):
                try:
                    out["Anomaly_Score"] = estimator.decision_function(X)
                except Exception:  # pragma: no cover — defensive
                    pass
        else:
            # supervised: classification / regression / time-series
            out[LABEL_COLUMN] = preds
            if self.task == TaskType.CLASSIFICATION and hasattr(estimator, "predict_proba"):
                try:
                    proba = estimator.predict_proba(X_for_pred)
                    classes = list(getattr(estimator, "classes_", range(proba.shape[1])))
                    if raw_score:
                        for i, cls in enumerate(classes):
                            out[f"{SCORE_COLUMN}_{cls}"] = np.round(proba[:, i], round)
                    elif proba.shape[1] == 2:
                        out[SCORE_COLUMN] = np.round(proba[:, 1], round)
                    else:
                        out[SCORE_COLUMN] = np.round(proba.max(axis=1), round)
                except Exception:  # pragma: no cover — not all classifiers have proba
                    pass

        # -------- metrics (supervised + y known)
        metrics_df: _pd.DataFrame | None = None
        if is_supervised and y is not None:
            metrics_df = self._compute_predict_metrics(
                estimator=estimator, X=X_for_pred, y=y, preds=preds, round=round
            )

        # -------- log + return
        self.logger.log(
            EventKind.MODEL_PREDICTED,
            duration_ms=(time.perf_counter() - t0) * 1000,
            payload={"n_rows": int(len(out))},
        )
        return PredictResult(predictions=out, metrics=metrics_df)

    # ---------- helpers for predict_model

    def _compute_predict_metrics(
        self,
        *,
        estimator: Any,
        X: pd.DataFrame,
        y: Any,
        preds: Any,
        round: int = 4,
    ) -> pd.DataFrame | None:
        """Compute one-row metrics DataFrame for a supervised prediction.

        Uses the same metric registry legacy used — classification gets
        Accuracy / AUC / Precision / Recall / F1 / ..., regression gets
        MAE / MSE / RMSE / R² / MAPE. Returns ``None`` if the registry
        disagrees with the task or anything else goes sideways (metrics
        are advisory, not load-bearing; a predict should never fail
        because a metric registry choked).
        """
        from pycaret.core.tasks import TaskType

        try:
            import pandas as _pd

            from pycaret.utils.generic import calculate_metrics

            if self.task == TaskType.CLASSIFICATION:
                from pycaret.containers.metrics.classification import (
                    get_all_metric_containers,
                )

                metrics_registry = get_all_metric_containers({}, raise_errors=False)
                try:
                    proba = estimator.predict_proba(X)
                    pred_proba = proba[:, 1] if proba.shape[1] == 2 else proba
                except Exception:
                    pred_proba = None
            elif self.task == TaskType.REGRESSION:
                from pycaret.containers.metrics.regression import (
                    get_all_metric_containers,
                )

                metrics_registry = get_all_metric_containers({}, raise_errors=False)
                pred_proba = None
            else:
                return None

            scores = calculate_metrics(
                metrics=metrics_registry,
                y_test=y,
                pred=preds,
                pred_proba=pred_proba,
            )
            if not scores:
                return None
            df = _pd.DataFrame(scores, index=[0])
            model_name = type(
                estimator.steps[-1][1] if hasattr(estimator, "steps") else estimator
            ).__name__
            df.insert(0, "Model", model_name)
            return df.round(round)
        except Exception:
            return None

    def plot_model(self, estimator: Any, *args: Any, **kwargs: Any) -> Any:
        """Delegates to the legacy plot dispatcher. Phase 3 of the roadmap
        replaces this with a Plotly-native registry."""
        self._require_fitted()
        return self._legacy.plot_model(estimator, *args, **kwargs)

    def evaluate_model(self, estimator: Any, *args: Any, **kwargs: Any) -> Any:
        """Interactive evaluation widget. Builds on plot_model — same Phase-3
        replacement target. Still delegates."""
        self._require_fitted()
        return self._legacy.evaluate_model(estimator, *args, **kwargs)

    # ----------------------------------------------- session-31 native verbs

    def pull(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Return the most recent metrics DataFrame.

        Session-31 drain: reads from ``self._fit_state["last_metrics"]``,
        which is updated by each native verb (``create_model``, ``tune_model``,
        ``compare_models``, ensemble / blend / stack / calibrate / finalize)
        before it returns. If the snapshot doesn't have a value (no native
        verb has run yet, or a TS-fallback path was taken), we fall back to
        the legacy ``self._legacy.pull()``.
        """
        self._require_fitted()
        last = self._fit_state.get("last_metrics") if hasattr(self, "_fit_state") else None
        if last is not None:
            return last
        # TS / unsupervised legacy fallback path — only reachable when a
        # non-drained verb populated the legacy display container.
        return self._legacy.pull(*args, **kwargs)

    def models(self, *args: Any, internal: bool = False, **kwargs: Any) -> pd.DataFrame:
        """Return a DataFrame describing the available models in the registry.

        Session-31 drain: builds the DataFrame from the model-registry
        snapshot in ``self._fit_state["model_registry"]``. ``internal=True``
        falls through to the legacy implementation, which exposes the full
        ``ModelContainer`` row (with engine-internal fields). Mirrors the
        legacy contract.
        """
        self._require_fitted()
        if internal:
            # The internal=True view exposes a richer set of fields that
            # come directly from the legacy holder; preserve that.
            return self._legacy.models(*args, internal=True, **kwargs)

        import pandas as _pd

        registry = self._fit_state.get("model_registry", {}) if hasattr(self, "_fit_state") else {}
        if not registry:
            return self._legacy.models(*args, **kwargs)

        rows: list[dict] = []
        for mid, container in registry.items():
            if getattr(container, "is_special", False):
                continue
            rows.append(
                {
                    "ID": mid,
                    "Name": getattr(container, "name", type(container).__name__),
                    "Reference": (
                        f"{container.class_def.__module__}.{container.class_def.__name__}"
                        if getattr(container, "class_def", None) is not None
                        else None
                    ),
                    "Turbo": bool(getattr(container, "is_turbo", False)),
                }
            )
        df = _pd.DataFrame(rows).set_index("ID")
        return df

    def get_metrics(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Return a DataFrame describing the available metrics for the task.

        Session-31 drain: reads from the task's metric registry (the same
        ``pycaret.containers.metrics.<task>.get_all_metric_containers``
        helper that ``create_model`` / ``predict_model`` use internally).
        """
        self._require_fitted()
        from pycaret.core.tasks import TaskType

        try:
            if self.task == TaskType.CLASSIFICATION:
                from pycaret.containers.metrics.classification import (
                    get_all_metric_containers,
                )

                metrics = get_all_metric_containers({}, raise_errors=False)
            elif self.task == TaskType.REGRESSION:
                from pycaret.containers.metrics.regression import (
                    get_all_metric_containers,
                )

                metrics = get_all_metric_containers({}, raise_errors=False)
            elif self.task == TaskType.CLUSTERING:
                from pycaret.containers.metrics.clustering import (
                    get_all_metric_containers,
                )

                metrics = get_all_metric_containers({}, raise_errors=False)
            elif self.task == TaskType.ANOMALY:
                from pycaret.containers.metrics.anomaly import (
                    get_all_metric_containers,
                )

                metrics = get_all_metric_containers({}, raise_errors=False)
            else:
                # Time-series falls through to the legacy registry.
                return self._legacy.get_metrics(*args, **kwargs)
        except Exception:
            return self._legacy.get_metrics(*args, **kwargs)

        import pandas as _pd

        rows: list[dict] = []
        for mid, container in metrics.items():
            score_func = getattr(container, "score_func", None)
            rows.append(
                {
                    "ID": mid,
                    "Name": getattr(container, "name", mid),
                    "Display Name": getattr(container, "display_name", mid),
                    "Score Function": (
                        f"{score_func.__module__}.{score_func.__name__}"
                        if score_func is not None
                        else None
                    ),
                    "Scorer": getattr(container, "scorer", None),
                    "Target": getattr(container, "target", "pred"),
                    "Args": dict(getattr(container, "args", {}) or {}),
                    "Greater is Better": bool(getattr(container, "greater_is_better", True)),
                    "Multiclass": bool(getattr(container, "is_multiclass", True)),
                    "Custom": bool(getattr(container, "is_custom", False)),
                }
            )
        return _pd.DataFrame(rows).set_index("ID")

    # The following verbs still delegate. Each has a non-trivial state
    # mutation path that the snapshot pattern doesn't cover well; they're
    # advisory escape hatches that don't sit in the predict/tune/compare
    # path. Drain candidates for a future polish session.

    def add_metric(self, *args: Any, **kwargs: Any) -> Any:
        return self._legacy.add_metric(*args, **kwargs)

    def remove_metric(self, *args: Any, **kwargs: Any) -> None:
        return self._legacy.remove_metric(*args, **kwargs)

    def get_config(self, *args: Any, **kwargs: Any) -> Any:
        return self._legacy.get_config(*args, **kwargs)

    def set_config(self, *args: Any, **kwargs: Any) -> Any:
        return self._legacy.set_config(*args, **kwargs)

    # -------------------------------------------------------- persistence verbs
    #
    # Session-22 drain: these four verbs no longer delegate to `self._legacy`.
    # They are now thin wrappers around the stateless helpers in
    # `pycaret.persistence` (a fitted sklearn Pipeline is just a picklable
    # object — there is nothing PyCaret-specific to persist). The legacy
    # persistence path ran a lot of code (cloud-credential injection, MLflow
    # artifact logging, 3.x-era metadata headers) that is either out of scope
    # for 4.0 or handled by the Control Plane. Dropping it removes ~200 LoC of
    # dependency surface.
    #
    # Contract:
    #   save_model(model, path)        → Path to the written `.pkl`.
    #   load_model(path)               → the loaded object (typically Pipeline).
    #   save_experiment(path)          → Path to the written `.pkl` of *self*.
    #   Experiment.load_experiment(path) → restored Experiment instance.
    #
    # ``save_model`` does NOT require the experiment to be fitted — a caller
    # may have loaded a pipeline from elsewhere and want a normalised save.
    # ``save_experiment`` DOES require fit — an unfitted Experiment is just
    # its constructor kwargs, which you already have.

    def save_model(self, model: Any, path: Any, *, verbose: bool = False) -> Any:
        """Persist a fitted model / pipeline to ``path`` (joblib-dumped).

        Returns the absolute `Path` of the file written.

        ``save_model`` is legal before ``fit`` — the verb is about the passed
        ``model``, not about experiment state. When the logger has not yet
        been installed (``fit`` is where that happens), the MODEL_SAVED event
        is silently dropped rather than raising.
        """
        from pycaret.persistence import save_model as _save

        written = _save(model, path, verbose=verbose)
        if self.logger is not None:
            self.logger.log(EventKind.MODEL_SAVED, payload={"path": str(written)})
        return written

    def load_model(self, path: Any, *, verbose: bool = False) -> Any:
        """Load a model previously written by ``save_model``."""
        from pycaret.persistence import load_model as _load

        return _load(path, verbose=verbose)

    def save_experiment(self, path: Any, *, verbose: bool = False) -> Any:
        """Persist the full Experiment (including fit state) to ``path``.

        The Experiment must be fitted. To re-hydrate, use
        ``Experiment.load_experiment(path)``.
        """
        self._require_fitted()
        from pycaret.persistence import save_model as _save

        written = _save(self, path, verbose=verbose)
        # After `fit()`, `self.logger` is guaranteed to be installed
        # (NullLogger at minimum). Still null-check for belt-and-braces.
        if self.logger is not None:
            self.logger.log(
                EventKind.MODEL_SAVED,
                payload={"path": str(written), "kind": "experiment"},
            )
        return written

    @staticmethod
    def load_experiment(path: Any, *, verbose: bool = False) -> Experiment:
        """Re-hydrate an Experiment previously saved by ``save_experiment``.

        Returns the loaded Experiment. Raises ``TypeError`` if the file on
        disk was not a PyCaret Experiment.
        """
        from pycaret.persistence import load_model as _load

        restored = _load(path, verbose=verbose)
        if not isinstance(restored, Experiment):
            raise TypeError(
                f"File at {path!r} contained a {type(restored).__name__!r}, "
                "not a PyCaret Experiment. Use `load_model` for plain models."
            )
        return restored

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

    # ----------------------------------------------- data-accessor properties
    #
    # Session-29 drain: post-fit, these read from ``self._fit_state`` (a
    # snapshot taken in ``fit()``) instead of dispatching to
    # ``self._legacy.X`` etc. on every access. The legacy holder is still
    # populated by ``setup()`` and used internally by the drained verbs
    # (which read transformed splits + the fold generator from it), but the
    # public API no longer requires legacy attribute lookups to function.

    def _snapshot_fit_state(self) -> None:
        """Cache references to the legacy state on ``self`` post-setup.

        Called once from ``fit()`` after ``self._legacy.setup()`` returns.
        We hold *references*, not copies — mutating ``self.X_train`` still
        propagates to the underlying frame, matching legacy semantics.

        Snapshot covers two tiers:

        1. **User-facing accessors** (drained in session 29): ``X``, ``X_train``,
           ``X_test``, ``y``, ``y_train``, ``y_test``, ``preprocess_pipeline``.
        2. **Internal training state** (drained in session 30): the post-
           preprocessing transformed splits, the CV fold generator, and the
           model-container registry. The drained verbs (``create_model``,
           ``tune_model``, ``compare_models``, ensemble / blend / stack /
           calibrate / finalize / unsupervised create_model) read these from
           ``self._fit_state`` instead of dispatching to ``self._legacy``.

        Some attributes are task-specific (clustering / anomaly don't have
        train/test splits); we ``getattr`` defensively so missing slots
        become ``None`` rather than raising.
        """
        legacy = self._legacy
        self._fit_state: dict[str, Any] = {
            # ---- user-facing (session 29)
            "X": getattr(legacy, "X", None),
            "X_train": getattr(legacy, "X_train", None),
            "X_test": getattr(legacy, "X_test", None),
            "y": getattr(legacy, "y", None),
            "y_train": getattr(legacy, "y_train", None),
            "y_test": getattr(legacy, "y_test", None),
            "preprocess_pipeline": getattr(legacy, "pipeline", None),
            # ---- internal training state (session 30)
            "X_transformed": getattr(legacy, "X_transformed", None),
            "X_train_transformed": getattr(legacy, "X_train_transformed", None),
            "y_transformed": getattr(legacy, "y_transformed", None),
            "y_train_transformed": getattr(legacy, "y_train_transformed", None),
            "fold_generator": getattr(legacy, "fold_generator", None),
            "model_registry": dict(getattr(legacy, "_all_models_internal", {})),
        }

    @property
    def X(self) -> pd.DataFrame:
        self._require_fitted()
        return self._fit_state["X"]

    @property
    def X_train(self) -> pd.DataFrame:
        self._require_fitted()
        return self._fit_state["X_train"]

    @property
    def X_test(self) -> pd.DataFrame:
        self._require_fitted()
        return self._fit_state["X_test"]

    @property
    def y(self):
        self._require_fitted()
        return self._fit_state["y"]

    @property
    def y_train(self):
        self._require_fitted()
        return self._fit_state["y_train"]

    @property
    def y_test(self):
        self._require_fitted()
        return self._fit_state["y_test"]

    @property
    def preprocess_pipeline(self) -> Pipeline:
        self._require_fitted()
        return self._fit_state["preprocess_pipeline"]

    @property
    def events(self) -> list:
        """Replay of everything the logger saw (empty if not using MemoryLogger)."""
        return list(getattr(self.logger, "events", []))

    # ------------------------------------------------------- internal helpers

    def _set_last_metrics(self, df: pd.DataFrame | None) -> None:
        """Stash the most recent metrics DataFrame for ``pull()``.

        Called by each native verb (``create_model``, ``tune_model``,
        ``compare_models``, ensemble / blend / stack / calibrate /
        finalize) before returning. ``pull()`` reads this slot.
        """
        if hasattr(self, "_fit_state") and df is not None:
            self._fit_state["last_metrics"] = df

    def _safe_pull(self):
        """Legacy-fallback pull for the TS code path. Native verbs use
        ``_set_last_metrics`` instead.
        """
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
