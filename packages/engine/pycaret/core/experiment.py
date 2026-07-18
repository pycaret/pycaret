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


class _ModelRegistryContext:
    """Minimal stand-in for an experiment, exposing only the attrs that
    the model-container constructors read from ``experiment.<x>``.

    Used by the session-35 native setup to build the model registry
    without instantiating the full legacy experiment + running its
    setup(). The container __init__ functions read:

    - ``seed`` — the random_state.
    - ``gpu_param`` — "force" / falsy.
    - ``n_jobs_param`` — parallelism.
    - ``X_train`` — used by size-aware models (KNN etc.) to pick defaults.
    - ``is_multiclass`` — picks classification-only knobs.

    Anything more exotic (e.g. a future container that reads
    ``experiment.dataset``) would need a new attr here. Keep this
    intentionally narrow — the proxy is a contract.
    """

    __slots__ = ("seed", "gpu_param", "n_jobs_param", "X_train", "is_multiclass")

    def __init__(
        self,
        *,
        seed: int,
        gpu_param: Any,
        n_jobs_param: int,
        X_train: Any,
        is_multiclass: bool = False,
    ) -> None:
        self.seed = seed
        self.gpu_param = gpu_param
        self.n_jobs_param = n_jobs_param
        self.X_train = X_train
        self.is_multiclass = is_multiclass

    def get_engine(self, id: str) -> str | None:
        """Always None — fall through to the model's default engine.

        Native setup phase 1 doesn't expose engine-selection plumbing.
        Users wanting alternate engines (sklearnex etc.) need the legacy
        path for now (set ``normalize=True`` or any complex flag to bypass
        native setup).
        """
        return None


class _TSContextProxy:
    """Minimal stand-in for a fitted TS legacy experiment, exposing only
    the attrs that ``pycaret.containers.models.time_series`` containers
    read from ``experiment.<x>`` during construction.

    Used by phase 5d to skip ``legacy.setup()`` entirely while still
    being able to build the sktime model registry. Each container reads
    a subset of: ``seed`` / ``gpu_param`` / ``n_jobs_param`` /
    ``seasonality_present`` / ``primary_sp_to_use`` /
    ``strictly_positive`` / ``seasonality_type`` /
    ``all_sps_to_use`` / ``X_train`` / ``is_multiclass``.

    Seasonality detection is a lightweight Fourier autocorrelation test
    via sktime's ``autocorrelation_seasonality_test`` — same as legacy
    uses internally. Far simpler than the full legacy detection chain
    (no harmonic removal, no PeriodIndex coercion), but covers the
    common case (univariate series with monthly / quarterly / yearly
    PeriodIndex) and gracefully degrades to ``sp=1`` otherwise.
    """

    __slots__ = (
        # Common (shared with _ModelRegistryContext)
        "seed",
        "gpu_param",
        "n_jobs_param",
        "X_train",
        "is_multiclass",
        # TS-specific
        "seasonality_present",
        "primary_sp_to_use",
        "strictly_positive",
        "seasonality_type",
        "all_sps_to_use",
        "enforce_pi",
        "enforce_exogenous",
        "exogenous_present",
        "fe_target_rr",
        "index_type",
    )

    def __init__(
        self,
        *,
        seed: int,
        gpu_param: Any,
        n_jobs_param: int,
        seasonality_present: bool,
        primary_sp_to_use: int,
        strictly_positive: bool,
        seasonality_type: str,
        all_sps_to_use: list,
        X_train: Any = None,
        is_multiclass: bool = False,
        enforce_pi: bool = False,
        enforce_exogenous: bool = False,
        exogenous_present: bool = False,
        fe_target_rr: list | None = None,
        index_type: str = "period",
    ) -> None:
        self.seed = seed
        self.gpu_param = gpu_param
        self.n_jobs_param = n_jobs_param
        self.seasonality_present = seasonality_present
        self.primary_sp_to_use = primary_sp_to_use
        self.strictly_positive = strictly_positive
        self.seasonality_type = seasonality_type
        self.all_sps_to_use = all_sps_to_use
        self.X_train = X_train
        self.is_multiclass = is_multiclass
        self.enforce_pi = enforce_pi
        self.enforce_exogenous = enforce_exogenous
        self.exogenous_present = exogenous_present
        # Default to None (matches legacy default) — `[]` breaks the cds_dt
        # recursive forecasters which expect either None or a real list of
        # WindowSummarizer / lag transformers.
        self.fe_target_rr = fe_target_rr
        self.index_type = index_type

    def get_engine(self, id: str) -> str | None:
        return None


class _LegacyShim:
    """Phase-6 placeholder for the deleted legacy experiment object.

    Test code uses the pattern::

        exp._legacy = exp._build_legacy_experiment()
        monkeypatch.setattr(exp._legacy, "setup", _poison)

    to assert that fit() and the verbs don't dispatch to legacy. With
    the legacy directory deleted, ``_build_legacy_experiment`` now
    returns this no-op shim. Each verb name is preregistered as an
    identity callable so ``monkeypatch.setattr`` (default ``raising=True``)
    works; the methods are never called by production code.
    """

    _VERB_NAMES = (
        "setup",
        "create_model",
        "predict_model",
        "assign_model",
        "compare_models",
        "tune_model",
        "ensemble_model",
        "blend_models",
        "stack_models",
        "calibrate_model",
        "finalize_model",
        "save_model",
        "load_model",
        "save_experiment",
        "load_experiment",
        "plot_model",
        "evaluate_model",
        "interpret_model",
        "automl",
        "get_leaderboard",
        "check_stats",
        "models",
        "get_metrics",
        "add_metric",
        "remove_metric",
        "pull",
        "get_config",
        "set_config",
    )

    def __init__(self) -> None:
        # Predefine each verb name so `monkeypatch.setattr(shim, name, ...)`
        # passes the strict-attribute check.
        for name in self._VERB_NAMES:
            setattr(self, name, _LegacyShim._noop)

    @staticmethod
    def _noop(*a: Any, **kw: Any) -> None:
        return None


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
    - Subclasses pre-configure `task` and may override the native setup hooks
      (`_native_setup_supervised` / `_native_setup_unsupervised` /
      `_native_setup_timeseries`) for task-specific shapes.
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
        # Phase 6: ``self._legacy`` is a no-op shim — present so existing
        # drain-lock test patterns (``monkeypatch.setattr(exp._legacy, ...)``)
        # keep working. Production code never reads anything off it.
        self._legacy = _LegacyShim()

        # Phase 6 (s46): setup_kwargs no longer fall through to legacy —
        # the legacy directory is gone. Reject any kwargs the native path
        # doesn't understand. Power users who need the removed knobs
        # should pin to PyCaret 3.x or wait for the post-4.0 reintroduction
        # of specific options as constructor params.
        if setup_kwargs:
            raise ConfigurationError(
                f"setup_kwargs are not supported in PyCaret 4.0: "
                f"{sorted(setup_kwargs.keys())}. The legacy escape hatch "
                "was removed in phase 6. If you need a removed option, "
                "use PyCaret 3.x or open an issue requesting it as a "
                "first-class constructor parameter."
            )

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

        # Drain dispatcher (sessions 35-38): every supported task has a
        # native setup path now that legacy is gone.
        self._native_setup_used = True
        if self._is_supervised():
            self._native_setup_supervised(data, setup_kwargs)
        else:
            self._native_setup_unsupervised(data, setup_kwargs)
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
        """Phase-6 stub. The real legacy class hierarchy was deleted.

        Returns a ``_LegacyShim`` instance with the verb names predefined
        as no-op identity methods so existing drain-lock test patterns
        keep working unchanged:

            exp._legacy = exp._build_legacy_experiment()
            monkeypatch.setattr(exp._legacy, "setup", _poison)
            exp.fit(df)  # native path — poison never fires

        Production code never reads attributes off this object — every
        native verb / setup path bypasses ``self._legacy``. The shim
        exists purely for back-compat with the established test pattern.
        """
        return _LegacyShim()

    # ------------------------------------------------- session-35 native setup

    def _can_use_native_setup(self, setup_kwargs: dict[str, Any]) -> bool:
        """Predicate: would fit() use the native (no-legacy) setup path?

        Phase 6 (s46) deleted the legacy directory, so the native path is
        the *only* path. This predicate is now informational — it tells
        callers whether the native path would handle their config
        (currently it handles every supported task except when
        ``setup_kwargs`` are passed, which now raises rather than
        falling back). Kept as a public surface for tests / introspection.

        Native phases (historical):
        - **Phase 1-3** (sessions 35-37): supervised tabular preprocessing.
        - **Phase 4** (s38): unsupervised tabular (clustering + anomaly).
        - **Phase 5a-d** (s39-s45): time-series (TS verbs drained s40-s44;
          ``legacy.setup()`` call stripped from native TS in s45).
        """
        from pycaret.core.tasks import TaskType

        # Supported task types: supervised tabular + unsupervised tabular
        # + time-series (phase 5a — soft drain).
        if self.task not in (
            TaskType.CLASSIFICATION,
            TaskType.REGRESSION,
            TaskType.CLUSTERING,
            TaskType.ANOMALY,
            TaskType.TIME_SERIES,
        ):
            return False
        # Any caller-supplied setup_kwargs forces legacy — we don't know what
        # those options do. Once we expose the knobs as constructor params
        # (e.g. `outliers_method`, `feature_selection_estimator`) we can
        # selectively allow the matching kwargs through here.
        if setup_kwargs:
            return False
        # Unsupervised tasks don't accept supervised-only flags; if the user
        # set them, fall back to legacy so the error path is consistent
        # with the legacy behavior.
        if not self._is_supervised():
            if self.remove_outliers or self.feature_selection:
                # These could in principle work on unsupervised X, but we
                # haven't wired them yet — Phase 4.5.
                return False
        return True

    def _native_setup_supervised(self, data: pd.DataFrame, setup_kwargs: dict[str, Any]) -> None:
        """Phase-1 native setup for classification + regression.

        Builds ``self._fit_state`` directly from the input DataFrame using
        sklearn primitives. Skips ``self._legacy.setup()`` entirely.

        Layout of the produced state:

        - ``X`` / ``X_train`` / ``X_test``: raw DataFrames (split via
          stratified sklearn split for clf, plain for reg).
        - ``y`` / ``y_train`` / ``y_test``: raw target series.
        - ``X_transformed`` / ``X_train_transformed``: post-preprocessor
          DataFrames (numeric imputation by mean + categorical imputation
          by mode + ordinal encoding).
        - ``y_train_transformed`` / ``y_transformed``: integer-encoded
          target for clf; raw for reg.
        - ``preprocess_pipeline``: sklearn ``Pipeline`` wrapping the
          ``ColumnTransformer``.
        - ``fold_generator``: ``StratifiedKFold`` (clf) or ``KFold`` (reg).
        - ``model_registry``: built via the per-task registry helper +
          a thin ``_ModelRegistryContext`` proxy, so the call doesn't
          require a fitted legacy.
        - ``last_metrics`` / ``metric_registry``: standard slots, lazily
          populated by the drained verbs.
        """
        import pandas as _pd
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
        from sklearn.pipeline import Pipeline as SkPipeline
        from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

        from pycaret.core.tasks import TaskType

        target_col = self.target
        if target_col is None or target_col not in data.columns:
            raise ConfigurationError(
                "Native setup requires the target column to be present in the "
                f"DataFrame. Got target={target_col!r}."
            )

        y = data[target_col]
        X = data.drop(columns=[target_col])

        # Detect numeric vs categorical columns.
        numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
        categorical_cols = [c for c in X.columns if c not in numeric_cols]

        # Train/test split. Stratify on y for classification.
        seed = self.session_id if self.session_id is not None else 0
        is_clf = self.task == TaskType.CLASSIFICATION
        stratify = y if is_clf else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            train_size=self.train_size,
            random_state=seed,
            stratify=stratify,
        )

        # Build preprocessing ColumnTransformer.
        # Phase 1: imputation + ordinal encoding.
        # Phase 2: optional StandardScaler (normalize) + PowerTransformer
        #          (transformation) chained inside the numeric branch.
        transformers: list = []
        if numeric_cols:
            num_steps: list = [("imputer", SimpleImputer(strategy="mean"))]
            if self.transformation:
                from sklearn.preprocessing import PowerTransformer

                # Yeo-Johnson: handles negatives. Same default as legacy
                # `transformation_method='yeo-johnson'`.
                num_steps.append(
                    ("transformer", PowerTransformer(method="yeo-johnson", standardize=False))
                )
            if self.normalize:
                from sklearn.preprocessing import StandardScaler

                # Z-score by default — same as legacy `normalize_method='zscore'`.
                num_steps.append(("scaler", StandardScaler()))
            num_pipe = SkPipeline(num_steps) if len(num_steps) > 1 else num_steps[0][1]
            transformers.append(("numerical_pipeline", num_pipe, numeric_cols))
        if categorical_cols:
            cat_pipe = SkPipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "encoder",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                        ),
                    ),
                ]
            )
            transformers.append(("categorical_pipeline", cat_pipe, categorical_cols))

        # Edge case: no columns at all (defensive — train_test_split would've
        # already failed). Fall through with an identity transform.
        if not transformers:
            preprocess_pipeline = SkPipeline([("preprocess", "passthrough")])
            X_train_transformed = X_train.copy()
            X_test_transformed = X_test.copy()
        else:
            ct = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
            ct.fit(X_train)
            preprocess_pipeline = SkPipeline([("preprocess", ct)])

            ordered_cols = numeric_cols + categorical_cols
            X_train_transformed = _pd.DataFrame(
                ct.transform(X_train),
                columns=ordered_cols,
                index=X_train.index,
            )
            X_test_transformed = _pd.DataFrame(
                ct.transform(X_test),
                columns=ordered_cols,
                index=X_test.index,
            )

        X_transformed = _pd.concat([X_train_transformed, X_test_transformed]).sort_index()

        # Label encoding for classification target; pass-through for regression.
        is_multiclass = False
        label_encoder: LabelEncoder | None = None
        if is_clf:
            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(y_train)
            y_test_encoded = label_encoder.transform(y_test)
            y_train_transformed = _pd.Series(
                y_train_encoded, index=y_train.index, name=y_train.name
            )
            y_test_transformed = _pd.Series(y_test_encoded, index=y_test.index, name=y_test.name)
            is_multiclass = len(label_encoder.classes_) > 2
        else:
            y_train_transformed = y_train
            y_test_transformed = y_test

        y_transformed = _pd.concat([y_train_transformed, y_test_transformed]).sort_index()

        # Phase 3a: outlier removal. Fit IsolationForest on the
        # transformed training set and drop the top `contamination`
        # fraction (5% by default — matches legacy `outliers_threshold`)
        # of training rows. Test set is left untouched. Both X_train
        # and the transformed splits stay in sync.
        outliers_dropped = 0
        if self.remove_outliers:
            from sklearn.ensemble import IsolationForest

            iso = IsolationForest(contamination=0.05, random_state=seed, n_jobs=self.n_jobs)
            inlier_mask = iso.fit_predict(X_train_transformed) == 1
            outliers_dropped = int((~inlier_mask).sum())
            X_train = X_train.loc[inlier_mask]
            X_train_transformed = X_train_transformed.loc[inlier_mask]
            y_train = y_train.loc[inlier_mask]
            y_train_transformed = y_train_transformed.loc[inlier_mask]
            # Recompute the union views.
            X_transformed = _pd.concat([X_train_transformed, X_test_transformed]).sort_index()
            y_transformed = _pd.concat([y_train_transformed, y_test_transformed]).sort_index()

        # Phase 3b: feature selection. SelectFromModel with a default
        # estimator picks features whose importances are above the median.
        # We append the fitted selector to ``preprocess_pipeline`` so that
        # predict-time preprocessing reapplies the same column drop. The
        # raw X / X_train / X_test keep all columns so user-facing
        # accessors don't lose information.
        feature_selector: Any | None = None
        selected_features: list[str] | None = None
        if self.feature_selection:
            from sklearn.feature_selection import SelectFromModel

            if is_clf:
                from sklearn.ensemble import ExtraTreesClassifier

                estimator = ExtraTreesClassifier(
                    n_estimators=100, n_jobs=self.n_jobs, random_state=seed
                )
            else:
                from sklearn.ensemble import ExtraTreesRegressor

                estimator = ExtraTreesRegressor(
                    n_estimators=100, n_jobs=self.n_jobs, random_state=seed
                )
            feature_selector = SelectFromModel(estimator, threshold="median")
            feature_selector.fit(X_train_transformed, y_train_transformed)
            keep_mask = feature_selector.get_support()
            selected_features = [
                col
                for col, keep in zip(X_train_transformed.columns, keep_mask, strict=True)
                if keep
            ]
            # Defensive: SelectFromModel can pick zero features for tiny / pathological
            # datasets. Keep at least one to avoid an empty matrix downstream.
            if not selected_features:
                selected_features = [X_train_transformed.columns[0]]
            X_train_transformed = X_train_transformed[selected_features]
            X_test_transformed = X_test_transformed[selected_features]
            X_transformed = _pd.concat([X_train_transformed, X_test_transformed]).sort_index()
            # Extend the pipeline so predict-time preprocessing applies
            # the same column drop to new data.
            preprocess_pipeline.steps.append(("feature_selection", feature_selector))

        # Fold generator.
        if is_clf:
            fold_generator = StratifiedKFold(n_splits=self.fold, shuffle=True, random_state=seed)
        else:
            fold_generator = KFold(n_splits=self.fold, shuffle=True, random_state=seed)

        # Model registry via thin proxy — skips legacy.
        proxy = _ModelRegistryContext(
            seed=seed,
            gpu_param="force" if self.use_gpu else False,
            n_jobs_param=self.n_jobs,
            X_train=X_train_transformed,
            is_multiclass=is_multiclass,
        )
        if is_clf:
            from pycaret.containers.models.classification import (
                get_all_model_containers,
            )
        else:
            from pycaret.containers.models.regression import (
                get_all_model_containers,
            )
        try:
            model_registry = dict(get_all_model_containers(proxy, raise_errors=False))
        except Exception:
            # If the registry helper can't run on the proxy for any reason,
            # fall back to using the legacy holder + minimal attrs.
            model_registry = {}

        self._fit_state = {
            # session 29 — user-facing data accessors
            "X": X,
            "X_train": X_train,
            "X_test": X_test,
            "y": y,
            "y_train": y_train,
            "y_test": y_test,
            "preprocess_pipeline": preprocess_pipeline,
            # session 30 — internal training state
            "X_transformed": X_transformed,
            "X_train_transformed": X_train_transformed,
            "y_transformed": y_transformed,
            "y_train_transformed": y_train_transformed,
            "fold_generator": fold_generator,
            "model_registry": model_registry,
            # session 31 — last metrics for pull()
            "last_metrics": None,
            # session 47 — last leaderboard from compare_models() for get_leaderboard()
            "last_leaderboard": None,
            # session 35 — extras specific to native setup
            "label_encoder": label_encoder,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            # session 37 — phase 3 extras (None when the corresponding flag
            # is off, so callers can introspect what ran).
            "feature_selector": feature_selector,
            "selected_features": selected_features,
            "outliers_dropped": outliers_dropped,
        }
        self.logger.log(
            EventKind.EXPERIMENT_FITTED,
            message=(
                f"Native fit ready (clf={is_clf}, n_train={len(X_train)}, "
                f"n_test={len(X_test)}, n_models={len(model_registry)})"
            ),
        )

    def _native_setup_unsupervised(self, data: pd.DataFrame, setup_kwargs: dict[str, Any]) -> None:
        """Phase-4 native setup for clustering + anomaly tasks.

        Unsupervised experiments don't have a train/test split — the
        whole frame is the training set. They also don't have a fold
        generator (CV is undefined for clustering / anomaly). The
        native preprocessing chain is the same as supervised, applied
        once to the full ``X``:

        - ``imputer`` (mean for numeric, mode for categorical)
        - ordinal encoding for categorical
        - optional ``StandardScaler`` (``normalize=True``)
        - optional ``PowerTransformer`` (``transformation=True``)

        Output state shape mirrors what ``UnsupervisedExperiment``'s
        verbs expect: ``X`` / ``X_transformed`` / ``preprocess_pipeline``
        / ``model_registry`` populated; ``X_train`` / ``y`` / etc. set
        to ``None`` because they don't apply.
        """
        import pandas as _pd
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline as SkPipeline
        from sklearn.preprocessing import OrdinalEncoder

        from pycaret.core.tasks import TaskType

        # Coerce single-Series input into a DataFrame.
        if isinstance(data, _pd.Series):
            data = data.to_frame()
        X = data

        # Detect numeric vs categorical columns.
        numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
        categorical_cols = [c for c in X.columns if c not in numeric_cols]

        seed = self.session_id if self.session_id is not None else 0

        # Build the ColumnTransformer — same shape as supervised but
        # without label encoding and without train/test splitting.
        transformers: list = []
        if numeric_cols:
            num_steps: list = [("imputer", SimpleImputer(strategy="mean"))]
            if self.transformation:
                from sklearn.preprocessing import PowerTransformer

                num_steps.append(
                    ("transformer", PowerTransformer(method="yeo-johnson", standardize=False))
                )
            if self.normalize:
                from sklearn.preprocessing import StandardScaler

                num_steps.append(("scaler", StandardScaler()))
            num_pipe = SkPipeline(num_steps) if len(num_steps) > 1 else num_steps[0][1]
            transformers.append(("numerical_pipeline", num_pipe, numeric_cols))
        if categorical_cols:
            cat_pipe = SkPipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "encoder",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                        ),
                    ),
                ]
            )
            transformers.append(("categorical_pipeline", cat_pipe, categorical_cols))

        if not transformers:
            preprocess_pipeline = SkPipeline([("preprocess", "passthrough")])
            X_transformed = X.copy()
        else:
            ct = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
            ct.fit(X)
            preprocess_pipeline = SkPipeline([("preprocess", ct)])
            ordered_cols = numeric_cols + categorical_cols
            X_transformed = _pd.DataFrame(ct.transform(X), columns=ordered_cols, index=X.index)

        # Model registry via the same proxy used in the supervised path.
        proxy = _ModelRegistryContext(
            seed=seed,
            gpu_param="force" if self.use_gpu else False,
            n_jobs_param=self.n_jobs,
            X_train=X_transformed,
            is_multiclass=False,
        )
        if self.task == TaskType.CLUSTERING:
            from pycaret.containers.models.clustering import (
                get_all_model_containers,
            )
        else:  # TaskType.ANOMALY
            from pycaret.containers.models.anomaly import (
                get_all_model_containers,
            )
        try:
            model_registry = dict(get_all_model_containers(proxy, raise_errors=False))
        except Exception:
            model_registry = {}

        self._fit_state = {
            # session 29 — user-facing accessors. Unsupervised has no
            # supervised target / train/test split, so the supervised
            # slots are None.
            "X": X,
            "X_train": None,
            "X_test": None,
            "y": None,
            "y_train": None,
            "y_test": None,
            "preprocess_pipeline": preprocess_pipeline,
            # session 30 — internal training state.
            "X_transformed": X_transformed,
            "X_train_transformed": None,
            "y_transformed": None,
            "y_train_transformed": None,
            "fold_generator": None,  # unsupervised tasks don't CV.
            "model_registry": model_registry,
            # session 31
            "last_metrics": None,
            # session 47 — last leaderboard from compare_models() for get_leaderboard()
            "last_leaderboard": None,
            # session 35 extras
            "label_encoder": None,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            # session 37 extras (None — outliers / feature_selection don't
            # apply unsupervised in this phase).
            "feature_selector": None,
            "selected_features": None,
            "outliers_dropped": 0,
        }
        self.logger.log(
            EventKind.EXPERIMENT_FITTED,
            message=(
                f"Native unsupervised fit ready (task={self.task.value}, "
                f"n_rows={len(X)}, n_models={len(model_registry)})"
            ),
        )

    def _native_setup_timeseries(self, data: Any, setup_kwargs: dict[str, Any]) -> None:
        """Phase-5d native setup for time-series — fully drained.

        Builds ``_fit_state`` directly from sktime primitives. **Does
        not** call ``self._legacy.setup()``. Same architectural shape as
        the supervised / unsupervised native setups (s35-s38), with TS-
        specific slots (``fh``, ``seasonal_period``).

        Steps:

        1. **Coerce input** to a univariate Series + (optional) exogenous
           DataFrame using ``self.target``.
        2. **Auto-detect seasonality** via sktime's
           ``autocorrelation_seasonality_test`` on candidates derived
           from the index frequency or ``self.seasonal_period``.
        3. **Build ForecastingHorizon** from ``self.fh``.
        4. **Train/test split** via sktime's ``temporal_train_test_split``.
        5. **Build CV fold generator** —
           ``ExpandingWindowSplitter`` (default) or
           ``SlidingWindowSplitter``.
        6. **Build model registry** from
           ``pycaret.containers.models.time_series.get_all_model_containers``
           through a ``_TSContextProxy``.
        7. **Build a minimal ForecastingPipeline** as
           ``preprocess_pipeline`` — empty target / exogenous transformer
           steps + a ``NaiveForecaster`` placeholder. Drained verbs swap
           the placeholder out via ``_add_model_to_pipeline``.
        """
        import numpy as np
        import pandas as pd
        from sktime.forecasting.base import ForecastingHorizon
        from sktime.forecasting.compose import ForecastingPipeline, TransformedTargetForecaster
        from sktime.forecasting.model_selection import (
            temporal_train_test_split,
        )
        from sktime.forecasting.naive import NaiveForecaster

        seed = self.session_id if self.session_id is not None else 0

        # ---- coerce input → univariate y + optional exogenous X
        if isinstance(data, pd.Series):
            y = data
            X = None
        elif isinstance(data, pd.DataFrame):
            if self.target is not None and self.target in data.columns:
                y = data[self.target]
                X = data.drop(columns=[self.target])
                if X.shape[1] == 0:
                    X = None
            elif data.shape[1] == 1:
                y = data.iloc[:, 0]
                X = None
            else:
                raise ConfigurationError(
                    "TS native setup: when passing a multi-column DataFrame, "
                    "either set `target=` to the column name to forecast, or "
                    "pass a univariate Series."
                )
        else:
            raise ConfigurationError(
                f"TS native setup: data must be a pandas Series or DataFrame, "
                f"got {type(data).__name__!r}."
            )

        # ---- forecast horizon
        fh_value = self.fh
        if isinstance(fh_value, int):
            fh = ForecastingHorizon(list(range(1, fh_value + 1)), is_relative=True)
        elif isinstance(fh_value, ForecastingHorizon):
            fh = fh_value
        else:
            try:
                fh = ForecastingHorizon(fh_value, is_relative=True)
            except Exception:
                fh = ForecastingHorizon([1], is_relative=True)

        # ---- seasonality auto-detection (lightweight port of legacy)
        seasonality_present, primary_sp_to_use, all_sps_to_use = self._auto_detect_seasonality(y)
        strictly_positive = bool(np.all(y > 0))
        seasonality_type = "mul" if (seasonality_present and strictly_positive) else "add"

        # ---- temporal train/test split
        try:
            if X is not None:
                y_train, y_test, X_train, X_test = temporal_train_test_split(y=y, X=X, fh=fh)
            else:
                y_train, y_test = temporal_train_test_split(y=y, fh=fh)
                X_train = X_test = None
        except Exception as e:
            raise ConfigurationError(
                f"TS native setup: temporal_train_test_split failed — {e}"
            ) from e

        # ---- CV fold generator
        fold_generator = self._build_ts_fold_generator(
            y_train=y_train, fh=fh, fold=self.fold, fold_strategy=self.fold_strategy
        )

        # ---- model registry via proxy (no legacy state)
        # Detect index_type for the proxy: "period" if PeriodIndex, else
        # "datetime" or "integer". TS containers branch on this.
        if hasattr(y_train, "index"):
            idx = y_train.index
            if isinstance(idx, pd.PeriodIndex):
                index_type = "period"
            elif isinstance(idx, pd.DatetimeIndex):
                index_type = "datetime"
            else:
                index_type = "integer"
        else:
            index_type = "period"

        proxy = _TSContextProxy(
            seed=seed,
            gpu_param="force" if self.use_gpu else False,
            n_jobs_param=self.n_jobs,
            seasonality_present=seasonality_present,
            primary_sp_to_use=primary_sp_to_use,
            strictly_positive=strictly_positive,
            seasonality_type=seasonality_type,
            all_sps_to_use=all_sps_to_use,
            X_train=X_train,
            is_multiclass=False,
            enforce_pi=False,
            enforce_exogenous=False,
            exogenous_present=X_train is not None,
            fe_target_rr=None,
            index_type=index_type,
        )
        from pycaret.containers.models.time_series import (
            get_all_model_containers,
        )

        try:
            model_registry = dict(get_all_model_containers(proxy, raise_errors=False))
        except Exception:
            # Defensive: if any container fails to construct on the proxy
            # (e.g. needs a never-exposed attr), fall through with what we
            # got. Empty registry would surface as a clear error in
            # create_model rather than a confusing AttributeError here.
            model_registry = {}

        # ---- minimal ForecastingPipeline (placeholder model swapped out
        # by drained verbs via _add_model_to_pipeline)
        preprocess_pipeline = ForecastingPipeline(
            steps=[
                (
                    "forecaster",
                    TransformedTargetForecaster(
                        steps=[("model", NaiveForecaster())],
                    ),
                )
            ]
        )

        self._fit_state = {
            # User-facing accessors.
            "X": X,
            "X_train": X_train,
            "X_test": X_test,
            "y": y,
            "y_train": y_train,
            "y_test": y_test,
            "preprocess_pipeline": preprocess_pipeline,
            # Internal training state.
            "X_transformed": X,
            "X_train_transformed": X_train,
            "y_transformed": y,
            "y_train_transformed": y_train,
            "fold_generator": fold_generator,
            "model_registry": model_registry,
            "last_metrics": None,
            # session 47 — last leaderboard from compare_models() for get_leaderboard()
            "last_leaderboard": None,
            # TS-specific slots.
            "fh": fh,
            "seasonal_period": (
                primary_sp_to_use if self.seasonal_period is None else self.seasonal_period
            ),
            "seasonality_present": seasonality_present,
            "strictly_positive": strictly_positive,
            "seasonality_type": seasonality_type,
            "all_sps_to_use": all_sps_to_use,
            # Shape-uniform slots unused by TS.
            "label_encoder": None,
            "numeric_cols": None,
            "categorical_cols": None,
            "feature_selector": None,
            "selected_features": None,
            "outliers_dropped": 0,
        }
        self.logger.log(
            EventKind.EXPERIMENT_FITTED,
            message=(
                f"Native time-series fit ready (n_train={len(y_train)}, "
                f"n_test={len(y_test)}, sp={primary_sp_to_use}, "
                f"n_models={len(model_registry)})"
            ),
        )

    @staticmethod
    def _auto_detect_seasonality(y: Any) -> tuple[bool, int, list]:
        """Lightweight seasonality detection — Fourier autocorrelation.

        Returns ``(seasonality_present, primary_sp_to_use, all_sps_to_use)``.

        Strategy:
        1. Derive candidate sp from the series' index frequency (e.g. monthly
           PeriodIndex → 12; quarterly → 4; weekly → 52). Default fallback:
           ``[1]`` when the index has no recognisable frequency.
        2. Run sktime's ``autocorrelation_seasonality_test`` on each
           candidate. Significant ones survive.
        3. ``seasonality_present`` is True iff at least one candidate is
           significant. ``primary_sp_to_use`` is the largest significant sp
           (or 1 if none).
        """
        try:
            from sktime.utils.seasonality import autocorrelation_seasonality_test

            from pycaret.utils.time_series import get_sp_from_str
        except Exception:
            return False, 1, [1]

        candidates: list[int] = []
        try:
            freqstr = getattr(y.index, "freqstr", None)
            if freqstr:
                try:
                    candidates.append(get_sp_from_str(freqstr))
                except Exception:
                    pass
        except Exception:
            pass

        if not candidates:
            return False, 1, [1]

        # Filter out any garbage values.
        candidates = [int(sp) for sp in candidates if isinstance(sp, int) and sp > 1]
        if not candidates:
            return False, 1, [1]

        sig_sps: list[int] = []
        for sp in candidates:
            try:
                if autocorrelation_seasonality_test(y, sp):
                    sig_sps.append(sp)
            except Exception:
                continue

        if not sig_sps:
            return False, 1, [1]
        primary = max(sig_sps)  # legacy default — pick the largest significant sp
        return True, primary, sig_sps

    def _build_ts_fold_generator(
        self,
        *,
        y_train: Any,
        fh: Any,
        fold: int,
        fold_strategy: Any,
    ) -> Any:
        """Build the sktime fold generator — same math legacy uses.

        ``initial_window = len(y_train) - ((fold - 1) * step_length + max(fh))``
        ``step_length    = len(fh)``
        """
        from sktime.forecasting.model_selection import (
            ExpandingWindowSplitter,
            SlidingWindowSplitter,
        )

        # `fold` may have been passed in as the legacy fold_generator
        # already (e.g. user reused an existing splitter); short-circuit.
        if hasattr(fold_strategy, "split") and not isinstance(fold_strategy, str):
            return fold_strategy

        # ForecastingHorizon is iterable but doesn't expose __iter__ as
        # an attribute (it has __getitem__-based iteration). Use try/except
        # over iter() to cover both ForecastingHorizon and plain int.
        try:
            fh_values = [int(v) for v in iter(fh)]
        except TypeError:
            fh_values = [int(fh)]
        step_length = max(1, len(fh_values))
        fh_max = max(fh_values) if fh_values else 1
        n_train = len(y_train)
        initial_window = n_train - ((fold - 1) * step_length + 1 * fh_max)
        if initial_window < 1:
            # Defensive: shrink folds rather than blowing up.
            initial_window = max(1, n_train // 2)

        if fold_strategy == "sliding":
            return SlidingWindowSplitter(
                step_length=step_length,
                window_length=initial_window,
                fh=fh,
                start_with_window=True,
            )
        # Default: "expanding" / "rolling" (and unknown strategies).
        return ExpandingWindowSplitter(
            initial_window=initial_window,
            step_length=step_length,
            fh=fh,
        )

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

        # Phase 6: clustering / anomaly / time-series all override
        # create_model in their task subclasses (s28 / s40). The base
        # implementation here is the supervised native path only.

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
            message=f"Training {model_id}…",
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
        # Widen the event payload with the mean-row + per-fold metrics
        # so a UI live-chart widget can render per-model and per-fold
        # progress without re-querying the database after the fact.
        mean_metrics: dict[str, float] = {}
        fold_metrics: list[dict[str, Any]] = []
        if metrics_df is not None and hasattr(metrics_df, "loc"):
            try:
                if "Mean" in metrics_df.index:
                    mean_metrics = {
                        k: (float(v) if isinstance(v, (int, float)) else v)
                        for k, v in metrics_df.loc["Mean"].to_dict().items()
                    }
                # Per-fold rows. Indexed by fold number (0..k-1) plus the
                # summary rows (Mean/Std) which we filter out so a chart
                # caller can iterate straight through.
                for idx, row in metrics_df.iterrows():
                    if isinstance(idx, str) and idx in ("Mean", "Std"):
                        continue
                    fold_metrics.append(
                        {
                            "fold": int(idx) if str(idx).isdigit() else idx,
                            **{
                                k: (float(v) if isinstance(v, (int, float)) else v)
                                for k, v in row.to_dict().items()
                            },
                        }
                    )
            except Exception:  # noqa: BLE001 — event-widening must never block
                mean_metrics = {}
                fold_metrics = []

        self.logger.log(
            EventKind.MODEL_CREATED,
            message=f"Trained {model_id}",
            duration_ms=(time.perf_counter() - t0) * 1000,
            payload={
                "estimator": model_id,
                "model_id": model_id,
                "metrics": mean_metrics,
                "fold_metrics": fold_metrics,
                "cross_validation": bool(cross_validation),
            },
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

        from pycaret.utils.generic import calculate_metrics

        try:
            metrics_registry = self._get_metric_registry()
            if not metrics_registry:
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

        # Phase 6: TimeSeriesExperiment overrides predict_model (s41) so
        # the TS branch never reaches this code path. Base implementation
        # is supervised + unsupervised tabular only.

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
                    out["Anomaly_Score"] = estimator.decision_function(X_for_pred)
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

            metrics_registry = self._get_metric_registry()
            if not metrics_registry:
                return None
            if self.task == TaskType.CLASSIFICATION:
                try:
                    proba = estimator.predict_proba(X)
                    pred_proba = proba[:, 1] if proba.shape[1] == 2 else proba
                except Exception:
                    pred_proba = None
            elif self.task == TaskType.REGRESSION:
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

    def plot_model(
        self,
        estimator: Any,
        plot: str | None = None,
        *,
        save: bool | str = False,
        **kwargs: Any,
    ) -> Any:
        """Render a single Plotly diagnostic for a fitted estimator.

        Session 54 reimplementation. Each task class registers a dictionary
        of named plots in ``_build_plot_registry(estimator)``. ``plot=`` looks
        up the entry and returns a ``plotly.graph_objects.Figure``.

        Parameters
        ----------
        estimator
            A fitted ``sklearn.pipeline.Pipeline`` (or ``ForecastingPipeline``
            for time-series). Some plots — like classification's
            ``class_distribution`` — ignore the estimator and use the
            experiment's data only; pass any fitted pipeline.
        plot : str, optional
            Plot kind. Defaults to the task's canonical first-look diagnostic
            (``'auc'`` for classification, ``'residuals'`` for regression,
            ``'cluster'`` for clustering, ``'score'`` for anomaly,
            ``'forecast'`` for time-series). Pass an unknown kind to see the
            full set in the ``ValueError`` message.
        save : bool | str, default False
            ``False`` → return the Figure. ``True`` → write to
            ``f"{plot}.png"`` and return that path. A string → write to that
            path. Static export requires ``pycaret[export]`` (pulls
            ``kaleido``).
        **kwargs
            Passed through to the underlying plot function.

        Returns
        -------
        plotly.graph_objects.Figure | str
            The figure, or the saved path when ``save`` is truthy.
        """
        self._require_fitted()
        registry = self._build_plot_registry(estimator)
        if plot is None:
            plot = self._default_plot_kind()
        if plot not in registry:
            raise ValueError(
                f"Unknown plot kind {plot!r} for task {self.task.value!r}. "
                f"Valid: {sorted(registry.keys())}"
            )
        fig = registry[plot](**kwargs)
        if save:
            path = save if isinstance(save, str) else f"{plot}.png"
            try:
                fig.write_image(path)
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(
                    f"plot_model(save={save!r}) requires kaleido. "
                    "Install with: pip install pycaret[export]. "
                    f"Original error: {e}"
                ) from e
            return path
        return fig

    def evaluate_model(self, estimator: Any, **kwargs: Any) -> dict:
        """Render the curated diagnostic bundle for a task.

        Session 54 reimplementation. Returns a ``dict`` mapping plot kind →
        ``plotly.graph_objects.Figure``. Iterates the curated subset of
        plots that the task class declares in ``_evaluate_plot_set()``,
        calling each plot function defensively — any one that raises (e.g.
        a SHAP entry without ``shap`` installed) is simply skipped, not
        propagated.

        The 3.x version showed an interactive ipywidget tab strip. The 4.0
        version returns the underlying figures so callers can render them
        however they want — ``fig.show()`` in notebooks, ``fig.to_dict()``
        for HTTP transport, etc.

        Returns
        -------
        dict[str, plotly.graph_objects.Figure]
            One Figure per plot kind that succeeded.
        """
        self._require_fitted()
        registry = self._build_plot_registry(estimator)
        out: dict[str, Any] = {}
        for kind in self._evaluate_plot_set():
            if kind not in registry:
                continue
            try:
                out[kind] = registry[kind](**kwargs)
            except Exception:  # noqa: BLE001
                # Best-effort: skip plots that fail (e.g. shap not installed,
                # estimator lacking feature_importances_).
                continue
        return out

    # --- subclass hooks for plot_model / evaluate_model -----------------------

    def _build_plot_registry(self, estimator: Any) -> dict[str, Any]:
        """Map plot kind → zero-arg-or-kwargs callable returning a Figure.

        Each leaf task class overrides this to register the plot kinds it
        supports. The base implementation raises so an experiment without
        plot wiring is loud instead of silently empty.
        """
        raise NotImplementedError(
            f"plot_model() is not wired for task {self.task.value!r}. "
            "Open an issue if you need this."
        )

    def _default_plot_kind(self) -> str:
        """Plot kind used when the caller passes ``plot=None``."""
        raise NotImplementedError(
            f"_default_plot_kind() is not wired for task {self.task.value!r}."
        )

    def _evaluate_plot_set(self) -> list[str]:
        """Curated list of plot kinds rendered by ``evaluate_model``."""
        raise NotImplementedError(
            f"_evaluate_plot_set() is not wired for task {self.task.value!r}."
        )

    # ----------------------------------------------- session-31 native verbs

    def pull(self, *args: Any, **kwargs: Any) -> pd.DataFrame | None:
        """Return the most recent metrics DataFrame.

        Reads from ``self._fit_state["last_metrics"]``, which is updated by
        each native verb (``create_model``, ``tune_model``,
        ``compare_models``, ensemble / blend / stack / calibrate /
        finalize) before it returns. ``None`` if no metrics-emitting verb
        has run yet — phase 6 removed the legacy ``self._legacy.pull()``
        fallback.
        """
        self._require_fitted()
        return self._fit_state.get("last_metrics") if hasattr(self, "_fit_state") else None

    def models(self, *args: Any, internal: bool = False, **kwargs: Any) -> pd.DataFrame:
        """Return a DataFrame describing the available models in the registry.

        Session-31 / 35 drain: builds the DataFrame from the snapshot's
        model registry. With ``internal=True`` returns the engine-internal
        view (``Special`` / ``Class`` / ``Equality`` / ``Args``) by reading
        each container's ``get_dict(internal=True)`` directly. Falls back
        to the legacy holder only when the snapshot is empty.

        Session-45 (phase 5d) added native TS filter for
        ``model_type ∈ TSModelTypes`` so this method no longer defers to
        legacy for any task. Phase 6 removed the legacy fallback for
        empty registries — that case is now an error condition.
        """
        self._require_fitted()
        import pandas as _pd

        from pycaret.core.tasks import TaskType

        registry = self._fit_state.get("model_registry", {}) if hasattr(self, "_fit_state") else {}
        if not registry:
            return _pd.DataFrame()

        # Time-series filters: legacy excludes containers whose model_type
        # isn't in TSModelTypes (specifically `ensemble_forecaster` whose
        # model_type='ensemble'). Replicate that filter here so the native
        # path matches legacy semantics — phase 5d (s45) unblocks this.
        if self.task == TaskType.TIME_SERIES:
            try:
                from pycaret.utils.time_series import TSModelTypes

                allowed_types = set(TSModelTypes)
                registry = {
                    mid: c
                    for mid, c in registry.items()
                    if getattr(c, "model_type", None) in allowed_types
                }
            except Exception:
                pass  # fall through with the unfiltered registry

        if internal:
            rows: list[dict] = []
            for mid, container in registry.items():
                d = dict(container.get_dict(internal=True))
                d.setdefault("ID", mid)
                # Surface 'Turbo' too — useful for filter logic in tests.
                if "Turbo" not in d:
                    d["Turbo"] = bool(getattr(container, "is_turbo", False))
                rows.append(d)
            return _pd.DataFrame(rows).set_index("ID")

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

        Session-31 drain + session-32: reads from the per-experiment metric
        registry stored in ``self._fit_state["metric_registry"]``. That
        registry is initialised at fit() time from the task helper and
        mutated by ``add_metric`` / ``remove_metric``.
        """
        self._require_fitted()
        import pandas as _pd

        metrics = self._get_metric_registry() or {}
        if not metrics:
            # Phase 6: legacy fallback removed. Empty registry → empty
            # DataFrame. (TS now builds its registry natively in s40.)
            return _pd.DataFrame()

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

    def add_metric(
        self,
        id: str,
        name: str,
        score_func: Any,
        target: str = "pred",
        greater_is_better: bool = True,
        args: dict | None = None,
        is_multiclass: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Register a custom metric with the experiment.

        Session-32 drain. Builds the right ``<Task>MetricContainer`` for the
        experiment's task and stashes it in ``self._fit_state["metric_registry"]``.
        Subsequent ``create_model`` / ``tune_model`` / ``compare_models``
        calls will compute this metric on every fold + include it in the
        leaderboard.

        Parameters mirror the legacy contract:

        Parameters
        ----------
        id : str
            Short identifier; used as the key in the registry.
        name : str
            Long name (e.g. ``"Mean Absolute Percentage Error"``).
        score_func : callable
            ``(y_true, y_pred) -> float`` (or with ``y_pred_proba`` if
            ``target="pred_proba"``).
        target : str, default="pred"
            ``"pred"`` / ``"pred_proba"`` / ``"threshold"``. Maps to the
            input the score function expects.
        greater_is_better : bool, default=True
            False for error metrics where lower is better.
        args : dict, optional
            Extra kwargs always passed to ``score_func``.
        is_multiclass : bool, default=True
            (Classification only.) Whether the metric supports multiclass.

        Returns
        -------
        The registered container. Suitable for downstream introspection.
        """
        self._require_fitted()
        from pycaret.core.tasks import TaskType

        registry = self._get_metric_registry()
        if registry is None:
            raise NotImplementedError(
                f"add_metric() is not supported for task {self.task.value!r}: "
                "no native metric registry yet. Open an issue if you need "
                "custom metrics for this task."
            )

        # Build the right container for the task.
        if self.task == TaskType.CLASSIFICATION:
            from pycaret.containers.metrics.classification import (
                ClassificationMetricContainer,
            )

            container = ClassificationMetricContainer(
                id=id,
                name=name,
                score_func=score_func,
                target=target,
                args=args,
                greater_is_better=greater_is_better,
                is_multiclass=is_multiclass,
                is_custom=True,
            )
        elif self.task == TaskType.REGRESSION:
            from pycaret.containers.metrics.regression import (
                RegressionMetricContainer,
            )

            container = RegressionMetricContainer(
                id=id,
                name=name,
                score_func=score_func,
                args=args,
                greater_is_better=greater_is_better,
                is_custom=True,
            )
        else:
            raise NotImplementedError(
                f"add_metric() not yet supported for task {self.task.value!r}. "
                "Phase 6 removed the legacy fallback. Open an issue if you "
                "need custom metrics for clustering / anomaly / time-series."
            )

        registry[id] = container
        return container

    def remove_metric(self, name_or_id: str) -> None:
        """Remove a metric from the experiment's registry.

        Session-32 drain. Pops from ``self._fit_state["metric_registry"]``;
        accepts either the metric's ``id`` or its display name (matching
        legacy semantics).
        """
        self._require_fitted()
        registry = self._get_metric_registry()
        if registry is None:
            raise NotImplementedError(
                f"remove_metric() is not supported for task {self.task.value!r}: "
                "no native metric registry yet."
            )

        # Try direct ID match first.
        if name_or_id in registry:
            del registry[name_or_id]
            return None

        # Fall back to name match.
        for key, container in list(registry.items()):
            container_name = getattr(container, "name", None)
            if container_name == name_or_id:
                del registry[key]
                return None

        raise ValueError(f"No metric matching {name_or_id!r} in the experiment's registry.")

    # Constructor parameters that ``set_config`` is allowed to mutate
    # post-fit. These are the knobs that don't invalidate the fit-time
    # snapshot (e.g. tweaking ``n_jobs`` or ``verbose`` mid-experiment is
    # safe; mutating ``target`` or ``train_size`` is not).
    _SETTABLE_CONFIG_KEYS = frozenset(
        {
            "session_id",
            "n_jobs",
            "verbose",
            "fold",
            "log_experiment",
        }
    )

    def get_config(self, variable: str | None = None) -> Any:
        """Return a configured experiment variable.

        Session-33 drain. Reads from ``self._fit_state`` (snapshot of all
        the data accessors + transformed splits + registries) and the
        constructor parameters stored on ``self``. Raises ``ValueError``
        for unknown names; with ``variable=None`` returns the full list
        of accessible names.

        Parameters
        ----------
        variable : str, optional
            Name to look up. ``None`` returns the list of accessible names.

        Returns
        -------
        Either the value (or the list of names if ``variable=None``).
        """
        self._require_fitted()

        # Build the accessible-names set: everything in _fit_state +
        # constructor params + a couple of computed aliases.
        snapshot_keys = set(self._fit_state.keys()) if hasattr(self, "_fit_state") else set()
        ctor_keys = {
            "task",
            "target",
            "session_id",
            "train_size",
            "fold",
            "fold_strategy",
            "preprocess",
            "normalize",
            "transformation",
            "remove_outliers",
            "feature_selection",
            "n_jobs",
            "use_gpu",
            "log_experiment",
            "verbose",
        }
        accessible = snapshot_keys | ctor_keys | {"pipeline", "seed"}

        if variable is None:
            return sorted(accessible)

        if variable not in accessible:
            raise ValueError(
                f"Variable {variable!r} not found. Accessible variables: {sorted(accessible)}"
            )

        # Aliases for legacy / convenience.
        if variable == "pipeline":
            return self._fit_state["preprocess_pipeline"]
        if variable == "seed":
            return self.session_id

        # Snapshot first, then fall back to constructor params on self.
        if variable in snapshot_keys:
            return self._fit_state[variable]
        return getattr(self, variable)

    def set_config(self, variable: str | None = None, value: Any = None, **kwargs: Any) -> None:
        """Update a configured experiment variable.

        Session-33 drain. Restricted to a small allowlist of constructor
        params that can be safely mutated post-fit
        (``_SETTABLE_CONFIG_KEYS``). Anything else raises ``ValueError``.

        Two call shapes:

        - Single: ``set_config("n_jobs", 4)``
        - Bulk:  ``set_config(n_jobs=4, verbose=True)``
        """
        self._require_fitted()

        if kwargs and variable:
            raise ValueError("variable parameter cannot be used together with keyword arguments.")
        if kwargs:
            updates = kwargs
        elif variable is not None:
            updates = {variable: value}
        else:
            return None

        for k, v in updates.items():
            if k.startswith("_"):
                raise ValueError(f"Variable {k!r} is read-only (starts with '_').")
            if k not in self._SETTABLE_CONFIG_KEYS:
                raise ValueError(
                    f"Variable {k!r} is not settable post-fit. "
                    f"Settable variables: {sorted(self._SETTABLE_CONFIG_KEYS)}. "
                    "Mutating other constructor params would invalidate the "
                    "fit-time snapshot — re-create the Experiment instead."
                )
            setattr(self, k, v)
        return None

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

    # Phase 6: removed `_snapshot_fit_state` — it was only used by the
    # legacy-fallback path in fit(). Native setup methods build _fit_state
    # directly.

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

    def _get_metric_registry(self) -> dict | None:
        """Return the per-experiment metric registry, lazily building it.

        Behavior:

        - Post-fit (``_fit_state`` exists): caches the registry in
          ``_fit_state["metric_registry"]`` so ``add_metric`` /
          ``remove_metric`` can mutate it.
        - Pre-fit (or fit-sentinel test setup): builds a fresh registry
          on every call from the task helper. No caching, no mutation
          surface — but the metric-using verbs (``predict_model``'s
          metric path) still work for one-shot use cases.

        Returns ``None`` for tasks where the native registry isn't yet
        wired (time-series). Callers fall through to the legacy registry
        in that case.
        """
        from pycaret.core.tasks import TaskType

        # Cached path — only available post-fit.
        cached = self._fit_state.get("metric_registry") if hasattr(self, "_fit_state") else None
        if cached is not None:
            return cached

        # Build from the task helper.
        try:
            if self.task == TaskType.CLASSIFICATION:
                from pycaret.containers.metrics.classification import (
                    get_all_metric_containers,
                )
            elif self.task == TaskType.REGRESSION:
                from pycaret.containers.metrics.regression import (
                    get_all_metric_containers,
                )
            elif self.task == TaskType.CLUSTERING:
                from pycaret.containers.metrics.clustering import (
                    get_all_metric_containers,
                )
            elif self.task == TaskType.ANOMALY:
                from pycaret.containers.metrics.anomaly import (
                    get_all_metric_containers,
                )
            else:
                # Time-series — no native registry yet.
                return None
            registry = dict(get_all_metric_containers({}, raise_errors=False))
        except Exception:
            return None

        # Cache only when fit-state exists. Otherwise return the fresh
        # build without persisting (one-shot test / pre-fit path).
        if hasattr(self, "_fit_state"):
            self._fit_state["metric_registry"] = registry
        return registry

    def _set_last_metrics(self, df: pd.DataFrame | None) -> None:
        """Stash the most recent metrics DataFrame for ``pull()``.

        Called by each native verb (``create_model``, ``tune_model``,
        ``compare_models``, ensemble / blend / stack / calibrate /
        finalize) before returning. ``pull()`` reads this slot.
        """
        if hasattr(self, "_fit_state") and df is not None:
            self._fit_state["last_metrics"] = df

    def _safe_pull(self):
        """Phase 6: legacy fallback removed. Now just reads from the
        snapshot — same as ``pull()``. Kept as a name for callers; may
        be merged into ``pull()`` in a future cleanup.
        """
        if hasattr(self, "_fit_state"):
            return self._fit_state.get("last_metrics")
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
