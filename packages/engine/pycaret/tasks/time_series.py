"""`TimeSeriesExperiment` — the PyCaret 4.0 time-series forecasting engine.

Supervised (the target is a time-indexed series). Verb surface mirrors
`SupervisedExperiment` plus time-series-specific additions (`check_stats`,
forecast-horizon handling).

Canonical import:

    from pycaret.tasks import TimeSeriesExperiment
    # or (backward-compat):
    from pycaret.time_series import TimeSeriesExperiment

Example
-------

>>> from pycaret.datasets import get_data
>>> from pycaret.tasks import TimeSeriesExperiment
>>> y = get_data("airline")
>>> exp = TimeSeriesExperiment(fh=12, session_id=42).fit(y)
>>> best = exp.compare_models().best
>>> forecast = exp.predict_model(best).predictions
"""

from __future__ import annotations

from typing import Any

from pycaret.core.supervised import SupervisedExperiment
from pycaret.core.tasks import TaskType
from pycaret.logging.base import BaseLogger


class TimeSeriesExperiment(SupervisedExperiment):
    """PyCaret 4.0 time-series forecasting experiment (sklearn-compatible).

    Unlike the tabular supervised tasks, `fit()` takes either a univariate
    Series / single-column DataFrame (target), or a DataFrame plus an `fh`
    (forecast horizon) parameter. `target` is optional for univariate data.
    """

    def __init__(
        self,
        *,
        target: str | None = None,
        fh: Any = 1,  # forecast horizon
        seasonal_period: Any = None,
        session_id: int | None = None,
        fold: int = 3,
        fold_strategy: str | object = "expanding",
        preprocess: bool = True,
        n_jobs: int = -1,
        use_gpu: bool = False,
        logger: BaseLogger | None = None,
        log_experiment: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            task=TaskType.TIME_SERIES,
            target=target,
            session_id=session_id,
            fold=fold,
            fold_strategy=fold_strategy,
            preprocess=preprocess,
            n_jobs=n_jobs,
            use_gpu=use_gpu,
            logger=logger,
            log_experiment=log_experiment,
            verbose=verbose,
        )
        self.fh = fh
        self.seasonal_period = seasonal_period

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        try:
            tags.estimator_type = "regressor"  # forecasts are continuous
        except AttributeError:
            pass
        return tags

    def _build_legacy_experiment(self):
        from pycaret.time_series.forecasting.oop import TSForecastingExperiment

        return TSForecastingExperiment()

    # TS data shape is different — a univariate Series or DataFrame + target
    # column — and the legacy setup() has its own param list. Skip the tabular
    # supervised (X, y) coercion in the base fit().
    def fit(self, X, y=None, **setup_kwargs):
        """TS-specific fit. Dispatches to the native phase-5a path or to
        ``legacy.setup()`` with state snapshot, depending on the predicate.
        """
        import time
        import uuid

        from pycaret.logging.base import NullLogger
        from pycaret.logging.events import EventKind
        from pycaret.logging.memory import MemoryLogger

        if y is not None:
            # Allow fit(X, y) where y is the target Series; treat y as data.
            data = y
        else:
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
            message="Starting time_series experiment",
            payload={"target": self.target},
        )

        # Phase 5a (s39): TS fit() goes through the same native dispatcher
        # as the other tasks. The "native" TS path still calls legacy.setup
        # underneath because TS verbs aren't drained yet — what we get is
        # _fit_state population so accessors (y_train, y_test, fh, ...) work
        # consistently across all task types.
        self._native_setup_used = False
        if self._can_use_native_setup(setup_kwargs):
            self._native_setup_timeseries(data, setup_kwargs)
            self._native_setup_used = True
        else:
            self._legacy.setup(**self._build_legacy_setup_kwargs(data, setup_kwargs))
            self._snapshot_fit_state()

        self._fitted = True
        self.logger.log(
            EventKind.EXPERIMENT_FITTED,
            message="Experiment fitted and ready",
            duration_ms=(time.perf_counter() - t0) * 1000,
        )
        return self

    def _build_legacy_setup_kwargs(self, data, extra):
        """TS legacy setup() has a different shape — override.

        The TS setup signature does not accept the tabular `preprocess` /
        `normalize` / `transformation` / `remove_outliers` flags; it has
        its own fine-grained knobs (`transform_target`, `scale_target`,
        `numeric_imputation_target`, etc.). Only the overlapping kwargs
        are forwarded here; user-supplied extras are merged last.
        """
        kwargs = {
            "data": data,
            "target": self.target,
            "fh": self.fh,
            "seasonal_period": self.seasonal_period,
            "session_id": self.session_id,
            "fold": self.fold,
            "fold_strategy": self.fold_strategy,
            "n_jobs": self.n_jobs,
            "use_gpu": self.use_gpu,
            "verbose": self.verbose,
            "html": False,
            "log_experiment": False,
        }
        kwargs.update(extra)
        return kwargs

    def check_stats(self, *args: Any, **kwargs: Any) -> Any:
        """Time-series-specific statistical tests (ADF, KPSS, etc.)."""
        self._require_fitted()
        return self._legacy.check_stats(*args, **kwargs)

    # ------------------------------------------------- session 40 (phase 5b)
    # Native TS create_model — drains legacy.create_model. Reads from
    # _fit_state and uses sktime forecasters directly.

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
    ):
        """Native time-series create_model (phase 5b).

        Resolves the estimator from the sktime registry, wires it into the
        experiment's preprocess pipeline (a ``ForecastingPipeline``), runs
        cross-validation via the existing ``cross_validate`` helper +
        per-fold metrics dict, then refits on the full ``y_train``.

        Returns a ``CreateResult`` whose ``pipeline`` is a real
        ``sktime.forecasting.compose.ForecastingPipeline`` — same as legacy.

        This drains ``self._legacy.create_model`` so that users who only
        want to fit and forecast no longer pay for any legacy code path
        (besides the legacy.setup() call still inside
        ``_native_setup_timeseries``; phase 5c removes that too).
        """
        from copy import deepcopy

        import pandas as pd
        from sktime.forecasting.compose import ForecastingPipeline

        from pycaret.core.errors import ConfigurationError
        from pycaret.core.results import CreateResult
        from pycaret.logging.events import EventKind
        from pycaret.utils.time_series.forecasting.model_selection import (
            cross_validate as _ts_cross_validate,
        )
        from pycaret.utils.time_series.forecasting.pipeline import (
            _add_model_to_pipeline,
        )

        self._require_fitted()

        import time as _time

        t0 = _time.perf_counter()

        # ---- resolve estimator → fitted forecaster instance + model_id
        if isinstance(estimator, str):
            registry = self._fit_state.get("model_registry", {})
            if estimator not in registry:
                raise ConfigurationError(
                    f"Unknown TS model id {estimator!r}. Call `Experiment.list_models()`."
                )
            container = registry[estimator]
            init_kwargs = dict(container.args)
            init_kwargs.update(estimator_kwargs)
            try:
                model = container.class_def(**init_kwargs)
            except TypeError:
                # Defensive: unexpected kwargs → retry with registry defaults.
                model = container.class_def(**dict(container.args))
            model_id = estimator
        elif isinstance(estimator, ForecastingPipeline):
            # Already a pipeline — pull the last forecaster from the
            # nested TransformedTargetForecaster.
            inner = estimator.steps[-1][1]
            model = deepcopy(inner.steps[-1][1])
            model_id = type(model).__name__
        else:
            if not hasattr(estimator, "fit"):
                raise TypeError(
                    "estimator must be a registry ID or a sktime forecaster (with .fit)."
                )
            model = deepcopy(estimator)
            model_id = type(estimator).__name__

        self.logger.log(
            EventKind.MODEL_CREATE_STARTED,
            payload={"estimator": model_id},
        )

        # ---- wire into preprocess pipeline
        base_pipeline = self._fit_state["preprocess_pipeline"]
        if not isinstance(base_pipeline, ForecastingPipeline):
            # Defensive — should always be a ForecastingPipeline post-setup.
            raise RuntimeError(
                "TS native create_model expected a ForecastingPipeline as "
                f"preprocess_pipeline; got {type(base_pipeline).__name__}."
            )
        pipeline_with_model = _add_model_to_pipeline(pipeline=base_pipeline, model=model)

        # ---- CV (when requested)
        y_train = self._fit_state["y_train"]
        x_train = self._fit_state.get("X_train")  # may be None for univariate
        cv = fold if fold is not None else self._fit_state["fold_generator"]
        fit_kwargs = fit_kwargs or {}
        # sktime forecasters that need fh-in-fit get fh from CV.
        if "fh" not in fit_kwargs and cv is not None:
            fit_kwargs["fh"] = cv.fh

        metrics_df: pd.DataFrame | None = None
        if cross_validation and cv is not None:
            metrics_registry = self._build_ts_metric_registry()
            scoring_dict = {k: v.scorer for k, v in metrics_registry.items()}
            additional_scorer_kwargs = {"sp": self._primary_sp_to_use()}
            try:
                scores, _cutoffs = _ts_cross_validate(
                    pipeline=pipeline_with_model,
                    y=y_train,
                    X=x_train,
                    cv=cv,
                    scoring=scoring_dict,
                    fit_params=dict(fit_kwargs),
                    n_jobs=self.n_jobs,
                    return_train_score=False,
                    alpha=0.05,
                    coverage=0.9,
                    error_score=0,
                    **additional_scorer_kwargs,
                )
            except Exception as e:  # noqa: BLE001 — surface CV failure
                self.logger.log(
                    EventKind.MODEL_CREATE_STARTED,
                    message=f"TS native CV failed for {model_id}: {e}",
                )
                scores = None

            if scores is not None:
                # Build display-name DataFrame: rows = Fold 0..N-1, Mean, Std.
                score_df = pd.DataFrame(
                    {v.display_name: scores[k] for k, v in metrics_registry.items()}
                )
                score_df.index = [f"Fold {i}" for i in range(len(score_df))]
                # Append Mean + Std rows.
                try:
                    score_df.loc["Mean"] = score_df.mean(numeric_only=True)
                    score_df.loc["Std"] = score_df.std(numeric_only=True)
                except TypeError:  # all-None columns
                    pass
                metrics_df = score_df.round(round)
                self._set_last_metrics(metrics_df)

        # ---- refit on full y_train so the returned pipeline is fitted
        # (cross_validate above clones per fold and doesn't mutate the
        # original).
        try:
            pipeline_with_model.fit(
                y=y_train, X=x_train, fh=cv.fh if cv is not None else self._fit_state["fh"]
            )
        except Exception:
            # Some forecasters don't accept fh in fit — retry without it.
            pipeline_with_model.fit(y=y_train, X=x_train)

        self.logger.log(
            EventKind.MODEL_CREATED,
            duration_ms=(_time.perf_counter() - t0) * 1000,
            payload={"estimator": model_id},
        )
        return CreateResult(
            pipeline=pipeline_with_model,
            model_id=model_id,
            metrics=metrics_df,
            params=self._safe_params(model),
        )

    def _build_ts_metric_registry(self) -> dict:
        """Build the TS metric registry from the container helper. Cached
        in ``_fit_state["metric_registry"]`` so add_metric / remove_metric
        can mutate it.
        """
        if self._fit_state.get("metric_registry"):
            return self._fit_state["metric_registry"]
        from pycaret.containers.metrics.time_series import (
            get_all_metric_containers,
        )

        registry = dict(get_all_metric_containers({}, raise_errors=False))
        self._fit_state["metric_registry"] = registry
        return registry

    def _primary_sp_to_use(self) -> int:
        """Resolve the seasonal period for MASE/RMSSE scorers.

        Priority: explicit ``self.seasonal_period`` constructor arg →
        ``self._legacy.primary_sp_to_use`` (legacy auto-detected) → 1.
        """
        if self.seasonal_period is not None:
            try:
                return int(self.seasonal_period)
            except (TypeError, ValueError):
                pass
        legacy_sp = getattr(self._legacy, "primary_sp_to_use", None)
        if legacy_sp is not None:
            try:
                return int(legacy_sp)
            except (TypeError, ValueError):
                pass
        return 1

    # ------------------------------------------------- session 41 (phase 5c)
    # Native TS predict_model — drains legacy.predict_model. Reads from
    # _fit_state and uses the standalone get_predictions_with_intervals
    # helper directly; computes test metrics via the registry +
    # calculate_metrics utility.

    def predict_model(
        self,
        estimator: Any,
        data: Any = None,
        *,
        fh: Any = None,
        X: Any = None,
        return_pred_int: bool = False,
        alpha: float | None = None,
        coverage: float = 0.9,
        round: int = 4,
        verbose: bool = False,
    ):
        """Native time-series predict_model (phase 5c).

        Computes forecasts and (when ground truth is available)
        per-metric scores against ``y_test``. Returns a typed
        ``PredictResult`` whose ``predictions`` DataFrame has columns
        ``["y_pred"]`` (or ``["y_pred", "lower", "upper"]`` when
        ``return_pred_int=True``) and whose ``metrics`` is a one-row
        DataFrame keyed by display name.

        Parameters
        ----------
        estimator : sktime forecaster or ForecastingPipeline
            Fitted forecaster. Typically the ``CreateResult.pipeline``
            from ``create_model``.
        data : pd.DataFrame, optional
            Compat shim — when supplied, treated as ``X`` (exogenous).
            Mirrors the supervised ``predict_model`` signature.
        fh : optional
            Override forecast horizon. Falls through to
            ``_fit_state["fh"]`` then ``estimator.fh``.
        X : pd.DataFrame, optional
            Exogenous variables for prediction. Falls through to
            ``_fit_state["X_test"]`` for non-finalized estimators.
        return_pred_int : bool, default=False
            Include ``lower`` / ``upper`` quantile columns.
        alpha : float, optional
            Quantile for point predictions. ``None`` = standard
            ``.predict()``; otherwise ``predict_quantiles(alpha=alpha)``.
        coverage : float, default=0.9
            Coverage level for prediction intervals.
        round : int, default=4
            Decimal places for predictions + metrics.
        verbose : bool, default=False
            Reserved (legacy progress-bar hook).
        """
        from copy import deepcopy

        import pandas as pd
        from sktime.forecasting.base import ForecastingHorizon
        from sktime.forecasting.compose import ForecastingPipeline

        from pycaret.core.results import PredictResult
        from pycaret.logging.events import EventKind
        from pycaret.utils.generic import calculate_metrics
        from pycaret.utils.time_series.forecasting import (
            get_predictions_with_intervals,
            update_additional_scorer_kwargs,
        )
        from pycaret.utils.time_series.forecasting.pipeline import (
            _add_model_to_pipeline,
        )

        self._require_fitted()

        import time as _time

        t0 = _time.perf_counter()

        # ---- reconcile X (exogenous data)
        if X is None and data is not None:
            X = data

        # ---- reconcile pipeline + estimator_
        if isinstance(estimator, ForecastingPipeline):
            pipeline_with_model = deepcopy(estimator)
            # Final inner model lives at .steps[-1][1].steps[-1][1].
            inner = pipeline_with_model.steps[-1][1]
            estimator_ = inner.steps[-1][1] if hasattr(inner, "steps") else inner
        else:
            if not hasattr(estimator, "predict"):
                raise TypeError(
                    "predict_model expects a fitted sktime forecaster or "
                    f"ForecastingPipeline; got {type(estimator).__name__!r}."
                )
            estimator_ = deepcopy(estimator)
            # Wire the bare forecaster into the experiment's preprocess
            # pipeline so prediction goes through the same transforms.
            base_pipeline = self._fit_state["preprocess_pipeline"]
            if isinstance(base_pipeline, ForecastingPipeline):
                pipeline_with_model = _add_model_to_pipeline(
                    pipeline=base_pipeline, model=estimator_
                )
            else:
                # Defensive fallback — wrap in a minimal pipeline.
                pipeline_with_model = estimator_

        # ---- reconcile fh
        if fh is None:
            fh = self._fit_state.get("fh")
            if fh is None:
                fh = getattr(estimator_, "fh", None)
        if fh is None:
            raise ValueError(
                "predict_model: cannot determine forecast horizon. "
                "Pass fh= or refit the experiment."
            )
        if not isinstance(fh, ForecastingHorizon):
            try:
                fh = ForecastingHorizon(fh, is_relative=True)
            except Exception:
                pass  # fall through; sktime will raise a clearer error if invalid

        # ---- reconcile X for exogenous
        if X is None:
            X = self._fit_state.get("X_test")

        # ---- get predictions + intervals
        result = get_predictions_with_intervals(
            forecaster=pipeline_with_model,
            alpha=alpha,
            coverage=coverage,
            X=X,
            fh=fh,
            merge=True,
            round=round,
        )
        y_pred_df = pd.DataFrame(result["y_pred"])
        if return_pred_int:
            predictions = result  # full DF with y_pred / lower / upper
        else:
            predictions = y_pred_df

        # ---- compute test metrics if y_test is available
        metrics_df: pd.DataFrame | None = None
        y_test = self._fit_state.get("y_test")
        if y_test is not None and len(y_test) > 0:
            # Align y_test to predictions index — when the user's fh exceeds
            # the test horizon we just compute on the overlap (and skip if
            # there's no overlap).
            try:
                aligned_test = y_test.loc[y_pred_df.index]
            except (KeyError, TypeError):
                aligned_test = None

            if aligned_test is not None and len(aligned_test) > 0:
                metrics_registry = self._build_ts_metric_registry()
                additional_kwargs = update_additional_scorer_kwargs(
                    initial_kwargs={"sp": self._primary_sp_to_use()},
                    y_train=self._fit_state.get("y_train"),
                    lower=result["lower"],
                    upper=result["upper"],
                )
                try:
                    metric_dict = calculate_metrics(
                        metrics=metrics_registry,
                        y_test=aligned_test,
                        pred=y_pred_df["y_pred"],
                        pred_proba=None,
                        **additional_kwargs,
                    )
                    if metric_dict:
                        full_name = type(estimator_).__name__
                        metrics_df = pd.DataFrame(metric_dict, index=[0])
                        metrics_df.insert(0, "Model", full_name)
                        metrics_df = metrics_df.round(round)
                        self._set_last_metrics(metrics_df)
                except Exception:  # noqa: BLE001 — defensive
                    metrics_df = None

        self.logger.log(
            EventKind.MODEL_PREDICTED,
            duration_ms=(_time.perf_counter() - t0) * 1000,
            payload={"estimator": self._describe_estimator(estimator)},
        )
        return PredictResult(predictions=predictions, metrics=metrics_df)
