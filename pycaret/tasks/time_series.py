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
        self._legacy.setup(**self._build_legacy_setup_kwargs(data, setup_kwargs))
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
