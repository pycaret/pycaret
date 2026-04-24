"""`SupervisedExperiment` — base for classification / regression / time-series.

Adds the supervised-learning verb set on top of `Experiment`:
`compare_models`, `tune_model`, `ensemble_model`, `blend_models`,
`stack_models`, `calibrate_model`, `finalize_model`, `interpret_model`,
`automl`, `get_leaderboard`.

Every verb returns a typed result dataclass and emits a structured event
through `self.logger`.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from pycaret.core.experiment import Experiment
from pycaret.core.results import (
    BlendResult,
    CalibrateResult,
    CompareResult,
    EnsembleResult,
    FinalizeResult,
    StackResult,
    TuneResult,
)
from pycaret.logging.events import EventKind

if TYPE_CHECKING:
    import pandas as pd


class SupervisedExperiment(Experiment):
    """Base for supervised tasks."""

    # --------------------------------------------------------- comparison

    def compare_models(self, *args: Any, **kwargs: Any) -> CompareResult:
        self._require_fitted()
        n_select = kwargs.get("n_select", 1)
        t0 = time.perf_counter()
        self.logger.log(EventKind.MODEL_COMPARE_STARTED)
        models = self._legacy.compare_models(*args, **kwargs)
        leaderboard = self._safe_pull()
        self.logger.log(
            EventKind.MODEL_COMPARE_FINISHED,
            duration_ms=(time.perf_counter() - t0) * 1000,
            payload={"n_select": n_select},
        )
        models_list = models if isinstance(models, list) else [models]
        return CompareResult(
            best=models_list[0],
            models=models_list,
            leaderboard=leaderboard,
            ranked_ids=self._ranked_ids_from_leaderboard(leaderboard),
        )

    # --------------------------------------------------------- tuning

    def tune_model(self, estimator: Any, *args: Any, **kwargs: Any) -> TuneResult:
        self._require_fitted()
        t0 = time.perf_counter()
        self.logger.log(EventKind.MODEL_TUNE_STARTED)
        tuned = self._legacy.tune_model(estimator, *args, **kwargs)
        metrics = self._safe_pull()
        self.logger.log(
            EventKind.MODEL_TUNED,
            duration_ms=(time.perf_counter() - t0) * 1000,
        )
        return TuneResult(
            pipeline=tuned,
            best_params=self._safe_params(tuned),
            search=None,
            cv_results=metrics,
            metrics=metrics,
        )

    # --------------------------------------------------------- ensembling

    def ensemble_model(self, estimator: Any, *args: Any, **kwargs: Any) -> EnsembleResult:
        self._require_fitted()
        self.logger.log(EventKind.MODEL_ENSEMBLE_STARTED)
        out = self._legacy.ensemble_model(estimator, *args, **kwargs)
        metrics = self._safe_pull()
        self.logger.log(EventKind.MODEL_ENSEMBLED)
        return EnsembleResult(
            pipeline=out,
            method=kwargs.get("method", "Bagging"),
            metrics=metrics,
        )

    def blend_models(self, estimators: list[Any], *args: Any, **kwargs: Any) -> BlendResult:
        self._require_fitted()
        self.logger.log(
            EventKind.MODEL_BLEND_STARTED,
            payload={"n_models": len(estimators)},
        )
        out = self._legacy.blend_models(estimators, *args, **kwargs)
        metrics = self._safe_pull()
        self.logger.log(EventKind.MODEL_BLENDED)
        return BlendResult(pipeline=out, metrics=metrics)

    def stack_models(self, estimators: list[Any], *args: Any, **kwargs: Any) -> StackResult:
        self._require_fitted()
        self.logger.log(
            EventKind.MODEL_STACK_STARTED,
            payload={"n_models": len(estimators)},
        )
        out = self._legacy.stack_models(estimators, *args, **kwargs)
        metrics = self._safe_pull()
        self.logger.log(EventKind.MODEL_STACKED)
        return StackResult(pipeline=out, metrics=metrics)

    # --------------------------------------------------------- calibration / finalize

    def calibrate_model(self, estimator: Any, *args: Any, **kwargs: Any) -> CalibrateResult:
        self._require_fitted()
        self.logger.log(EventKind.MODEL_CALIBRATE_STARTED)
        out = self._legacy.calibrate_model(estimator, *args, **kwargs)
        metrics = self._safe_pull()
        self.logger.log(EventKind.MODEL_CALIBRATED)
        return CalibrateResult(
            pipeline=out,
            method=kwargs.get("method", "sigmoid"),
            metrics=metrics,
        )

    def finalize_model(self, estimator: Any, *args: Any, **kwargs: Any) -> FinalizeResult:
        self._require_fitted()
        out = self._legacy.finalize_model(estimator, *args, **kwargs)
        self.logger.log(EventKind.MODEL_FINALIZED)
        return FinalizeResult(pipeline=out)

    # --------------------------------------------------------- interpretation

    def interpret_model(self, estimator: Any, *args: Any, **kwargs: Any) -> Any:
        self._require_fitted()
        return self._legacy.interpret_model(estimator, *args, **kwargs)

    # --------------------------------------------------------- leaderboard / automl

    def automl(self, *args: Any, **kwargs: Any) -> Any:
        self._require_fitted()
        return self._legacy.automl(*args, **kwargs)

    def get_leaderboard(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        self._require_fitted()
        return self._legacy.get_leaderboard(*args, **kwargs)
