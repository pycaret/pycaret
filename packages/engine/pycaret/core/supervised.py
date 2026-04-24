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

    # Map PyCaret metric display names → sklearn built-in scorer strings.
    # Classification. Unknown → fall through to "accuracy" default.
    _CLF_OPTIMIZE_SKLEARN = {
        "Accuracy": "accuracy",
        "accuracy": "accuracy",
        "AUC": "roc_auc",
        "roc_auc": "roc_auc",
        "Recall": "recall",
        "recall": "recall",
        "Prec.": "precision",
        "Precision": "precision",
        "precision": "precision",
        "F1": "f1",
        "f1": "f1",
    }
    # Regression. Default "R2".
    _REG_OPTIMIZE_SKLEARN = {
        "R2": "r2",
        "r2": "r2",
        "MAE": "neg_mean_absolute_error",
        "MSE": "neg_mean_squared_error",
        "RMSE": "neg_root_mean_squared_error",
        "MAPE": "neg_mean_absolute_percentage_error",
    }

    def tune_model(
        self,
        estimator: Any,
        *,
        fold: Any | None = None,
        n_iter: int = 10,
        custom_grid: dict | None = None,
        optimize: str | None = None,
        fit_kwargs: dict | None = None,
        round: int = 4,
        verbose: bool = False,
    ) -> TuneResult:
        """Hyperparameter search via ``RandomizedSearchCV``.

        Session-25 drain (supervised path): no longer delegates to
        ``self._legacy.tune_model``. Pulls the model's search space from
        the engine's registry (or accepts a ``custom_grid``), runs
        ``RandomizedSearchCV`` on the base estimator, and returns a fitted
        Pipeline (preprocessor + best estimator). Clustering / anomaly /
        time-series fall back to legacy delegation (future drain).

        Parameters
        ----------
        estimator : sklearn-compatible object, Pipeline, or registry ID
            From ``create_model``, a Pipeline; its last step is the model
            being tuned. A registry ID (``"lr"``) or a bare estimator
            are also accepted.
        fold : int or cross-validator, optional
            Defaults to the experiment's configured CV generator.
        n_iter : int, default=10
            Random-search iterations.
        custom_grid : dict, optional
            Replaces the registry's default search space. Use un-prefixed
            parameter names (we unwrap to the bare estimator first).
        optimize : str, optional
            Metric to optimise. Accepts either PyCaret-style names
            (``"Accuracy"``, ``"AUC"``, ``"MAE"``, ``"R2"``) or sklearn
            scorer names. Defaults to ``"accuracy"`` (clf) / ``"r2"`` (reg).
        fit_kwargs : dict, optional
            Forwarded to the underlying ``.fit()`` inside the search.
        round : int, default=4
            Decimal places for the returned metrics DataFrame.
        verbose : bool, default=False
            Reserved; currently ignored (legacy progress hook).

        Returns
        -------
        TuneResult
            - ``pipeline`` — fitted Pipeline with the tuned model.
            - ``best_params`` — dict of winning hyperparameters.
            - ``search`` — the ``RandomizedSearchCV`` instance.
            - ``cv_results`` — DataFrame of sklearn's ``cv_results_``.
            - ``metrics`` — per-fold metrics (same shape as
              ``CreateResult.metrics``) for the tuned model.
        """
        self._require_fitted()

        from pycaret.core.tasks import TaskType

        if self.task not in (TaskType.CLASSIFICATION, TaskType.REGRESSION):
            return self._tune_model_legacy(
                estimator,
                fold=fold,
                n_iter=n_iter,
                custom_grid=custom_grid,
                optimize=optimize,
                verbose=verbose,
            )

        return self._tune_model_supervised_native(
            estimator,
            fold=fold,
            n_iter=n_iter,
            custom_grid=custom_grid,
            optimize=optimize,
            fit_kwargs=fit_kwargs or {},
            round=round,
            verbose=verbose,
        )

    # ---------------------- tune_model — native supervised path

    def _tune_model_supervised_native(
        self,
        estimator: Any,
        *,
        fold: Any | None,
        n_iter: int,
        custom_grid: dict | None,
        optimize: str | None,
        fit_kwargs: dict,
        round: int,
        verbose: bool,
    ) -> TuneResult:
        from copy import deepcopy

        import pandas as _pd
        from sklearn.model_selection import RandomizedSearchCV
        from sklearn.pipeline import Pipeline as SkPipeline

        from pycaret.core.tasks import TaskType

        t0 = time.perf_counter()

        # ---- resolve estimator → (bare model, model_id)
        if isinstance(estimator, SkPipeline):
            model_id, bare_model = estimator.steps[-1]
        elif isinstance(estimator, str):
            bare_model, model_id = self._resolve_supervised_estimator(estimator, {})
        else:
            bare_model = estimator
            model_id = type(estimator).__name__

        self.logger.log(
            EventKind.MODEL_TUNE_STARTED,
            payload={"estimator": model_id, "n_iter": n_iter},
        )

        # ---- search space: custom_grid > registry.tune_distributions > tune_grid
        search_space = custom_grid
        if search_space is None:
            registry = getattr(self._legacy, "_all_models_internal", {})
            container = registry.get(model_id)
            if container is not None:
                # Prefer `tune_grid` (explicit dict[str, list]) because
                # `tune_distribution` uses a custom PyCaret distribution type
                # that sklearn's RandomizedSearchCV doesn't accept (needs
                # iterable-or-scipy-dist). Callers wanting continuous scipy
                # distributions pass `custom_grid=` directly. Adapting
                # `tune_distribution` → scipy is a future polish.
                search_space = getattr(container, "tune_grid", None) or {}
        if not search_space:
            # No tuning possible — fall through to a plain CV fit.
            tuned = self.create_model(
                bare_model,
                fold=fold,
                cross_validation=True,
                fit_kwargs=fit_kwargs,
                round=round,
                verbose=verbose,
            )
            self.logger.log(
                EventKind.MODEL_TUNED,
                duration_ms=(time.perf_counter() - t0) * 1000,
                payload={"estimator": model_id, "note": "no search space"},
            )
            return TuneResult(
                pipeline=tuned.pipeline,
                best_params=tuned.params,
                search=None,
                cv_results=None,
                metrics=tuned.metrics,
            )

        # ---- resolve scoring
        if self.task == TaskType.CLASSIFICATION:
            scoring = self._CLF_OPTIMIZE_SKLEARN.get(optimize or "Accuracy", "accuracy")
        else:
            scoring = self._REG_OPTIMIZE_SKLEARN.get(optimize or "R2", "r2")

        # ---- pull transformed train data + CV generator
        X_train = self._legacy.X_train_transformed
        y_train = self._legacy.y_train_transformed
        cv = fold if fold is not None else self._legacy.fold_generator

        # ---- run the search
        search = RandomizedSearchCV(
            estimator=deepcopy(bare_model),
            param_distributions=search_space,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=self.n_jobs,
            refit=True,
            random_state=self.session_id,
            error_score=0.0,
        )
        search.fit(X_train, y_train, **fit_kwargs)

        # ---- assemble the tuned Pipeline
        tuned_pipeline = deepcopy(self.preprocess_pipeline)
        tuned_pipeline.steps.append((model_id, search.best_estimator_))

        # ---- cv_results_ DataFrame + per-fold metrics (re-uses session-24 helper)
        try:
            cv_results_df = _pd.DataFrame(search.cv_results_)
        except Exception:
            cv_results_df = None
        try:
            metrics_df = self._cross_validate_supervised(
                model=search.best_estimator_,
                X=X_train,
                y=y_train,
                cv=cv,
                round_=round,
            )
        except Exception:
            metrics_df = None

        self.logger.log(
            EventKind.MODEL_TUNED,
            duration_ms=(time.perf_counter() - t0) * 1000,
            payload={
                "estimator": model_id,
                "best_params": dict(search.best_params_),
            },
        )

        return TuneResult(
            pipeline=tuned_pipeline,
            best_params=dict(search.best_params_),
            search=search,
            cv_results=cv_results_df,
            metrics=metrics_df,
        )

    def _tune_model_legacy(
        self,
        estimator: Any,
        *,
        fold: Any | None = None,
        n_iter: int = 10,
        custom_grid: dict | None = None,
        optimize: str | None = None,
        verbose: bool = False,
    ) -> TuneResult:
        """Fallback for tasks whose tune_model hasn't been drained yet."""
        t0 = time.perf_counter()
        self.logger.log(EventKind.MODEL_TUNE_STARTED)
        kwargs: dict[str, Any] = {"n_iter": n_iter, "verbose": verbose}
        if fold is not None:
            kwargs["fold"] = fold
        if custom_grid is not None:
            kwargs["custom_grid"] = custom_grid
        if optimize is not None:
            kwargs["optimize"] = optimize
        tuned = self._legacy.tune_model(estimator, **kwargs)
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
