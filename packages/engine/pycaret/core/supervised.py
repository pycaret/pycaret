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

    # Models excluded by default when turbo=True — known to be slow on
    # datasets larger than a few hundred rows. Mirrors the legacy default.
    _TURBO_EXCLUDE = frozenset({"rbfsvm", "gpc", "mlp"})

    def compare_models(
        self,
        *,
        include: list[Any] | None = None,
        exclude: list[str] | None = None,
        fold: Any | None = None,
        cross_validation: bool = True,
        sort: str | None = None,
        n_select: int = 1,
        turbo: bool = True,
        errors: str = "ignore",
        fit_kwargs: dict | None = None,
        round: int = 4,
        verbose: bool = False,
    ) -> CompareResult:
        """Train every (or every selected) model in the registry + rank.

        Session-26 drain (supervised path): no longer delegates to
        ``self._legacy.compare_models``. Iterates the model registry,
        calls ``self.create_model`` per model (which already runs CV via
        the shared metric registry), and assembles the leaderboard from
        each model's ``Mean`` metrics row.

        Parameters
        ----------
        include : list of str or estimator objects, optional
            Restrict the comparison to these models. Strings are looked
            up in the registry; objects are wrapped via ``create_model``.
            If ``None``, every active registry entry is used.
        exclude : list of str, optional
            Registry IDs to omit. Applied after ``include``.
        fold : int or cross-validator, optional
            Defaults to the experiment's configured CV generator.
        cross_validation : bool, default=True
            If False, each model is fit-only (no CV); leaderboard rows
            still come from a single train-only metric pass.
        sort : str, optional
            Metric column to rank by. Accepts PyCaret display names
            (``"Accuracy"``, ``"AUC"``, ``"R2"``, ...). Default:
            ``"Accuracy"`` for classification, ``"R2"`` for regression.
        n_select : int, default=1
            How many top-ranked models to return.
        turbo : bool, default=True
            If True, ``rbfsvm`` / ``gpc`` / ``mlp`` are skipped (slow on
            anything but tiny datasets).
        errors : str, default="ignore"
            ``"ignore"`` skips a model on per-model failure; ``"raise"``
            propagates.
        fit_kwargs : dict, optional
            Forwarded to each model's ``.fit()``.
        round : int, default=4
            Decimal places for the leaderboard.
        verbose : bool, default=False
            Reserved; currently ignored (legacy progress hook).

        Returns
        -------
        CompareResult
            - ``best`` — top-1 fitted Pipeline (preprocessor + best model).
            - ``models`` — top-K Pipelines.
            - ``leaderboard`` — DataFrame indexed by ranked model name,
              with metric-Mean columns + a ``Model`` ID column.
            - ``ranked_ids`` — list of model IDs in rank order.
        """
        self._require_fitted()

        from pycaret.core.tasks import TaskType

        if self.task not in (TaskType.CLASSIFICATION, TaskType.REGRESSION):
            return self._compare_models_legacy(
                include=include,
                exclude=exclude,
                fold=fold,
                cross_validation=cross_validation,
                sort=sort,
                n_select=n_select,
                turbo=turbo,
                fit_kwargs=fit_kwargs,
                round=round,
                verbose=verbose,
            )

        return self._compare_models_supervised_native(
            include=include,
            exclude=exclude,
            fold=fold,
            cross_validation=cross_validation,
            sort=sort,
            n_select=n_select,
            turbo=turbo,
            errors=errors,
            fit_kwargs=fit_kwargs or {},
            round=round,
            verbose=verbose,
        )

    def _compare_models_supervised_native(
        self,
        *,
        include: list[Any] | None,
        exclude: list[str] | None,
        fold: Any | None,
        cross_validation: bool,
        sort: str | None,
        n_select: int,
        turbo: bool,
        errors: str,
        fit_kwargs: dict,
        round: int,
        verbose: bool,
    ) -> CompareResult:
        """Native compare_models for classification + regression."""
        import pandas as _pd

        from pycaret.core.tasks import TaskType

        t0 = time.perf_counter()

        # ---- decide which models to compare
        registry = self._fit_state.get("model_registry", {})
        if include is not None:
            candidates = list(include)
        else:
            candidates = [mid for mid, c in registry.items() if not getattr(c, "is_special", False)]
        if exclude:
            candidates = [c for c in candidates if c not in set(exclude)]
        if turbo:
            candidates = [
                c for c in candidates if not (isinstance(c, str) and c in self._TURBO_EXCLUDE)
            ]

        # ---- default sort metric per task
        if sort is None:
            sort = "Accuracy" if self.task == TaskType.CLASSIFICATION else "R2"

        self.logger.log(
            EventKind.MODEL_COMPARE_STARTED,
            payload={"n_candidates": len(candidates), "sort": sort},
        )

        # ---- per-model training loop
        rows: list[dict] = []
        pipelines: dict[str, Any] = {}
        for cand in candidates:
            try:
                created = self.create_model(
                    cand,
                    fold=fold,
                    cross_validation=cross_validation,
                    fit_kwargs=fit_kwargs,
                    round=round,
                    verbose=False,
                )
            except Exception:
                if errors == "raise":
                    raise
                continue

            mid = created.model_id
            pipelines[mid] = created.pipeline

            row: dict[str, Any] = {"Model": mid}
            if created.metrics is not None and "Mean" in created.metrics.index:
                mean_row = created.metrics.loc["Mean"].to_dict()
                row.update(mean_row)
            rows.append(row)

        if not rows:
            # Nothing succeeded — return an empty CompareResult rather than
            # raising, matching the documented `errors="ignore"` semantics.
            self.logger.log(
                EventKind.MODEL_COMPARE_FINISHED,
                duration_ms=(time.perf_counter() - t0) * 1000,
                payload={"n_select": n_select, "n_succeeded": 0},
            )
            return CompareResult(
                best=None,
                models=[],
                leaderboard=_pd.DataFrame(),
                ranked_ids=[],
            )

        # ---- assemble leaderboard
        leaderboard = _pd.DataFrame(rows)
        # Sort: descending for greater-is-better metrics, ascending for error.
        ascending = self._sort_metric_is_ascending(sort)
        if sort in leaderboard.columns:
            leaderboard = leaderboard.sort_values(by=sort, ascending=ascending).reset_index(
                drop=True
            )
        leaderboard = leaderboard.round(round)

        ranked_ids: list[str] = leaderboard["Model"].astype(str).tolist()
        top_ids = ranked_ids[: max(1, n_select)]
        models = [pipelines[mid] for mid in top_ids]

        self.logger.log(
            EventKind.MODEL_COMPARE_FINISHED,
            duration_ms=(time.perf_counter() - t0) * 1000,
            payload={
                "n_select": n_select,
                "n_succeeded": len(rows),
                "winner": ranked_ids[0] if ranked_ids else None,
            },
        )

        self._set_last_metrics(leaderboard)
        return CompareResult(
            best=models[0] if models else None,
            models=models,
            leaderboard=leaderboard,
            ranked_ids=ranked_ids,
        )

    @staticmethod
    def _sort_metric_is_ascending(sort: str) -> bool:
        """Return True if the metric is "smaller is better" (errors etc.)."""
        ascending_metrics = {
            "MAE",
            "mae",
            "MSE",
            "mse",
            "RMSE",
            "rmse",
            "RMSLE",
            "rmsle",
            "MAPE",
            "mape",
            "neg_mean_absolute_error",
            "neg_mean_squared_error",
            "neg_root_mean_squared_error",
            "neg_mean_absolute_percentage_error",
        }
        # neg_* sklearn names: bigger is better (less negative). Treat as desc.
        if sort.startswith("neg_"):
            return False
        return sort in ascending_metrics

    def _compare_models_legacy(self, **kwargs: Any) -> CompareResult:
        """Phase 6 stub. The legacy fallback was removed when
        ``pycaret/internal/pycaret_experiment/`` was deleted. Reachable
        only when called on a non-supervised, non-TS experiment that
        doesn't override ``compare_models`` itself — i.e. clustering /
        anomaly. Drain those if needed.
        """
        raise NotImplementedError(
            f"compare_models() is not yet supported natively for task "
            f"{self.task.value!r}. Phase 6 removed the legacy fallback. "
            "Open an issue if you need cross-model comparison for this task."
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
            registry = self._fit_state.get("model_registry", {})
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

        # ---- pull transformed train data + CV generator from fit-time snapshot
        X_train = self._fit_state["X_train_transformed"]
        y_train = self._fit_state["y_train_transformed"]
        cv = fold if fold is not None else self._fit_state["fold_generator"]

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

        self._set_last_metrics(metrics_df)
        return TuneResult(
            pipeline=tuned_pipeline,
            best_params=dict(search.best_params_),
            search=search,
            cv_results=cv_results_df,
            metrics=metrics_df,
        )

    def _tune_model_legacy(self, estimator: Any, **kwargs: Any) -> TuneResult:
        """Phase 6 stub. The legacy fallback was removed when
        ``pycaret/internal/pycaret_experiment/`` was deleted.
        Clustering / anomaly don't have a native tune path yet.
        """
        raise NotImplementedError(
            f"tune_model() is not yet supported natively for task "
            f"{self.task.value!r}. Phase 6 removed the legacy fallback."
        )

    # --------------------------------------------------------- ensembling
    # Sessions 27 (drain): ensemble_model / blend_models / stack_models /
    # calibrate_model / finalize_model are all native for supervised tasks.
    # Each is a thin wrapper around an sklearn meta-estimator + the same
    # `deepcopy(self.preprocess_pipeline) + [(name, model)]` Pipeline assembly
    # used by `create_model`. Time-series / clustering / anomaly delegate.

    def _unwrap_estimator(self, estimator: Any) -> tuple[Any, str]:
        """Return (bare estimator, model_id) from a Pipeline or a bare model.

        Pipelines come in from create_model / tune_model / compare_models;
        their last step is the model. Bare estimators are accepted too —
        registry IDs are resolved via `_resolve_supervised_estimator`.
        """
        from sklearn.pipeline import Pipeline as SkPipeline

        if isinstance(estimator, SkPipeline):
            model_id, bare = estimator.steps[-1]
            return bare, model_id
        if isinstance(estimator, str):
            return self._resolve_supervised_estimator(estimator, {})
        return estimator, type(estimator).__name__

    def _wrap_in_pipeline(self, model: Any, name: str) -> Any:
        """Build the canonical Pipeline shape: preprocessing + (name, model)."""
        from copy import deepcopy

        pipeline = deepcopy(self.preprocess_pipeline)
        pipeline.steps.append((name, model))
        return pipeline

    def ensemble_model(
        self,
        estimator: Any,
        *,
        method: str = "Bagging",
        n_estimators: int = 10,
        fold: Any | None = None,
        round: int = 4,
        fit_kwargs: dict | None = None,
        verbose: bool = False,
    ) -> EnsembleResult:
        """Bagging or Boosting wrapper around a base estimator.

        Session-27 drain. ``method="Bagging"`` uses
        ``BaggingClassifier``/``BaggingRegressor``; ``method="Boosting"`` uses
        ``AdaBoostClassifier``/``AdaBoostRegressor``. Returns a fitted
        Pipeline (preprocessor + ensemble).
        """
        self._require_fitted()
        from pycaret.core.tasks import TaskType

        if self.task not in (TaskType.CLASSIFICATION, TaskType.REGRESSION):
            return self._ensemble_model_legacy(
                estimator,
                method=method,
                n_estimators=n_estimators,
                fold=fold,
                verbose=verbose,
            )

        from copy import deepcopy

        t0 = time.perf_counter()
        bare, model_id = self._unwrap_estimator(estimator)
        self.logger.log(
            EventKind.MODEL_ENSEMBLE_STARTED,
            payload={"estimator": model_id, "method": method},
        )

        if method == "Bagging":
            if self.task == TaskType.CLASSIFICATION:
                from sklearn.ensemble import BaggingClassifier

                meta = BaggingClassifier(
                    estimator=deepcopy(bare),
                    n_estimators=n_estimators,
                    random_state=self.session_id,
                    n_jobs=self.n_jobs,
                )
            else:
                from sklearn.ensemble import BaggingRegressor

                meta = BaggingRegressor(
                    estimator=deepcopy(bare),
                    n_estimators=n_estimators,
                    random_state=self.session_id,
                    n_jobs=self.n_jobs,
                )
            wrapped_name = f"Bagging[{model_id}]"
        elif method == "Boosting":
            if self.task == TaskType.CLASSIFICATION:
                from sklearn.ensemble import AdaBoostClassifier

                meta = AdaBoostClassifier(
                    estimator=deepcopy(bare),
                    n_estimators=n_estimators,
                    random_state=self.session_id,
                )
            else:
                from sklearn.ensemble import AdaBoostRegressor

                meta = AdaBoostRegressor(
                    estimator=deepcopy(bare),
                    n_estimators=n_estimators,
                    random_state=self.session_id,
                )
            wrapped_name = f"AdaBoost[{model_id}]"
        else:
            raise ValueError(f"method must be 'Bagging' or 'Boosting', got {method!r}")

        # Train via create_model so we get CV metrics + a Pipeline.
        created = self.create_model(
            meta,
            fold=fold,
            cross_validation=True,
            fit_kwargs=fit_kwargs or {},
            round=round,
            verbose=False,
        )
        # Replace the auto-named pipeline step with our descriptive name.
        created.pipeline.steps[-1] = (wrapped_name, created.pipeline.steps[-1][1])

        self.logger.log(
            EventKind.MODEL_ENSEMBLED,
            duration_ms=(time.perf_counter() - t0) * 1000,
            payload={"estimator": wrapped_name},
        )
        return EnsembleResult(pipeline=created.pipeline, method=method, metrics=created.metrics)

    def blend_models(
        self,
        estimators: list[Any],
        *,
        method: str = "auto",
        weights: list[float] | None = None,
        fold: Any | None = None,
        round: int = 4,
        fit_kwargs: dict | None = None,
        verbose: bool = False,
    ) -> BlendResult:
        """Soft / hard voting ensemble across multiple estimators.

        Session-27 drain. Wraps sklearn ``VotingClassifier`` / ``VotingRegressor``.
        Classification ``method="auto"`` picks ``"soft"`` when every base
        model has ``predict_proba``, else ``"hard"``.
        """
        self._require_fitted()
        from pycaret.core.tasks import TaskType

        if self.task not in (TaskType.CLASSIFICATION, TaskType.REGRESSION):
            return self._blend_models_legacy(
                estimators,
                method=method,
                weights=weights,
                fold=fold,
                verbose=verbose,
            )

        from copy import deepcopy

        t0 = time.perf_counter()
        unwrapped: list[tuple[str, Any]] = []
        for i, est in enumerate(estimators):
            bare, mid = self._unwrap_estimator(est)
            # Voting estimator names must be unique strings.
            unwrapped.append((f"{mid}_{i}", deepcopy(bare)))

        self.logger.log(
            EventKind.MODEL_BLEND_STARTED,
            payload={"n_models": len(unwrapped), "method": method},
        )

        if self.task == TaskType.CLASSIFICATION:
            from sklearn.ensemble import VotingClassifier

            voting = method
            if voting == "auto":
                voting = (
                    "soft" if all(hasattr(m, "predict_proba") for _, m in unwrapped) else "hard"
                )
            meta = VotingClassifier(
                estimators=unwrapped,
                voting=voting,
                weights=weights,
                n_jobs=self.n_jobs,
            )
        else:
            from sklearn.ensemble import VotingRegressor

            meta = VotingRegressor(estimators=unwrapped, weights=weights, n_jobs=self.n_jobs)

        created = self.create_model(
            meta,
            fold=fold,
            cross_validation=True,
            fit_kwargs=fit_kwargs or {},
            round=round,
            verbose=False,
        )
        created.pipeline.steps[-1] = ("Voting", created.pipeline.steps[-1][1])
        self.logger.log(
            EventKind.MODEL_BLENDED,
            duration_ms=(time.perf_counter() - t0) * 1000,
            payload={"n_models": len(unwrapped)},
        )
        return BlendResult(pipeline=created.pipeline, metrics=created.metrics)

    def stack_models(
        self,
        estimators: list[Any],
        *,
        meta_model: Any | None = None,
        fold: Any | None = None,
        round: int = 4,
        fit_kwargs: dict | None = None,
        verbose: bool = False,
    ) -> StackResult:
        """Two-layer stacking ensemble with a meta-learner.

        Session-27 drain. Wraps sklearn ``StackingClassifier`` /
        ``StackingRegressor``. The meta-learner defaults to ``LogisticRegression``
        for classification, ``LinearRegression`` for regression — matching
        the legacy default.
        """
        self._require_fitted()
        from pycaret.core.tasks import TaskType

        if self.task not in (TaskType.CLASSIFICATION, TaskType.REGRESSION):
            return self._stack_models_legacy(
                estimators,
                meta_model=meta_model,
                fold=fold,
                verbose=verbose,
            )

        from copy import deepcopy

        t0 = time.perf_counter()
        unwrapped: list[tuple[str, Any]] = []
        for i, est in enumerate(estimators):
            bare, mid = self._unwrap_estimator(est)
            unwrapped.append((f"{mid}_{i}", deepcopy(bare)))

        if meta_model is not None:
            meta_bare, meta_id = self._unwrap_estimator(meta_model)
        else:
            if self.task == TaskType.CLASSIFICATION:
                from sklearn.linear_model import LogisticRegression

                meta_bare = LogisticRegression(max_iter=1000, random_state=self.session_id)
                meta_id = "LogisticRegression"
            else:
                from sklearn.linear_model import LinearRegression

                meta_bare = LinearRegression()
                meta_id = "LinearRegression"

        self.logger.log(
            EventKind.MODEL_STACK_STARTED,
            payload={"n_models": len(unwrapped), "meta": meta_id},
        )

        cv = fold if fold is not None else self._fit_state["fold_generator"]
        if self.task == TaskType.CLASSIFICATION:
            from sklearn.ensemble import StackingClassifier

            meta = StackingClassifier(
                estimators=unwrapped,
                final_estimator=deepcopy(meta_bare),
                cv=cv,
                n_jobs=self.n_jobs,
            )
        else:
            from sklearn.ensemble import StackingRegressor

            meta = StackingRegressor(
                estimators=unwrapped,
                final_estimator=deepcopy(meta_bare),
                cv=cv,
                n_jobs=self.n_jobs,
            )

        created = self.create_model(
            meta,
            fold=fold,
            cross_validation=True,
            fit_kwargs=fit_kwargs or {},
            round=round,
            verbose=False,
        )
        created.pipeline.steps[-1] = (
            f"Stacking[{meta_id}]",
            created.pipeline.steps[-1][1],
        )
        self.logger.log(
            EventKind.MODEL_STACKED,
            duration_ms=(time.perf_counter() - t0) * 1000,
            payload={"n_models": len(unwrapped), "meta": meta_id},
        )
        return StackResult(pipeline=created.pipeline, metrics=created.metrics)

    # --------------------------------------------------------- calibration / finalize

    def calibrate_model(
        self,
        estimator: Any,
        *,
        method: str = "sigmoid",
        cv: int | Any | None = None,
        fold: Any | None = None,
        round: int = 4,
        fit_kwargs: dict | None = None,
        verbose: bool = False,
    ) -> CalibrateResult:
        """Probability calibration via ``CalibratedClassifierCV``.

        Session-27 drain. Classification only — calibration is undefined for
        regression. ``method`` is ``"sigmoid"`` (Platt scaling) or
        ``"isotonic"``.
        """
        self._require_fitted()
        from pycaret.core.tasks import TaskType

        if self.task != TaskType.CLASSIFICATION:
            raise ValueError(
                "calibrate_model is only valid for classification tasks. "
                f"This is a {self.task.value} experiment."
            )

        from copy import deepcopy

        from sklearn.calibration import CalibratedClassifierCV

        t0 = time.perf_counter()
        bare, model_id = self._unwrap_estimator(estimator)
        self.logger.log(
            EventKind.MODEL_CALIBRATE_STARTED,
            payload={"estimator": model_id, "method": method},
        )

        meta = CalibratedClassifierCV(
            estimator=deepcopy(bare),
            method=method,
            cv=cv if cv is not None else (fold or self._fit_state["fold_generator"]),
            n_jobs=self.n_jobs,
        )

        created = self.create_model(
            meta,
            fold=fold,
            cross_validation=True,
            fit_kwargs=fit_kwargs or {},
            round=round,
            verbose=False,
        )
        created.pipeline.steps[-1] = (
            f"Calibrated[{model_id}]",
            created.pipeline.steps[-1][1],
        )
        self.logger.log(
            EventKind.MODEL_CALIBRATED,
            duration_ms=(time.perf_counter() - t0) * 1000,
            payload={"estimator": model_id, "method": method},
        )
        return CalibrateResult(pipeline=created.pipeline, method=method, metrics=created.metrics)

    def finalize_model(self, estimator: Any) -> FinalizeResult:
        """Re-fit ``estimator`` on the **full** training set (train + holdout).

        Session-27 drain. Used right before deploying — the holdout has
        already served its purpose, so squeeze it back into training. Returns
        a fresh fitted Pipeline; the input is left untouched.
        """
        self._require_fitted()
        from pycaret.core.tasks import TaskType

        if self.task not in (TaskType.CLASSIFICATION, TaskType.REGRESSION):
            return self._finalize_model_legacy(estimator)

        from copy import deepcopy

        t0 = time.perf_counter()
        bare, model_id = self._unwrap_estimator(estimator)
        # Re-fit on the FULL transformed dataset (train + test combined).
        # `_fit_state["X_transformed"]` is the union; same for y.
        X_full = self._fit_state["X_transformed"]
        y_full = self._fit_state["y_transformed"]
        finalized_model = deepcopy(bare)
        finalized_model.fit(X_full, y_full)

        pipeline = self._wrap_in_pipeline(finalized_model, model_id)
        self.logger.log(
            EventKind.MODEL_FINALIZED,
            duration_ms=(time.perf_counter() - t0) * 1000,
            payload={"estimator": model_id, "n_rows": int(len(X_full))},
        )
        return FinalizeResult(pipeline=pipeline)

    # --- phase 6 stubs for non-supervised tasks
    # The legacy fallbacks were removed when ``pycaret/internal/
    # pycaret_experiment/`` was deleted. Clustering / anomaly don't have
    # native paths for these meta-estimator verbs (they don't really apply
    # to unsupervised tasks). TS overrides finalize_model in its subclass
    # (s44) so reaches a different entry point.

    def _ensemble_model_legacy(self, estimator: Any, **kwargs: Any) -> EnsembleResult:
        raise NotImplementedError(
            f"ensemble_model() not supported for task {self.task.value!r}. "
            "Bagging / boosting are supervised-only; phase 6 removed the legacy fallback."
        )

    def _blend_models_legacy(self, estimators: list[Any], **kwargs: Any) -> BlendResult:
        raise NotImplementedError(
            f"blend_models() not supported for task {self.task.value!r}. "
            "Voting is supervised-only; phase 6 removed the legacy fallback."
        )

    def _stack_models_legacy(self, estimators: list[Any], **kwargs: Any) -> StackResult:
        raise NotImplementedError(
            f"stack_models() not supported for task {self.task.value!r}. "
            "Stacking is supervised-only; phase 6 removed the legacy fallback."
        )

    def _finalize_model_legacy(self, estimator: Any) -> FinalizeResult:
        raise NotImplementedError(
            f"finalize_model() not yet supported natively for task "
            f"{self.task.value!r}. Phase 6 removed the legacy fallback."
        )

    # --------------------------------------------------------- interpretation

    def interpret_model(self, estimator: Any, *args: Any, **kwargs: Any) -> Any:
        """Removed in PyCaret 4.0 (phase 6). The legacy SHAP / lime path was
        deleted with the rest of ``pycaret/internal/pycaret_experiment/``.
        """
        raise NotImplementedError(
            "interpret_model() was removed in PyCaret 4.0. Use SHAP "
            "directly: `import shap; shap.Explainer(pipeline.steps[-1][1])"
            "(X_test)` is the canonical replacement."
        )

    # --------------------------------------------------------- leaderboard / automl

    def automl(self, *args: Any, **kwargs: Any) -> Any:
        """Removed in PyCaret 4.0 (phase 6). Use ``compare_models`` +
        ``tune_model`` for the same workflow with explicit control.
        """
        raise NotImplementedError(
            "automl() was removed in PyCaret 4.0. Equivalent: "
            "`exp.compare_models(n_select=N).best` then `exp.tune_model(best)`."
        )

    def get_leaderboard(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Removed in PyCaret 4.0 (phase 6). The leaderboard is the
        ``leaderboard`` attribute of ``CompareResult``; no separate accessor.
        """
        raise NotImplementedError(
            "get_leaderboard() was removed in PyCaret 4.0. The leaderboard "
            "DataFrame is on `compare_models(...).leaderboard`."
        )
