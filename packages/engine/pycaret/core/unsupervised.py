"""`UnsupervisedExperiment` — base for clustering and anomaly detection.

Unsupervised tasks don't have a target column. They add `assign_model` — the
verb that labels every row in the dataset with its predicted cluster / anomaly
score using a fitted model.

Session-28 drain: `create_model` and `assign_model` no longer delegate to
`self._legacy.<verb>`. They run sklearn / pyod estimators directly on
`self._legacy.X_transformed` and return real sklearn Pipelines.
"""

from __future__ import annotations

import time
from typing import Any

from pycaret.core.errors import ConfigurationError
from pycaret.core.experiment import Experiment
from pycaret.core.results import CreateResult
from pycaret.core.tasks import TaskType
from pycaret.logging.events import EventKind


class UnsupervisedExperiment(Experiment):
    """Base for clustering / anomaly experiments."""

    def _is_supervised(self) -> bool:  # override
        return False

    # ------------------------------------------------------- create_model

    def create_model(
        self,
        estimator: Any,
        *,
        num_clusters: int | None = None,
        fraction: float | None = None,
        fit_kwargs: dict | None = None,
        round: int = 4,
        verbose: bool = False,
        **estimator_kwargs: Any,
    ) -> CreateResult:
        """Train an unsupervised model + return a ``CreateResult``.

        Session-28 drain: this verb no longer delegates to
        ``self._legacy.create_model``. It resolves the estimator from the
        engine's clustering / anomaly registry, fits it on
        ``self._legacy.X_transformed``, and assembles a sklearn Pipeline
        (preprocessor + fitted model).

        Parameters
        ----------
        estimator : str or sklearn-compatible object
            Registry ID (``"kmeans"``, ``"iforest"``, ...) or a
            pre-constructed object with ``.fit``.
        num_clusters : int, optional
            Clustering only. Passed as ``n_clusters`` to algorithms that
            accept it (KMeans, Agglomerative, etc.).
        fraction : float, optional
            Anomaly only. Passed as ``contamination`` to algorithms that
            accept it (IForest, LOF, etc.).
        fit_kwargs : dict, optional
            Forwarded to ``model.fit``.
        round : int, default=4
            Reserved (unsupervised tasks don't currently emit metrics).
        verbose : bool, default=False
            Reserved; legacy progress hook.
        **estimator_kwargs
            Forwarded to the constructor when ``estimator`` is a registry
            ID. Merged over the registry's defaults.
        """
        self._require_fitted()
        from copy import deepcopy

        from sklearn.pipeline import Pipeline as SkPipeline

        t0 = time.perf_counter()

        # ---- resolve estimator → instance + model_id
        if isinstance(estimator, str):
            registry = getattr(self._legacy, "_all_models_internal", {})
            if estimator not in registry:
                raise ConfigurationError(
                    f"Unknown model id {estimator!r}. Call "
                    "Experiment.list_models() for available IDs."
                )
            container = registry[estimator]
            init_kwargs = dict(container.args)
            init_kwargs.update(estimator_kwargs)
            # Translate clustering num_clusters → n_clusters when the
            # constructor accepts it (most clustering algos do).
            if num_clusters is not None:
                init_kwargs.setdefault("n_clusters", num_clusters)
            # Anomaly fraction → contamination.
            if fraction is not None and self.task == TaskType.ANOMALY:
                init_kwargs.setdefault("contamination", fraction)
            try:
                model = container.class_def(**init_kwargs)
            except TypeError:
                # Constructor doesn't accept one of our forwarded kwargs
                # (e.g. AffinityPropagation has no n_clusters). Drop them
                # and retry with the registry defaults.
                model = container.class_def(**dict(container.args))
            model_id = estimator
        elif isinstance(estimator, SkPipeline):
            # Already a Pipeline — pull the last step + use its name.
            model_id, model = estimator.steps[-1]
            model = deepcopy(model)
        else:
            if not hasattr(estimator, "fit"):
                raise TypeError(
                    "estimator must be a registry ID string or an object with a `.fit` method."
                )
            model = deepcopy(estimator)
            model_id = type(estimator).__name__

        self.logger.log(
            EventKind.MODEL_CREATE_STARTED,
            payload={"estimator": model_id},
        )

        # ---- fit on the transformed data
        X = self._legacy.X_transformed
        fit_kwargs = fit_kwargs or {}
        # CBLOF (`cluster` anomaly detector) can fail when the default
        # n_clusters yields a degenerate small/large cluster separation.
        # Mirrors the legacy retry: bump n_clusters to 12 and try once more.
        is_cblof = isinstance(estimator, str) and estimator == "cluster"
        try:
            model.fit(X, **fit_kwargs)
        except ValueError:
            if is_cblof and hasattr(model, "set_params"):
                try:
                    model.set_params(n_clusters=12)
                    model.fit(X, **fit_kwargs)
                except Exception as e:  # noqa: BLE001 — surface the real cause
                    raise RuntimeError(
                        "Could not form valid cluster separation. Try a different dataset or model."
                    ) from e
            else:
                raise

        # ---- assemble Pipeline (preprocessor + fitted model)
        pipeline = deepcopy(self.preprocess_pipeline)
        pipeline.steps.append((model_id, model))

        self.logger.log(
            EventKind.MODEL_CREATED,
            duration_ms=(time.perf_counter() - t0) * 1000,
            payload={"estimator": model_id},
        )
        return CreateResult(
            pipeline=pipeline,
            model_id=model_id,
            metrics=None,  # unsupervised tasks don't emit a metrics DF in v1
            params=self._safe_params(model),
        )

    # ------------------------------------------------------- assign_model

    def assign_model(
        self,
        estimator: Any,
        *,
        transformation: bool = False,
        score: bool = True,
        verbose: bool = False,
    ) -> Any:
        """Decorate the training data with predicted cluster / anomaly labels.

        Session-28 drain: no longer delegates to
        ``self._legacy.assign_model``. Reads ``model.labels_`` (and, for
        anomaly, ``model.decision_scores_``) and attaches them to a copy
        of ``self.X``.

        Parameters
        ----------
        estimator : sklearn Pipeline or fitted estimator
            Typically the ``CreateResult.pipeline`` from ``create_model``.
        transformation : bool, default=False
            If True, return rows from the transformed (post-preprocessor)
            data instead of the raw ``self.X``. Mostly useful for
            debugging the preprocessing chain.
        score : bool, default=True
            Anomaly only. If True, the ``Anomaly_Score`` column is
            attached.
        verbose : bool, default=False
            Reserved; legacy progress hook.
        """
        self._require_fitted()
        import pandas as _pd
        from sklearn.pipeline import Pipeline as SkPipeline

        # Unwrap Pipeline → bare fitted model. The labels live on the
        # estimator that was actually fit on the data.
        if isinstance(estimator, SkPipeline):
            model = estimator.steps[-1][1]
        else:
            model = estimator

        if not hasattr(model, "labels_"):
            raise ValueError(
                f"{type(model).__name__} doesn't expose `.labels_`; was the "
                "model fit on the experiment's data?"
            )

        data: _pd.DataFrame = self._legacy.X_transformed.copy() if transformation else self.X.copy()

        if self.task == TaskType.CLUSTERING:
            data["Cluster"] = [f"Cluster {i}" for i in model.labels_]
        elif self.task == TaskType.ANOMALY:
            data["Anomaly"] = model.labels_
            if score and hasattr(model, "decision_scores_"):
                data["Anomaly_Score"] = model.decision_scores_
        else:
            raise ConfigurationError("assign_model is only valid for clustering / anomaly tasks.")

        self.logger.log(EventKind.MODEL_PREDICTED, payload={"shape": list(data.shape)})
        return data
