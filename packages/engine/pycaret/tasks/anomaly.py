"""`AnomalyExperiment` — the PyCaret 4.0 anomaly-detection engine.

Unsupervised. Verb surface mirrors `ClusteringExperiment`.

Canonical import:

    from pycaret.tasks import AnomalyExperiment
    # or (backward-compat):
    from pycaret.anomaly import AnomalyExperiment

Example
-------

>>> from pycaret.datasets import get_data
>>> from pycaret.tasks import AnomalyExperiment
>>> df = get_data("anomaly")
>>> exp = AnomalyExperiment(session_id=42).fit(df)
>>> iforest = exp.create_model("iforest").pipeline
>>> labelled = exp.assign_model(iforest)  # DataFrame with "Anomaly" + score
"""

from __future__ import annotations

from pycaret.core.tasks import TaskType
from pycaret.core.unsupervised import UnsupervisedExperiment
from pycaret.logging.base import BaseLogger


class AnomalyExperiment(UnsupervisedExperiment):
    """PyCaret 4.0 anomaly-detection experiment (sklearn-compatible)."""

    def __init__(
        self,
        *,
        session_id: int | None = None,
        preprocess: bool = True,
        normalize: bool = False,
        transformation: bool = False,
        feature_selection: bool = False,
        n_jobs: int = -1,
        use_gpu: bool = False,
        logger: BaseLogger | None = None,
        log_experiment: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            task=TaskType.ANOMALY,
            target=None,
            session_id=session_id,
            preprocess=preprocess,
            normalize=normalize,
            transformation=transformation,
            feature_selection=feature_selection,
            n_jobs=n_jobs,
            use_gpu=use_gpu,
            logger=logger,
            log_experiment=log_experiment,
            verbose=verbose,
        )

    # Phase 6: removed _build_legacy_experiment. Native setup only.

    # ---------- session 54: plot_model / evaluate_model wiring ----------

    def _default_plot_kind(self) -> str:
        return "score"

    def _evaluate_plot_set(self) -> list[str]:
        return ["score", "anomaly_map"]

    def _build_plot_registry(self, estimator):
        from pycaret.plots import anomaly as ap

        feature_names = list(self.X.columns)

        return {
            "score": lambda **kw: ap.score_distribution(estimator, self.X, **kw),
            "anomaly_map": lambda **kw: ap.anomaly_map(estimator, self.X, **kw),
            "feature_anomaly": lambda feature_x, feature_y, **kw: ap.feature_anomaly_scatter(
                estimator,
                self.X,
                feature_x,
                feature_y,
                feature_names=feature_names,
                **kw,
            ),
            "score_vs_feature": lambda feature, **kw: ap.score_vs_feature(
                estimator,
                self.X,
                feature,
                feature_names=feature_names,
                **kw,
            ),
        }
