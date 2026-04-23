"""`ClusteringExperiment` — the PyCaret 4.0 clustering engine.

Unsupervised. Verb surface: `fit`, `create_model`, `assign_model`,
`predict_model`, `plot_model`, `save_model`, `load_model`, `pull`, `models`.

Canonical import:

    from pycaret.tasks import ClusteringExperiment
    # or (backward-compat):
    from pycaret.clustering import ClusteringExperiment

Example
-------

>>> from pycaret.datasets import get_data
>>> from pycaret.tasks import ClusteringExperiment
>>> df = get_data("jewellery")
>>> exp = ClusteringExperiment(session_id=42).fit(df)
>>> km = exp.create_model("kmeans", num_clusters=4).pipeline
>>> labelled = exp.assign_model(km)   # DataFrame with a new "Cluster" column
"""

from __future__ import annotations

from pycaret.core.tasks import TaskType
from pycaret.core.unsupervised import UnsupervisedExperiment
from pycaret.logging.base import BaseLogger


class ClusteringExperiment(UnsupervisedExperiment):
    """PyCaret 4.0 clustering experiment (sklearn-compatible)."""

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
            task=TaskType.CLUSTERING,
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

    def _build_legacy_experiment(self):
        from pycaret.clustering.oop import ClusteringExperiment as _LegacyClusterExp

        return _LegacyClusterExp()
