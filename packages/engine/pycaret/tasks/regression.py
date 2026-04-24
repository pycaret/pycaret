"""`RegressionExperiment` — the PyCaret 4.0 regression engine.

Canonical import:

    from pycaret.tasks import RegressionExperiment
    # or (backward-compat):
    from pycaret.regression import RegressionExperiment

Example
-------

>>> from pycaret.datasets import get_data
>>> from pycaret.tasks import RegressionExperiment
>>> df = get_data("boston")
>>> exp = RegressionExperiment(target="medv", session_id=42).fit(df)
>>> result = exp.compare_models()
>>> best = result.best
>>> preds = exp.predict_model(best).predictions
"""

from __future__ import annotations

from pycaret.core.supervised import SupervisedExperiment
from pycaret.core.tasks import TaskType
from pycaret.logging.base import BaseLogger


class RegressionExperiment(SupervisedExperiment):
    """PyCaret 4.0 regression experiment (sklearn-compatible)."""

    def __init__(
        self,
        *,
        target: str | None = None,
        session_id: int | None = None,
        train_size: float = 0.7,
        fold: int = 10,
        fold_strategy: str | object = "kfold",
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
        super().__init__(
            task=TaskType.REGRESSION,
            target=target,
            session_id=session_id,
            train_size=train_size,
            fold=fold,
            fold_strategy=fold_strategy,
            preprocess=preprocess,
            normalize=normalize,
            transformation=transformation,
            remove_outliers=remove_outliers,
            feature_selection=feature_selection,
            n_jobs=n_jobs,
            use_gpu=use_gpu,
            logger=logger,
            log_experiment=log_experiment,
            verbose=verbose,
        )

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        try:
            tags.estimator_type = "regressor"
        except AttributeError:
            pass
        return tags

    def _build_legacy_experiment(self):
        from pycaret.regression.oop import RegressionExperiment as _LegacyRegExp

        return _LegacyRegExp()
