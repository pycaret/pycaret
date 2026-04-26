"""`ClassificationExperiment` — the PyCaret 4.0 classification engine.

Canonical import:

    from pycaret.tasks import ClassificationExperiment
    # or (backward-compat):
    from pycaret.classification import ClassificationExperiment

Example
-------

>>> from pycaret.datasets import get_data
>>> from pycaret.tasks import ClassificationExperiment
>>> df = get_data("juice")
>>> exp = ClassificationExperiment(target="Purchase", session_id=42).fit(df)
>>> result = exp.compare_models()          # -> CompareResult
>>> best = result.best
>>> tuned = exp.tune_model(best).pipeline  # -> Pipeline
>>> preds = exp.predict_model(tuned).predictions
>>> exp.save_model(tuned, "my_model")
"""

from __future__ import annotations

from pycaret.core.supervised import SupervisedExperiment
from pycaret.core.tasks import TaskType
from pycaret.logging.base import BaseLogger


class ClassificationExperiment(SupervisedExperiment):
    """PyCaret 4.0 classification experiment (sklearn-compatible)."""

    def __init__(
        self,
        *,
        target: str | None = None,
        session_id: int | None = None,
        train_size: float = 0.7,
        fold: int = 10,
        fold_strategy: str | object = "stratifiedkfold",
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
            task=TaskType.CLASSIFICATION,
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
            tags.estimator_type = "classifier"
        except AttributeError:
            pass
        return tags

    # Phase 6: removed _build_legacy_experiment. Native setup only.
