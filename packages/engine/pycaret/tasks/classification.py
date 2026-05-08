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

    # ---------- session 54: plot_model / evaluate_model wiring ----------

    def _default_plot_kind(self) -> str:
        return "auc"

    def _evaluate_plot_set(self) -> list[str]:
        return ["auc", "pr", "confusion_matrix", "calibration", "class_distribution"]

    def _build_plot_registry(self, estimator):
        from pycaret.plots import classification as cp
        from pycaret.plots import feature as fp

        feature_names = list(self.X_test.columns)

        return {
            "auc": lambda **kw: cp.roc_curve(estimator, self.X_test, self.y_test, **kw),
            "roc": lambda **kw: cp.roc_curve(estimator, self.X_test, self.y_test, **kw),
            "pr": lambda **kw: cp.pr_curve(estimator, self.X_test, self.y_test, **kw),
            "confusion_matrix": lambda **kw: cp.confusion_matrix(
                estimator, self.X_test, self.y_test, **kw
            ),
            "calibration": lambda **kw: cp.calibration_curve(
                estimator, self.X_test, self.y_test, **kw
            ),
            "threshold": lambda **kw: cp.threshold_curve(
                estimator, self.X_test, self.y_test, **kw
            ),
            "lift": lambda **kw: cp.lift_curve(estimator, self.X_test, self.y_test, **kw),
            "gain": lambda **kw: cp.gain_curve(estimator, self.X_test, self.y_test, **kw),
            "class_distribution": lambda **kw: cp.class_distribution(self.y, **kw),
            "permutation": lambda **kw: fp.permutation_importance(
                estimator, self.X_test, self.y_test, feature_names=feature_names, **kw
            ),
            "feature": lambda **kw: fp.permutation_importance(
                estimator, self.X_test, self.y_test, feature_names=feature_names, **kw
            ),
            "pdp": lambda feature, **kw: fp.partial_dependence(
                estimator, self.X_test, feature, feature_names=feature_names, **kw
            ),
            "ice": lambda feature, **kw: fp.ice_curve(
                estimator, self.X_test, feature, feature_names=feature_names, **kw
            ),
            "shap_summary": lambda **kw: fp.shap_summary(
                estimator, self.X_test, feature_names=feature_names, **kw
            ),
            "summary": lambda **kw: fp.shap_summary(
                estimator, self.X_test, feature_names=feature_names, **kw
            ),
            "shap_beeswarm": lambda **kw: fp.shap_beeswarm(
                estimator, self.X_test, feature_names=feature_names, **kw
            ),
        }
