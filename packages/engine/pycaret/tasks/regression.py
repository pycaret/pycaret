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

    # Phase 6: removed _build_legacy_experiment. Native setup only.

    # ---------- session 54: plot_model / evaluate_model wiring ----------

    def _default_plot_kind(self) -> str:
        return "residuals"

    def _evaluate_plot_set(self) -> list[str]:
        return ["residuals", "residuals_distribution", "prediction_error", "feature"]

    def _build_plot_registry(self, estimator):
        from pycaret.plots import feature as fp
        from pycaret.plots import regression as rp

        feature_names = list(self.X_test.columns)

        return {
            "residuals": lambda **kw: rp.residuals(estimator, self.X_test, self.y_test, **kw),
            "residuals_distribution": lambda **kw: rp.residuals_distribution(
                estimator, self.X_test, self.y_test, **kw
            ),
            "prediction_error": lambda **kw: rp.prediction_error(
                estimator, self.X_test, self.y_test, **kw
            ),
            "error": lambda **kw: rp.prediction_error(
                estimator, self.X_test, self.y_test, **kw
            ),
            "learning": lambda **kw: rp.learning_curve(
                estimator, self.X_train, self.y_train, **kw
            ),
            "learning_curve": lambda **kw: rp.learning_curve(
                estimator, self.X_train, self.y_train, **kw
            ),
            "feature": lambda **kw: rp.feature_importance(estimator, feature_names, **kw),
            "permutation": lambda **kw: fp.permutation_importance(
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
