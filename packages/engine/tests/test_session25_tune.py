"""Session 25 — god-class drain: ``tune_model`` (supervised path).

Supervised ``Experiment.tune_model`` no longer delegates to
``self._legacy.tune_model``. It runs ``sklearn.model_selection
.RandomizedSearchCV`` over the base estimator on the experiment's
transformed training data + returns a fitted Pipeline + the search
object + a CV metrics DataFrame with identical schema to
``CreateResult.metrics``.
"""

from __future__ import annotations

import pytest


@pytest.mark.slow
def test_tune_model_returns_pipeline_with_tuned_estimator():
    """tune_model on a create_model pipeline returns a fresh Pipeline whose
    last step is the best estimator from the search.
    """
    import pycaret.datasets
    from pycaret.core import TuneResult
    from pycaret.tasks import ClassificationExperiment
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.pipeline import Pipeline as SkPipeline

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    created = exp.create_model("lr", verbose=False)

    tuned: TuneResult = exp.tune_model(created.pipeline, n_iter=5, verbose=False)
    assert isinstance(tuned.pipeline, SkPipeline)
    # Tuned pipeline's last step is still named "lr" and is a LogisticRegression.
    name, step = tuned.pipeline.steps[-1]
    assert name == "lr"
    assert type(step).__name__ == "LogisticRegression"
    # best_params non-empty (the search space for LR is {"C": [...], "class_weight": [...]}).
    assert isinstance(tuned.best_params, dict)
    assert len(tuned.best_params) > 0
    assert isinstance(tuned.search, RandomizedSearchCV)


@pytest.mark.slow
def test_tune_model_cv_results_and_metrics_dataframes():
    """The cv_results + metrics DataFrames both come back populated."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    created = exp.create_model("lr", verbose=False)
    tuned = exp.tune_model(created.pipeline, n_iter=4, verbose=False)

    assert tuned.cv_results is not None
    # RandomizedSearchCV with n_iter=4 → 4 rows.
    assert len(tuned.cv_results) == 4
    # Standard sklearn cv_results_ columns include "mean_test_score".
    assert "mean_test_score" in tuned.cv_results.columns

    # metrics is the same shape as CreateResult.metrics.
    assert tuned.metrics is not None
    assert list(tuned.metrics.index[-2:]) == ["Mean", "Std"]
    cols = [c.lower() for c in tuned.metrics.columns]
    assert any("accuracy" in c for c in cols)


@pytest.mark.slow
def test_tune_model_custom_grid_overrides_registry():
    """custom_grid= takes precedence over the registry's default search space."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    created = exp.create_model("lr", verbose=False)
    # Pin C to a tiny grid — tuned.best_params['C'] must come from this set.
    grid = {"C": [0.1, 1.0, 10.0]}
    tuned = exp.tune_model(created.pipeline, custom_grid=grid, n_iter=3, verbose=False)

    assert tuned.best_params["C"] in grid["C"]


@pytest.mark.slow
def test_tune_model_optimize_mapping_uses_sklearn_scorer():
    """`optimize="AUC"` maps to sklearn's `"roc_auc"` scorer."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    created = exp.create_model("lr", verbose=False)
    tuned = exp.tune_model(created.pipeline, n_iter=3, optimize="AUC", verbose=False)
    # The internal SearchCV should've been configured with roc_auc.
    assert tuned.search is not None
    assert tuned.search.scoring == "roc_auc"


@pytest.mark.slow
def test_tune_model_regression_default_optimize_is_r2():
    """Regression tune_model defaults to r2 when `optimize=` is not given."""
    import pycaret.datasets
    from pycaret.tasks import RegressionExperiment

    df = pycaret.datasets.get_data("boston", verbose=False)
    exp = RegressionExperiment(target="medv", session_id=42, n_jobs=1, fold=3).fit(df)
    created = exp.create_model("lasso", verbose=False)
    tuned = exp.tune_model(created.pipeline, n_iter=3, verbose=False)
    assert tuned.search is not None
    assert tuned.search.scoring == "r2"


@pytest.mark.slow
def test_tune_model_does_not_call_legacy_tune_model(monkeypatch):
    """Drain-lock: self._legacy.tune_model must NOT be invoked."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    created = exp.create_model("lr", verbose=False)

    def _poison(*a, **kw):
        raise AssertionError(
            "Session-25 drain regression: Experiment.tune_model called "
            "self._legacy.tune_model on a supervised task. The native "
            "path must use RandomizedSearchCV directly."
        )

    monkeypatch.setattr(exp._legacy, "tune_model", _poison)

    tuned = exp.tune_model(created.pipeline, n_iter=3, verbose=False)
    assert tuned.pipeline is not None


@pytest.mark.slow
def test_tune_model_predict_chain_from_tuned_pipeline():
    """End-to-end: create_model → tune_model → predict_model all on a
    sklearn Pipeline, no transitional branches.
    """
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    created = exp.create_model("lr", verbose=False)
    tuned = exp.tune_model(created.pipeline, n_iter=3, verbose=False)

    preds = exp.predict_model(tuned.pipeline)
    assert "prediction_label" in preds.predictions.columns
    assert "prediction_score" in preds.predictions.columns
    assert preds.metrics is not None


@pytest.mark.slow
def test_tune_model_accepts_registry_id_directly():
    """tune_model('lr', ...) works without a prior create_model call."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment
    from sklearn.pipeline import Pipeline as SkPipeline

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    tuned = exp.tune_model("lr", n_iter=3, verbose=False)
    assert isinstance(tuned.pipeline, SkPipeline)
    assert tuned.pipeline.steps[-1][0] == "lr"


def test_tune_model_requires_fit():
    from pycaret.core.errors import NotFittedError
    from pycaret.tasks import ClassificationExperiment

    exp = ClassificationExperiment(target="Purchase", session_id=0)
    with pytest.raises(NotFittedError):
        exp.tune_model("lr")
