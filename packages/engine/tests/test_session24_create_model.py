"""Session 24 — god-class drain: ``create_model`` (supervised path).

Supervised ``Experiment.create_model`` no longer delegates to
``self._legacy.create_model``. It resolves the estimator from the engine's
model registry, runs cross-validation, refits on the full training set,
and returns a **real sklearn Pipeline** (preprocessor + trained model).
Clustering / anomaly / time-series still delegate to the legacy engine
for now — separate drain sessions.

Unlock: ``CreateResult.pipeline`` is now a real Pipeline for supervised
tasks. ``predict_model`` can use it directly (no bare-estimator branch).
"""

from __future__ import annotations

import pytest


@pytest.mark.slow
def test_create_model_returns_real_sklearn_pipeline_classification():
    """create_model('lr') on a classification exp returns a fitted Pipeline
    whose last step is the trained estimator + preprocessing steps before.
    """
    import pycaret.datasets
    from pycaret.core import CreateResult
    from pycaret.tasks import ClassificationExperiment
    from sklearn.pipeline import Pipeline as SkPipeline

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1).fit(df)
    result: CreateResult = exp.create_model("lr", verbose=False)

    assert result.model_id == "lr"
    assert isinstance(result.pipeline, SkPipeline)
    # Last step should be the trained model under the model_id name.
    last_name, last_step = result.pipeline.steps[-1]
    assert last_name == "lr"
    assert type(last_step).__name__ == "LogisticRegression"
    # Pipeline should carry preprocessing steps before the model.
    assert len(result.pipeline.steps) >= 2


@pytest.mark.slow
def test_create_model_cv_metrics_have_mean_and_std_rows():
    """CV metrics DataFrame: ``Fold 0..N-1``, ``Mean``, ``Std``.

    The row count should be `fold + 2` (defaults to fold=10 → 12 rows).
    """
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=5).fit(df)
    result = exp.create_model("lr", verbose=False)

    assert result.metrics is not None
    assert list(result.metrics.index[-2:]) == ["Mean", "Std"]
    # 5 folds + Mean + Std = 7 rows
    assert len(result.metrics) == 7
    # Classification metrics registry — should include Accuracy + AUC.
    cols = [c.lower() for c in result.metrics.columns]
    assert any("accuracy" in c for c in cols)
    assert any("auc" in c for c in cols)


@pytest.mark.slow
def test_create_model_no_cross_validation_skips_metrics():
    """cross_validation=False → fit-only, metrics DataFrame is None."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1).fit(df)
    result = exp.create_model("lr", cross_validation=False, verbose=False)
    assert result.metrics is None
    # Pipeline is still a fitted Pipeline.
    from sklearn.pipeline import Pipeline as SkPipeline

    assert isinstance(result.pipeline, SkPipeline)


@pytest.mark.slow
def test_create_model_regression_uses_regression_metric_registry():
    """Regression create_model picks up MAE / MSE / RMSE / R² via the reg
    metric registry.
    """
    import pycaret.datasets
    from pycaret.tasks import RegressionExperiment
    from sklearn.pipeline import Pipeline as SkPipeline

    df = pycaret.datasets.get_data("boston", verbose=False)
    exp = RegressionExperiment(target="medv", session_id=42, n_jobs=1, fold=3).fit(df)
    result = exp.create_model("lr", verbose=False)

    assert isinstance(result.pipeline, SkPipeline)
    assert result.metrics is not None
    cols = result.metrics.columns.tolist()
    # Regression metric names (the registry canonicalises these)
    assert any("MAE" in c for c in cols)
    assert any("R2" in c or "R²" in c for c in cols)


@pytest.mark.slow
def test_create_model_unknown_id_raises():
    """Passing a bogus registry ID raises ConfigurationError."""
    import pycaret.datasets
    from pycaret.core.errors import ConfigurationError
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1).fit(df)
    with pytest.raises(ConfigurationError, match="Unknown model id"):
        exp.create_model("zzzz_not_a_real_model_id", verbose=False)


@pytest.mark.slow
def test_create_model_accepts_preconstructed_estimator():
    """A user can pass a bare sklearn estimator; it's fit + wrapped in a Pipeline."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline as SkPipeline

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    user_model = LogisticRegression(max_iter=500, C=2.0)
    result = exp.create_model(user_model, verbose=False)
    assert result.model_id == "LogisticRegression"
    assert isinstance(result.pipeline, SkPipeline)
    # The user's hyperparameter survived into the final fitted step.
    _, fitted = result.pipeline.steps[-1]
    assert fitted.C == 2.0


@pytest.mark.slow
def test_create_model_predict_model_roundtrip_no_bare_branch():
    """The killer test: create_model + predict_model work as a clean
    pipeline-in, pipeline-out chain.

    Since create_model now returns a real Pipeline, predict_model can call
    .predict on it directly — no transitional `preprocessor.transform`.
    We assert that by monkeypatching `self.preprocess_pipeline` to raise:
    if predict_model tries to use the preprocessor, we'll hear about it.
    """
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1).fit(df)
    created = exp.create_model("lr", verbose=False)

    # If predict_model reaches for self.preprocess_pipeline, this blows up
    # — but it shouldn't, because created.pipeline is already a full
    # Pipeline with the preprocessor baked in.
    class _Blow:
        def transform(self, *a, **kw):
            raise AssertionError(
                "predict_model reached for self.preprocess_pipeline even "
                "though the estimator was already a Pipeline. The "
                "transitional bare-estimator branch is misbehaving."
            )

    # Can't replace the property directly; shadow it on the instance
    # via object.__setattr__ since Experiment stores `_legacy` there.
    original = exp._legacy
    try:
        # Shadow the property via a fake legacy proxy that returns our blow-up.
        class _LegacyProxy:
            def __getattr__(self, name):
                if name == "pipeline":
                    return _Blow()
                return getattr(original, name)

        exp._legacy = _LegacyProxy()
        result = exp.predict_model(created.pipeline)
    finally:
        exp._legacy = original

    assert "prediction_label" in result.predictions.columns


@pytest.mark.slow
def test_create_model_does_not_call_legacy_create_model(monkeypatch):
    """The drain: supervised Experiment.create_model must NOT call
    self._legacy.create_model.

    Poison the legacy method + verify create_model still comes back.
    """
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)

    def _poison(*a, **kw):
        raise AssertionError(
            "Session-24 drain regression: Experiment.create_model called "
            "self._legacy.create_model on a supervised task. The native "
            "path must instantiate the estimator + cross_validate directly."
        )

    monkeypatch.setattr(exp._legacy, "create_model", _poison)

    # Should NOT raise — native path runs without legacy.create_model.
    result = exp.create_model("lr", verbose=False)
    assert result.model_id == "lr"
    assert result.pipeline is not None


@pytest.mark.slow
def test_create_model_clustering_still_delegates_to_legacy():
    """Unsupervised clustering's create_model still goes through legacy
    (its drain is a later session). Make sure the fallback path works.
    """
    import pycaret.datasets
    from pycaret.core import CreateResult
    from pycaret.tasks import ClusteringExperiment

    df = pycaret.datasets.get_data("jewellery", verbose=False)
    exp = ClusteringExperiment(session_id=42, n_jobs=1).fit(df)
    result = exp.create_model("kmeans", num_clusters=4, verbose=False)
    assert isinstance(result, CreateResult)
    assert result.model_id == "kmeans"
    assert result.pipeline is not None


def test_create_model_requires_fit():
    """create_model on an unfit Experiment raises NotFittedError."""
    from pycaret.core.errors import NotFittedError
    from pycaret.tasks import ClassificationExperiment

    exp = ClassificationExperiment(target="Purchase", session_id=0)
    with pytest.raises(NotFittedError):
        exp.create_model("lr")
