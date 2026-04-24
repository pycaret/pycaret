"""Session 27 — god-class drain: ``ensemble_model`` / ``blend_models`` /
``stack_models`` / ``calibrate_model`` / ``finalize_model``.

The five remaining supervised model-combination verbs no longer delegate
to ``self._legacy.*``. They wrap sklearn meta-estimators (``BaggingClassifier``
/ ``AdaBoostClassifier`` / ``VotingClassifier`` / ``StackingClassifier`` /
``CalibratedClassifierCV``) and reuse the already-drained
``self.create_model`` to assemble Pipelines + run CV.

This finishes the supervised drain — only the unsupervised /
time-series fallbacks remain, and `pycaret/internal/pycaret_experiment/`
becomes deletable in a follow-up cleanup session.
"""

from __future__ import annotations

import pytest

# ============================================================ ensemble_model


@pytest.mark.slow
def test_ensemble_model_bagging():
    """Bagging wrapper returns a Pipeline whose last step is a BaggingClassifier."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment
    from sklearn.pipeline import Pipeline as SkPipeline

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    base = exp.create_model("dt", verbose=False)
    result = exp.ensemble_model(base.pipeline, method="Bagging", n_estimators=3)

    assert isinstance(result.pipeline, SkPipeline)
    assert result.method == "Bagging"
    name, step = result.pipeline.steps[-1]
    assert "Bagging" in name
    assert type(step).__name__ == "BaggingClassifier"
    # Predict chain works on the wrapped Pipeline.
    preds = exp.predict_model(result.pipeline)
    assert "prediction_label" in preds.predictions.columns


@pytest.mark.slow
def test_ensemble_model_boosting():
    """Boosting → AdaBoostClassifier."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    base = exp.create_model("dt", verbose=False)
    result = exp.ensemble_model(base.pipeline, method="Boosting", n_estimators=3)
    assert "AdaBoost" in result.pipeline.steps[-1][0]
    assert type(result.pipeline.steps[-1][1]).__name__ == "AdaBoostClassifier"


@pytest.mark.slow
def test_ensemble_model_does_not_call_legacy_ensemble_model(monkeypatch):
    """Drain-lock: self._legacy.ensemble_model must NOT be invoked."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    base = exp.create_model("dt", verbose=False)

    def _poison(*a, **kw):
        raise AssertionError("Session-27 drain regression: ensemble_model called legacy.")

    monkeypatch.setattr(exp._legacy, "ensemble_model", _poison)
    result = exp.ensemble_model(base.pipeline, method="Bagging", n_estimators=2)
    assert result.pipeline is not None


# ============================================================ blend_models


@pytest.mark.slow
def test_blend_models_voting_classifier_soft():
    """Soft-voting blend across 2 models that both have predict_proba."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment
    from sklearn.ensemble import VotingClassifier

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    lr = exp.create_model("lr", verbose=False)
    dt = exp.create_model("dt", verbose=False)
    result = exp.blend_models([lr.pipeline, dt.pipeline])

    name, step = result.pipeline.steps[-1]
    assert name == "Voting"
    assert isinstance(step, VotingClassifier)
    # Auto-detect picks soft when all base models support predict_proba.
    assert step.voting == "soft"


@pytest.mark.slow
def test_blend_models_regressor():
    """Regression blend uses VotingRegressor."""
    import pycaret.datasets
    from pycaret.tasks import RegressionExperiment
    from sklearn.ensemble import VotingRegressor

    df = pycaret.datasets.get_data("boston", verbose=False)
    exp = RegressionExperiment(target="medv", session_id=42, n_jobs=1, fold=3).fit(df)
    lr = exp.create_model("lr", verbose=False)
    ridge = exp.create_model("ridge", verbose=False)
    result = exp.blend_models([lr.pipeline, ridge.pipeline])
    assert isinstance(result.pipeline.steps[-1][1], VotingRegressor)


@pytest.mark.slow
def test_blend_models_does_not_call_legacy_blend_models(monkeypatch):
    """Drain-lock for blend_models."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    lr = exp.create_model("lr", verbose=False)
    dt = exp.create_model("dt", verbose=False)

    def _poison(*a, **kw):
        raise AssertionError("Session-27 drain regression: blend_models called legacy.")

    monkeypatch.setattr(exp._legacy, "blend_models", _poison)
    result = exp.blend_models([lr.pipeline, dt.pipeline])
    assert result.pipeline is not None


# ============================================================ stack_models


@pytest.mark.slow
def test_stack_models_classifier_with_default_meta():
    """Default meta-learner is LogisticRegression for classification."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment
    from sklearn.ensemble import StackingClassifier

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    lr = exp.create_model("lr", verbose=False)
    dt = exp.create_model("dt", verbose=False)
    result = exp.stack_models([lr.pipeline, dt.pipeline])
    name, step = result.pipeline.steps[-1]
    assert "LogisticRegression" in name
    assert isinstance(step, StackingClassifier)
    assert type(step.final_estimator).__name__ == "LogisticRegression"


@pytest.mark.slow
def test_stack_models_does_not_call_legacy_stack_models(monkeypatch):
    """Drain-lock for stack_models."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    lr = exp.create_model("lr", verbose=False)
    dt = exp.create_model("dt", verbose=False)

    def _poison(*a, **kw):
        raise AssertionError("Session-27 drain regression: stack_models called legacy.")

    monkeypatch.setattr(exp._legacy, "stack_models", _poison)
    result = exp.stack_models([lr.pipeline, dt.pipeline])
    assert result.pipeline is not None


# ============================================================ calibrate_model


@pytest.mark.slow
def test_calibrate_model_classification():
    """CalibratedClassifierCV wraps the base estimator with sigmoid calibration."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment
    from sklearn.calibration import CalibratedClassifierCV

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    lr = exp.create_model("lr", verbose=False)
    result = exp.calibrate_model(lr.pipeline, method="sigmoid")
    assert isinstance(result.pipeline.steps[-1][1], CalibratedClassifierCV)
    assert result.method == "sigmoid"


@pytest.mark.slow
def test_calibrate_model_rejects_regression():
    """Calibration is undefined for regression — raises ValueError."""
    import pycaret.datasets
    from pycaret.tasks import RegressionExperiment

    df = pycaret.datasets.get_data("boston", verbose=False)
    exp = RegressionExperiment(target="medv", session_id=42, n_jobs=1, fold=3).fit(df)
    lr = exp.create_model("lr", verbose=False)
    with pytest.raises(ValueError, match="only valid for classification"):
        exp.calibrate_model(lr.pipeline)


# ============================================================ finalize_model


@pytest.mark.slow
def test_finalize_model_refits_on_full_data():
    """finalize_model refits the estimator on train+test combined."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment
    from sklearn.pipeline import Pipeline as SkPipeline

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    lr = exp.create_model("lr", verbose=False)
    finalized = exp.finalize_model(lr.pipeline)
    assert isinstance(finalized.pipeline, SkPipeline)
    # Predict on the holdout (which is now part of training data) — still works.
    preds = exp.predict_model(finalized.pipeline)
    assert "prediction_label" in preds.predictions.columns


@pytest.mark.slow
def test_finalize_model_does_not_call_legacy_finalize_model(monkeypatch):
    """Drain-lock for finalize_model."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    lr = exp.create_model("lr", verbose=False)

    def _poison(*a, **kw):
        raise AssertionError("Session-27 drain regression: finalize_model called legacy.")

    monkeypatch.setattr(exp._legacy, "finalize_model", _poison)
    finalized = exp.finalize_model(lr.pipeline)
    assert finalized.pipeline is not None


# ============================================================ misc


def test_combine_verbs_require_fit():
    """All five verbs raise NotFittedError on an unfit experiment."""
    from pycaret.core.errors import NotFittedError
    from pycaret.tasks import ClassificationExperiment

    exp = ClassificationExperiment(target="Purchase", session_id=0)
    with pytest.raises(NotFittedError):
        exp.ensemble_model("lr")
    with pytest.raises(NotFittedError):
        exp.blend_models(["lr", "dt"])
    with pytest.raises(NotFittedError):
        exp.stack_models(["lr", "dt"])
    with pytest.raises(NotFittedError):
        exp.calibrate_model("lr")
    with pytest.raises(NotFittedError):
        exp.finalize_model("lr")
