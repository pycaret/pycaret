"""Session 23 — god-class drain: ``predict_model``.

``Experiment.predict_model`` no longer delegates to
``self._legacy.predict_model``. It now calls ``estimator.predict`` and
(for classifiers) ``estimator.predict_proba`` directly, on the assumption
that the estimator is a fitted sklearn Pipeline with preprocessing baked
in — which is what ``create_model`` / ``compare_models`` / ``tune_model``
return in 4.0.

These tests lock the contract + the drain. They use small sklearn-native
fixtures (no full engine ``setup()``) where possible to run in
milliseconds, and fall back to the slow E2E marker only when holdout-set
semantics are tested.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------- helpers


def _tiny_clf_pipeline(n_samples: int = 120, n_classes: int = 2, seed: int = 0):
    """Fit a StandardScaler+LogReg pipeline on toy data. Returns (pipeline, X_df, y_series)."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=seed,
    )
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    y_ser = pd.Series(y, name="target")
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500))])
    pipe.fit(X_df, y_ser)
    return pipe, X_df, y_ser


def _tiny_reg_pipeline(n_samples: int = 120, seed: int = 0):
    X, y = make_regression(n_samples=n_samples, n_features=4, noise=5.0, random_state=seed)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    y_ser = pd.Series(y, name="target")
    pipe = Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())])
    pipe.fit(X_df, y_ser)
    return pipe, X_df, y_ser


# =================================================================== contract


def test_predict_model_rejects_object_without_predict():
    """predict_model rejects anything that isn't a fitted estimator with
    a `.predict` method. (A bare estimator is *accepted* transitionally —
    see `test_predict_model_accepts_bare_estimator_transitionally` — but
    a random dict / string is not.)
    """
    from pycaret.tasks import ClassificationExperiment

    exp = ClassificationExperiment(target="y", session_id=0)
    exp._fitted = True  # pretend fitted so we bypass NotFittedError
    with pytest.raises(TypeError, match="`.predict` method"):
        exp.predict_model({"not": "an_estimator"})


def test_predict_model_requires_fit():
    """predict_model on an unfit Experiment raises NotFittedError."""
    from pycaret.core.errors import NotFittedError
    from pycaret.tasks import ClassificationExperiment

    exp = ClassificationExperiment(target="target", session_id=0)
    pipe, X, _ = _tiny_clf_pipeline()
    with pytest.raises(NotFittedError):
        exp.predict_model(pipe, data=X)


# ============================================================== classification


@pytest.mark.slow
def test_predict_model_classification_binary_adds_label_and_score():
    """Binary classification: prediction_label + single prediction_score column."""
    import pycaret.datasets
    from pycaret.core import CreateResult, PredictResult
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1).fit(df)
    created: CreateResult = exp.create_model("lr", verbose=False)

    result: PredictResult = exp.predict_model(created.pipeline)
    assert isinstance(result, PredictResult)
    assert "prediction_label" in result.predictions.columns
    assert "prediction_score" in result.predictions.columns
    # Binary → no per-class score columns when raw_score=False
    per_class = [c for c in result.predictions.columns if c.startswith("prediction_score_")]
    assert per_class == []
    # Metrics computed on the holdout because y is known.
    assert result.metrics is not None
    assert "Model" in result.metrics.columns
    assert len(result.metrics) == 1


@pytest.mark.slow
def test_predict_model_classification_raw_score_emits_per_class_columns():
    """raw_score=True gives per-class probability columns instead of one."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1).fit(df)
    created = exp.create_model("lr", verbose=False)

    result = exp.predict_model(created.pipeline, raw_score=True)
    per_class = [c for c in result.predictions.columns if c.startswith("prediction_score_")]
    assert len(per_class) == 2  # binary → 2 class columns


# =================================================================== regression


@pytest.mark.slow
def test_predict_model_regression_has_no_score_column():
    """Regression: prediction_label only, no prediction_score."""
    import pycaret.datasets
    from pycaret.tasks import RegressionExperiment

    df = pycaret.datasets.get_data("boston", verbose=False)
    exp = RegressionExperiment(target="medv", session_id=42, n_jobs=1).fit(df)
    created = exp.create_model("lr", verbose=False)

    result = exp.predict_model(created.pipeline)
    assert "prediction_label" in result.predictions.columns
    assert "prediction_score" not in result.predictions.columns
    # Regression metrics DF
    assert result.metrics is not None
    # MAE is a reliable, always-present regression metric
    assert any("MAE" in c or "MSE" in c for c in result.metrics.columns)


# ====================================================================== drain-lock


@pytest.mark.slow
def test_predict_model_does_not_call_legacy_predict_model(monkeypatch):
    """The drain: ``self._legacy.predict_model`` must NOT be invoked.

    We monkeypatch the legacy bound method to raise, then call
    predict_model + verify it comes back successfully. If any future
    refactor accidentally re-adds a delegation, this test catches it on
    the matrix.
    """
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1).fit(df)
    created = exp.create_model("lr", verbose=False)

    def _poison(*a, **kw):
        raise AssertionError(
            "Session-23 drain regression: Experiment.predict_model called "
            "self._legacy.predict_model. The native path must use "
            "estimator.predict directly."
        )

    monkeypatch.setattr(exp._legacy, "predict_model", _poison)

    # This should NOT raise — the native path ignores self._legacy.predict_model.
    result = exp.predict_model(created.pipeline)
    assert "prediction_label" in result.predictions.columns


# ============================================================ event stream


@pytest.mark.slow
def test_predict_model_emits_model_predicted_event():
    """The event stream records the predict with row count + duration."""
    import pycaret.datasets
    from pycaret.logging import EventKind, MemoryLogger
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    log = MemoryLogger()
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, logger=log).fit(df)
    created = exp.create_model("lr", verbose=False)
    result = exp.predict_model(created.pipeline)

    preds = [e for e in log.events if e.kind == EventKind.MODEL_PREDICTED]
    assert len(preds) >= 1
    assert preds[-1].payload["n_rows"] == len(result.predictions)
    assert preds[-1].duration_ms is not None
    assert preds[-1].duration_ms >= 0.0


# =================================================================== fast path
# These tests don't need a full engine fit — they fabricate a fitted pipeline
# + a minimal fitted Experiment (fit=True sentinel) to exercise the raw
# predict dispatch. Runs in ms.


def test_predict_model_on_unseen_data_no_target(monkeypatch):
    """When `data` has no target column, predict returns only predictions
    (no metrics), even on a supervised Experiment.
    """
    from pycaret.core.results import PredictResult
    from pycaret.tasks import ClassificationExperiment

    exp = ClassificationExperiment(target="target", session_id=0)
    # Short-circuit fit() so we don't build the legacy. The drained
    # predict_model reads self.X_test only when data is None; here we
    # pass data explicitly, so we're entirely on the native path.
    exp._fitted = True
    exp.logger = None  # predict_model null-checks before logging

    pipe, X, _ = _tiny_clf_pipeline()
    new_data = X.sample(20, random_state=1)

    # Avoid depending on self.logger — inline a minimal sink.
    class _Sink:
        def __init__(self):
            self.events = []

        def log(self, *a, **kw):
            self.events.append((a, kw))

    exp.logger = _Sink()
    result = exp.predict_model(pipe, data=new_data)
    assert isinstance(result, PredictResult)
    assert "prediction_label" in result.predictions.columns
    assert result.metrics is None  # y was absent
    assert exp.logger.events, "MODEL_PREDICTED event should have been emitted"


def test_predict_model_on_unseen_data_with_target_computes_metrics():
    """When `data` includes the target column, metrics are computed."""
    from pycaret.tasks import ClassificationExperiment

    exp = ClassificationExperiment(target="target", session_id=0)
    exp._fitted = True

    class _Sink:
        def __init__(self):
            self.events = []

        def log(self, *a, **kw):
            self.events.append((a, kw))

    exp.logger = _Sink()

    pipe, X, y = _tiny_clf_pipeline()
    holdout = X.copy()
    holdout["target"] = y.values

    result = exp.predict_model(pipe, data=holdout)
    assert result.metrics is not None
    assert len(result.metrics) == 1
    # Binary classification → Accuracy metric must be present
    assert any(c.lower() == "accuracy" for c in result.metrics.columns)


def test_predict_model_regression_on_unseen_data_has_only_label():
    from pycaret.tasks import RegressionExperiment

    exp = RegressionExperiment(target="target", session_id=0)
    exp._fitted = True

    class _Sink:
        def __init__(self):
            self.events = []

        def log(self, *a, **kw):
            self.events.append((a, kw))

    exp.logger = _Sink()

    pipe, X, y = _tiny_reg_pipeline()
    holdout = X.copy()
    holdout["target"] = y.values

    result = exp.predict_model(pipe, data=holdout)
    assert "prediction_label" in result.predictions.columns
    assert "prediction_score" not in result.predictions.columns
    # Sanity: predictions have the same length as X
    assert len(result.predictions) == len(X)
    # Metrics are one row with some regression columns
    assert result.metrics is not None
    assert any("MAE" in c or "MSE" in c for c in result.metrics.columns)


def test_predict_model_classification_multiclass_score_is_winning_prob():
    """Multi-class, raw_score=False → prediction_score is max-probability."""
    from pycaret.tasks import ClassificationExperiment

    exp = ClassificationExperiment(target="target", session_id=0)
    exp._fitted = True

    class _Sink:
        def log(self, *a, **kw):
            pass

    exp.logger = _Sink()

    pipe, X, _ = _tiny_clf_pipeline(n_samples=120, n_classes=3)
    result = exp.predict_model(pipe, data=X)
    scores = result.predictions["prediction_score"].to_numpy()
    # Each row's max probability should be >= 1/3 (non-trivially peaked)
    assert np.all(scores >= 1.0 / 3 - 1e-9)
    assert np.all(scores <= 1.0 + 1e-9)


def test_predict_model_accepts_raw_score_for_multiclass():
    """raw_score=True on multi-class emits one column per class."""
    from pycaret.tasks import ClassificationExperiment

    exp = ClassificationExperiment(target="target", session_id=0)
    exp._fitted = True

    class _Sink:
        def log(self, *a, **kw):
            pass

    exp.logger = _Sink()

    pipe, X, _ = _tiny_clf_pipeline(n_samples=120, n_classes=3)
    result = exp.predict_model(pipe, data=X, raw_score=True)
    per_class = [c for c in result.predictions.columns if c.startswith("prediction_score_")]
    assert len(per_class) == 3
    # Each row's per-class probabilities should sum to ~1 (the default
    # round=4 introduces small float error; 1e-3 is the right tolerance).
    sums = result.predictions[per_class].sum(axis=1).to_numpy()
    assert np.allclose(sums, 1.0, atol=1e-3)
