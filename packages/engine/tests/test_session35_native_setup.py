"""Session 35 — drain ``self._legacy.setup()`` for the simple supervised case.

The biggest remaining drain target. When the user passes a basic
classification or regression experiment with no complex preprocessing
flags (``normalize`` / ``transformation`` / ``remove_outliers`` /
``feature_selection``) and no extra ``setup_kwargs``, ``fit()`` now
skips ``self._legacy.setup()`` entirely and builds the experiment state
natively:

- Train/test split via sklearn's ``train_test_split`` (stratified for clf).
- Numeric imputation by mean + categorical imputation by mode + ordinal
  encoding via ``ColumnTransformer``.
- ``LabelEncoder`` for the classification target.
- ``StratifiedKFold`` (clf) / ``KFold`` (reg) as the fold generator.
- Model registry built via the per-task container helper using a thin
  ``_ModelRegistryContext`` proxy (no legacy needed).

Legacy ``setup()`` is still used for:
- Time-series + clustering + anomaly tasks (Phase-2 native setup work).
- Any complex preprocessing flag (normalize / transformation /
  remove_outliers / feature_selection).
- Any user-supplied ``setup_kwargs``.

The drain-lock test poisons ``self._legacy.setup`` to raise; the simple
flow still succeeds.
"""

from __future__ import annotations

import pytest


@pytest.mark.slow
def test_native_setup_used_for_basic_classification(monkeypatch):
    """A basic classification fit() does NOT call self._legacy.setup()."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3)

    def _poison(*a, **kw):
        raise AssertionError(
            "Session-35 drain regression: fit() called self._legacy.setup() "
            "for a simple classification experiment. The native path must "
            "build _fit_state without legacy.setup()."
        )

    # Build the legacy holder first (so the constructor runs), then poison.
    exp._legacy = exp._build_legacy_experiment()
    monkeypatch.setattr(exp._legacy, "setup", _poison)
    # We override _build_legacy_experiment to just return the existing
    # poisoned legacy so fit() doesn't replace it.
    monkeypatch.setattr(exp, "_build_legacy_experiment", lambda: exp._legacy)

    exp.fit(df)
    assert exp._native_setup_used is True


@pytest.mark.slow
def test_native_setup_used_for_basic_regression(monkeypatch):
    """A basic regression fit() does NOT call self._legacy.setup()."""
    import pycaret.datasets
    from pycaret.tasks import RegressionExperiment

    df = pycaret.datasets.get_data("boston", verbose=False)
    exp = RegressionExperiment(target="medv", session_id=42, n_jobs=1, fold=3)

    def _poison(*a, **kw):
        raise AssertionError(
            "Session-35 drain regression: fit() called self._legacy.setup() "
            "for a simple regression experiment."
        )

    exp._legacy = exp._build_legacy_experiment()
    monkeypatch.setattr(exp._legacy, "setup", _poison)
    monkeypatch.setattr(exp, "_build_legacy_experiment", lambda: exp._legacy)

    exp.fit(df)
    assert exp._native_setup_used is True


@pytest.mark.slow
def test_complex_preprocessing_falls_back_to_legacy():
    """``setup_kwargs`` (caller-supplied) still forces legacy.

    Originally session 35 used ``normalize=True`` here; sessions 36 + 37
    made every constructor preprocessing flag native. The remaining
    "still legacy" route is caller-supplied ``setup_kwargs`` — those keys
    pass through to the legacy ``setup()``'s 100+ knobs and aren't
    handled by the native chain yet.
    """
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(
        target="Purchase",
        session_id=42,
        n_jobs=1,
        fold=3,
    ).fit(df, html=False)  # extra setup_kwarg → forces legacy
    # Native NOT used when extra setup_kwargs are passed.
    assert exp._native_setup_used is False


@pytest.mark.slow
def test_unsupervised_uses_legacy_setup():
    """Clustering tasks fall back to legacy setup (Phase-2 work)."""
    import pycaret.datasets
    from pycaret.tasks import ClusteringExperiment

    df = pycaret.datasets.get_data("jewellery", verbose=False)
    exp = ClusteringExperiment(session_id=42, n_jobs=1).fit(df)
    assert exp._native_setup_used is False


@pytest.mark.slow
def test_native_setup_full_chain_works():
    """End-to-end via native setup: create_model → tune_model → predict_model."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment
    from sklearn.pipeline import Pipeline as SkPipeline

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    assert exp._native_setup_used

    created = exp.create_model("lr", verbose=False)
    assert isinstance(created.pipeline, SkPipeline)
    assert created.metrics is not None

    tuned = exp.tune_model(created.pipeline, n_iter=3, verbose=False)
    assert tuned.pipeline is not None

    preds = exp.predict_model(tuned.pipeline)
    assert "prediction_label" in preds.predictions.columns
    assert preds.metrics is not None


@pytest.mark.slow
def test_native_setup_label_encoded_y():
    """Classification target is integer-encoded in y_train_transformed."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    assert exp._native_setup_used

    yt = exp._fit_state["y_train_transformed"]
    # Original target is 'CH' / 'MM' (categorical strings).
    assert exp.y_train.dtype.name in ("category", "object", "string")
    # Transformed is integer 0/1.
    assert str(yt.dtype).startswith("int")
    assert sorted(yt.unique().tolist()) == [0, 1]


@pytest.mark.slow
def test_native_setup_regression_y_unchanged():
    """Regression target is NOT label-encoded (no LabelEncoder applied)."""
    import pycaret.datasets
    from pycaret.tasks import RegressionExperiment

    df = pycaret.datasets.get_data("boston", verbose=False)
    exp = RegressionExperiment(target="medv", session_id=42, n_jobs=1, fold=3).fit(df)
    assert exp._native_setup_used

    yt = exp._fit_state["y_train_transformed"]
    # Regression: y_train_transformed is the raw y_train.
    import pandas as pd

    pd.testing.assert_series_equal(yt, exp.y_train, check_names=False)


@pytest.mark.slow
def test_model_registry_proxy_populates_lr():
    """The _ModelRegistryContext proxy must yield an `lr` entry."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    assert exp._native_setup_used
    assert "lr" in exp._fit_state["model_registry"]
    # And the registry instance must be the expected class.
    lr_container = exp._fit_state["model_registry"]["lr"]
    assert lr_container.class_def.__name__ == "LogisticRegression"


@pytest.mark.slow
def test_native_setup_models_internal_view():
    """models(internal=True) builds a richer DataFrame from the snapshot
    even when legacy.setup hasn't run.
    """
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    assert exp._native_setup_used
    df_models = exp.models(internal=True)
    # internal=True surface includes Special / Equality / Args.
    assert {"Special", "Equality", "Args"}.issubset(set(df_models.columns))
    assert "lr" in df_models.index


def test_can_use_native_setup_predicate():
    """The predicate after sessions 35 + 36 + 37.

    Every supervised constructor preprocessing flag (normalize /
    transformation / remove_outliers / feature_selection) is now native
    — see test_session36_native_setup_phase2 + test_session37_native_setup_phase3
    for those drains. Here we lock the still-legacy paths: caller-supplied
    ``setup_kwargs`` and unsupervised tasks.
    """
    from pycaret.tasks import (
        AnomalyExperiment,
        ClassificationExperiment,
        ClusteringExperiment,
    )

    # Simple supervised — yes.
    assert ClassificationExperiment(target="t", session_id=0)._can_use_native_setup({}) is True
    # Extra setup kwargs — still forces legacy (we don't know what they do).
    assert (
        ClassificationExperiment(target="t", session_id=0)._can_use_native_setup({"foo": 1})
        is False
    )
    # Unsupervised — still legacy (phase-4 work).
    assert ClusteringExperiment(session_id=0)._can_use_native_setup({}) is False
    assert AnomalyExperiment(session_id=0)._can_use_native_setup({}) is False
