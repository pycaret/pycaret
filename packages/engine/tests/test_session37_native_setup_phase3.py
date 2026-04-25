"""Session 37 — native setup() phase 3: ``remove_outliers`` + ``feature_selection``.

Phase 1 (s35) shipped impute + ordinal-encode + label-encode + split +
fold + registry. Phase 2 (s36) added ``normalize`` + ``transformation``.
Phase 3 finishes the supervised tabular preprocessing flags:

- ``remove_outliers=True`` → ``IsolationForest(contamination=0.05)`` is
  fit on the transformed training set; the top 5% most anomalous rows
  are dropped from both raw and transformed splits. Test set is
  untouched. Mirrors legacy's ``outliers_threshold=0.05`` default.
- ``feature_selection=True`` → ``SelectFromModel(threshold="median")``
  with an ``ExtraTrees{Classifier|Regressor}`` estimator (100 trees).
  Above-median importance features are kept on the *transformed*
  splits; raw splits keep all columns so user-facing accessors don't
  lose information.

After this session, every preprocessing flag on the supervised
``Experiment`` constructor runs natively. Only ``setup_kwargs`` and
non-supervised tasks still force legacy.
"""

from __future__ import annotations

import pytest


@pytest.mark.slow
def test_remove_outliers_drops_training_rows(monkeypatch):
    """remove_outliers=True drops ~5% of training rows; legacy.setup
    is NOT called.
    """
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(
        target="Purchase",
        session_id=42,
        n_jobs=1,
        fold=3,
        remove_outliers=True,
    )

    def _poison(*a, **kw):
        raise AssertionError("Session-37 drain regression: remove_outliers=True hit legacy.setup.")

    exp._legacy = exp._build_legacy_experiment()
    monkeypatch.setattr(exp._legacy, "setup", _poison)
    monkeypatch.setattr(exp, "_build_legacy_experiment", lambda: exp._legacy)

    exp.fit(df)
    assert exp._native_setup_used is True

    # ~5% of 749 ≈ 37; tolerate +/- a couple due to ties.
    dropped = exp._fit_state["outliers_dropped"]
    assert 30 <= dropped <= 50, f"unexpected drop count: {dropped}"
    assert exp.X_train.shape[0] == 749 - dropped
    assert exp._fit_state["X_train_transformed"].shape[0] == 749 - dropped
    # Test set is untouched.
    assert exp.X_test.shape[0] == 1070 - 749


@pytest.mark.slow
def test_feature_selection_trims_columns(monkeypatch):
    """feature_selection=True keeps a subset of transformed columns."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment
    from sklearn.feature_selection import SelectFromModel

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(
        target="Purchase",
        session_id=42,
        n_jobs=1,
        fold=3,
        feature_selection=True,
    )

    def _poison(*a, **kw):
        raise AssertionError(
            "Session-37 drain regression: feature_selection=True hit legacy.setup."
        )

    exp._legacy = exp._build_legacy_experiment()
    monkeypatch.setattr(exp._legacy, "setup", _poison)
    monkeypatch.setattr(exp, "_build_legacy_experiment", lambda: exp._legacy)

    exp.fit(df)
    assert exp._native_setup_used is True

    selector = exp._fit_state["feature_selector"]
    assert isinstance(selector, SelectFromModel)
    selected = exp._fit_state["selected_features"]
    assert len(selected) > 0
    assert len(selected) < 18, "feature selector should drop something with median threshold"
    # Transformed splits are trimmed; raw splits keep every column.
    assert exp._fit_state["X_train_transformed"].shape[1] == len(selected)
    assert exp.X_train.shape[1] == 18


@pytest.mark.slow
def test_remove_outliers_and_feature_selection_combined():
    """Both flags compose: outliers dropped first, then features selected
    on the (now-smaller) train.
    """
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(
        target="Purchase",
        session_id=42,
        n_jobs=1,
        fold=3,
        remove_outliers=True,
        feature_selection=True,
    ).fit(df)
    assert exp._native_setup_used is True
    assert exp._fit_state["outliers_dropped"] > 0
    assert exp._fit_state["selected_features"] is not None
    Xt = exp._fit_state["X_train_transformed"]
    assert Xt.shape[0] < 749
    assert Xt.shape[1] < 18


@pytest.mark.slow
def test_phase3_end_to_end_supervised():
    """create + tune + predict chain works under all phase-3 flags."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(
        target="Purchase",
        session_id=42,
        n_jobs=1,
        fold=3,
        normalize=True,
        transformation=True,
        remove_outliers=True,
        feature_selection=True,
    ).fit(df)
    assert exp._native_setup_used is True

    created = exp.create_model("lr", verbose=False)
    assert created.metrics is not None
    tuned = exp.tune_model(created.pipeline, n_iter=3, verbose=False)
    assert tuned.pipeline is not None
    preds = exp.predict_model(tuned.pipeline)
    assert "prediction_label" in preds.predictions.columns


@pytest.mark.slow
def test_regression_remove_outliers_native():
    """Regression with remove_outliers=True works natively."""
    import pycaret.datasets
    from pycaret.tasks import RegressionExperiment

    df = pycaret.datasets.get_data("boston", verbose=False)
    exp = RegressionExperiment(
        target="medv", session_id=42, n_jobs=1, fold=3, remove_outliers=True
    ).fit(df)
    assert exp._native_setup_used is True
    dropped = exp._fit_state["outliers_dropped"]
    # 5% of train_size * 506 ≈ 18 +/- a few.
    assert dropped > 0


@pytest.mark.slow
def test_regression_feature_selection_uses_extra_trees_regressor():
    """The selector for regression wraps an ExtraTreesRegressor."""
    import pycaret.datasets
    from pycaret.tasks import RegressionExperiment
    from sklearn.ensemble import ExtraTreesRegressor

    df = pycaret.datasets.get_data("boston", verbose=False)
    exp = RegressionExperiment(
        target="medv", session_id=42, n_jobs=1, fold=3, feature_selection=True
    ).fit(df)
    assert exp._native_setup_used is True
    selector = exp._fit_state["feature_selector"]
    assert isinstance(selector.estimator, ExtraTreesRegressor)


@pytest.mark.slow
def test_outlier_removal_keeps_y_train_aligned():
    """y_train and y_train_transformed match X_train's row count after drop."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(
        target="Purchase",
        session_id=42,
        n_jobs=1,
        fold=3,
        remove_outliers=True,
    ).fit(df)
    n = exp.X_train.shape[0]
    assert exp.y_train.shape[0] == n
    assert exp._fit_state["X_train_transformed"].shape[0] == n
    assert exp._fit_state["y_train_transformed"].shape[0] == n


def test_can_use_native_setup_predicate_phase3():
    """All four preprocessing flags now route to native."""
    from pycaret.tasks import ClassificationExperiment, RegressionExperiment

    for flag in ("normalize", "transformation", "remove_outliers", "feature_selection"):
        exp = ClassificationExperiment(target="t", session_id=0, **{flag: True})
        assert exp._can_use_native_setup({}) is True, f"{flag} should be native"
        exp = RegressionExperiment(target="t", session_id=0, **{flag: True})
        assert exp._can_use_native_setup({}) is True, f"{flag} (reg) should be native"

    # All four together — still native.
    exp = ClassificationExperiment(
        target="t",
        session_id=0,
        normalize=True,
        transformation=True,
        remove_outliers=True,
        feature_selection=True,
    )
    assert exp._can_use_native_setup({}) is True

    # setup_kwargs still forces legacy.
    exp = ClassificationExperiment(target="t", session_id=0)
    assert exp._can_use_native_setup({"foo": 1}) is False
