"""Session 36 — native setup() phase 2: ``normalize`` + ``transformation``.

Phase 1 (session 35) shipped native setup for the simple supervised
case. Phase 2 extends the native preprocessing chain with two more
common options:

- ``normalize=True`` → ``StandardScaler`` on numeric columns (z-score).
- ``transformation=True`` → ``PowerTransformer(yeo-johnson)`` on numeric
  columns (handles negatives; Yeo-Johnson matches the legacy default
  ``transformation_method='yeo-johnson'``).

Both can be combined. The preprocessing order is **transform → scale**
so the scaler sees the post-power values and produces ~zero mean / unit
std.

The two heavier options (``remove_outliers``, ``feature_selection``)
still fall through to legacy — Phase 3 work.
"""

from __future__ import annotations

import pytest


@pytest.mark.slow
def test_native_setup_used_with_normalize(monkeypatch):
    """normalize=True stays on the native path (was forcing legacy in phase 1)."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(
        target="Purchase",
        session_id=42,
        n_jobs=1,
        fold=3,
        normalize=True,
    )

    def _poison(*a, **kw):
        raise AssertionError(
            "Session-36 drain regression: normalize=True triggered "
            "legacy.setup() instead of using native phase-2 path."
        )

    exp._legacy = exp._build_legacy_experiment()
    monkeypatch.setattr(exp._legacy, "setup", _poison)
    monkeypatch.setattr(exp, "_build_legacy_experiment", lambda: exp._legacy)

    exp.fit(df)
    assert exp._native_setup_used is True


@pytest.mark.slow
def test_native_setup_used_with_transformation(monkeypatch):
    """transformation=True stays on the native path."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(
        target="Purchase",
        session_id=42,
        n_jobs=1,
        fold=3,
        transformation=True,
    )

    def _poison(*a, **kw):
        raise AssertionError("Session-36 drain regression: transformation=True hit legacy.")

    exp._legacy = exp._build_legacy_experiment()
    monkeypatch.setattr(exp._legacy, "setup", _poison)
    monkeypatch.setattr(exp, "_build_legacy_experiment", lambda: exp._legacy)

    exp.fit(df)
    assert exp._native_setup_used is True


@pytest.mark.slow
def test_normalize_produces_zero_mean_unit_std():
    """After native normalize, numeric X_train_transformed has mean ~0, std ~1."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(
        target="Purchase", session_id=42, n_jobs=1, fold=3, normalize=True
    ).fit(df)
    assert exp._native_setup_used is True
    Xt = exp._fit_state["X_train_transformed"]
    # Numeric cols sit at the front; check at least the first 3 columns.
    means = Xt.iloc[:, :3].mean().abs()
    stds = Xt.iloc[:, :3].std()
    assert (means < 1e-6).all(), f"means not ~0: {means.tolist()}"
    # Train std comes out very close to 1 (sample stat).
    assert all(0.99 < s < 1.02 for s in stds), f"stds not ~1: {stds.tolist()}"


@pytest.mark.slow
def test_transformation_applies_power_transformer():
    """transformation=True puts a PowerTransformer in the numeric pipeline."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment
    from sklearn.preprocessing import PowerTransformer

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(
        target="Purchase", session_id=42, n_jobs=1, fold=3, transformation=True
    ).fit(df)
    assert exp._native_setup_used is True
    # Walk the fitted preprocessor's named steps to find the transformer.
    ct = exp._fit_state["preprocess_pipeline"].named_steps["preprocess"]
    num_pipe = ct.named_transformers_["numerical_pipeline"]
    # The pipeline should have a 'transformer' step that is a PowerTransformer.
    pt = num_pipe.named_steps["transformer"]
    assert isinstance(pt, PowerTransformer)
    assert pt.method == "yeo-johnson"


@pytest.mark.slow
def test_normalize_and_transformation_combined():
    """Combined: PowerTransformer then StandardScaler."""
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
    ).fit(df)
    assert exp._native_setup_used is True
    ct = exp._fit_state["preprocess_pipeline"].named_steps["preprocess"]
    num_pipe = ct.named_transformers_["numerical_pipeline"]
    step_names = [n for n, _ in num_pipe.steps]
    assert step_names == ["imputer", "transformer", "scaler"]
    # Combined output should be ~ unit-std on the first numeric cols.
    Xt = exp._fit_state["X_train_transformed"]
    stds = Xt.iloc[:, :3].std()
    assert all(0.99 < s < 1.02 for s in stds)


@pytest.mark.slow
def test_remove_outliers_now_native_post_phase3():
    """As of session 37 (phase 3), remove_outliers=True runs natively.

    Originally a "still legacy" assertion — kept for historical context;
    the actual phase-3 drain-locks live in test_session37.
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
    ).fit(df)
    assert exp._native_setup_used is True


def test_feature_selection_now_native_post_phase3():
    """As of session 37 (phase 3), feature_selection=True runs natively.

    Originally a "still legacy" predicate-only check (legacy lacked
    lightgbm); kept here for historical context.
    """
    from pycaret.tasks import ClassificationExperiment

    exp = ClassificationExperiment(target="t", session_id=0, feature_selection=True)
    assert exp._can_use_native_setup({}) is True


@pytest.mark.slow
def test_normalize_predict_chain_works():
    """End-to-end with normalize=True: create + predict round-trip."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(
        target="Purchase", session_id=42, n_jobs=1, fold=3, normalize=True
    ).fit(df)
    created = exp.create_model("lr", verbose=False)
    preds = exp.predict_model(created.pipeline)
    assert "prediction_label" in preds.predictions.columns


@pytest.mark.slow
def test_regression_with_transformation_native():
    """Regression with transformation=True works natively."""
    import pycaret.datasets
    from pycaret.tasks import RegressionExperiment

    df = pycaret.datasets.get_data("boston", verbose=False)
    exp = RegressionExperiment(
        target="medv", session_id=42, n_jobs=1, fold=3, transformation=True
    ).fit(df)
    assert exp._native_setup_used is True
    created = exp.create_model("lr", verbose=False)
    assert created.metrics is not None


def test_can_use_native_setup_predicate_phase2():
    """Phase 2: normalize + transformation are native (the canonical
    phase-2 flags). Other flags get their own session's coverage.
    """
    from pycaret.tasks import ClassificationExperiment

    # Phase 2 — these now go native.
    assert (
        ClassificationExperiment(target="t", session_id=0, normalize=True)._can_use_native_setup({})
        is True
    )
    assert (
        ClassificationExperiment(
            target="t", session_id=0, transformation=True
        )._can_use_native_setup({})
        is True
    )
    assert (
        ClassificationExperiment(
            target="t", session_id=0, normalize=True, transformation=True
        )._can_use_native_setup({})
        is True
    )
