"""Session 38 — native setup() phase 4: unsupervised tabular tasks.

Phase 1 (s35): supervised impute + ordinal-encode + label-encode + split + fold.
Phase 2 (s36): supervised normalize + transformation.
Phase 3 (s37): supervised remove_outliers + feature_selection.

Phase 4 finishes the tabular preprocessing drain by adding native setup
for **unsupervised** tasks (clustering + anomaly):

- No train/test split — the whole frame is the training set.
- No fold generator — clustering/anomaly don't CV in the usual sense.
- Same preprocessing chain as supervised (imputer + optional scaler +
  optional power transformer + ordinal encoder for categoricals).
- Model registry built from clustering / anomaly containers via the same
  ``_ModelRegistryContext`` proxy.

After this session, every tabular task (clf / reg / clustering / anomaly)
runs ``fit()`` natively when no ``setup_kwargs`` are passed and no
unsupervised-incompatible flags (``remove_outliers`` /
``feature_selection``) are set. Time-series remains the only task that
falls through to legacy.setup() — phase 5.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Drain-lock tests: legacy.setup MUST NOT be called by native unsupervised.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_clustering_native_setup_skips_legacy(monkeypatch):
    """ClusteringExperiment.fit() must NOT touch legacy.setup."""
    import pycaret.datasets
    from pycaret.tasks import ClusteringExperiment

    df = pycaret.datasets.get_data("jewellery", verbose=False)
    exp = ClusteringExperiment(session_id=42, n_jobs=1)

    def _poison(*a, **kw):
        raise AssertionError("Session-38 drain regression: clustering hit legacy.setup.")

    exp._legacy = exp._build_legacy_experiment()
    monkeypatch.setattr(exp._legacy, "setup", _poison)
    monkeypatch.setattr(exp, "_build_legacy_experiment", lambda: exp._legacy)

    exp.fit(df)
    assert exp._native_setup_used is True
    # No train/test split for clustering.
    assert exp._fit_state["X_train"] is None
    assert exp._fit_state["X_test"] is None
    assert exp._fit_state["fold_generator"] is None
    # X / X_transformed / preprocess_pipeline + a non-empty model registry.
    assert exp._fit_state["X"] is not None
    assert exp._fit_state["X_transformed"] is not None
    assert exp._fit_state["preprocess_pipeline"] is not None
    assert len(exp._fit_state["model_registry"]) > 0


@pytest.mark.slow
def test_anomaly_native_setup_skips_legacy(monkeypatch):
    """AnomalyExperiment.fit() must NOT touch legacy.setup."""
    import pycaret.datasets
    from pycaret.tasks import AnomalyExperiment

    df = pycaret.datasets.get_data("jewellery", verbose=False)
    exp = AnomalyExperiment(session_id=42, n_jobs=1)

    def _poison(*a, **kw):
        raise AssertionError("Session-38 drain regression: anomaly hit legacy.setup.")

    exp._legacy = exp._build_legacy_experiment()
    monkeypatch.setattr(exp._legacy, "setup", _poison)
    monkeypatch.setattr(exp, "_build_legacy_experiment", lambda: exp._legacy)

    exp.fit(df)
    assert exp._native_setup_used is True
    assert exp._fit_state["X_train"] is None
    assert exp._fit_state["fold_generator"] is None
    # Anomaly registry has ≥ a dozen pyod estimators.
    assert len(exp._fit_state["model_registry"]) >= 8


# ---------------------------------------------------------------------------
# End-to-end: native fit + create_model + assign_model.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_clustering_create_model_under_native_setup(monkeypatch):
    """create_model('kmeans') runs on natively-built _fit_state."""
    import pycaret.datasets
    from pycaret.tasks import ClusteringExperiment
    from sklearn.pipeline import Pipeline as SkPipeline

    df = pycaret.datasets.get_data("jewellery", verbose=False)
    exp = ClusteringExperiment(session_id=42, n_jobs=1)

    def _poison(*a, **kw):
        raise AssertionError("Session-38 drain regression: legacy.setup was called.")

    exp._legacy = exp._build_legacy_experiment()
    monkeypatch.setattr(exp._legacy, "setup", _poison)
    monkeypatch.setattr(exp, "_build_legacy_experiment", lambda: exp._legacy)
    exp.fit(df)

    res = exp.create_model("kmeans", verbose=False)
    assert isinstance(res.pipeline, SkPipeline)
    assert res.model_id == "kmeans"
    # Pipeline ends in the fitted KMeans model.
    last_step = res.pipeline.steps[-1][1]
    assert hasattr(last_step, "labels_")


@pytest.mark.slow
def test_anomaly_create_and_assign_model_under_native_setup(monkeypatch):
    """create_model('iforest') + assign_model on native unsupervised setup."""
    import pycaret.datasets
    from pycaret.tasks import AnomalyExperiment
    from sklearn.pipeline import Pipeline as SkPipeline

    df = pycaret.datasets.get_data("jewellery", verbose=False)
    exp = AnomalyExperiment(session_id=42, n_jobs=1)

    def _poison(*a, **kw):
        raise AssertionError("Session-38 drain regression: legacy.setup was called.")

    exp._legacy = exp._build_legacy_experiment()
    monkeypatch.setattr(exp._legacy, "setup", _poison)
    monkeypatch.setattr(exp, "_build_legacy_experiment", lambda: exp._legacy)
    exp.fit(df)

    res = exp.create_model("iforest", verbose=False)
    assert isinstance(res.pipeline, SkPipeline)
    assert res.model_id == "iforest"

    labeled = exp.assign_model(res.pipeline)
    assert "Anomaly" in labeled.columns
    assert "Anomaly_Score" in labeled.columns
    # Labels match the row count of the original data.
    assert labeled.shape[0] == df.shape[0]


# ---------------------------------------------------------------------------
# normalize / transformation flags compose with the unsupervised path.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_clustering_native_setup_with_normalize(monkeypatch):
    """normalize=True on clustering: numeric branch grows a StandardScaler."""
    import pycaret.datasets
    from pycaret.tasks import ClusteringExperiment
    from sklearn.preprocessing import StandardScaler

    df = pycaret.datasets.get_data("jewellery", verbose=False)
    exp = ClusteringExperiment(session_id=42, n_jobs=1, normalize=True)

    def _poison(*a, **kw):
        raise AssertionError("Session-38 drain regression: normalize=True hit legacy.setup.")

    exp._legacy = exp._build_legacy_experiment()
    monkeypatch.setattr(exp._legacy, "setup", _poison)
    monkeypatch.setattr(exp, "_build_legacy_experiment", lambda: exp._legacy)
    exp.fit(df)

    assert exp._native_setup_used is True
    ct = exp._fit_state["preprocess_pipeline"].named_steps["preprocess"]
    num_pipe = next(t for name, t, _cols in ct.transformers if name == "numerical_pipeline")
    assert any(isinstance(s, StandardScaler) for _, s in num_pipe.steps)


@pytest.mark.slow
def test_anomaly_native_setup_with_transformation(monkeypatch):
    """transformation=True on anomaly: numeric branch grows a PowerTransformer."""
    import pycaret.datasets
    from pycaret.tasks import AnomalyExperiment
    from sklearn.preprocessing import PowerTransformer

    df = pycaret.datasets.get_data("jewellery", verbose=False)
    exp = AnomalyExperiment(session_id=42, n_jobs=1, transformation=True)

    def _poison(*a, **kw):
        raise AssertionError("Session-38 drain regression: transformation=True hit legacy.setup.")

    exp._legacy = exp._build_legacy_experiment()
    monkeypatch.setattr(exp._legacy, "setup", _poison)
    monkeypatch.setattr(exp, "_build_legacy_experiment", lambda: exp._legacy)
    exp.fit(df)

    assert exp._native_setup_used is True
    ct = exp._fit_state["preprocess_pipeline"].named_steps["preprocess"]
    num_pipe = next(t for name, t, _cols in ct.transformers if name == "numerical_pipeline")
    assert any(isinstance(s, PowerTransformer) for _, s in num_pipe.steps)


# ---------------------------------------------------------------------------
# Predicate: which task/flag combos route to native?
# ---------------------------------------------------------------------------


def test_can_use_native_setup_unsupervised_predicate():
    """Predicate: clustering + anomaly are native by default."""
    from pycaret.tasks import AnomalyExperiment, ClusteringExperiment

    for cls in (ClusteringExperiment, AnomalyExperiment):
        # Bare experiment → native path.
        assert cls(session_id=0)._can_use_native_setup({}) is True
        # normalize / transformation → still native.
        assert cls(session_id=0, normalize=True)._can_use_native_setup({}) is True
        assert cls(session_id=0, transformation=True)._can_use_native_setup({}) is True
        # remove_outliers / feature_selection are NOT wired for unsupervised
        # in phase 4, so they force legacy. (Unsupervised constructors don't
        # surface those kwargs, so we set them post-init to test the predicate
        # branch that handles users who reach in.)
        exp = cls(session_id=0)
        exp.remove_outliers = True
        assert exp._can_use_native_setup({}) is False
        exp = cls(session_id=0)
        exp.feature_selection = True
        assert exp._can_use_native_setup({}) is False
        # Caller-supplied setup_kwargs always force legacy.
        assert cls(session_id=0)._can_use_native_setup({"foo": 1}) is False


def test_time_series_still_falls_back_to_legacy():
    """Time-series doesn't yet have a native path; predicate must say no."""
    from pycaret.tasks import TimeSeriesExperiment

    exp = TimeSeriesExperiment(session_id=0)
    assert exp._can_use_native_setup({}) is False


# ---------------------------------------------------------------------------
# Categorical handling on the unsupervised native path.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_unsupervised_native_setup_encodes_categoricals(monkeypatch):
    """Categorical columns are ordinal-encoded in the native path."""
    import numpy as np
    import pandas as pd
    from pycaret.tasks import ClusteringExperiment

    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "x": rng.normal(size=80),
            "y": rng.normal(size=80),
            "color": rng.choice(["red", "green", "blue"], size=80),
        }
    )
    exp = ClusteringExperiment(session_id=42, n_jobs=1)

    def _poison(*a, **kw):
        raise AssertionError("Session-38 drain regression: categorical hit legacy.setup.")

    exp._legacy = exp._build_legacy_experiment()
    monkeypatch.setattr(exp._legacy, "setup", _poison)
    monkeypatch.setattr(exp, "_build_legacy_experiment", lambda: exp._legacy)
    exp.fit(df)

    Xt = exp._fit_state["X_transformed"]
    # All three columns survive; categorical is encoded as numeric.
    assert list(Xt.columns) == ["x", "y", "color"]
    assert np.issubdtype(Xt["color"].dtype, np.number)
    # And we have a real preprocess Pipeline.
    assert exp._fit_state["preprocess_pipeline"] is not None
