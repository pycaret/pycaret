"""Session 48 — Track A · feature attribution plots.

Model-agnostic explainability built on sklearn primitives so they work
on any fitted model without SHAP. SHAP-based plots layered on top as
soft-deps.

Coverage:

- ``permutation_importance`` — drop-in feature-importance bar chart
  with ±σ error bars (works on any model).
- ``partial_dependence`` — 1-D + 2-D PDP showing how the model output
  changes with one or two features.
- ``ice_curve`` — individual conditional expectation curves; shows
  interaction effects when ICE traces are non-parallel.
- SHAP plots (``shap_summary``, ``shap_beeswarm``) raise a clear
  ``ImportError`` when shap isn't installed; UI side disables those
  cards in that case.
"""

from __future__ import annotations

import plotly.graph_objects as go
import pytest


@pytest.fixture(scope="module")
def reg_pipeline():
    import pycaret.datasets
    from pycaret.tasks import RegressionExperiment

    df = pycaret.datasets.get_data("boston", verbose=False)
    exp = RegressionExperiment(target="medv", session_id=42, n_jobs=1, fold=3).fit(df)
    res = exp.create_model("rf", verbose=False)
    return exp, res.pipeline


# ---------------------------------------------------------------------------
# Permutation importance — model-agnostic.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_permutation_importance_basic(reg_pipeline):
    from pycaret.plots import feature

    exp, pipeline = reg_pipeline
    fig = feature.permutation_importance(
        pipeline,
        exp.X_test,
        exp.y_test,
        n_repeats=3,
        random_state=0,
    )
    assert isinstance(fig, go.Figure)
    bar = fig.data[0]
    assert bar.type == "bar"
    assert bar.orientation == "h"
    # Error bars are populated.
    assert bar.error_x is not None and bar.error_x.array is not None


@pytest.mark.slow
def test_permutation_importance_top_n(reg_pipeline):
    from pycaret.plots import feature

    exp, pipeline = reg_pipeline
    fig = feature.permutation_importance(
        pipeline,
        exp.X_test,
        exp.y_test,
        n_repeats=3,
        top_n=5,
    )
    # Returned bar count matches top_n (limited by feature count).
    assert len(fig.data[0].y) <= 5


# ---------------------------------------------------------------------------
# Partial dependence.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_partial_dependence_1d(reg_pipeline):
    from pycaret.plots import feature

    exp, pipeline = reg_pipeline
    feat = list(exp.X_train.columns)[0]
    fig = feature.partial_dependence(pipeline, exp.X_train, feat, grid_resolution=20)
    assert isinstance(fig, go.Figure)
    # Default kind="average" → 1 trace.
    assert len(fig.data) == 1
    # Title includes the feature name.
    assert feat in fig.layout.title.text


@pytest.mark.slow
def test_partial_dependence_2d(reg_pipeline):
    from pycaret.plots import feature

    exp, pipeline = reg_pipeline
    cols = list(exp.X_train.columns)[:2]
    fig = feature.partial_dependence(pipeline, exp.X_train, tuple(cols), grid_resolution=15)
    assert isinstance(fig, go.Figure)
    # Heatmap.
    assert fig.data[0].type == "heatmap"
    assert " × " in fig.layout.title.text


@pytest.mark.slow
def test_partial_dependence_unknown_feature_raises(reg_pipeline):
    from pycaret.plots import feature

    exp, pipeline = reg_pipeline
    with pytest.raises(ValueError, match="not in feature_names"):
        feature.partial_dependence(pipeline, exp.X_train, "definitely_not_a_real_feature")


# ---------------------------------------------------------------------------
# ICE curves.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_ice_curve_default(reg_pipeline):
    from pycaret.plots import feature

    exp, pipeline = reg_pipeline
    feat = list(exp.X_train.columns)[0]
    fig = feature.ice_curve(pipeline, exp.X_train, feat, sample=20, grid_resolution=15)
    assert isinstance(fig, go.Figure)
    # ≤ 20 ICE traces + 1 mean trace (sampled to 20).
    assert 1 < len(fig.data) <= 21


@pytest.mark.slow
def test_ice_curve_centered(reg_pipeline):
    from pycaret.plots import feature

    exp, pipeline = reg_pipeline
    feat = list(exp.X_train.columns)[0]
    fig = feature.ice_curve(
        pipeline,
        exp.X_train,
        feat,
        sample=10,
        centered=True,
        grid_resolution=12,
    )
    assert isinstance(fig, go.Figure)
    assert fig.layout.yaxis.title.text == "Centered prediction"


# ---------------------------------------------------------------------------
# SHAP plots — soft dependency.
# ---------------------------------------------------------------------------


def test_shap_summary_raises_helpful_error_when_shap_missing(monkeypatch):
    """When shap isn't installed, ``shap_summary`` raises ImportError with
    install instructions. UI side disables the card in that case.
    """
    # Force the import path to act as if shap is missing.
    import builtins

    from pycaret.plots import feature

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "shap":
            raise ModuleNotFoundError("No module named 'shap'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(ImportError, match="shap is not installed"):
        feature.shap_summary(object(), object())


def test_shap_beeswarm_raises_helpful_error_when_shap_missing(monkeypatch):
    import builtins

    from pycaret.plots import feature

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "shap":
            raise ModuleNotFoundError("No module named 'shap'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(ImportError, match="shap is not installed"):
        feature.shap_beeswarm(object(), object())


# ---------------------------------------------------------------------------
# Module export contract.
# ---------------------------------------------------------------------------


def test_feature_module_exports():
    from pycaret.plots import feature

    for name in (
        "permutation_importance",
        "partial_dependence",
        "ice_curve",
        "shap_summary",
        "shap_beeswarm",
    ):
        assert hasattr(feature, name), f"feature module missing {name}"
