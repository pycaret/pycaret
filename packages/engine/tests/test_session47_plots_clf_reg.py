"""Session 47 — Track A · plot module: classification + regression diagnostics.

The legacy ``plot_model`` was deleted in s46. This session begins the
Plotly-native rewrite: every function in ``pycaret.plots.<task>``
returns a ``plotly.graph_objects.Figure`` that the React dashboard
consumes via ``react-plotly.js`` and notebooks render via ``.show()``.

Coverage:

- **Classification (binary + multiclass)**: confusion_matrix, roc_curve,
  pr_curve, calibration_curve, threshold_curve, lift_curve, gain_curve,
  class_distribution.
- **Regression**: residuals, residuals_distribution, prediction_error,
  learning_curve, feature_importance.

The chart layouts share a palette + typography aligned with the React
dashboard's design tokens (see ``pycaret.plots._base.PALETTE``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def binary_pipeline():
    """Fitted binary classifier on the juice dataset."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    res = exp.create_model("lr", verbose=False)
    return exp, res.pipeline


@pytest.fixture(scope="module")
def multiclass_pipeline():
    """Fitted multiclass classifier on iris."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("iris", verbose=False)
    exp = ClassificationExperiment(target="species", session_id=42, n_jobs=1, fold=3).fit(df)
    res = exp.create_model("lr", verbose=False)
    return exp, res.pipeline


@pytest.fixture(scope="module")
def regression_pipeline():
    """Fitted regressor on boston."""
    import pycaret.datasets
    from pycaret.tasks import RegressionExperiment

    df = pycaret.datasets.get_data("boston", verbose=False)
    exp = RegressionExperiment(target="medv", session_id=42, n_jobs=1, fold=3).fit(df)
    res = exp.create_model("rf", verbose=False)  # tree → has feature_importances_
    return exp, res.pipeline


# ---------------------------------------------------------------------------
# Module-level surface.
# ---------------------------------------------------------------------------


def test_plots_module_has_palette_and_helpers():
    """``pycaret.plots`` exposes design tokens + the layout helper."""
    from pycaret.plots import PALETTE, classification, default_layout, regression

    assert isinstance(PALETTE, dict)
    assert "accent" in PALETTE
    assert callable(default_layout)
    assert hasattr(classification, "confusion_matrix")
    assert hasattr(regression, "residuals")


def test_default_layout_includes_brand_font():
    """The shared layout uses Inter (matches the React dashboard)."""
    from pycaret.plots import default_layout

    layout = default_layout(title="x")
    assert layout["font"]["family"].startswith("Inter")
    # Title text propagates.
    assert layout["title"]["text"] == "x"


# ---------------------------------------------------------------------------
# Classification — binary.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_confusion_matrix_binary(binary_pipeline):
    from pycaret.plots import classification as cplots

    exp, pipeline = binary_pipeline
    fig = cplots.confusion_matrix(pipeline, exp.X_test, exp.y_test)
    assert isinstance(fig, go.Figure)
    # Heatmap z is the cm matrix.
    assert len(fig.data) == 1
    assert fig.data[0].type == "heatmap"


@pytest.mark.slow
def test_roc_curve_binary(binary_pipeline):
    from pycaret.plots import classification as cplots

    exp, pipeline = binary_pipeline
    fig = cplots.roc_curve(pipeline, exp.X_test, exp.y_test)
    assert isinstance(fig, go.Figure)
    # One curve + one diagonal baseline.
    assert len(fig.data) == 2
    # Trace 0 has AUC text in its name.
    assert "AUC" in fig.data[0].name


@pytest.mark.slow
def test_pr_curve_binary(binary_pipeline):
    from pycaret.plots import classification as cplots

    exp, pipeline = binary_pipeline
    fig = cplots.pr_curve(pipeline, exp.X_test, exp.y_test)
    assert isinstance(fig, go.Figure)
    assert "AP" in fig.data[0].name


@pytest.mark.slow
def test_calibration_curve_binary(binary_pipeline):
    from pycaret.plots import classification as cplots

    exp, pipeline = binary_pipeline
    fig = cplots.calibration_curve(pipeline, exp.X_test, exp.y_test, n_bins=8)
    assert isinstance(fig, go.Figure)
    # Model curve + reference diagonal.
    assert len(fig.data) == 2


@pytest.mark.slow
def test_threshold_curve_binary(binary_pipeline):
    from pycaret.plots import classification as cplots

    exp, pipeline = binary_pipeline
    fig = cplots.threshold_curve(pipeline, exp.X_test, exp.y_test)
    assert isinstance(fig, go.Figure)
    # 4 metrics — precision / recall / f1 / accuracy.
    assert len(fig.data) == 4


@pytest.mark.slow
def test_lift_and_gain_curves(binary_pipeline):
    from pycaret.plots import classification as cplots

    exp, pipeline = binary_pipeline
    lift = cplots.lift_curve(pipeline, exp.X_test, exp.y_test)
    gain = cplots.gain_curve(pipeline, exp.X_test, exp.y_test)
    assert isinstance(lift, go.Figure)
    assert isinstance(gain, go.Figure)
    # Both have model line + random baseline.
    assert len(lift.data) == 2
    assert len(gain.data) == 2


def test_class_distribution_independent_of_estimator():
    """class_distribution operates on y alone (no estimator needed)."""
    from pycaret.plots import classification as cplots

    y = pd.Series(["a", "a", "b", "c", "c", "c"])
    fig = cplots.class_distribution(y)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    # Three unique labels.
    assert len(fig.data[0].x) == 3


# ---------------------------------------------------------------------------
# Classification — multiclass.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_roc_curve_multiclass(multiclass_pipeline):
    from pycaret.plots import classification as cplots

    exp, pipeline = multiclass_pipeline
    fig = cplots.roc_curve(pipeline, exp.X_test, exp.y_test)
    assert isinstance(fig, go.Figure)
    # 3 classes (one-vs-rest) + 1 random baseline = 4 traces.
    assert len(fig.data) == 4


@pytest.mark.slow
def test_calibration_rejects_multiclass(multiclass_pipeline):
    from pycaret.plots import classification as cplots

    exp, pipeline = multiclass_pipeline
    with pytest.raises(ValueError, match="binary"):
        cplots.calibration_curve(pipeline, exp.X_test, exp.y_test)


# ---------------------------------------------------------------------------
# Regression.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_residuals_plot(regression_pipeline):
    from pycaret.plots import regression as rplots

    exp, pipeline = regression_pipeline
    fig = rplots.residuals(pipeline, exp.X_test, exp.y_test)
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "scatter"
    # Number of points equals test size.
    assert len(fig.data[0].x) == len(exp.y_test)


@pytest.mark.slow
def test_residuals_distribution(regression_pipeline):
    from pycaret.plots import regression as rplots

    exp, pipeline = regression_pipeline
    fig = rplots.residuals_distribution(pipeline, exp.X_test, exp.y_test, nbins=20)
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "histogram"


@pytest.mark.slow
def test_prediction_error(regression_pipeline):
    from pycaret.plots import regression as rplots

    exp, pipeline = regression_pipeline
    fig = rplots.prediction_error(pipeline, exp.X_test, exp.y_test)
    assert isinstance(fig, go.Figure)
    # Scatter + diagonal reference line.
    assert len(fig.data) == 2
    # R² annotated in the title.
    assert "R²" in fig.layout.title.text


@pytest.mark.slow
def test_feature_importance_tree(regression_pipeline):
    from pycaret.plots import regression as rplots

    exp, pipeline = regression_pipeline
    fig = rplots.feature_importance(pipeline, feature_names=list(exp.X_train.columns))
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "bar"
    # Bars are horizontal.
    assert fig.data[0].orientation == "h"


@pytest.mark.slow
def test_feature_importance_linear():
    """Linear models use |coef_| as the importance proxy."""
    import pycaret.datasets
    from pycaret.plots import regression as rplots
    from pycaret.tasks import RegressionExperiment

    df = pycaret.datasets.get_data("boston", verbose=False)
    exp = RegressionExperiment(target="medv", session_id=42, n_jobs=1, fold=3).fit(df)
    res = exp.create_model("lr", verbose=False)
    fig = rplots.feature_importance(res.pipeline, feature_names=list(exp.X_train.columns))
    assert isinstance(fig, go.Figure)
    # x-axis title reflects the source.
    assert "coef" in fig.layout.xaxis.title.text


def test_feature_importance_rejects_naive_model():
    """Models without importance/coef raise a clear error."""
    from pycaret.plots import regression as rplots
    from sklearn.dummy import DummyRegressor

    model = DummyRegressor()
    rng = np.random.default_rng(0)
    model.fit(rng.normal(size=(20, 3)), rng.normal(size=20))
    with pytest.raises(ValueError, match="feature_importances_ nor coef_"):
        rplots.feature_importance(model)


# ---------------------------------------------------------------------------
# Serialization — every figure must round-trip through JSON.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_figure_to_dict_serializable(binary_pipeline):
    """The React UI consumes ``fig.to_dict()`` over HTTP. Make sure it works."""
    import json

    from pycaret.plots import classification as cplots

    exp, pipeline = binary_pipeline
    fig = cplots.confusion_matrix(pipeline, exp.X_test, exp.y_test)
    payload = fig.to_dict()
    assert "data" in payload
    assert "layout" in payload
    # Plotly's `to_json` handles numpy arrays — we use that on the API side.
    js = fig.to_json()
    assert isinstance(js, str)
    parsed = json.loads(js)
    assert parsed["data"]
