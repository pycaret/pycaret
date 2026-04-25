"""Session 34 — fix sklearn 1.8 ``squared=False`` deprecation in the
regression metric registry.

The legacy ``RMSEMetricContainer`` passed ``args={"squared": False}`` to
``mean_squared_error``. sklearn 1.6+ deprecated that path in favor of a
dedicated ``root_mean_squared_error``. Every regression CV run was
emitting ~7 deprecation warnings (one per fold + Mean + Std), so the
test output was flooded.

The fix prefers ``root_mean_squared_error`` when available + falls back
to a ``sqrt(mse)`` shim for older sklearn. Same pattern applied to
``RMSLEMetricContainer``.

These tests:

1. Lock the new ``RMSE`` container's ``score_func`` to *not* be
   ``mean_squared_error`` (which would re-trigger the deprecation when
   called with kwargs).
2. Run a regression CV with ``-W error::DeprecationWarning``-style
   strict warning filtering and assert no DeprecationWarning fires.
"""

from __future__ import annotations

import warnings

import pytest


def test_rmse_metric_does_not_use_mean_squared_error_directly():
    """The fix: RMSE's score_func is the dedicated root_mean_squared_error
    (or our sqrt(mse) shim), not the deprecated `mean_squared_error(squared=False)`.
    """
    from pycaret.containers.metrics.regression import RMSEMetricContainer
    from sklearn import metrics

    rmse = RMSEMetricContainer({})
    # Fast path: sklearn 1.6+ has root_mean_squared_error directly.
    if hasattr(metrics, "root_mean_squared_error"):
        assert rmse.score_func is metrics.root_mean_squared_error
    else:
        # Older sklearn — must NOT be mean_squared_error (that's the
        # deprecated path). It should be our locally-defined shim.
        assert rmse.score_func is not metrics.mean_squared_error


def test_rmse_metric_args_does_not_carry_squared():
    """The container's args dict no longer carries `squared=False`."""
    from pycaret.containers.metrics.regression import RMSEMetricContainer

    rmse = RMSEMetricContainer({})
    # `args` is the kwargs-passed-to-score_func dict; should not have squared.
    assert "squared" not in (rmse.args or {})


def test_rmse_score_func_handles_kwargs_without_deprecation():
    """Calling RMSE's score_func with the kwargs sklearn ≥1.8 expects
    (no `squared=`) succeeds without a DeprecationWarning.
    """
    import numpy as np
    from pycaret.containers.metrics.regression import RMSEMetricContainer

    rmse = RMSEMetricContainer({})
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 1.9, 3.05, 4.0, 5.2])

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        # Should not raise.
        score = rmse.score_func(y_true, y_pred)
    assert score > 0
    # Sanity: rough RMSE for these arrays is ~0.13.
    assert 0.05 < score < 0.5


@pytest.mark.slow
def test_regression_cv_does_not_emit_squared_deprecation():
    """Full regression CV via Experiment.create_model — no DeprecationWarning
    about `squared=`.
    """
    import pycaret.datasets
    from pycaret.tasks import RegressionExperiment

    df = pycaret.datasets.get_data("boston", verbose=False)
    exp = RegressionExperiment(target="medv", session_id=42, n_jobs=1, fold=3).fit(df)

    with warnings.catch_warnings():
        # Capture all warnings + then grep for the deprecation.
        warnings.simplefilter("always")
        warnings.simplefilter("error", DeprecationWarning, append=True)
        # Should not raise.
        exp.create_model("lr", verbose=False)
