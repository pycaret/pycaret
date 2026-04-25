"""Session 41 — phase 5c: drain ``TimeSeriesExperiment.predict_model``.

Phase 5b (s40) drained ``create_model``. This session drains the second
TS verb. ``exp.predict_model(forecaster)`` no longer touches
``self._legacy.predict_model`` — predictions go directly through
``get_predictions_with_intervals`` and metrics are computed via the
standalone ``calculate_metrics`` utility against ``_fit_state["y_test"]``.

After this session, only ``compare_models`` / ``tune_model`` /
``finalize_model`` / ``assign_model`` still delegate to legacy for TS.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Drain-lock: legacy.predict_model MUST NOT be called.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_predict_model_drain_lock(monkeypatch):
    """predict_model on a drained-create pipeline doesn't touch legacy."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)
    res = exp.create_model("naive", verbose=False)

    def _poison(*a, **kw):
        raise AssertionError("Phase 5c regression: predict_model hit legacy.")

    monkeypatch.setattr(exp._legacy, "predict_model", _poison)

    preds = exp.predict_model(res.pipeline)
    assert preds.predictions is not None
    assert "y_pred" in preds.predictions.columns
    assert len(preds.predictions) == 12


# ---------------------------------------------------------------------------
# Metrics path: ground-truth metrics computed against y_test.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_predict_model_returns_metrics_against_y_test(monkeypatch):
    """When y_test is known, metrics is a one-row DataFrame keyed by display name."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)
    res = exp.create_model("naive", verbose=False)

    def _poison(*a, **kw):
        raise AssertionError("regression")

    monkeypatch.setattr(exp._legacy, "predict_model", _poison)

    preds = exp.predict_model(res.pipeline)
    assert preds.metrics is not None
    # Standard TS metrics (display names from container).
    cols = set(preds.metrics.columns)
    for expected in ("Model", "MAE", "RMSE", "MAPE", "MASE", "R2"):
        assert expected in cols, f"missing column {expected}"
    assert preds.metrics.iloc[0]["Model"] == "NaiveForecaster"


# ---------------------------------------------------------------------------
# pull() returns the predict_model metrics.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_predict_model_metrics_appear_in_pull(monkeypatch):
    """pull() returns the metrics from native predict_model."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)
    res = exp.create_model("naive", verbose=False)

    def _poison(*a, **kw):
        raise AssertionError("regression")

    monkeypatch.setattr(exp._legacy, "predict_model", _poison)

    exp.predict_model(res.pipeline)
    pulled = exp.pull()
    assert pulled is not None
    assert "MAE" in pulled.columns


# ---------------------------------------------------------------------------
# Prediction intervals.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_predict_model_with_return_pred_int(monkeypatch):
    """return_pred_int=True yields lower / upper interval columns."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)
    # Use a forecaster that supports prediction intervals.
    res = exp.create_model("arima", verbose=False)

    def _poison(*a, **kw):
        raise AssertionError("regression")

    monkeypatch.setattr(exp._legacy, "predict_model", _poison)

    preds = exp.predict_model(res.pipeline, return_pred_int=True)
    assert "y_pred" in preds.predictions.columns
    assert "lower" in preds.predictions.columns
    assert "upper" in preds.predictions.columns


# ---------------------------------------------------------------------------
# Custom fh override.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_predict_model_with_custom_fh(monkeypatch):
    """Passing fh= overrides the experiment's default forecast horizon."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)
    res = exp.create_model("naive", verbose=False)

    def _poison(*a, **kw):
        raise AssertionError("regression")

    monkeypatch.setattr(exp._legacy, "predict_model", _poison)

    preds = exp.predict_model(res.pipeline, fh=[1, 2, 3])
    assert len(preds.predictions) == 3


# ---------------------------------------------------------------------------
# Bare forecaster (without pipeline) gets wired into the experiment's preprocess.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_predict_model_bare_forecaster(monkeypatch):
    """A bare fitted sktime forecaster is wired into the experiment pipeline."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment
    from sktime.forecasting.naive import NaiveForecaster

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)

    def _poison(*a, **kw):
        raise AssertionError("regression")

    monkeypatch.setattr(exp._legacy, "predict_model", _poison)

    fc = NaiveForecaster()
    fc.fit(exp.y_train, fh=list(range(1, 13)))
    preds = exp.predict_model(fc)
    assert preds.predictions is not None
    assert len(preds.predictions) == 12


# ---------------------------------------------------------------------------
# Bad input handling.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_predict_model_non_estimator_raises():
    """Object without .predict raises TypeError."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)

    with pytest.raises(TypeError, match="sktime forecaster or"):
        exp.predict_model(object())


# ---------------------------------------------------------------------------
# Integration: create + predict end-to-end without ever touching legacy verbs.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_create_and_predict_end_to_end_drained(monkeypatch):
    """Full create→predict chain works with both legacy verbs poisoned."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)

    def _poison(name):
        def _f(*a, **kw):
            raise AssertionError(f"Phase 5c regression: hit legacy.{name}.")

        return _f

    monkeypatch.setattr(exp._legacy, "create_model", _poison("create_model"))
    monkeypatch.setattr(exp._legacy, "predict_model", _poison("predict_model"))

    res = exp.create_model("ets", verbose=False)
    preds = exp.predict_model(res.pipeline)
    assert preds.predictions is not None
    assert preds.metrics is not None
    assert "MAE" in preds.metrics.columns
