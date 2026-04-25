"""Session 40 — phase 5b: drain ``TimeSeriesExperiment.create_model``.

Phase 5a (s39) brought TS into the native dispatcher and gave us
``_fit_state`` parity with the other tasks, but ``create_model`` still
delegated to ``self._legacy.create_model`` for TS. Phase 5b drains it.

What this session ships:

- ``TimeSeriesExperiment.create_model`` resolves the estimator from the
  sktime registry, wires it into ``_fit_state["preprocess_pipeline"]``
  via ``_add_model_to_pipeline``, runs cross-validation through the
  existing ``cross_validate`` helper (which clones the pipeline per
  fold), builds the metrics DataFrame in the standard
  ``Fold 0..N / Mean / Std`` shape, refits on the full ``y_train``,
  and returns a ``CreateResult`` whose ``pipeline`` is a real
  ``sktime.forecasting.compose.ForecastingPipeline``.
- A new ``_build_ts_metric_registry`` helper caches the TS metric
  registry in ``_fit_state["metric_registry"]`` so ``add_metric`` /
  ``remove_metric`` can mutate it (parity with the supervised drain).
- A new ``_primary_sp_to_use`` helper resolves the seasonal-period
  scorer kwarg required by MASE / RMSSE.

After this session, only ``predict_model`` / ``compare_models`` /
``tune_model`` / ``finalize_model`` / ``assign_model`` still delegate to
legacy for TS. Phase 5c will drain those, then ``_native_setup_timeseries``
can stop calling ``legacy.setup()`` and the legacy directory becomes
deletable.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Drain-lock: legacy.create_model MUST NOT be called.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_create_model_drain_lock_naive(monkeypatch):
    """create_model('naive') runs without touching legacy.create_model."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment
    from sktime.forecasting.compose import ForecastingPipeline

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)

    def _poison(*a, **kw):
        raise AssertionError("Phase 5b regression: TS create_model hit legacy.")

    monkeypatch.setattr(exp._legacy, "create_model", _poison)

    res = exp.create_model("naive", verbose=False)
    assert isinstance(res.pipeline, ForecastingPipeline)
    assert res.model_id == "naive"
    assert res.metrics is not None
    # 3 folds + Mean + Std rows.
    assert "Fold 0" in res.metrics.index
    assert "Mean" in res.metrics.index
    assert "Std" in res.metrics.index


@pytest.mark.slow
def test_create_model_drain_lock_classical(monkeypatch):
    """ARIMA + ETS both run natively (classical sktime forecasters)."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment
    from sktime.forecasting.compose import ForecastingPipeline

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)

    def _poison(*a, **kw):
        raise AssertionError("Phase 5b regression: classical hit legacy.")

    monkeypatch.setattr(exp._legacy, "create_model", _poison)

    for mid in ("arima", "ets", "exp_smooth", "theta"):
        res = exp.create_model(mid, verbose=False)
        assert isinstance(res.pipeline, ForecastingPipeline)
        assert res.model_id == mid


# ---------------------------------------------------------------------------
# cross_validation=False: skip CV, refit on full y_train.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_create_model_no_cv(monkeypatch):
    """cross_validation=False returns a fitted pipeline + metrics=None."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment
    from sktime.forecasting.compose import ForecastingPipeline

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)

    def _poison(*a, **kw):
        raise AssertionError("Phase 5b regression: no-CV hit legacy.")

    monkeypatch.setattr(exp._legacy, "create_model", _poison)

    res = exp.create_model("naive", cross_validation=False, verbose=False)
    assert isinstance(res.pipeline, ForecastingPipeline)
    assert res.metrics is None


# ---------------------------------------------------------------------------
# Pre-constructed forecaster path.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_create_model_with_preconstructed_forecaster(monkeypatch):
    """Passing a fitted/unfit sktime forecaster wraps it in a pipeline."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment
    from sktime.forecasting.compose import ForecastingPipeline
    from sktime.forecasting.naive import NaiveForecaster

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)

    def _poison(*a, **kw):
        raise AssertionError("Phase 5b regression: preconstructed hit legacy.")

    monkeypatch.setattr(exp._legacy, "create_model", _poison)

    custom = NaiveForecaster(strategy="mean")
    res = exp.create_model(custom, verbose=False)
    assert isinstance(res.pipeline, ForecastingPipeline)
    # When a bare estimator is passed, model_id is the class name.
    assert res.model_id == "NaiveForecaster"


# ---------------------------------------------------------------------------
# Metrics DataFrame shape.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_create_model_metrics_dataframe_shape():
    """Metrics has Fold rows + Mean + Std + standard TS metric columns."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)

    res = exp.create_model("naive", verbose=False)
    cols = set(res.metrics.columns)
    # Standard TS metrics (display names from container).
    for expected in ("MAE", "RMSE", "MAPE", "MASE", "R2"):
        assert expected in cols, f"missing column {expected}"
    # 3 folds (default fold=3) + Mean + Std rows.
    assert len(res.metrics) == 5


# ---------------------------------------------------------------------------
# pull() returns the metrics DataFrame after native create_model.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_create_model_metrics_appear_in_pull(monkeypatch):
    """pull() returns the metrics from the native create_model run."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)

    def _poison(*a, **kw):
        raise AssertionError("Phase 5b regression: pull hit legacy.")

    monkeypatch.setattr(exp._legacy, "create_model", _poison)

    exp.create_model("naive", verbose=False)
    pulled = exp.pull()
    # Same data — pull just returns the last-set metrics.
    assert pulled is not None
    assert "MAE" in pulled.columns


# ---------------------------------------------------------------------------
# Predict_model still works on the drained-create pipeline.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_predict_model_works_with_drained_create_pipeline(monkeypatch):
    """predict_model (still legacy in 5b) accepts the native pipeline."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)

    def _poison(*a, **kw):
        raise AssertionError("Phase 5b regression: create hit legacy.")

    monkeypatch.setattr(exp._legacy, "create_model", _poison)

    res = exp.create_model("naive", verbose=False)
    preds = exp.predict_model(res.pipeline)
    assert preds.predictions is not None
    assert len(preds.predictions) == 12  # fh=12


# ---------------------------------------------------------------------------
# Bad input handling.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_create_model_unknown_id_raises():
    """Unknown registry ID raises ConfigurationError."""
    import pycaret.datasets
    from pycaret.core.errors import ConfigurationError
    from pycaret.tasks import TimeSeriesExperiment

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)

    with pytest.raises(ConfigurationError, match="Unknown TS model id"):
        exp.create_model("not_a_real_model", verbose=False)


@pytest.mark.slow
def test_create_model_non_estimator_raises():
    """Object without .fit raises TypeError."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)

    with pytest.raises(TypeError, match="registry ID or a sktime forecaster"):
        exp.create_model(object(), verbose=False)


# ---------------------------------------------------------------------------
# Metric registry is now built natively for TS too.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_ts_metric_registry_caches_on_fit_state():
    """First call builds the registry; subsequent calls hit the cache."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)

    # Pre-fit, no registry cached yet.
    assert exp._fit_state.get("metric_registry") is None

    # create_model triggers build.
    exp.create_model("naive", verbose=False)
    cached = exp._fit_state.get("metric_registry")
    assert cached is not None
    assert "mae" in cached
    assert "rmse" in cached
    assert "mape" in cached
    # Second call to the build helper returns the same dict object.
    assert exp._build_ts_metric_registry() is cached
