"""Session 43 — phase 5c (cont.): drain ``TimeSeriesExperiment.tune_model``.

Phases 5b/5c (s40-s42) drained ``create_model``, ``predict_model``, and
``compare_models``. This session drains the fourth TS verb. Native
``tune_model`` wraps sktime's ``ForecastingGridSearchCV`` /
``ForecastingRandomizedSearchCV`` around the experiment's preprocess
pipeline, uses the registry container's ``tune_grid`` /
``tune_distributions`` (or ``custom_grid=``), and refits the best
hyperparameters via the drained ``create_model``.

After this session, only ``finalize_model`` and ``assign_model`` still
delegate to legacy for TS.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Drain-lock: legacy.tune_model MUST NOT be called.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_tune_model_drain_lock(monkeypatch):
    """tune_model on a drained-create pipeline doesn't touch legacy."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment
    from sktime.forecasting.compose import ForecastingPipeline

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)
    res = exp.create_model("naive", verbose=False)

    def _poison(*a, **kw):
        raise AssertionError("Phase 5c regression: tune_model hit legacy.")

    monkeypatch.setattr(exp._legacy, "tune_model", _poison)

    tuned = exp.tune_model(res.pipeline, n_iter=3, verbose=False)
    assert isinstance(tuned.pipeline, ForecastingPipeline)
    assert tuned.best_params is not None
    assert isinstance(tuned.best_params, dict)


# ---------------------------------------------------------------------------
# Best params come from the registry container's tune_distributions.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_tune_model_returns_grid_keys(monkeypatch):
    """best_params keys match the container's tune_grid / distributions."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)
    res = exp.create_model("naive", verbose=False)

    def _poison(*a, **kw):
        raise AssertionError("regression")

    monkeypatch.setattr(exp._legacy, "tune_model", _poison)

    tuned = exp.tune_model(res.pipeline, n_iter=2, verbose=False)
    # NaiveContainer's tune_distribution has 'strategy' and 'sp'.
    assert set(tuned.best_params.keys()).issubset({"strategy", "sp"})


# ---------------------------------------------------------------------------
# Search algorithm: grid vs random.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_tune_model_grid_search(monkeypatch):
    """search_algorithm='grid' uses ForecastingGridSearchCV."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment
    from pycaret.utils.time_series.forecasting.model_selection import (
        ForecastingGridSearchCV,
    )

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)
    res = exp.create_model("naive", verbose=False)

    def _poison(*a, **kw):
        raise AssertionError("regression")

    monkeypatch.setattr(exp._legacy, "tune_model", _poison)

    tuned = exp.tune_model(res.pipeline, search_algorithm="grid", verbose=False)
    assert isinstance(tuned.search, ForecastingGridSearchCV)


@pytest.mark.slow
def test_tune_model_random_search_default(monkeypatch):
    """Default search_algorithm='random' uses ForecastingRandomizedSearchCV."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment
    from pycaret.utils.time_series.forecasting.model_selection import (
        ForecastingRandomizedSearchCV,
    )

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)
    res = exp.create_model("naive", verbose=False)

    def _poison(*a, **kw):
        raise AssertionError("regression")

    monkeypatch.setattr(exp._legacy, "tune_model", _poison)

    tuned = exp.tune_model(res.pipeline, n_iter=3, verbose=False)
    assert isinstance(tuned.search, ForecastingRandomizedSearchCV)


# ---------------------------------------------------------------------------
# custom_grid path.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_tune_model_custom_grid(monkeypatch):
    """custom_grid= overrides the container's defaults."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)
    res = exp.create_model("naive", verbose=False)

    def _poison(*a, **kw):
        raise AssertionError("regression")

    monkeypatch.setattr(exp._legacy, "tune_model", _poison)

    tuned = exp.tune_model(
        res.pipeline,
        custom_grid={"strategy": ["mean"], "sp": [1]},
        search_algorithm="grid",
        verbose=False,
    )
    assert tuned.best_params["strategy"] == "mean"
    assert tuned.best_params["sp"] == 1


# ---------------------------------------------------------------------------
# Metrics + pull() integration.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_tune_model_metrics_appear_in_pull(monkeypatch):
    """pull() returns the tuned model's CV metrics."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)
    res = exp.create_model("naive", verbose=False)

    def _poison(*a, **kw):
        raise AssertionError("regression")

    monkeypatch.setattr(exp._legacy, "tune_model", _poison)

    exp.tune_model(res.pipeline, n_iter=2, verbose=False)
    pulled = exp.pull()
    assert pulled is not None
    assert "MAE" in pulled.columns


# ---------------------------------------------------------------------------
# Bad input handling.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_tune_model_string_raises():
    """tune_model with a registry ID string raises (not supported)."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)

    with pytest.raises(TypeError, match="fitted forecaster or pipeline"):
        exp.tune_model("naive", verbose=False)


@pytest.mark.slow
def test_tune_model_bad_optimize_metric_raises():
    """Unknown optimize metric → ConfigurationError."""
    import pycaret.datasets
    from pycaret.core.errors import ConfigurationError
    from pycaret.tasks import TimeSeriesExperiment

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)
    res = exp.create_model("naive", verbose=False)

    with pytest.raises(ConfigurationError, match="not found in the metric registry"):
        exp.tune_model(res.pipeline, optimize="NOT_A_REAL_METRIC", verbose=False)


# ---------------------------------------------------------------------------
# return_tuner=True returns (TuneResult, search_obj) tuple.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_tune_model_return_tuner(monkeypatch):
    """return_tuner=True returns (result, search) tuple."""
    import pycaret.datasets
    from pycaret.core.results import TuneResult
    from pycaret.tasks import TimeSeriesExperiment

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)
    res = exp.create_model("naive", verbose=False)

    def _poison(*a, **kw):
        raise AssertionError("regression")

    monkeypatch.setattr(exp._legacy, "tune_model", _poison)

    out = exp.tune_model(res.pipeline, n_iter=2, return_tuner=True, verbose=False)
    assert isinstance(out, tuple)
    assert len(out) == 2
    result, search = out
    assert isinstance(result, TuneResult)
    assert search is result.search


# ---------------------------------------------------------------------------
# End-to-end: create + tune + predict, all 4 legacy verbs poisoned.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_create_tune_predict_end_to_end_drained(monkeypatch):
    """Full create → tune → predict chain with 4 legacy verbs poisoned."""
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
    monkeypatch.setattr(exp._legacy, "compare_models", _poison("compare_models"))
    monkeypatch.setattr(exp._legacy, "tune_model", _poison("tune_model"))

    res = exp.create_model("naive", verbose=False)
    tuned = exp.tune_model(res.pipeline, n_iter=2, verbose=False)
    preds = exp.predict_model(tuned.pipeline)
    assert preds.predictions is not None
    assert len(preds.predictions) == 12  # fh=12
