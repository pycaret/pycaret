"""Session 45 — phase 5d: strip ``legacy.setup()`` from
``_native_setup_timeseries``.

Phase 5a (s39) brought TS into the native dispatcher but the soft drain
still called ``legacy.setup()`` underneath. Phases 5b/5c (s40-s44)
drained every callable TS verb. This session takes the final step:
``_native_setup_timeseries`` no longer calls ``legacy.setup()`` at all.

What's new:

- A new ``_TSContextProxy`` class (in ``pycaret.core.experiment``)
  exposes the 14 attrs that TS containers and util helpers read off
  the experiment object — ``seed`` / ``gpu_param`` / ``n_jobs_param`` /
  ``seasonality_present`` / ``primary_sp_to_use`` / ``strictly_positive`` /
  ``seasonality_type`` / ``all_sps_to_use`` / ``X_train`` /
  ``is_multiclass`` / ``enforce_pi`` / ``enforce_exogenous`` /
  ``exogenous_present`` / ``fe_target_rr`` / ``index_type``.
- Seasonality auto-detection via sktime's
  ``autocorrelation_seasonality_test`` on candidates derived from the
  index frequency. ``primary_sp_to_use`` is the largest significant sp
  (legacy default).
- ``temporal_train_test_split`` for the y_train / y_test split.
- ``ExpandingWindowSplitter`` (default) or ``SlidingWindowSplitter``
  for the fold generator. Same math legacy uses for ``initial_window``
  / ``step_length``.
- A minimal ``ForecastingPipeline`` (placeholder ``NaiveForecaster``)
  serves as ``preprocess_pipeline``; drained verbs swap the placeholder
  in via ``_add_model_to_pipeline``.

After this session, ``pycaret/internal/pycaret_experiment/`` is no
longer needed for any default TS workflow — phase 6 deletes the 10K-line
legacy directory.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Total drain-lock: legacy.setup is the only one left for TS, and it's
# now bypassed by the native path.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_native_ts_setup_bypasses_legacy_setup(monkeypatch):
    """fit() does not touch legacy.setup when the native path runs."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42)

    def _poison(*a, **kw):
        raise AssertionError("Phase 5d regression: legacy.setup was called.")

    exp._legacy = exp._build_legacy_experiment()
    monkeypatch.setattr(exp._legacy, "setup", _poison)
    monkeypatch.setattr(exp, "_build_legacy_experiment", lambda: exp._legacy)

    exp.fit(df)
    assert exp._native_setup_used is True
    assert exp._fit_state["y_train"] is not None
    assert exp._fit_state["y_test"] is not None
    assert len(exp._fit_state["model_registry"]) > 0


# ---------------------------------------------------------------------------
# Seasonality auto-detection.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_native_ts_setup_detects_monthly_seasonality(monkeypatch):
    """Airline data (monthly PeriodIndex) → seasonality_present=True, sp=12."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42)

    def _poison(*a, **kw):
        raise AssertionError("regression")

    exp._legacy = exp._build_legacy_experiment()
    monkeypatch.setattr(exp._legacy, "setup", _poison)
    monkeypatch.setattr(exp, "_build_legacy_experiment", lambda: exp._legacy)

    exp.fit(df)
    assert exp._fit_state["seasonality_present"] is True
    assert exp._fit_state["seasonal_period"] == 12
    assert exp._fit_state["seasonality_type"] in ("mul", "add")
    assert exp._fit_state["strictly_positive"] is True


def test_auto_detect_seasonality_no_freq_returns_default():
    """Series with no recognisable index frequency → (False, 1, [1])."""
    import numpy as np
    import pandas as pd
    from pycaret.core.experiment import Experiment

    # Plain RangeIndex — no frequency.
    y = pd.Series(np.random.default_rng(0).standard_normal(50))
    seasonality, primary_sp, all_sps = Experiment._auto_detect_seasonality(y)
    assert seasonality is False
    assert primary_sp == 1
    assert all_sps == [1]


# ---------------------------------------------------------------------------
# Fold generator math matches legacy.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_native_ts_fold_generator_is_expanding_window_splitter(monkeypatch):
    """Default fold_strategy='expanding' → ExpandingWindowSplitter."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment
    from sktime.split import ExpandingWindowSplitter

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42)

    def _poison(*a, **kw):
        raise AssertionError("regression")

    exp._legacy = exp._build_legacy_experiment()
    monkeypatch.setattr(exp._legacy, "setup", _poison)
    monkeypatch.setattr(exp, "_build_legacy_experiment", lambda: exp._legacy)

    exp.fit(df)
    fg = exp._fit_state["fold_generator"]
    assert isinstance(fg, ExpandingWindowSplitter)
    # n_splits should match self.fold (default 3).
    assert fg.get_n_splits(exp._fit_state["y_train"]) == 3


@pytest.mark.slow
def test_native_ts_fold_generator_sliding(monkeypatch):
    """fold_strategy='sliding' → SlidingWindowSplitter."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment
    from sktime.split import SlidingWindowSplitter

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42, fold_strategy="sliding")

    def _poison(*a, **kw):
        raise AssertionError("regression")

    exp._legacy = exp._build_legacy_experiment()
    monkeypatch.setattr(exp._legacy, "setup", _poison)
    monkeypatch.setattr(exp, "_build_legacy_experiment", lambda: exp._legacy)

    exp.fit(df)
    assert isinstance(exp._fit_state["fold_generator"], SlidingWindowSplitter)


# ---------------------------------------------------------------------------
# Model registry is populated through the proxy.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_native_ts_model_registry_populated(monkeypatch):
    """The full sktime model registry builds through the proxy (no legacy)."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42)

    def _poison(*a, **kw):
        raise AssertionError("regression")

    exp._legacy = exp._build_legacy_experiment()
    monkeypatch.setattr(exp._legacy, "setup", _poison)
    monkeypatch.setattr(exp, "_build_legacy_experiment", lambda: exp._legacy)

    exp.fit(df)
    registry = exp._fit_state["model_registry"]
    # All standard sktime forecasters should be present.
    for mid in ("naive", "snaive", "polytrend", "arima", "ets", "theta", "exp_smooth"):
        assert mid in registry, f"missing {mid}"


# ---------------------------------------------------------------------------
# End-to-end: full chain with EVERY legacy verb (including setup) poisoned.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_full_chain_no_legacy_verbs_called(monkeypatch):
    """create + tune + finalize + predict + setup all poisoned → still works."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42)

    def _poison(name):
        def _f(*a, **kw):
            raise AssertionError(f"Phase 5d regression: legacy.{name} called.")

        return _f

    exp._legacy = exp._build_legacy_experiment()
    monkeypatch.setattr(exp, "_build_legacy_experiment", lambda: exp._legacy)
    for verb in (
        "setup",
        "create_model",
        "predict_model",
        "compare_models",
        "tune_model",
        "finalize_model",
    ):
        monkeypatch.setattr(exp._legacy, verb, _poison(verb))

    exp.fit(df)
    res = exp.create_model("naive", verbose=False)
    tuned = exp.tune_model(res.pipeline, n_iter=2, verbose=False)
    final = exp.finalize_model(tuned.pipeline)
    preds = exp.predict_model(final.pipeline, fh=[1, 2, 3])
    assert preds.predictions is not None
    assert len(preds.predictions) == 3


# ---------------------------------------------------------------------------
# setup_kwargs raise in PyCaret 4.0 (phase 6 deleted the legacy escape).
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_setup_kwargs_raises_in_phase6():
    """Phase 6 deleted the legacy directory. setup_kwargs no longer fall
    through — they raise ConfigurationError pointing users at 3.x or at
    requesting a first-class constructor param.
    """
    import pycaret.datasets
    from pycaret.core.errors import ConfigurationError
    from pycaret.tasks import TimeSeriesExperiment

    df = pycaret.datasets.get_data("airline", verbose=False)
    with pytest.raises(ConfigurationError, match="setup_kwargs are not supported"):
        TimeSeriesExperiment(fh=12, session_id=42).fit(df, html=False)


# ---------------------------------------------------------------------------
# Coerces univariate Series input.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_native_ts_setup_accepts_univariate_series(monkeypatch):
    """Pass a Series directly (no DataFrame) — works."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment

    df = pycaret.datasets.get_data("airline", verbose=False)
    # Coerce to univariate Series.
    if isinstance(df, type(df)) and hasattr(df, "iloc"):
        # df is a 1-col DataFrame; pull as Series.
        y = df.iloc[:, 0] if df.ndim > 1 else df

    exp = TimeSeriesExperiment(fh=12, session_id=42)

    def _poison(*a, **kw):
        raise AssertionError("regression")

    exp._legacy = exp._build_legacy_experiment()
    monkeypatch.setattr(exp._legacy, "setup", _poison)
    monkeypatch.setattr(exp, "_build_legacy_experiment", lambda: exp._legacy)

    exp.fit(y)
    assert exp._native_setup_used is True
    assert exp._fit_state["y_train"] is not None
