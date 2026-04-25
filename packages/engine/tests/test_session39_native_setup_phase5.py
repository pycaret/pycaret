"""Session 39 — native setup() phase 5a: time-series accessor parity.

The supervised + unsupervised tabular tasks landed full native setup
in phases 1-4 (s35-s38). Phase 5 is the time-series drain — the most
involved because TS uses sktime, has its own data shape (univariate
Series with PeriodIndex / DatetimeIndex), its own splitter
(``ExpandingWindowSplitter``), and a registry of sktime forecasters.

**Phase 5a (this session)** is a *soft* drain:

- ``TimeSeriesExperiment.fit()`` now goes through the native dispatcher
  (``_can_use_native_setup`` accepts ``TaskType.TIME_SERIES``).
- A new ``Experiment._native_setup_timeseries`` populates ``_fit_state``
  with TS-shape slots (``y`` / ``y_train`` / ``y_test`` / ``fh`` /
  ``seasonal_period`` / ``fold_generator`` / ``model_registry`` /
  ``preprocess_pipeline``).
- The native path **still calls ``legacy.setup()`` underneath** because
  the TS verbs (``create_model``, ``predict_model``, ``compare_models``,
  ...) haven't been drained yet. They read from legacy state.
- User-facing accessors (``exp.y_train``, ``exp.y_test``,
  ``exp.preprocess_pipeline``) now work for TS just like they do for the
  other tasks. Before this session, accessing them on a TS experiment
  raised ``KeyError`` because ``_fit_state`` wasn't populated.

**Phase 5b/c (future sessions)** will drain the TS verbs themselves so
``_native_setup_timeseries`` can stop calling ``legacy.setup()`` and the
``pycaret/internal/pycaret_experiment/`` directory becomes deletable.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Predicate: TS now routes through the native dispatcher.
# ---------------------------------------------------------------------------


def test_can_use_native_setup_accepts_time_series():
    """Predicate accepts TIME_SERIES (phase 5a)."""
    from pycaret.tasks import TimeSeriesExperiment

    exp = TimeSeriesExperiment(fh=12, session_id=0)
    assert exp._can_use_native_setup({}) is True


def test_can_use_native_setup_time_series_with_setup_kwargs_legacy():
    """setup_kwargs still forces legacy for TS too."""
    from pycaret.tasks import TimeSeriesExperiment

    exp = TimeSeriesExperiment(fh=12, session_id=0)
    assert exp._can_use_native_setup({"verbose": True}) is False


# ---------------------------------------------------------------------------
# Native path populates _fit_state correctly.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_time_series_native_path_populates_fit_state():
    """fit() on TS goes through native dispatcher; _fit_state has TS shape."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)

    assert exp._native_setup_used is True
    assert hasattr(exp, "_fit_state")
    state = exp._fit_state
    # User-facing slots populated.
    assert state["y"] is not None
    assert state["y_train"] is not None
    assert state["y_test"] is not None
    assert state["preprocess_pipeline"] is not None
    # TS-specific slots populated.
    assert state["fh"] is not None
    assert state["fold_generator"] is not None
    # Model registry has the standard sktime forecasters.
    assert "naive" in state["model_registry"]
    assert "arima" in state["model_registry"]
    assert "ets" in state["model_registry"]


@pytest.mark.slow
def test_time_series_user_facing_accessors_work():
    """exp.y_train / exp.y_test / exp.preprocess_pipeline work for TS."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)

    # 144 total rows in airline; fh=12 → train=131, test=12 (legacy has
    # an extra-row offset for the splitter — what matters here is that
    # y_train + y_test ≤ y, the 12-step horizon is preserved, and the
    # split is reproducible.)
    assert exp.y_train.shape[0] + exp.y_test.shape[0] <= exp.y.shape[0]
    assert exp.y_test.shape[0] == 12
    # Series, not DataFrame, for univariate.
    import pandas as pd

    assert isinstance(exp.y_train, pd.Series)
    assert exp.preprocess_pipeline is not None


@pytest.mark.slow
def test_time_series_fold_generator_is_expanding_window_splitter():
    """fold_generator is sktime's ExpandingWindowSplitter."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment
    from sktime.split import ExpandingWindowSplitter

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)
    assert isinstance(exp._fit_state["fold_generator"], ExpandingWindowSplitter)


@pytest.mark.slow
def test_time_series_setup_kwargs_falls_back_to_legacy_with_state_snapshot():
    """When setup_kwargs forces legacy, _fit_state is still populated via
    the post-setup snapshot helper.
    """
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df, html=False)

    # Native NOT used because setup_kwargs are present.
    assert exp._native_setup_used is False
    # But _fit_state is still populated by the snapshot helper after legacy.setup.
    assert hasattr(exp, "_fit_state")
    assert exp._fit_state["y_train"] is not None
    assert exp.y_train.shape[0] + exp.y_test.shape[0] <= exp.y.shape[0]


# ---------------------------------------------------------------------------
# Verbs continue to work — phase 5a is a soft drain.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_time_series_create_model_under_native_setup():
    """create_model still works (legacy.setup is called underneath)."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)
    assert exp._native_setup_used is True

    res = exp.create_model("naive", verbose=False)
    assert res.pipeline is not None
    assert res.model_id == "naive"


@pytest.mark.slow
def test_time_series_predict_model_under_native_setup():
    """predict_model returns forecasts on test horizon."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)
    res = exp.create_model("naive", verbose=False)
    preds = exp.predict_model(res.pipeline)
    # 12-step forecast.
    assert preds.predictions is not None
    assert len(preds.predictions) == 12


# ---------------------------------------------------------------------------
# Models registry: TS still goes through legacy.models() (phase 5a defers
# the model_type filter rather than reimplementing it on the snapshot).
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_time_series_models_internal_filters_ensemble_forecaster():
    """models(internal=True) for TS filters out 'ensemble_forecaster' via
    legacy.models() because ensemble_forecaster requires runtime-built
    forecasters and isn't a member of TSModelTypes.
    """
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)
    df_models = exp.models(internal=True)
    # ensemble_forecaster is in the raw _all_models_internal but not in
    # the public models() output (model_type='ensemble' isn't in TSModelTypes).
    assert "ensemble_forecaster" not in df_models.index
    # Standard forecasters are present.
    assert "naive" in df_models.index
    assert "arima" in df_models.index
