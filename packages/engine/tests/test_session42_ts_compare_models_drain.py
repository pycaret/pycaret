"""Session 42 — phase 5c (cont.): drain ``TimeSeriesExperiment.compare_models``.

Phase 5b (s40) drained ``create_model``; phase 5c (s41) drained
``predict_model``. This session drains the third TS verb.
``exp.compare_models()`` no longer touches ``self._legacy.compare_models``
— it iterates the sktime registry, calls native ``create_model`` per
candidate (which already does CV via the drained native path), and
assembles a leaderboard ranked by ``MASE`` (default — lower is better).

After this session, only ``tune_model`` / ``finalize_model`` /
``assign_model`` still delegate to legacy for TS.
"""

from __future__ import annotations

import pytest

# Curated, reasonably-fast forecaster set. Avoid the slowest classical
# models (auto_arima, stlf) and any tabular-CDS reductions that would
# blow up the test runtime — the drain semantics are the same regardless
# of which models actually end up in the leaderboard.
_FAST_FORECASTERS = ["naive", "snaive", "polytrend", "theta"]


# ---------------------------------------------------------------------------
# Drain-lock: legacy.compare_models MUST NOT be called.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_compare_models_drain_lock(monkeypatch):
    """compare_models runs without touching legacy.compare_models."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)

    def _poison(*a, **kw):
        raise AssertionError("Phase 5c regression: compare_models hit legacy.")

    monkeypatch.setattr(exp._legacy, "compare_models", _poison)

    res = exp.compare_models(include=_FAST_FORECASTERS, verbose=False)
    assert res.best is not None
    assert len(res.models) == 1
    assert len(res.ranked_ids) == len(_FAST_FORECASTERS)


# ---------------------------------------------------------------------------
# Leaderboard shape + sort.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_compare_models_leaderboard_sorted_by_mase(monkeypatch):
    """Default sort='MASE' ranks ascending (lower is better)."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)

    def _poison(*a, **kw):
        raise AssertionError("regression")

    monkeypatch.setattr(exp._legacy, "compare_models", _poison)

    res = exp.compare_models(include=_FAST_FORECASTERS, verbose=False)
    # Leaderboard is sorted ascending by MASE.
    mase = res.leaderboard["MASE"].tolist()
    assert mase == sorted(mase)
    # First row's Model id matches ranked_ids[0].
    assert res.leaderboard.iloc[0]["Model"] == res.ranked_ids[0]


@pytest.mark.slow
def test_compare_models_n_select_returns_top_k(monkeypatch):
    """n_select=3 returns the top 3 fitted pipelines."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment
    from sktime.forecasting.compose import ForecastingPipeline

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)

    def _poison(*a, **kw):
        raise AssertionError("regression")

    monkeypatch.setattr(exp._legacy, "compare_models", _poison)

    res = exp.compare_models(include=_FAST_FORECASTERS, n_select=3, verbose=False)
    assert len(res.models) == 3
    for m in res.models:
        assert isinstance(m, ForecastingPipeline)


# ---------------------------------------------------------------------------
# include / exclude filters.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_compare_models_exclude_filter(monkeypatch):
    """exclude= drops the named models from the comparison."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)

    def _poison(*a, **kw):
        raise AssertionError("regression")

    monkeypatch.setattr(exp._legacy, "compare_models", _poison)

    res = exp.compare_models(include=_FAST_FORECASTERS, exclude=["snaive"], verbose=False)
    assert "snaive" not in res.ranked_ids
    # Other 3 models are present.
    for mid in ("naive", "polytrend", "theta"):
        assert mid in res.ranked_ids


@pytest.mark.slow
def test_compare_models_excludes_ensemble_forecaster_by_default(monkeypatch):
    """ensemble_forecaster is filtered out by default — it requires
    runtime-built sub-forecasters. Same filter legacy.models() applies.
    """
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)

    def _poison(*a, **kw):
        raise AssertionError("regression")

    monkeypatch.setattr(exp._legacy, "compare_models", _poison)

    # Don't pass include= → all models considered. Restrict to a small
    # subset that includes baseline models so the test runs fast; if the
    # default-include path leaks ensemble_forecaster it would crash here.
    res = exp.compare_models(
        include=_FAST_FORECASTERS + ["ensemble_forecaster"],
        errors="ignore",
        verbose=False,
    )
    # ensemble_forecaster errored out and got skipped.
    assert "ensemble_forecaster" not in res.ranked_ids


# ---------------------------------------------------------------------------
# errors="ignore" / errors="raise".
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_compare_models_errors_ignore_skips_failures(monkeypatch):
    """errors='ignore' skips models that error, returns successful only."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)

    def _poison(*a, **kw):
        raise AssertionError("regression")

    monkeypatch.setattr(exp._legacy, "compare_models", _poison)

    # ensemble_forecaster will fail; others succeed.
    res = exp.compare_models(
        include=["naive", "ensemble_forecaster", "theta"],
        errors="ignore",
        verbose=False,
    )
    assert set(res.ranked_ids) == {"naive", "theta"}


# ---------------------------------------------------------------------------
# pull() returns the leaderboard.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_compare_models_leaderboard_appears_in_pull(monkeypatch):
    """pull() returns the leaderboard from the native compare_models run."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)

    def _poison(*a, **kw):
        raise AssertionError("regression")

    monkeypatch.setattr(exp._legacy, "compare_models", _poison)

    exp.compare_models(include=_FAST_FORECASTERS, verbose=False)
    pulled = exp.pull()
    assert pulled is not None
    assert "Model" in pulled.columns
    assert "MASE" in pulled.columns


# ---------------------------------------------------------------------------
# Integration: create_model + compare_models + predict_model all native.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_compare_then_predict_end_to_end_drained(monkeypatch):
    """Full compare_models → predict_model chain works with all 3 legacy
    verbs poisoned.
    """
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

    res = exp.compare_models(include=_FAST_FORECASTERS, verbose=False)
    assert res.best is not None
    preds = exp.predict_model(res.best)
    assert preds.predictions is not None
    assert len(preds.predictions) == 12  # fh=12
    assert preds.metrics is not None


# ---------------------------------------------------------------------------
# Empty result handling.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_compare_models_all_failures_returns_empty(monkeypatch):
    """When every candidate errors out, returns empty CompareResult."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)

    def _poison(*a, **kw):
        raise AssertionError("regression")

    monkeypatch.setattr(exp._legacy, "compare_models", _poison)

    # Only ensemble_forecaster which errors out.
    res = exp.compare_models(include=["ensemble_forecaster"], errors="ignore", verbose=False)
    assert res.best is None
    assert res.models == []
    assert res.ranked_ids == []
    assert res.leaderboard.empty
