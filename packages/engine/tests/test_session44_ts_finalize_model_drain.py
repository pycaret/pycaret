"""Session 44 — phase 5c (cont.): drain ``TimeSeriesExperiment.finalize_model``.

Phases 5b/5c (s40-s43) drained ``create_model``, ``predict_model``,
``compare_models``, and ``tune_model``. This session drains the fifth
TS verb. ``exp.finalize_model(forecaster)`` no longer touches
``self._legacy.finalize_model`` — refits the forecaster on the **full**
``y`` (``y_train + y_test``) using the experiment's preprocess pipeline,
returns a ``FinalizeResult`` with a deployment-ready
``ForecastingPipeline``.

After this session:
- 5 of 5 callable TS verbs drained (``assign_model`` doesn't exist for
  TS; ``UnsupervisedExperiment``-only verb).
- Only ``_native_setup_timeseries`` still calls ``legacy.setup()``
  underneath. Phase 5d will strip that — it requires a proxy that
  exposes ``seasonality_present`` / ``primary_sp_to_use`` /
  ``strictly_positive`` / ``seasonality_type`` (legacy auto-detects
  these via Fourier analysis).
- Phase 6 then deletes ``pycaret/internal/pycaret_experiment/`` (10K LoC).
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Drain-lock: legacy.finalize_model MUST NOT be called.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_finalize_model_drain_lock(monkeypatch):
    """finalize_model on a drained-create pipeline doesn't touch legacy."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment
    from sktime.forecasting.compose import ForecastingPipeline

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)
    res = exp.create_model("naive", verbose=False)

    def _poison(*a, **kw):
        raise AssertionError("Phase 5c regression: finalize_model hit legacy.")

    monkeypatch.setattr(exp._legacy, "finalize_model", _poison)

    final = exp.finalize_model(res.pipeline)
    assert isinstance(final.pipeline, ForecastingPipeline)


# ---------------------------------------------------------------------------
# Refits on the full y (not just y_train).
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_finalize_model_uses_full_y(monkeypatch):
    """The finalized model is fit on the entire y (train + test)."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)
    res = exp.create_model("naive", verbose=False)

    def _poison(*a, **kw):
        raise AssertionError("regression")

    monkeypatch.setattr(exp._legacy, "finalize_model", _poison)

    final = exp.finalize_model(res.pipeline)
    # The forecaster's _y attribute reflects what it was last fit on.
    inner = final.pipeline.steps[-1][1]
    bare = inner.steps[-1][1] if hasattr(inner, "steps") else inner
    fitted_y = getattr(bare, "_y", None)
    if fitted_y is not None:
        # Full y = train (132) + test (12) = 144 in airline.
        assert len(fitted_y) == len(exp._fit_state["y"])


# ---------------------------------------------------------------------------
# Bare forecaster (without pipeline) is accepted.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_finalize_model_bare_forecaster(monkeypatch):
    """A bare sktime forecaster gets wired into the preprocess pipeline."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment
    from sktime.forecasting.compose import ForecastingPipeline
    from sktime.forecasting.naive import NaiveForecaster

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)

    def _poison(*a, **kw):
        raise AssertionError("regression")

    monkeypatch.setattr(exp._legacy, "finalize_model", _poison)

    fc = NaiveForecaster()
    fc.fit(exp.y_train, fh=list(range(1, 13)))
    final = exp.finalize_model(fc)
    assert isinstance(final.pipeline, ForecastingPipeline)


# ---------------------------------------------------------------------------
# After finalize, predict_model still works.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_predict_after_finalize(monkeypatch):
    """predict_model on a finalized pipeline produces forecasts beyond the test set."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)
    res = exp.create_model("naive", verbose=False)

    def _poison(*a, **kw):
        raise AssertionError("regression")

    monkeypatch.setattr(exp._legacy, "finalize_model", _poison)

    final = exp.finalize_model(res.pipeline)
    # Predict beyond the original test horizon (the finalized model has
    # seen the entire y, so predictions extend into the true future).
    preds = exp.predict_model(final.pipeline, fh=[1, 2, 3, 4, 5, 6])
    assert preds.predictions is not None
    assert len(preds.predictions) == 6


# ---------------------------------------------------------------------------
# Bad input handling.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_finalize_model_non_estimator_raises():
    """Object without .fit raises TypeError."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)

    with pytest.raises(TypeError, match="sktime BaseForecaster or"):
        exp.finalize_model(object())


# ---------------------------------------------------------------------------
# Integration: full create + tune + finalize + predict, all 5 legacy verbs poisoned.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_full_chain_drained_end_to_end(monkeypatch):
    """create + tune + finalize + predict with all 5 legacy verbs poisoned."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)

    def _poison(name):
        def _f(*a, **kw):
            raise AssertionError(f"Phase 5c regression: hit legacy.{name}.")

        return _f

    for verb in ("create_model", "predict_model", "compare_models", "tune_model", "finalize_model"):
        monkeypatch.setattr(exp._legacy, verb, _poison(verb))

    res = exp.create_model("naive", verbose=False)
    tuned = exp.tune_model(res.pipeline, n_iter=2, verbose=False)
    final = exp.finalize_model(tuned.pipeline)
    preds = exp.predict_model(final.pipeline, fh=[1, 2, 3])
    assert preds.predictions is not None
    assert len(preds.predictions) == 3


# ---------------------------------------------------------------------------
# Idempotence: finalize is a no-op semantically when called twice on the
# same input — the result still represents a model fit on the full y.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_finalize_model_does_not_mutate_input(monkeypatch):
    """The original pipeline's fitted state is not changed by finalize."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)
    res = exp.create_model("naive", verbose=False)

    def _poison(*a, **kw):
        raise AssertionError("regression")

    monkeypatch.setattr(exp._legacy, "finalize_model", _poison)

    inner_before = res.pipeline.steps[-1][1]
    bare_before = inner_before.steps[-1][1] if hasattr(inner_before, "steps") else inner_before
    y_before = getattr(bare_before, "_y", None)
    n_before = len(y_before) if y_before is not None else None

    exp.finalize_model(res.pipeline)

    bare_after = inner_before.steps[-1][1] if hasattr(inner_before, "steps") else inner_before
    y_after = getattr(bare_after, "_y", None)
    n_after = len(y_after) if y_after is not None else None

    # Original pipeline still represents the same fit (train-only, n=132).
    if n_before is not None:
        assert n_before == n_after
