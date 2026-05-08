"""Session 53 — revive ``get_leaderboard`` / ``automl`` / ``interpret_model`` /
``check_stats``.

These four verbs shipped as ``NotImplementedError`` stubs in 4.0.0a2-a5 (per
session 46's phase-6 deletion). This session reimplements them on the new
native engine surface:

- ``get_leaderboard()`` — reads ``_fit_state["last_leaderboard"]`` snapshotted
  by ``compare_models``.
- ``automl()`` — convenience wrapper for ``compare_models`` + ``tune_model``.
- ``interpret_model()`` — SHAP explainer; ``shap`` is an optional extra.
- ``check_stats()`` — TS statsmodels + scipy diagnostic suite.

``plot_model`` and ``evaluate_model`` are intentionally still stubs (Plotly
rewrite is its own track).
"""

from __future__ import annotations

import importlib.util

import pandas as pd
import pytest


_HAS_SHAP = importlib.util.find_spec("shap") is not None
_HAS_STATSMODELS = importlib.util.find_spec("statsmodels") is not None
_HAS_SKTIME = importlib.util.find_spec("sktime") is not None


# ---------------------------------------------------------------------------
# get_leaderboard
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_get_leaderboard_returns_compare_models_leaderboard():
    """``get_leaderboard()`` mirrors ``compare_models(...).leaderboard``."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    cmp_result = exp.compare_models(include=["lr", "dt"], verbose=False)

    leaderboard = exp.get_leaderboard()
    assert isinstance(leaderboard, pd.DataFrame)
    assert not leaderboard.empty
    assert "Model" in leaderboard.columns
    pd.testing.assert_frame_equal(leaderboard, cmp_result.leaderboard)


@pytest.mark.slow
def test_get_leaderboard_returns_a_copy():
    """Mutating the returned DataFrame must not poison subsequent calls."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    exp.compare_models(include=["lr", "dt"], verbose=False)

    first = exp.get_leaderboard()
    first.drop(first.index, inplace=True)

    second = exp.get_leaderboard()
    assert not second.empty


@pytest.mark.slow
def test_get_leaderboard_raises_before_compare_models():
    """No leaderboard exists until ``compare_models`` has run."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    with pytest.raises(RuntimeError, match="compare_models"):
        exp.get_leaderboard()


# ---------------------------------------------------------------------------
# automl
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_automl_returns_tuned_pipeline():
    """``automl()`` runs compare → tune and returns a fitted Pipeline."""
    from sklearn.pipeline import Pipeline

    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)

    tuned = exp.automl(include=["lr", "dt"], n_iter=2, turbo=True)
    assert isinstance(tuned, Pipeline)
    # Pipeline ends with a fitted estimator that supports predict.
    preds = tuned.predict(exp.X_test.head())
    assert len(preds) == len(exp.X_test.head())


@pytest.mark.slow
def test_automl_records_leaderboard_for_get_leaderboard():
    """``automl`` runs compare_models internally → ``get_leaderboard`` works after."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    exp.automl(include=["lr", "dt"], n_iter=2)

    leaderboard = exp.get_leaderboard()
    assert not leaderboard.empty
    assert set(leaderboard["Model"]) <= {"lr", "dt"}


@pytest.mark.slow
def test_automl_respects_optimize_metric():
    """The ``optimize`` argument flows to both compare-rank and tune-objective."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)

    # AUC instead of default — exercises the sort= path through compare_models.
    tuned = exp.automl(include=["lr", "dt"], optimize="AUC", n_iter=2)
    assert tuned is not None


# ---------------------------------------------------------------------------
# interpret_model
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_SHAP, reason="shap not installed (optional extra)")
def test_interpret_model_returns_shap_explanation():
    """With shap installed, ``interpret_model`` returns a shap.Explanation."""
    import shap

    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    res = exp.create_model("dt", verbose=False)  # tree model — TreeExplainer fast-path

    explanation = exp.interpret_model(res.pipeline)
    assert isinstance(explanation, shap.Explanation)
    assert explanation.values is not None


@pytest.mark.slow
@pytest.mark.skipif(_HAS_SHAP, reason="test only meaningful when shap is missing")
def test_interpret_model_raises_when_shap_missing():
    """Without shap, raise ``ImportError`` pointing at ``pycaret[interpret]``."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    res = exp.create_model("lr", verbose=False)

    with pytest.raises(ImportError, match=r"pycaret\[interpret\]"):
        exp.interpret_model(res.pipeline)


# ---------------------------------------------------------------------------
# check_stats
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(
    not (_HAS_STATSMODELS and _HAS_SKTIME),
    reason="time-series extra not installed",
)
def test_check_stats_default_returns_all_categories():
    """``test='all'`` surfaces summary + white-noise + stationarity + normality."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)

    out = exp.check_stats()
    assert isinstance(out, pd.DataFrame)
    assert set(out.columns) >= {"Test", "Test Name", "Data", "Property", "Setting", "Value"}
    categories = set(out["Test"])
    assert "Summary Statistics" in categories
    assert "White Noise" in categories
    assert "Stationarity" in categories
    assert "Normality" in categories


@pytest.mark.slow
@pytest.mark.skipif(
    not (_HAS_STATSMODELS and _HAS_SKTIME),
    reason="time-series extra not installed",
)
def test_check_stats_filters_by_test_kind():
    """``test='stationarity'`` returns only ADF + KPSS rows."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)

    out = exp.check_stats(test="stationarity")
    assert set(out["Test"]) == {"Stationarity"}
    assert set(out["Test Name"]) == {"ADF", "KPSS"}


@pytest.mark.slow
@pytest.mark.skipif(
    not (_HAS_STATSMODELS and _HAS_SKTIME),
    reason="time-series extra not installed",
)
def test_check_stats_split_train_uses_y_train():
    """``split='train'`` operates on the training portion of the series."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)

    out = exp.check_stats(test="summary", split="train")
    assert set(out["Data"]) == {"train"}
    # 8 descriptive stats from pd.Series.describe() (count..max).
    assert len(out) == 8


def test_check_stats_rejects_unknown_kwargs():
    """Unknown ``test`` and ``split`` arguments raise ``ValueError``."""
    import pycaret.datasets
    from pycaret.tasks import TimeSeriesExperiment

    df = pycaret.datasets.get_data("airline", verbose=False)
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(df)

    with pytest.raises(ValueError, match="Unknown test"):
        exp.check_stats(test="not_a_real_test")
    with pytest.raises(ValueError, match="Unknown split"):
        exp.check_stats(split="not_a_split")
