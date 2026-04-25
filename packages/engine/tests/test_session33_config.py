"""Session 33 — drain get_config / set_config off self._legacy.

After session 32, `add_metric` / `remove_metric` were drained. The only
secondary verbs left were `get_config` / `set_config` (escape hatches
for advanced users) and `plot_model` / `evaluate_model` (Phase 3
Plotly-native rewrite).

Session 33 drains `get_config` / `set_config`:
- ``get_config(name)`` reads from ``self._fit_state`` (data accessors,
  transformed splits, registries) + constructor params on ``self``.
  Raises ``ValueError`` for unknown names.
- ``set_config(name, value)`` writes to a small allowlist of constructor
  params (`_SETTABLE_CONFIG_KEYS`). Mutating snapshot-invalidating
  params raises with a pointer to re-creating the Experiment.
"""

from __future__ import annotations

import pytest


@pytest.mark.slow
def test_get_config_returns_known_variables(monkeypatch):
    """get_config(None) returns the full accessible-names list."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)

    def _poison(*a, **kw):
        raise AssertionError("Session-33 drain regression: get_config called legacy.")

    monkeypatch.setattr(exp._legacy, "get_config", _poison)

    names = exp.get_config()
    assert isinstance(names, list)
    # Sanity — every snapshot key + key constructor params should be there.
    expected = {"X", "X_train", "X_test", "y", "y_train", "y_test", "session_id", "n_jobs"}
    assert expected.issubset(set(names))


@pytest.mark.slow
def test_get_config_returns_snapshot_state():
    """get_config('X_train') returns the fit-time X_train snapshot."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    assert exp.get_config("X_train") is exp.X_train
    assert exp.get_config("y_test") is exp.y_test


@pytest.mark.slow
def test_get_config_supports_seed_pipeline_aliases():
    """Aliases: 'seed' → session_id; 'pipeline' → preprocess_pipeline."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    assert exp.get_config("seed") == 42
    assert exp.get_config("pipeline") is exp.preprocess_pipeline


@pytest.mark.slow
def test_get_config_unknown_raises():
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    with pytest.raises(ValueError, match="not found"):
        exp.get_config("zzzz_not_real")


@pytest.mark.slow
def test_set_config_mutates_settable_keys(monkeypatch):
    """set_config('n_jobs', 4) updates the constructor param."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)

    def _poison(*a, **kw):
        raise AssertionError("Session-33 drain regression: set_config called legacy.")

    monkeypatch.setattr(exp._legacy, "set_config", _poison)

    exp.set_config("n_jobs", 4)
    assert exp.n_jobs == 4
    assert exp.get_config("n_jobs") == 4


@pytest.mark.slow
def test_set_config_bulk_via_kwargs():
    """set_config(verbose=True, fold=5) updates multiple keys at once."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    exp.set_config(verbose=True, fold=5)
    assert exp.verbose is True
    assert exp.fold == 5


@pytest.mark.slow
def test_set_config_rejects_non_settable_keys():
    """`set_config('target', 'newcol')` raises — that would invalidate the snapshot."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    with pytest.raises(ValueError, match="not settable"):
        exp.set_config("target", "newcol")


@pytest.mark.slow
def test_set_config_rejects_underscore_prefix():
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    with pytest.raises(ValueError, match="read-only"):
        exp.set_config("_fitted", False)


@pytest.mark.slow
def test_set_config_rejects_mixed_variable_and_kwargs():
    """Cannot pass both positional `variable` + kwargs."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    with pytest.raises(ValueError, match="cannot be used together"):
        exp.set_config("n_jobs", 4, fold=5)


def test_config_verbs_require_fit():
    from pycaret.core.errors import NotFittedError
    from pycaret.tasks import ClassificationExperiment

    exp = ClassificationExperiment(target="Purchase", session_id=0)
    with pytest.raises(NotFittedError):
        exp.get_config()
    with pytest.raises(NotFittedError):
        exp.set_config("n_jobs", 4)
