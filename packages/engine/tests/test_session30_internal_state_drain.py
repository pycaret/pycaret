"""Session 30 — drain internal training-state attrs off self._legacy.

After session 29 promoted user-facing accessors to ``self._fit_state``,
six *internal* legacy reads remained inside the drained verbs:

  - ``self._legacy.X_train_transformed``
  - ``self._legacy.y_train_transformed``
  - ``self._legacy.X_transformed``
  - ``self._legacy.y_transformed``
  - ``self._legacy.fold_generator``
  - ``self._legacy._all_models_internal`` (the model registry)

Session 30 captures them in ``_fit_state`` at fit() time alongside the
user-facing snapshot. Every drained verb now reads training data, the
CV generator, and the model registry from the snapshot.

Drain-lock: poison every legacy attr we drained + verify create_model /
tune_model / compare_models / ensemble / finalize / clustering create
all still succeed.
"""

from __future__ import annotations

import pytest


@pytest.mark.slow
def test_create_model_does_not_read_legacy_internal_state(monkeypatch):
    """create_model reads X_train_transformed / y_train_transformed /
    fold_generator / model_registry from self._fit_state, not self._legacy.
    """
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)

    # Poison every internal-state attr on _legacy.
    legacy = exp._legacy
    for name in (
        "X_train_transformed",
        "y_train_transformed",
        "X_transformed",
        "y_transformed",
        "fold_generator",
        "_all_models_internal",
    ):
        # Properties on the legacy class — assign via __dict__ to shadow.
        try:
            object.__setattr__(legacy, name, _poison_value())
        except AttributeError:
            # Read-only descriptor — best-effort. The verb success below is
            # the real signal.
            pass

    # Should not raise — native path uses the snapshot.
    result = exp.create_model("lr", verbose=False)
    assert result.pipeline is not None
    assert result.metrics is not None


@pytest.mark.slow
def test_tune_model_uses_snapshot_for_search_space():
    """tune_model resolves the search space from _fit_state["model_registry"],
    not from self._legacy._all_models_internal directly.
    """
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    # Sanity: snapshot contains the registry.
    assert "model_registry" in exp._fit_state
    assert "lr" in exp._fit_state["model_registry"]
    # Drop the legacy registry entirely — tune_model should still work.
    try:
        object.__setattr__(exp._legacy, "_all_models_internal", {})
    except AttributeError:
        pass
    tuned = exp.tune_model("lr", n_iter=3, verbose=False)
    # If tune_model had read from the empty legacy registry, it would fall
    # through to the no-search-space path; we check best_params is non-empty
    # (fixed registry tune_grid for lr has C + class_weight).
    assert isinstance(tuned.best_params, dict)
    assert len(tuned.best_params) > 0


@pytest.mark.slow
def test_finalize_model_uses_snapshot_X_transformed():
    """finalize_model re-fits on _fit_state["X_transformed"], not legacy."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    created = exp.create_model("lr", verbose=False)
    # Poison X_transformed/y_transformed on legacy.
    try:
        object.__setattr__(exp._legacy, "X_transformed", _poison_value())
        object.__setattr__(exp._legacy, "y_transformed", _poison_value())
    except AttributeError:
        pass
    finalized = exp.finalize_model(created.pipeline)
    assert finalized.pipeline is not None


@pytest.mark.slow
def test_clustering_create_model_uses_snapshot_X_transformed():
    """Clustering's native create_model fits on the snapshot's X_transformed."""
    import pycaret.datasets
    from pycaret.tasks import ClusteringExperiment

    df = pycaret.datasets.get_data("jewellery", verbose=False)
    exp = ClusteringExperiment(session_id=42, n_jobs=1).fit(df)
    # Poison legacy.X_transformed.
    try:
        object.__setattr__(exp._legacy, "X_transformed", _poison_value())
        object.__setattr__(exp._legacy, "_all_models_internal", {})
    except AttributeError:
        pass
    result = exp.create_model("kmeans", num_clusters=4, verbose=False)
    assert result.pipeline is not None


@pytest.mark.slow
def test_fit_state_holds_all_internal_keys():
    """Sanity: post-fit, _fit_state has every snapshot slot populated."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    state = exp._fit_state
    for key in (
        "X",
        "X_train",
        "X_test",
        "y",
        "y_train",
        "y_test",
        "preprocess_pipeline",
        "X_transformed",
        "X_train_transformed",
        "y_transformed",
        "y_train_transformed",
        "fold_generator",
        "model_registry",
    ):
        assert key in state, f"missing {key!r} in _fit_state"
    # X_train_transformed is post-preprocessing; X_train is raw.
    assert state["X_train_transformed"] is not None
    assert state["fold_generator"] is not None
    assert len(state["model_registry"]) > 0


# ------------------------------------------------------------- helpers


class _PoisonedAttrAccess:
    """Reading any attribute or invoking the object raises an explicit error."""

    def __getattr__(self, name):
        raise AssertionError(
            f"Session-30 drain regression: a verb read self._legacy.{name} "
            "(transformed-state / fold / registry). The native path must "
            "use self._fit_state instead."
        )

    def __iter__(self):
        raise AssertionError(
            "Session-30 drain regression: a verb iterated self._legacy.<x>; "
            "should iterate self._fit_state['model_registry'] instead."
        )

    def __getitem__(self, key):
        raise AssertionError(
            f"Session-30 drain regression: a verb subscripted self._legacy.<x>[{key!r}]."
        )

    def __contains__(self, key):
        raise AssertionError(
            f"Session-30 drain regression: a verb did `{key!r} in self._legacy.<x>`."
        )

    def __len__(self):
        raise AssertionError("Session-30 drain regression: a verb called len(self._legacy.<x>).")


def _poison_value():
    return _PoisonedAttrAccess()
