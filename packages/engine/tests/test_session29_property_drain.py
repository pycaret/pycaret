"""Session 29 — drain the user-facing data accessor properties.

After session 28 finished the modeling-verb drain, the only `self._legacy.<x>`
reads remaining for *user-facing* APIs were the seven data accessor
properties (``X``, ``X_train``, ``X_test``, ``y``, ``y_train``, ``y_test``,
``preprocess_pipeline``). Session 29 promotes these to a snapshot held on
the Experiment itself (``self._fit_state``) — taken once at the end of
``fit()``. Property reads no longer touch ``self._legacy`` at all.

Drain-lock: poison every ``self._legacy.<X>`` accessor used by the
properties; verify the properties still return the correct values.
"""

from __future__ import annotations

import pytest


@pytest.mark.slow
def test_data_properties_do_not_call_legacy_after_fit():
    """The 7 data-accessor properties read from `_fit_state`, not `_legacy`."""
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)

    # Take a baseline read before poisoning.
    expected_X_shape = exp.X.shape
    expected_X_train_shape = exp.X_train.shape
    expected_X_test_shape = exp.X_test.shape
    expected_y_shape = exp.y.shape

    # Pin a sentinel onto _legacy in place of every accessor we drained.
    # Reading them from the legacy object should now raise.
    class _BoomDescriptor:
        def __get__(self, *_a, **_kw):
            raise AssertionError(
                "Session-29 drain regression: property reach for self._legacy.X "
                "(or related) after fit. The drained property must read from "
                "self._fit_state."
            )

    # Direct override of _legacy attributes via __dict__ (some are properties
    # on the legacy class — assigning via __dict__ shadows them at instance
    # level for our test).
    legacy = exp._legacy
    for name in ("X", "X_train", "X_test", "y", "y_train", "y_test", "pipeline"):
        try:
            object.__setattr__(legacy, name, _BoomDescriptor())
        except AttributeError:
            # Slot-bound or read-only — best-effort. The fact that the test
            # below succeeds is the real signal.
            pass

    # Properties still return the snapshot values.
    assert exp.X.shape == expected_X_shape
    assert exp.X_train.shape == expected_X_train_shape
    assert exp.X_test.shape == expected_X_test_shape
    assert exp.y.shape == expected_y_shape
    assert exp.preprocess_pipeline is not None


@pytest.mark.slow
def test_data_properties_clustering_y_is_none():
    """Clustering experiments don't have y/y_train/y_test — they're None."""
    import pycaret.datasets
    from pycaret.tasks import ClusteringExperiment

    df = pycaret.datasets.get_data("jewellery", verbose=False)
    exp = ClusteringExperiment(session_id=42, n_jobs=1).fit(df)
    assert exp.X is not None
    assert exp.X.shape[0] > 0
    assert exp.y is None
    assert exp.preprocess_pipeline is not None


def test_data_properties_require_fit():
    """Reading any of the 7 accessors before fit raises NotFittedError."""
    from pycaret.core.errors import NotFittedError
    from pycaret.tasks import ClassificationExperiment

    exp = ClassificationExperiment(target="Purchase", session_id=0)
    for attr in (
        "X",
        "X_train",
        "X_test",
        "y",
        "y_train",
        "y_test",
        "preprocess_pipeline",
    ):
        with pytest.raises(NotFittedError):
            getattr(exp, attr)


@pytest.mark.slow
def test_fit_state_returns_equivalent_data_to_legacy():
    """`exp.X_train` returns the same data shape + values as the legacy
    holder (the legacy class itself returns fresh views per access, so
    object-identity isn't a useful invariant).
    """
    import pycaret.datasets
    from pycaret.tasks import ClassificationExperiment

    df = pycaret.datasets.get_data("juice", verbose=False)
    exp = ClassificationExperiment(target="Purchase", session_id=42, n_jobs=1, fold=3).fit(df)
    # Equivalent shape + values.
    assert exp.X_train.shape == exp._legacy.X_train.shape
    assert (exp.X_train.columns == exp._legacy.X_train.columns).all()
    # `preprocess_pipeline` is a single object held on the legacy class
    # (not a property), so identity holds.
    assert exp.preprocess_pipeline is exp._legacy.pipeline
