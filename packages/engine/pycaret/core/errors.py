"""PyCaret 4.0 exception hierarchy.

Every error an engine operation raises should be (a subclass of) `PyCaretError`
so that a UI / agent caller can catch the engine's errors without also catching
unrelated upstream library errors.
"""

from __future__ import annotations


class PyCaretError(Exception):
    """Root of the PyCaret engine exception hierarchy."""


class ConfigurationError(PyCaretError, ValueError):
    """Raised when an experiment is configured with inconsistent parameters.

    Inherits from `ValueError` as well so existing sklearn-style try/except
    blocks continue to work.
    """


class NotFittedError(PyCaretError, RuntimeError):
    """Raised when an operation is invoked on an unfitted Experiment.

    Inherits from `RuntimeError` to match sklearn's `NotFittedError` behaviour
    (which is itself a `ValueError` — we stop short of that because it's
    semantically closer to a runtime-state mismatch).
    """


class UnknownModelError(PyCaretError, KeyError):
    """Raised when `create_model(id)` is called with an id not in the registry."""


class UnknownMetricError(PyCaretError, KeyError):
    """Raised when a metric name is referenced that isn't in the registry."""
