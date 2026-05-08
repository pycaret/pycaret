"""Session 46 — phase 6: delete ``pycaret/internal/pycaret_experiment/``.

The legacy directory (10K LoC) and the five ``oop.py`` thin wrappers
(12K LoC across classification / regression / clustering / anomaly /
time_series/forecasting) were deleted. The default 4.0 workflow runs
without ever importing legacy code.

Phase-6 trade-offs:
- ``setup_kwargs`` no longer fall through to a legacy escape hatch —
  they raise ``ConfigurationError`` (the kwargs were the only paths into
  legacy ``setup()``).
- All six legacy-only verbs (``plot_model``, ``evaluate_model``,
  ``interpret_model``, ``automl``, ``get_leaderboard``, ``check_stats``)
  originally shipped as ``NotImplementedError`` stubs and have since been
  reimplemented — those positive-path tests live in their own session
  files. This file only locks the deletion of the legacy directory and
  the shim namespace.
- ``_LegacyShim`` exists only as a back-compat namespace for the
  drain-lock test pattern; production code never reads off it.

After this session, MVP 1 (engine) is feature-complete for 4.0.0.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Import-side: legacy modules no longer importable.
# ---------------------------------------------------------------------------


def test_legacy_internal_directory_deleted():
    """``pycaret.internal.pycaret_experiment`` no longer importable."""
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("pycaret.internal.pycaret_experiment")


def test_legacy_classification_oop_deleted():
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("pycaret.classification.oop")


def test_legacy_regression_oop_deleted():
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("pycaret.regression.oop")


def test_legacy_clustering_oop_deleted():
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("pycaret.clustering.oop")


def test_legacy_anomaly_oop_deleted():
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("pycaret.anomaly.oop")


def test_legacy_ts_oop_deleted():
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("pycaret.time_series.forecasting.oop")


def test_4x_oop_classes_still_importable():
    """The 4.0 ``Experiment`` classes are unaffected by the deletion."""
    from pycaret.tasks import (
        AnomalyExperiment,
        ClassificationExperiment,
        ClusteringExperiment,
        RegressionExperiment,
        TimeSeriesExperiment,
    )

    for cls in (
        ClassificationExperiment,
        RegressionExperiment,
        ClusteringExperiment,
        AnomalyExperiment,
        TimeSeriesExperiment,
    ):
        assert callable(cls)


# ---------------------------------------------------------------------------
# Back-compat shim: drain-lock pattern still works.
# ---------------------------------------------------------------------------


def test_legacy_shim_has_verb_attrs():
    """``_LegacyShim`` predefines all verb names so monkeypatch works."""
    from pycaret.core.experiment import _LegacyShim

    shim = _LegacyShim()
    for name in (
        "setup",
        "create_model",
        "predict_model",
        "compare_models",
        "tune_model",
        "finalize_model",
        "ensemble_model",
        "blend_models",
        "stack_models",
        "calibrate_model",
        "assign_model",
        "save_model",
        "load_model",
    ):
        assert hasattr(shim, name), f"_LegacyShim missing verb {name!r}"
        # Each is a no-op callable.
        assert callable(getattr(shim, name))
