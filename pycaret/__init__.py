"""PyCaret 4.0 — lean, modern, agent-native open-source AutoML engine.

Canonical 4.0 API
-----------------

    from pycaret.tasks import (
        ClassificationExperiment,
        RegressionExperiment,
        ClusteringExperiment,
        AnomalyExperiment,
        TimeSeriesExperiment,
    )
    from pycaret import save_model, load_model

    exp = ClassificationExperiment(target="y", session_id=42).fit(df)
    best = exp.compare_models().best
    preds = exp.predict_model(best).predictions
    save_model(best, "best.pkl")

Agent / UI introspection
------------------------

    from pycaret.api import (
        list_models, describe_model,
        list_metrics, describe_setup_params,
    )

Event-stream logging
--------------------

    from pycaret.logging import MemoryLogger, BaseLogger, EventKind
"""

import sys

from pycaret.persistence import load_model, save_model
from pycaret.utils._show_versions import show_versions

__version__ = "4.0.0.dev0"
version_ = __version__

__all__ = [
    "__version__",
    "load_model",
    "save_model",
    "show_versions",
]
