"""`UnsupervisedExperiment` — base for clustering and anomaly detection.

Unsupervised tasks don't have a target column. They add `assign_model` — the
verb that labels every row in the dataset with its predicted cluster / anomaly
score using a fitted model.
"""

from __future__ import annotations

from typing import Any

from pycaret.core.experiment import Experiment
from pycaret.logging.events import EventKind


class UnsupervisedExperiment(Experiment):
    """Base for clustering / anomaly experiments."""

    def _is_supervised(self) -> bool:  # override
        return False

    def assign_model(self, estimator: Any, *args: Any, **kwargs: Any) -> Any:
        """Label every row in the dataset with its predicted cluster/anomaly."""
        self._require_fitted()
        out = self._legacy.assign_model(estimator, *args, **kwargs)
        self.logger.log(EventKind.MODEL_PREDICTED, payload={"shape": getattr(out, "shape", None)})
        return out

    # Predict / plot / evaluate inherit from Experiment via self._legacy.
