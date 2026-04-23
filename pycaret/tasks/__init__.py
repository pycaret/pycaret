"""Task-specific PyCaret 4.0 experiments.

Each task subclass pre-configures its `TaskType` and `__sklearn_tags__`. The
legacy import paths (`pycaret.classification.ClassificationExperiment` etc.)
re-export from here so backward-compatible docs still work.
"""

from pycaret.tasks.anomaly import AnomalyExperiment
from pycaret.tasks.classification import ClassificationExperiment
from pycaret.tasks.clustering import ClusteringExperiment
from pycaret.tasks.regression import RegressionExperiment
from pycaret.tasks.time_series import TimeSeriesExperiment

__all__ = [
    "AnomalyExperiment",
    "ClassificationExperiment",
    "ClusteringExperiment",
    "RegressionExperiment",
    "TimeSeriesExperiment",
]
