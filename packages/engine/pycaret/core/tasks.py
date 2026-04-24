"""TaskType — the enum of ML tasks PyCaret supports.

Using a real Enum instead of a string makes it easy for an LLM agent or a React
UI to enumerate what's available without guessing.
"""

from __future__ import annotations

from enum import StrEnum


class TaskType(StrEnum):
    """The machine-learning task a PyCaret experiment is configured for.

    Inheriting from `str` so that instances serialize cleanly to JSON and
    survive pickling without a custom __reduce__.
    """

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY = "anomaly"
    TIME_SERIES = "time_series"

    @property
    def is_supervised(self) -> bool:
        return self in (TaskType.CLASSIFICATION, TaskType.REGRESSION, TaskType.TIME_SERIES)

    @property
    def is_classification(self) -> bool:
        return self is TaskType.CLASSIFICATION

    @property
    def is_regression(self) -> bool:
        return self is TaskType.REGRESSION

    def __str__(self) -> str:  # pragma: no cover — convenience
        return self.value
