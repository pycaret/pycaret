"""PyCaret 4.0 engine primitives.

This package hosts the task-agnostic core of the PyCaret 4.0 engine:

- `Experiment` — sklearn-compatible base class. Task-specific subclasses in
  `pycaret.tasks.*`.
- `SupervisedExperiment` / `UnsupervisedExperiment` — verb-set mixins used as
  intermediate bases.
- `TaskType` — enum of supported ML tasks.
- Result dataclasses — `CompareResult`, `CreateResult`, `TuneResult`, ...
- `PyCaretError` hierarchy — the engine exception surface.

See `docs/revamp/ARCHITECTURE.md` for the design rationale.
"""

from pycaret.core.errors import (
    ConfigurationError,
    NotFittedError,
    PyCaretError,
    UnknownMetricError,
    UnknownModelError,
)
from pycaret.core.experiment import Experiment
from pycaret.core.results import (
    BlendResult,
    CalibrateResult,
    CompareResult,
    CreateResult,
    EnsembleResult,
    FinalizeResult,
    PredictResult,
    StackResult,
    TuneResult,
)
from pycaret.core.supervised import SupervisedExperiment
from pycaret.core.tasks import TaskType
from pycaret.core.unsupervised import UnsupervisedExperiment

__all__ = [
    "BlendResult",
    "CalibrateResult",
    "CompareResult",
    "ConfigurationError",
    "CreateResult",
    "EnsembleResult",
    "Experiment",
    "FinalizeResult",
    "NotFittedError",
    "PredictResult",
    "PyCaretError",
    "StackResult",
    "SupervisedExperiment",
    "TaskType",
    "TuneResult",
    "UnknownMetricError",
    "UnknownModelError",
    "UnsupervisedExperiment",
]
