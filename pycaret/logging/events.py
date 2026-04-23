"""Structured event types emitted by PyCaret 4.0 operations.

Events are the single source of truth for "what happened during this run":

- Every `Experiment` method emits at least one event.
- Events are immutable dataclasses that can be serialized to JSON.
- A React UI subscribes to the stream (via `BaseLogger.subscribe(...)`) and
  renders progress; an LLM agent reads the same stream as trace data.

`EventKind` is a string-enum so events can round-trip through JSON without
custom encoders.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class EventKind(StrEnum):
    """Canonical event kinds. String-valued for JSON round-tripping."""

    # Experiment lifecycle
    EXPERIMENT_STARTED = "experiment.started"
    EXPERIMENT_FITTED = "experiment.fitted"  # after setup/preprocessing done
    EXPERIMENT_FINISHED = "experiment.finished"  # after close()

    # Preprocessing
    PREPROCESSOR_STARTED = "preprocessor.started"
    PREPROCESSOR_FITTED = "preprocessor.fitted"
    DATA_SPLIT = "data.split"  # train/test/holdout split

    # Model operations
    MODEL_CREATE_STARTED = "model.create.started"
    MODEL_CREATED = "model.created"
    MODEL_COMPARE_STARTED = "model.compare.started"
    MODEL_COMPARED = "model.compared"  # per-model, emits n times
    MODEL_COMPARE_FINISHED = "model.compare.finished"
    MODEL_TUNE_STARTED = "model.tune.started"
    MODEL_TUNED = "model.tuned"
    MODEL_ENSEMBLE_STARTED = "model.ensemble.started"
    MODEL_ENSEMBLED = "model.ensembled"
    MODEL_BLEND_STARTED = "model.blend.started"
    MODEL_BLENDED = "model.blended"
    MODEL_STACK_STARTED = "model.stack.started"
    MODEL_STACKED = "model.stacked"
    MODEL_CALIBRATE_STARTED = "model.calibrate.started"
    MODEL_CALIBRATED = "model.calibrated"
    MODEL_FINALIZED = "model.finalized"
    MODEL_PREDICTED = "model.predicted"
    MODEL_SAVED = "model.saved"
    MODEL_LOADED = "model.loaded"

    # Diagnostics
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class Event:
    """An immutable record of something that happened during an experiment.

    Attributes
    ----------
    kind : EventKind
        Canonical kind string.
    timestamp : float
        Unix seconds since the epoch, captured via `time.time()`.
    duration_ms : float, optional
        Wall-clock duration for "finished" events. `None` for point events.
    message : str
        Short human-readable summary.
    payload : dict
        Structured fields specific to this event kind (e.g. `{"model_id": "lr",
        "score": 0.83}`). JSON-serializable by convention.
    experiment_id : str, optional
        Stable id of the emitting experiment (defaults to the experiment's
        `session_id` stringified).
    """

    kind: EventKind
    message: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    duration_ms: float | None = None
    timestamp: float = field(default_factory=time.time)
    experiment_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation."""
        return {
            "kind": self.kind.value,
            "message": self.message,
            "payload": dict(self.payload),
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
            "experiment_id": self.experiment_id,
        }
