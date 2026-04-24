"""PyCaret 4.0 structured event-stream logger.

Replaces the 3.x mlflow/comet/wandb/dagshub tracker adapters with a lean
in-process event stream designed to be consumed by:

- The forthcoming PyCaret React UI (over websocket or polling).
- LLM agents that reason over engine traces.
- Notebook users who want a human-readable log after a long run.

Two public types:

- `Event` — an immutable record of something that happened.
- `BaseLogger` — the hook interface. Default is an in-memory buffer; subclass
  for file output, custom sinks, or (in a later release) external trackers as
  optional adapters.

All callers inside `pycaret.core.*` / `pycaret.tasks.*` go through `BaseLogger`;
no direct `print` statements.
"""

from pycaret.logging.base import BaseLogger, NullLogger
from pycaret.logging.events import Event, EventKind
from pycaret.logging.memory import MemoryLogger

__all__ = [
    "BaseLogger",
    "Event",
    "EventKind",
    "MemoryLogger",
    "NullLogger",
]
