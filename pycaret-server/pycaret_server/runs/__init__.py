"""Run-execution subsystem for the PyCaret platform.

Exposes one class:

- `RunOrchestrator` — submits runs to an in-process thread pool, wires the
  engine's `BaseLogger` to the database as `Event` rows, writes artifacts,
  and broadcasts events to WebSocket subscribers via the `event_broker`.

Submodules:

- `broker`      — async/thread-safe event broker (fan-out to WebSockets).
- `logger_bridge` — `DBEventLogger(BaseLogger)` persisting events + notifying the broker.
- `orchestrator` — the worker pool + plan dispatcher.
- `plans`       — pure functions that turn a `RunPlan` into engine verb calls.
"""

from pycaret_server.runs.broker import EventBroker, event_broker
from pycaret_server.runs.orchestrator import RunOrchestrator, get_orchestrator

__all__ = [
    "EventBroker",
    "RunOrchestrator",
    "event_broker",
    "get_orchestrator",
]
