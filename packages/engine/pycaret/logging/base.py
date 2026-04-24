"""Logger interface + default null implementation for PyCaret 4.0.

`BaseLogger` is the hook surface an Experiment uses to emit structured events.
Subclass it to:

- Persist events to disk / a database.
- Forward events to mlflow / comet / a custom tracker (as a separate package).
- Power a React UI via websocket fan-out.

The default instance is `NullLogger`, which silently drops everything. Use
`MemoryLogger` (from `pycaret.logging.memory`) when you want events replayable
in the notebook or consumed by an in-process UI.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pycaret.logging.events import Event, EventKind


class BaseLogger:
    """The logger hook surface.

    Subclass and override `emit(event)` to customise what happens to events.
    `BaseLogger.emit` is a no-op, so the default subclass is the null logger.

    `log(...)` is the convenience entry point: callers construct an `Event`
    from kwargs. If you only need to persist or route events, override `emit`;
    you don't need to touch `log`.
    """

    def __init__(self, experiment_id: str | None = None) -> None:
        self._experiment_id = experiment_id
        self._subscribers: list[Callable[[Event], None]] = []

    # ---------- public emit surface ----------

    def log(
        self,
        kind: EventKind,
        *,
        message: str = "",
        payload: dict[str, Any] | None = None,
        duration_ms: float | None = None,
    ) -> Event:
        """Build an `Event` and call `emit`. Returns the constructed event so
        callers can reuse its timestamp/metadata."""
        event = Event(
            kind=kind,
            message=message,
            payload=payload or {},
            duration_ms=duration_ms,
            experiment_id=self._experiment_id,
        )
        self.emit(event)
        for cb in list(self._subscribers):
            try:
                cb(event)
            except Exception:  # pragma: no cover — don't let subscribers break emission
                pass
        return event

    def emit(self, event: Event) -> None:
        """Consume an event. Default: no-op (null logger)."""

    # ---------- subscriber pattern for UI fan-out ----------

    def subscribe(self, callback: Callable[[Event], None]) -> Callable[[], None]:
        """Register a callback that receives every emitted event.

        Returns an unsubscribe function. This is how a React-UI backend or a
        long-running notebook cell attaches itself to the engine's event stream.
        """
        self._subscribers.append(callback)

        def _unsubscribe() -> None:
            try:
                self._subscribers.remove(callback)
            except ValueError:
                pass

        return _unsubscribe

    # ---------- backward-compat shim ----------
    #
    # 3.x code paths call `logger.log_experiment(...)`, `log_model(...)`, etc.
    # directly on a `BaseLogger`. Phase 1 of the 4.0 revamp collapsed those
    # external-tracker adapters; the shim methods below keep the call sites
    # (inside the legacy god-class, which the new Experiment delegates to)
    # working until Phase 5 retires that code. They are no-ops here.

    def log_experiment(self, *args, **kwargs) -> None: ...
    def log_model(self, *args, **kwargs) -> None: ...
    def log_model_comparison(self, *args, **kwargs) -> None: ...
    def log_plot(self, *args, **kwargs) -> None: ...
    def log_params(self, *args, **kwargs) -> None: ...
    def log_metrics(self, *args, **kwargs) -> None: ...
    def log_artifact(self, *args, **kwargs) -> None: ...
    def log_hpram_grid(self, *args, **kwargs) -> None: ...
    def log_sklearn_pipeline(self, *args, **kwargs) -> None: ...
    def init_logger(self) -> None: ...
    def init_experiment(self, *args, **kwargs) -> None: ...
    def finish_experiment(self) -> None: ...
    def set_tags(self, *args, **kwargs) -> None: ...

    @property
    def loggers(self) -> list[BaseLogger]:
        """Kept for symmetry with the removed 3.x `DashboardLogger` fan-out."""
        return [self]


class NullLogger(BaseLogger):
    """Discards every event. Default logger for experiments that don't opt in."""


# Export a module-level singleton so callers can pass a default cheaply.
NULL = NullLogger()
