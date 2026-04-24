"""Bridge engine events into the platform database and the WebSocket broker.

`DBEventLogger` subclasses `pycaret.logging.BaseLogger` and overrides `emit()`
to:

1. Serialize the engine `Event` into our `events` table row.
2. Broadcast the same payload through `event_broker.publish(run_id, ...)` so
   connected WebSocket clients see live progress.

One instance is attached to one `Experiment` for the lifetime of one `Run`;
it holds a SQLAlchemy session dedicated to event writes (separate from the
HTTP request's session so concurrent commits don't deadlock).
"""

from __future__ import annotations

from datetime import UTC, datetime

from pycaret.logging import BaseLogger
from pycaret.logging.events import Event as EngineEvent

from pycaret_server.db import Event as DBEvent
from pycaret_server.db import get_session
from pycaret_server.runs.broker import event_broker


class DBEventLogger(BaseLogger):
    """Persist engine events into the `events` table + fan out to WebSocket subscribers.

    The worker thread owns its own short-lived SQLAlchemy `Session`. Every
    `emit()` opens a session, commits a single row, and closes it — simple
    and robust under SQLite which doesn't love long-held connections.
    """

    def __init__(self, run_id: str, experiment_id: str | None = None) -> None:
        super().__init__(experiment_id=experiment_id)
        self._run_id = run_id

    def emit(self, event: EngineEvent) -> None:  # noqa: D401
        payload = dict(event.payload) if event.payload else None

        # 1) Write to DB in a scoped session.
        session = get_session()
        try:
            row = DBEvent(
                run_id=self._run_id,
                kind=str(event.kind),
                message=event.message or None,
                payload=payload,
                duration_ms=event.duration_ms,
                emitted_at=datetime.fromtimestamp(event.timestamp, tz=UTC),
            )
            session.add(row)
            session.commit()
        except Exception:  # pragma: no cover — never let a bad row crash a run
            session.rollback()
        finally:
            session.close()

        # 2) Fan out to WebSocket subscribers.
        try:
            event_broker.publish(self._run_id, event.to_dict())
        except Exception:  # pragma: no cover
            pass
