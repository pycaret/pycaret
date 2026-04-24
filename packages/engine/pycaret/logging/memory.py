"""In-memory + optional file-backed logger.

The default logger when a user does `Experiment(log=True)`. Stores every event
in a list so the UI / notebook can replay it afterwards, and optionally tees
each event to a JSON-lines file.

Cheap, thread-safe enough for single-process use (the event list uses a list
append which is atomic in CPython), and dependency-free.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from pycaret.logging.base import BaseLogger
from pycaret.logging.events import Event


class MemoryLogger(BaseLogger):
    """Captures emitted events in an in-memory list and (optionally) a JSONL file.

    Parameters
    ----------
    experiment_id : str, optional
        Stable id stamped on every event for correlating multi-experiment runs.
    file : str | Path, optional
        When provided, each event is also appended as a JSON line to this path.
        The file is flushed on every write so a UI tailing it sees progress.

    Usage
    -----
    >>> logger = MemoryLogger()
    >>> exp = ClassificationExperiment(target="y", logger=logger).fit(data)
    >>> exp.compare_models()
    >>> logger.events          # list of Event dataclasses
    >>> logger.as_jsonl()      # "\n"-joined JSONL string
    """

    def __init__(
        self,
        experiment_id: str | None = None,
        *,
        file: str | Path | None = None,
    ) -> None:
        super().__init__(experiment_id=experiment_id)
        self._events: list[Event] = []
        self._lock = threading.Lock()
        self._file = Path(file) if file else None
        if self._file is not None:
            self._file.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: Event) -> None:
        with self._lock:
            self._events.append(event)
            if self._file is not None:
                with self._file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(event.to_dict()) + "\n")

    # ---------- read access ----------

    @property
    def events(self) -> list[Event]:
        """Immutable-looking snapshot of all events captured so far."""
        with self._lock:
            return list(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()

    def as_jsonl(self) -> str:
        with self._lock:
            return "\n".join(json.dumps(e.to_dict()) for e in self._events)
