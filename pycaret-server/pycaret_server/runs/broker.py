"""Thread-safe event broker — fans events from worker threads to WebSocket clients.

Design
------

Events are emitted synchronously from a `concurrent.futures` worker thread
inside the `DBEventLogger`. WebSocket handlers are async coroutines running on
the asyncio event loop. The broker bridges the two worlds with `asyncio.Queue`:

- A WebSocket handler calls `subscribe(run_id)` on connect, receiving a queue
  it can `await queue.get()` from.
- The logger thread calls `publish(run_id, event)`, which uses
  `loop.call_soon_threadsafe` to hand the event off to each subscriber's
  asyncio.Queue from the correct thread.
- On disconnect, the handler calls `unsubscribe(run_id, queue)`.

The broker is a module-level singleton; tests can reset it via `event_broker.clear()`.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any


class EventBroker:
    """Per-run event fan-out. Thread-safe for the producer side."""

    # A sentinel object pushed to a subscriber's queue to signal "no more events"
    # (run reached a terminal state). WebSocket handlers recognize this and close.
    END = object()

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # run_id -> list of (asyncio.Queue, loop) pairs
        self._subscribers: dict[str, list[tuple[asyncio.Queue, asyncio.AbstractEventLoop]]] = {}

    # -------------------------------------------------------------- subscribe

    def subscribe(self, run_id: str) -> asyncio.Queue:
        """Register an asyncio.Queue for a run. Must be called from the event loop.

        Returns the queue; the caller should ``await queue.get()`` in a loop.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as e:
            raise RuntimeError("EventBroker.subscribe() must be called from a running loop") from e
        q: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self._subscribers.setdefault(run_id, []).append((q, loop))
        return q

    def unsubscribe(self, run_id: str, q: asyncio.Queue) -> None:
        """Remove a previously subscribed queue."""
        with self._lock:
            lst = self._subscribers.get(run_id, [])
            self._subscribers[run_id] = [(qq, ll) for (qq, ll) in lst if qq is not q]
            if not self._subscribers[run_id]:
                self._subscribers.pop(run_id, None)

    # ---------------------------------------------------------------- publish

    def publish(self, run_id: str, event: dict[str, Any]) -> None:
        """Publish an event to every subscriber of `run_id`.

        Safe to call from any thread — individual puts are scheduled on the
        owning event loop via `call_soon_threadsafe`.
        """
        with self._lock:
            targets = list(self._subscribers.get(run_id, []))
        for q, loop in targets:
            try:
                loop.call_soon_threadsafe(q.put_nowait, event)
            except RuntimeError:
                # Loop is closed — drop the event silently; the subscriber will
                # have already observed its disconnect.
                pass

    def close_run(self, run_id: str) -> None:
        """Send the END sentinel to every subscriber of `run_id`.

        Called once a run reaches a terminal state (succeeded/failed/cancelled)
        so WebSocket handlers can cleanly close the connection.
        """
        with self._lock:
            targets = list(self._subscribers.get(run_id, []))
        for q, loop in targets:
            try:
                loop.call_soon_threadsafe(q.put_nowait, self.END)
            except RuntimeError:
                pass

    # ------------------------------------------------------------------ admin

    def subscriber_count(self, run_id: str) -> int:
        with self._lock:
            return len(self._subscribers.get(run_id, []))

    def clear(self) -> None:
        """Drop all subscribers. For test fixtures."""
        with self._lock:
            self._subscribers.clear()


# Module-level singleton.
event_broker = EventBroker()
