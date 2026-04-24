"""In-process deployment registry.

A thin LRU-style cache that loads a fitted pipeline from disk on first
prediction and keeps it in memory until the deployment is deleted or the
process recycles. No out-of-process worker, no remote protocol — the whole
point of 4.0's serving story is "the same FastAPI process answers /predict
against the pipeline that was just trained".

The registry is a singleton. Tests reset it via `reset_registry()`.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pandas as pd


class DeploymentRegistry:
    """Slug → loaded pipeline. All operations are thread-safe; prediction is
    hot-path so we trade a little lock contention for correctness."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # slug -> (pipeline, source_path_when_loaded)
        self._loaded: dict[str, tuple[Any, str]] = {}
        # slug -> rolling latency list (last 100). Kept tiny; full histogram
        # would require a richer dependency we don't want in core.
        self._latencies: dict[str, list[float]] = {}

    # ------------------------------------------------------------ load / evict

    def get(self, slug: str, path: str) -> Any:
        """Return the loaded pipeline, loading from `path` if needed.

        If the cached entry was loaded from a different path (e.g. the
        deployment repointed at a new Pipeline row), reload from disk.
        """
        with self._lock:
            cached = self._loaded.get(slug)
            if cached and cached[1] == path:
                return cached[0]
        pipe = self._load(path)
        with self._lock:
            self._loaded[slug] = (pipe, path)
        return pipe

    def evict(self, slug: str) -> None:
        with self._lock:
            self._loaded.pop(slug, None)
            self._latencies.pop(slug, None)

    def clear(self) -> None:
        with self._lock:
            self._loaded.clear()
            self._latencies.clear()

    # --------------------------------------------------------------- predict

    def predict(self, slug: str, path: str, rows: list[dict]) -> tuple[list[dict], float]:
        """Predict a batch of row-dicts. Returns (predictions, latency_ms)."""
        if not rows:
            raise ValueError("rows must be a non-empty list of record dicts")
        pipe = self.get(slug, path)
        df = pd.DataFrame.from_records(rows)
        t0 = time.perf_counter()
        y = pipe.predict(df)
        latency = (time.perf_counter() - t0) * 1000
        with self._lock:
            self._latencies.setdefault(slug, []).append(latency)
            # Cap the rolling window at 100 to bound memory.
            if len(self._latencies[slug]) > 100:
                self._latencies[slug] = self._latencies[slug][-100:]
        preds = [{"index": i, "prediction": _jsonify(v)} for i, v in enumerate(y)]
        return preds, latency

    def latency_percentiles(self, slug: str) -> tuple[float | None, float | None]:
        """Rough p50/p95 over the last 100 predictions. ``None`` until we have
        samples."""
        with self._lock:
            samples = list(self._latencies.get(slug, []))
        if not samples:
            return None, None
        samples.sort()
        n = len(samples)
        p50 = samples[int(n * 0.5)]
        p95 = samples[min(n - 1, int(n * 0.95))]
        return p50, p95

    # ------------------------------------------------------------- internals

    def _load(self, path: str) -> Any:
        """Unpickle a Pipeline from disk. Uses cloudpickle to match how we save."""
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"pipeline artifact not found: {path}")
        try:
            import cloudpickle
        except ImportError:
            import pickle as cloudpickle  # type: ignore[no-redef]
        return cloudpickle.loads(p.read_bytes())


def _jsonify(v: Any) -> Any:
    """Best-effort conversion so sklearn's numpy outputs survive JSON."""
    try:
        import numpy as np

        if isinstance(v, np.generic):
            return v.item()
    except ImportError:
        pass
    return v


# -------------------------------------------------------------- singleton


_registry: DeploymentRegistry | None = None
_lock = threading.Lock()


def get_registry() -> DeploymentRegistry:
    global _registry
    with _lock:
        if _registry is None:
            _registry = DeploymentRegistry()
        return _registry


def reset_registry() -> None:
    """For test fixtures."""
    global _registry
    with _lock:
        if _registry is not None:
            _registry.clear()
            _registry = None
