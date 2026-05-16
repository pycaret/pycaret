"""GPU device detection.

Single-source-of-truth for "does this worker have a GPU?". Used by:

- The worker entrypoint: refuses to claim ``queue=gpu`` Jobs unless
  ``detect_gpus().available`` is True, so a CPU worker started against
  the GPU queue can't drain it into failures.
- The admin ``/system/info`` route: surfaces the live capabilities to
  the UI so users see why their submitted ``use_gpu=True`` runs are
  queued (no GPU worker registered yet).

We probe in this order, stopping at the first hit:

1. ``CUDA_VISIBLE_DEVICES`` env override (returns the explicit count).
2. ``pynvml`` library if importable (proper NVML query — gives names + VRAM).
3. ``nvidia-smi -L`` shelled out (covers boxes without pynvml installed).

Anything that throws is treated as "no GPU". The function is cached
per-process because hardware doesn't change at runtime.
"""

from __future__ import annotations

import functools
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GpuInventory:
    """What this worker can offer to a GPU-queue job."""

    available: bool
    count: int = 0
    devices: list[str] = field(default_factory=list)
    source: str = "none"  # "env" | "pynvml" | "nvidia-smi" | "none"
    error: str | None = None


def _from_env() -> GpuInventory | None:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    # Empty string is the convention for "no GPUs visible to this process".
    cleaned = raw.strip()
    if cleaned == "":
        return GpuInventory(available=False, source="env")
    ids = [x.strip() for x in cleaned.split(",") if x.strip()]
    if not ids:
        return GpuInventory(available=False, source="env")
    return GpuInventory(
        available=True,
        count=len(ids),
        devices=[f"cuda:{i}" for i in ids],
        source="env",
    )


def _from_pynvml() -> GpuInventory | None:
    try:
        import pynvml  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return None
    try:
        pynvml.nvmlInit()
    except Exception as exc:  # noqa: BLE001
        return GpuInventory(
            available=False, source="pynvml", error=f"nvmlInit: {exc}"
        )
    try:
        n = int(pynvml.nvmlDeviceGetCount())
        names: list[str] = []
        for i in range(n):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            raw = pynvml.nvmlDeviceGetName(h)
            names.append(raw.decode() if isinstance(raw, bytes) else str(raw))
        return GpuInventory(
            available=n > 0, count=n, devices=names, source="pynvml"
        )
    except Exception as exc:  # noqa: BLE001
        return GpuInventory(
            available=False, source="pynvml", error=f"nvml query: {exc}"
        )
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass


def _from_nvidia_smi() -> GpuInventory | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        res = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as exc:  # noqa: BLE001
        return GpuInventory(
            available=False, source="nvidia-smi", error=str(exc)
        )
    if res.returncode != 0:
        return GpuInventory(
            available=False,
            source="nvidia-smi",
            error=res.stderr.strip() or "nvidia-smi non-zero",
        )
    # ``GPU 0: Tesla T4 (UUID: ...)`` per line.
    names: list[str] = []
    for line in res.stdout.splitlines():
        m = re.match(r"GPU \d+:\s+(.+?)\s*(?:\(UUID:.*)?$", line.strip())
        if m:
            names.append(m.group(1))
    return GpuInventory(
        available=bool(names),
        count=len(names),
        devices=names,
        source="nvidia-smi",
    )


@functools.lru_cache(maxsize=1)
def detect_gpus() -> GpuInventory:
    """Return the cached GpuInventory for this process."""
    for probe in (_from_env, _from_pynvml, _from_nvidia_smi):
        result = probe()
        if result is not None:
            if result.available:
                _log.info(
                    "GPU detected via %s: %d device(s) — %s",
                    result.source,
                    result.count,
                    ", ".join(result.devices) or "<unnamed>",
                )
            return result
    return GpuInventory(available=False, source="none")


def reset_for_tests() -> None:
    """Drop the cache so a test can re-probe after monkeypatching env."""
    detect_gpus.cache_clear()
