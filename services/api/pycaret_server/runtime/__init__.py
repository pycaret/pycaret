"""Runtime introspection — GPU detection, system probes.

Lightweight, side-effect-free helpers the worker + admin routes pull
from to figure out "what can this process do". No dependency on the
DB or the FastAPI app; safe to import from anywhere.
"""

from pycaret_server.runtime.gpu import (
    GpuInventory,
    detect_gpus,
    reset_for_tests,
)

__all__ = ["GpuInventory", "detect_gpus", "reset_for_tests"]
