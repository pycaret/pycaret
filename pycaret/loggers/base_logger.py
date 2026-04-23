"""Legacy import path for `BaseLogger`.

In 4.0 the canonical location is `pycaret.logging.base`. This module is a thin
re-export so `from pycaret.loggers.base_logger import BaseLogger` (the 3.x
import path seen in user subclasses) still works.
"""

from pycaret.logging.base import BaseLogger

# 3.x also exposed this module-level constant — keep it visible for compat.
SETUP_TAG = "Session Initialized"

__all__ = ["BaseLogger", "SETUP_TAG"]
