"""Legacy import path — re-exports the 4.0 logger surface.

The real logger lives at `pycaret.logging` as of 4.0. This module exists so
that 3.x user subclasses of `pycaret.loggers.base_logger.BaseLogger` keep
importing cleanly. New code should import from `pycaret.logging` directly.
"""

from pycaret.logging.base import BaseLogger

__all__ = ["BaseLogger"]
