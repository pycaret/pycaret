from abc import ABC, abstractmethod
from typing import Any, TypeVar
import logging
import pandas as pd

from pycaret.internal.Display import Display

T = TypeVar("T")


class AbstractDataProfiler(ABC):
    """Abstract data profiler class."""

    @abstractmethod
    def profile(
        self,
        data: pd.DataFrame,
        verbose: bool,
        logger: logging.Logger,
        **profiler_kwargs
    ) -> T:
        """Profile data and return report object.

        In case of an error, this method should fail
        gracefully and return None."""
        return

    @abstractmethod
    def display(self, data: T, display: Display) -> None:
        """Display the report data."""
        return
