from abc import ABC
from typing import List, Optional, Union

import pandas as pd

from pycaret.internal.display.display_backend import (
    DisplayBackend,
    SilentBackend,
    detect_backend,
)
from pycaret.internal.logging import get_logger


class DisplayComponent(ABC):
    def __init__(
        self,
        *,
        verbose: bool = True,
        backend: Optional[Union[str, DisplayBackend]] = None,
    ) -> None:
        self.logger = get_logger()
        self.verbose = verbose
        self._backend = detect_backend(backend)

    @property
    def backend(self) -> DisplayBackend:
        if not self.verbose:
            return SilentBackend()
        return self._backend

    @backend.setter
    def backend(self, val: DisplayBackend):
        self._backend = val

    def close(self):
        self.backend.clear_display()


class MonitorDisplay(DisplayComponent):
    def __init__(
        self,
        monitor_rows: List[List[str]],
        *,
        verbose: bool = True,
        backend: Optional[Union[str, DisplayBackend]] = None,
    ):
        super().__init__(verbose=verbose, backend=backend)
        if not self._backend.can_update_rich:
            self.backend = SilentBackend()

        self.monitor = pd.DataFrame(
            monitor_rows, columns=[" " * i for i in range(len(monitor_rows[0]))]
        ).set_index("")

    def display(self, clear: bool = False):
        if clear:
            self.backend.clear_display()
        self.backend.display(self.monitor)

    def update(self, row_idx: int, message: str):
        self.monitor.iloc[row_idx, 1:] = str(message)
        self.display()
