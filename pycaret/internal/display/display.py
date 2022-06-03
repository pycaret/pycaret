# Module: internal.display class
# Author: Antoni Baum (Yard1) <antoni.baum@protonmail.com>
# License: MIT

from typing import Any, Dict, List, Optional

from pycaret.internal.display.display_backend import SilentBackend, detect_backend
from pycaret.internal.display.display_component import MonitorDisplay
from pycaret.internal.display.progress_bar import ProgressBarDisplay
from pycaret.internal.logging import get_logger


class CommonDisplay:
    """
    Provides a common interface to handle method displays.
    """

    def display_progress(self):
        if not self.can_display:
            return
        if self._progress_bar_display:
            self._progress_bar_display.display()

    def move_progress(self, value: int = 1):
        if not self.can_display:
            return
        if self._progress_bar_display:
            self._progress_bar_display.step(value)

    def update_monitor(self, row_idx: int, message: str):
        if not self.can_display:
            return
        if self._monitor_display:
            self._monitor_display.update(row_idx, message)

    def display(self, df, *, clear: bool = False, final_display: bool = True):
        if not self.can_display:
            return
        if clear:
            self._general_display.clear_display()
        if final_display:
            self.close()
        self._general_display.display(df, final_display=final_display)

    def clear_output(self):
        if not self.can_display:
            return
        self._general_display.clear_output()

    def close(self):
        if not self.can_display:
            return
        if self._progress_bar_display:
            self._progress_bar_display.close()
        if self._monitor_display:
            self._monitor_display.close()

    @property
    def can_update_text(self) -> bool:
        return self._general_display.can_update_text

    @property
    def can_update_rich(self) -> bool:
        return self._general_display.can_update_rich

    @property
    def can_display(self) -> bool:
        return True

    def __init__(
        self,
        verbose: bool = True,
        html_param: bool = True,
        progress_args: Optional[Dict[str, Any]] = None,
        monitor_rows: Optional[List[List[str]]] = None,
    ):
        self.logger = get_logger()
        self.verbose = verbose
        self.html_param = html_param

        backend_id = "cli" if html_param is False else None

        if monitor_rows:
            self._monitor_display = MonitorDisplay(
                monitor_rows, backend=backend_id, verbose=self.verbose
            )
            self._monitor_display.display()
        else:
            self._monitor_display = None

        self._general_display = (
            detect_backend(backend_id) if self.verbose else SilentBackend()
        )
        self._general_display.display(None)

        if progress_args:
            self._progress_bar_display = ProgressBarDisplay(
                **progress_args, backend=backend_id, verbose=self.verbose
            )
            self._progress_bar_display.display()
        else:
            self._progress_bar_display = None


class DummyDisplay(CommonDisplay):
    """The Display class to completely turn off all displays"""

    def __init__(self):
        super().__init__(verbose=False)

    @property
    def can_display(self):
        return False
