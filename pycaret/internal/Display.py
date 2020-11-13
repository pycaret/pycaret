# Module: internal.Display class
# Author: Antoni Baum (Yard1) <antoni.baum@protonmail.com>
# License: MIT

from pycaret.internal.logging import get_logger
import pandas as pd
import ipywidgets as ipw
from IPython import get_ipython
from IPython.display import display, HTML, clear_output, update_display
from typing import Optional, List, Dict, Any
from pycaret.utils import enable_colab


class Display:
    """
    Provides a common interface to handle IPython displays.
    """

    default_progress_args = {
        "value": 0,
        "min": 0,
        "max": 10,
        "step": 1,
        "description": "Processing: ",
    }

    verbose = True
    html_param = True
    progress = None
    master_display = None
    master_display_id = None
    monitor = None
    monitor_id = None
    originator = ""

    def can_display(self, override):
        return (self.verbose and override != False) or override == True

    def display_progress(self, override=None):
        if self.progress is None:
            return
        if self.can_display(override):
            self._display(self.progress)

    def display_master_display(self, override=None):
        if self.master_display is None:
            return
        if self.can_display(override):
            if not self.master_display_id:
                display_ = self._display(self.master_display, display_id=True)
                if display_ is not None:
                    self.master_display_id = display_.display_id
            else:
                self._update_display(
                    self.master_display, display_id=self.master_display_id
                )

    def display_monitor(self, override=None):
        if self.monitor is None:
            return
        if self.can_display(override):
            if not self.monitor_id:
                self.monitor_id = "monitor"
                self._display(self.monitor, display_id=self.monitor_id)
            else:
                self._update_display(self.monitor, display_id=self.monitor_id)

    def move_progress(self, value: int = 1, override=None):
        if self.progress is None:
            return
        if self.can_display(override):
            self.progress.value += value

    def append_to_master_display(self, df_to_append, override=None):
        if self.master_display is None:
            return
        if self.can_display(override):
            self.master_display = pd.concat(
                [self.master_display, df_to_append], ignore_index=True
            )

    def replace_master_display(self, df):
        self.master_display = df

    def update_monitor(self, row_idx: int, message: str, override=None):
        if self.monitor is None:
            return
        if self.can_display(override):
            self.monitor.iloc[row_idx, 1:] = str(message)
            self.display_monitor()

    def display(self, df, clear=False, override=None):
        if self.can_display(override):
            if clear:
                self.clear_output(override=True)
            self._display(df)

    def clear_output(self, override: bool = False):
        if self.html_param and (self.verbose or override):
            clear_output()

    def _display(self, df, *args, **kwargs):
        if not self.html_param:
            try:
                obj = df.data
            except:
                obj = df
            if isinstance(obj, pd.DataFrame) and obj.empty:
                return
            try:
                display(obj, *args, **kwargs)
                return
            except:
                print(obj)
                return
        elif self.enviroment == "google.colab":
            try:
                return display(df.data, *args, **kwargs)
            except:
                return display(df, *args, **kwargs)
        else:
            return display(df, *args, **kwargs)

    def _update_display(self, df, *args, **kwargs):
        if not self.html_param:
            try:
                print(df.data)
            except:
                print(df)
            return
        elif self.enviroment == "google.colab":
            try:
                return update_display(df.data, *args, **kwargs)
            except:
                return update_display(df, *args, **kwargs)
        else:
            return update_display(df, *args, **kwargs)

    def __init__(
        self,
        verbose: bool = True,
        html_param: bool = True,
        progress_args: Optional[Dict[str, Any]] = None,
        master_display_columns: Optional[List[str]] = None,
        monitor_rows: Optional[List[List[str]]] = None,
        round: int = 4,
    ):
        self.logger = get_logger()
        self.verbose = verbose
        self.html_param = html_param
        self.round = round
        try:
            self.enviroment = str(get_ipython())
            self.enviroment = "google.colab" if is_in_colab() else self.enviroment
        except:
            self.enviroment = ""

        if not self.verbose:
            return

        self.logger.info("Preparing display monitor")

        # progress bar
        if progress_args and self.verbose and self.html_param:
            progress_args = {**self.default_progress_args, **progress_args}
            self.progress = ipw.IntProgress(**progress_args)

        if master_display_columns:
            self.master_display = pd.DataFrame(columns=master_display_columns)

        if monitor_rows and self.html_param:
            self.monitor = pd.DataFrame(
                monitor_rows, columns=[" " * i for i in range(len(monitor_rows[0]))],
            ).set_index("")


def is_in_colab():
    try:
        return "google.colab" in str(get_ipython())
    except:
        return False
