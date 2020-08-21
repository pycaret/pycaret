# Module: internal.Display class
# Author: Antoni Baum (Yard1) <antoni.baum@protonmail.com>
# License: MIT

import datetime
import pandas as pd
import pandas.io.formats.style
import ipywidgets as ipw
from IPython.display import display, HTML, clear_output, update_display


class Display:
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

    def display_progress(self, override=None):
        if self.progress is None:
            return
        if (self.verbose and self.html_param and override != False) or override == True:
            display(self.progress)

    def display_master_display(self, override=None):
        if self.master_display is None:
            return
        if (self.verbose and self.html_param and override != False) or override == True:
            if not self.master_display_id:
                display_ = display(self.master_display, display_id=True)
                self.master_display_id = display_.display_id
            else:
                update_display(self.master_display, display_id=self.master_display_id)

    def display_monitor(self, override=None):
        if self.monitor is None:
            return
        if (self.verbose and self.html_param and override != False) or override == True:
            if not self.monitor_id:
                self.monitor_id = "monitor"
                display(self.monitor, display_id=self.monitor_id)
            else:
                update_display(self.monitor, display_id=self.monitor_id)

    def move_progress(self, value: int = 1, override=None):
        if self.progress is None:
            return
        if (self.verbose and self.html_param and override != False) or override == True:
            self.progress.value += value

    def append_to_master_display(self, df_to_append, override=None):
        if self.master_display is None:
            return
        if (self.verbose and self.html_param and override != False) or override == True:
            self.master_display = pd.concat(
                [self.master_display, df_to_append], ignore_index=True
            )

    def replace_master_display(self, df):
        self.master_display = df

    def update_monitor(self, row_idx: int, message: str, override=None):
        if self.monitor is None:
            return
        if (self.verbose and self.html_param and override != False) or override == True:
            self.monitor.iloc[row_idx, 1:] = str(message)

    def display(self, df, clear=False, override=None):
        if (self.verbose and self.html_param and override != False) or override == True:
            if clear:
                self.clear_output()
            display(df)
        elif (
            self.verbose and not self.html_param and override != False
        ) or override == True:
            print(df.data)

    def clear_output(self):
        clear_output()

    def __init__(
        self,
        verbose: bool = True,
        html_param: bool = True,
        progress_args: dict = None,
        master_display_columns: list = None,
        monitor_rows: list = None,
        round: int = 4,
        logger=None,
    ):
        self.verbose = verbose
        self.html_param = html_param
        self.logger = logger
        self.round = round

        if not (self.verbose and self.html_param):
            return

        self.logger.info("Preparing display monitor")

        # progress bar
        if progress_args:
            progress_args = {**self.default_progress_args, **progress_args}
            self.progress = ipw.IntProgress(**progress_args)

        if master_display_columns:
            self.master_display = pd.DataFrame(columns=master_display_columns)

        if monitor_rows:
            self.monitor = pd.DataFrame(
                monitor_rows, columns=[" " * i for i in range(len(monitor_rows[0]))],
            ).set_index("")
