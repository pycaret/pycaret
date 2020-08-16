# Module: Utility
# Author: Moez Ali <moez.ali@queensu.ca>
# License: MIT

import datetime
import pandas as pd
import ipywidgets as ipw
from IPython.display import display, HTML, clear_output, update_display

version_ = "2.0"


def version():
    print(version_)


def __version__():
    return version_


def check_metric(actual, prediction, metric, round=4):

    """
    Function to evaluate classification and regression metrics.
    """

    # general dependencies
    import numpy as np

    # metric calculation starts here

    if metric == "Accuracy":

        from sklearn import metrics

        result = metrics.accuracy_score(actual, prediction)
        result = result.round(round)

    elif metric == "Recall":

        from sklearn import metrics

        result = metrics.recall_score(actual, prediction)
        result = result.round(round)

    elif metric == "Precision":

        from sklearn import metrics

        result = metrics.precision_score(actual, prediction)
        result = result.round(round)

    elif metric == "F1":

        from sklearn import metrics

        result = metrics.f1_score(actual, prediction)
        result = result.round(round)

    elif metric == "Kappa":

        from sklearn import metrics

        result = metrics.cohen_kappa_score(actual, prediction)
        result = result.round(round)

    elif metric == "AUC":

        from sklearn import metrics

        result = metrics.roc_auc_score(actual, prediction)
        result = result.round(round)

    elif metric == "MCC":

        from sklearn import metrics

        result = metrics.matthews_corrcoef(actual, prediction)
        result = result.round(round)

    elif metric == "MAE":

        from sklearn import metrics

        result = metrics.mean_absolute_error(actual, prediction)
        result = result.round(round)

    elif metric == "MSE":

        from sklearn import metrics

        result = metrics.mean_squared_error(actual, prediction)
        result = result.round(round)

    elif metric == "RMSE":

        from sklearn import metrics

        result = metrics.mean_squared_error(actual, prediction)
        result = np.sqrt(result)
        result = result.round(round)

    elif metric == "R2":

        from sklearn import metrics

        result = metrics.r2_score(actual, prediction)
        result = result.round(round)

    elif metric == "RMSLE":

        result = np.sqrt(
            np.mean(
                np.power(
                    np.log(np.array(abs(prediction)) + 1)
                    - np.log(np.array(abs(actual)) + 1),
                    2,
                )
            )
        )
        result = result.round(round)

    elif metric == "MAPE":

        mask = actual != 0
        result = (np.fabs(actual - prediction) / actual)[mask].mean()
        result = result.round(round)

    return result


def enable_colab():

    """
    Function to render plotly visuals in colab.
    """

    def configure_plotly_browser_state():

        import IPython

        display(
            IPython.core.display.HTML(
                """
            <script src="/static/components/requirejs/require.js"></script>
            <script>
              requirejs.config({
                paths: {
                  base: '/static/base',
                  plotly: 'https://cdn.plot.ly/plotly-latest.min.js?noext',
                },
              });
            </script>
            """
            )
        )

    import IPython

    IPython.get_ipython().events.register(
        "pre_run_cell", configure_plotly_browser_state
    )
    print("Colab mode activated.")


def get_logger():
    import logging

    try:
        hasattr(logger, "name")
    except:
        logger = logging.getLogger("logs")
        logger.setLevel(logging.DEBUG)

        # create console handler and set level to debug
        if logger.hasHandlers():
            logger.handlers.clear()

        ch = logging.FileHandler("logs.log")
        ch.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(ch)

    return logger


def color_df(df: pd.DataFrame, color: str, names: list, axis: int = 1) -> pd.DataFrame:
    return df.style.apply(
        lambda x: [f"background: {color}" if (x.name in names) else "" for i in x],
        axis=axis,
    )


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
        logger=None,
    ):
        self.verbose = verbose
        self.html_param = html_param
        self.logger = logger

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
