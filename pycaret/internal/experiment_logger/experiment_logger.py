import pandas as pd
from pandas.io.formats.style import Styler
from typing import Optional, Union
from abc import ABC

from pycaret.internal.Display import Display


class ExperimentLogger(ABC):
    name: str = "Abstract Logger"
    id: str = "abstract"

    def __init__(self) -> None:
        return

    def setup_logging(
        self,
        experiment: "_PyCaretExperiment",
        functions: Union[pd.DataFrame, Styler],
        runtime: float,
        log_profile: bool,
        profile_kwargs: dict,
        log_data: bool,
        display: Optional[Display] = None,
    ) -> None:
        """Called once in experiment setup"""
        return

    def log_model(
        self,
        experiment: "_PyCaretExperiment",
        model,
        model_results: Union[pd.DataFrame, Styler],
        score_dict: dict,
        source: str,
        runtime: float,
        model_fit_time: float,
        _prep_pipe,
        log_holdout: bool = True,
        log_plots: bool = False,
        tune_cv_results: Optional[dict] = None,
        URI: Optional[str] = None,
        display: Optional[Display] = None,
    ) -> None:
        """Called whenever a model must be logged"""
        return

    def get_logs(
        self,
        experiment: "_PyCaretExperiment",
    ) -> pd.DataFrame:
        """
        Returns a table with experiment logs consisting
        run details, parameter, metrics and tags.

        Example
        -------
        >>> logs = get_logs()

        This will return a Pandas dataframe.

        Returns
        -------
        pandas.DataFrame

        """
        return
