# Module: internal.utils
# Author: Moez Ali <moez.ali@queensu.ca> and Antoni Baum (Yard1) <antoni.baum@protonmail.com>
# License: MIT

import datetime
import pandas as pd
import pandas.io.formats.style
import ipywidgets as ipw
from IPython.display import display, HTML, clear_output, update_display
from pycaret.internal.logging import get_logger
from typing import Any, Optional, Dict, Union
import numpy as np


def get_config(variable: str, globals_d: dict):

    """
    This function is used to access global environment variables.

    Example
    -------
    >>> X_train = get_config('X_train') 

    This will return X_train transformed dataset.

    Returns
    -------
    variable

    """

    function_params_str = ", ".join(
        [f"{k}={v}" for k, v in locals().items() if not k == "globals_d"]
    )

    logger = get_logger()

    logger.info("Initializing get_config()")
    logger.info(f"get_config({function_params_str})")

    global_var = globals_d[variable]

    logger.info(f"Global variable: {variable} returned as {global_var}")
    logger.info(
        "get_config() succesfully completed......................................"
    )

    return global_var


def set_config(variable: str, value, globals_d: dict):

    """
    This function is used to reset global environment variables.

    Example
    -------
    >>> set_config('seed', 123) 

    This will set the global seed to '123'.

    """

    function_params_str = ", ".join(
        [f"{k}={v}" for k, v in locals().items() if not k == "globals_d"]
    )

    logger = get_logger()

    logger.info("Initializing set_config()")
    logger.info(f"set_config({function_params_str})")

    globals_d[variable] = value

    logger.info(f"Global variable: {variable} updated to {value}")
    logger.info(
        "set_config() succesfully completed......................................"
    )


def color_df(
    df: pd.DataFrame, color: str, names: list, axis: int = 1
) -> pandas.io.formats.style.Styler:
    return df.style.apply(
        lambda x: [f"background: {color}" if (x.name in names) else "" for i in x],
        axis=axis,
    )


def get_model_id(e, all_models: pd.DataFrame) -> str:
    for row in all_models.itertuples():
        if type(e) is row.Class:
            return row[0]

    return None


def get_model_name(e, all_models: pd.DataFrame) -> str:
    if isinstance(e, str) and e in all_models.index:
        model_id = e
    else:
        model_id = get_model_id(e, all_models)

    if model_id is not None:
        name = all_models.loc[model_id]["Name"]
    else:
        name = str(e).split("(")[0]

    return name


def is_special_model(e, all_models: pd.DataFrame) -> bool:
    for row in all_models.itertuples():
        if type(e) is row.Class:
            return row.Special

    return False


def get_class_name(class_var: Any) -> str:
    return str(class_var)[8:-2]


def get_package_name(class_var: Any) -> str:
    if not isinstance(str, class_var):
        class_var = get_class_name(class_var)
    return class_var.split(".")[0]


def param_grid_to_lists(param_grid: dict) -> dict:
    if param_grid:
        for k, v in param_grid.items():
            param_grid[k] = list(v)
    return param_grid


def make_internal_pipeline(internal_pipeline_steps, model):
    from imblearn.pipeline import Pipeline

    return Pipeline(internal_pipeline_steps + [("actual_estimator", model)])


def calculate_metrics(
    metrics: pd.DataFrame,
    ytest,
    pred_,
    pred_proba: Optional[float] = None,
    score_dict: Optional[Dict[str, np.array]] = None,
) -> Dict[str, np.array]:
    columns = list(metrics.columns)
    score_function_idx = columns.index("Score Function") + 1
    display_name_idx = columns.index("Display Name") + 1

    if not score_dict:
        score_dict = {
            metric._2: np.empty((0, 0))
            for metric in metrics.itertuples()
            if metric[score_function_idx]
        }

    for row in metrics.itertuples():
        if not row[score_function_idx]:
            continue
        target = pred_proba if row.Target == "pred_proba" else pred_
        try:
            calculated_metric = row[score_function_idx](ytest, target, **row.Args)
        except:
            calculated_metric = 0

        score_dict[row[display_name_idx]] = np.append(
            score_dict[row[display_name_idx]], calculated_metric
        )
    return score_dict
