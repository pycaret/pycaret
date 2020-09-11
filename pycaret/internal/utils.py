# Module: internal.utils
# Author: Moez Ali <moez.ali@queensu.ca> and Antoni Baum (Yard1) <antoni.baum@protonmail.com>
# License: MIT

import datetime
import pandas as pd
import pandas.io.formats.style
import ipywidgets as ipw
from IPython.display import display, HTML, clear_output, update_display
from pycaret.internal.logging import get_logger
from pycaret.internal.validation import *
from typing import Any, List, Optional, Dict, Tuple, Union
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, BaseCrossValidator


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

    if not variable in globals_d["pycaret_globals"]:
        raise ValueError(
            f"Variable {variable} not found. Possible variables are: {globals_d['pycaret_globals']}"
        )

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

    if not variable in globals_d["pycaret_globals"] or variable == "pycaret_globals":
        raise ValueError(
            f"Variable {variable} not found. Possible variables are: {globals_d['pycaret_globals']}"
        )

    globals_d[variable] = value

    logger.info(f"Global variable: {variable} updated to {value}")
    logger.info(
        "set_config() succesfully completed......................................"
    )


def save_config(file_name: str, globals_d: dict):
    """
    This function is used to save all enviroment variables to file,
    allowing to later resume modeling without rerunning setup().

    Example
    -------
    >>> save_config('myvars.pkl') 

    This will save all enviroment variables to 'myvars.pkl'.

    """

    function_params_str = ", ".join(
        [f"{k}={v}" for k, v in locals().items() if not k == "globals_d"]
    )

    logger = get_logger()

    logger.info("Initializing save_config()")
    logger.info(f"save_config({function_params_str})")

    globals_to_dump = {
        k: v for k, v in globals_d.items() if k in globals_d["pycaret_globals"]
    }

    import joblib

    joblib.dump(globals_to_dump, file_name)

    logger.info(f"Global variables dumped to {file_name}")
    logger.info(
        "save_config() succesfully completed......................................"
    )


def load_config(file_name: str, globals_d: dict):
    """
    This function is used to load enviroment variables from file created with save_config(),
    allowing to later resume modeling without rerunning setup().


    Example
    -------
    >>> load_config('myvars.pkl') 

    This will load all enviroment variables from 'myvars.pkl'.

    """

    function_params_str = ", ".join(
        [f"{k}={v}" for k, v in locals().items() if not k == "globals_d"]
    )

    logger = get_logger()

    logger.info("Initializing load_config()")
    logger.info(f"load_config({function_params_str})")

    import joblib

    loaded_globals = joblib.load(file_name)

    logger.info(f"Global variables loaded from {file_name}")

    for k, v in loaded_globals.items():
        globals_d[k] = v

    globals_d["logger"] = get_logger()

    logger.info(f"Global variables set to match those in {file_name}")

    logger.info(
        "load_config() succesfully completed......................................"
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


def get_model_name(e, all_models: pd.DataFrame, deep: bool = True) -> str:
    if isinstance(e, str) and e in all_models.index:
        model_id = e
    else:
        if deep:
            while True:
                if hasattr(e, "steps"):
                    e = e.steps[-1][1]
                elif hasattr(e, "base_estimator"):
                    e = e.base_estimator
                elif hasattr(e, "estimator"):
                    e = e.estimator
                else:
                    break

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
    from pycaret.internal.Pipeline import Pipeline, PartialFitPipeline

    if hasattr(model, "partial_fit"):
        return PartialFitPipeline(
            internal_pipeline_steps + [("actual_estimator", model)]
        )
    return Pipeline(internal_pipeline_steps + [("actual_estimator", model)])


def calculate_metrics(
    metrics: pd.DataFrame,
    ytest,
    pred_,
    pred_proba: Optional[float] = None,
    score_dict: Optional[Dict[str, np.array]] = None,
    weights: Optional[list] = None,
) -> Dict[str, np.array]:
    columns = list(metrics.columns)
    score_function_idx = columns.index("Score Function") + 1
    display_name_idx = columns.index("Display Name") + 1

    logger = get_logger()

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
            calculated_metric = row[score_function_idx](
                ytest, target, sample_weight=weights, **row.Args
            )
        except:
            logger.warning(
                f"sample_weight not supported for {row[score_function_idx]}."
            )
            try:
                calculated_metric = row[score_function_idx](ytest, target, **row.Args)
            except:
                logger.warning(f"{row[score_function_idx]} not supported, setting to 0")
                calculated_metric = 0

        score_dict[row[display_name_idx]] = np.append(
            score_dict[row[display_name_idx]], calculated_metric
        )
    return score_dict


def normalize_custom_transformers(
    transformers: Union[Any, Tuple[str, Any], List[Any], List[Tuple[str, Any]]]
) -> list:
    if isinstance(transformers, dict):
        transformers = list(transformers.items())
    if isinstance(transformers, list):
        for i, x in enumerate(transformers):
            _check_custom_transformer(x)
            if not isinstance(x, tuple):
                transformers[i] = (f"custom_step_{i}", x)
    else:
        _check_custom_transformer(transformers)
        if not isinstance(transformers, tuple):
            transformers = (f"custom_step", transformers)
        if is_sklearn_pipeline(transformers[0]):
            return transformers.steps
        transformers = [transformers]
    return transformers


def _check_custom_transformer(transformer):
    actual_transformer = transformer
    if isinstance(transformer, tuple):
        if len(transformer) != 2:
            raise ValueError("Transformer tuple must have a size of 2.")
        if not isinstance(transformer[0], str):
            raise TypeError("First element of transformer tuple must be a str.")
        actual_transformer = transformer[1]
    if not (
        hasattr(actual_transformer, "fit")
        and hasattr(actual_transformer, "transform")
        and hasattr(actual_transformer, "fit_transform")
    ):
        raise TypeError(
            "Transformer must be an object implementing methods 'fit', 'transform' and 'fit_transform'."
        )


def get_cv_splitter(
    fold: Optional[Union[int, BaseCrossValidator]],
    default: BaseCrossValidator,
    seed: int,
    shuffle: bool,
    int_default: str = "kfold",
) -> BaseCrossValidator:
    if not fold:
        return default
    if is_sklearn_cv_generator(fold):
        return fold
    if type(fold) is int:
        if int_default == "kfold":
            return KFold(fold, random_state=seed, shuffle=shuffle)
        elif int_default == "stratifiedkfold":
            return StratifiedKFold(fold, random_state=seed, shuffle=shuffle)
        else:
            raise ValueError(
                "Wrong value for int_default param. Needs to be either 'kfold' or 'stratifiedkfold'."
            )
    raise TypeError(
        f"{fold} is of type {type(fold)} while it needs to be either a CV generator or int."
    )


def get_cv_n_folds(
    fold: Optional[Union[int, BaseCrossValidator]], default_folds: int
) -> int:
    if not fold:
        return default_folds
    if isinstance(fold, int):
        return fold
    else:
        return fold.get_n_splits()
