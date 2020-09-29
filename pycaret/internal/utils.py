# Module: internal.utils
# Author: Moez Ali <moez.ali@queensu.ca> and Antoni Baum (Yard1) <antoni.baum@protonmail.com>
# License: MIT

import os
from pycaret.containers.metrics.base_metric import MetricContainer
from pycaret.containers.models.base_model import ModelContainer
import pandas as pd
import pandas.io.formats.style
import ipywidgets as ipw
from IPython.display import display, HTML, clear_output, update_display
from pycaret.internal.logging import get_logger
from pycaret.internal.validation import *
from typing import Any, List, Optional, Dict, Tuple, Union
from sklearn import clone
from sklearn.utils import check_random_state
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, BaseCrossValidator
import sklearn.model_selection._search


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

    if variable.startswith("_"):
        raise ValueError(f"Variable {variable} is read only ('_' prefix).")

    if not variable in globals_d["pycaret_globals"] or variable == "pycaret_globals":
        raise ValueError(
            f"Variable {variable} not found. Possible variables are: {globals_d['pycaret_globals']}"
        )

    globals_d[variable] = value

    # special case
    if not globals_d["gpu_param"] and variable == "n_jobs_param":
        globals_d["_gpu_n_jobs_param"] = value

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


def get_model_id(e, all_models: Dict[str, ModelContainer]) -> str:
    from pycaret.internal.meta_estimators import get_estimator_from_meta_estimator

    return next(
        (
            k
            for k, v in all_models.items()
            if v.is_estimator_equal(get_estimator_from_meta_estimator(e))
        ),
        None,
    )


def get_model_name(e, all_models: Dict[str, ModelContainer], deep: bool = True) -> str:
    old_e = e
    if isinstance(e, str) and e in all_models:
        model_id = e
    else:
        if deep:
            while hasattr(e, "get_params"):
                old_e = e
                params = e.get_params()
                if "steps" in params:
                    e = params["steps"][-1][1]
                elif "base_estimator" in params:
                    e = params["base_estimator"]
                elif "regressor" in params:
                    e = params["regressor"]
                elif "estimator" in params:
                    e = params["estimator"]
                else:
                    break
        if e is None:
            e = old_e
        model_id = get_model_id(e, all_models)

    if model_id is not None:
        name = all_models[model_id].name
    else:
        try:
            name = type(e).__name__
        except:
            name = str(e).split("(")[0]

    return name


def is_special_model(e, all_models: Dict[str, ModelContainer]) -> bool:
    try:
        return all_models[get_model_id(e, all_models)].is_special
    except:
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


def calculate_metrics(
    metrics: Dict[str, MetricContainer],
    y_test,
    pred,
    pred_proba: Optional[float] = None,
    score_dict: Optional[Dict[str, np.array]] = None,
    weights: Optional[list] = None,
) -> Dict[str, np.array]:

    score_dict = []

    for k, v in metrics.items():
        score_dict.append(
            _calculate_metric(
                v, v.score_func, v.display_name, y_test, pred, pred_proba, weights,
            )
        )

    score_dict = dict([x for x in score_dict if x is not None])
    return score_dict


def _calculate_metric(
    container, score_func, display_name, y_test, pred_, pred_proba, weights
):
    if not score_func:
        return None
    target = pred_proba if container.target == "pred_proba" else pred_
    try:
        calculated_metric = score_func(
            y_test, target, sample_weight=weights, **container.args
        )
    except:
        try:
            calculated_metric = score_func(y_test, target, **container.args)
        except:
            calculated_metric = 0

    return (display_name, calculated_metric)


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
    fold: Optional[Union[int, BaseCrossValidator]], default, X, y=None, groups=None
) -> int:
    if not fold:
        fold = default
    if isinstance(fold, int):
        return fold
    else:
        return fold.get_n_splits(X, y=y, groups=groups)


class none_n_jobs(object):
    """
    Context which sets `n_jobs` or `thread_count` to None for passed model.
    """

    def __init__(self, model):
        self.params = {}
        self.model = model
        try:
            self.params = {
                k: v
                for k, v in self.model.get_params().items()
                if k.endswith("n_jobs") or k.endswith("thread_count")
            }
        except:
            pass

    def __enter__(self):
        if self.params:
            self.model.set_params(**{k: None for k, v in self.params.items()})

    def __exit__(self, type, value, traceback):
        if self.params:
            self.model.set_params(**self.params)


class true_warm_start(object):
    """
    Context which sets `warm_start` to True for passed model.
    """

    def __init__(self, model):
        self.params = {}
        self.model = model
        try:
            self.params = {
                k: v
                for k, v in self.model.get_params().items()
                if k.endswith("warm_start")
            }
        except:
            pass

    def __enter__(self):
        if self.params:
            self.model.set_params(**{k: True for k, v in self.params.items()})

    def __exit__(self, type, value, traceback):
        if self.params:
            self.model.set_params(**self.params)


class nullcontext(object):
    def __init__(self, enter_result=None):
        self.enter_result = enter_result

    def __enter__(self):
        return self.enter_result

    def __exit__(self, *excinfo):
        pass


def get_groups(
    groups: Union[str, pd.DataFrame], X_train: pd.DataFrame, default: pd.DataFrame
):
    logger = get_logger()
    if groups is None:
        return default
    if isinstance(groups, str):
        if groups not in X_train.columns:
            raise ValueError(
                f"Column {groups} used for groups is not present in the dataset."
            )
        groups = X_train[groups]
    else:
        try:
            groups = groups[groups.index.isin(X_train.index)]
        except Exception as e:
            logger.warn(
                "Couldn't get the same rows in groups as in X_train. Exception below:"
            )
            logger.warn(e)
        if len(groups) != len(X_train):
            raise ValueError(
                f"groups has lenght {len(groups)} which doesn't match X_train length of {len(X_train)}."
            )
    return groups


def get_all_object_vars_and_properties(object):
    """
    Gets all class, static and dynamic attributes from an object.
    
    Calling ``vars()`` would only return static attributes.
    
    https://stackoverflow.com/a/59769926
    """
    return {
        k: getattr(object, k, "")
        for k in object.__dir__()
        if k[:2] != "__" and type(getattr(object, k, "")).__name__ != "method"
    }


# adapted from https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/_random.pyx
def _sample_without_replacement_check_input(n_population: int, n_samples: int):
    """ Check that input are consistent for sample_without_replacement"""
    if n_population < 0:
        raise ValueError(
            "n_population should be greater than 0, got %s." % n_population
        )

    if n_samples > n_population:
        raise ValueError(
            "n_population should be greater or equal than "
            "n_samples, got n_samples > n_population (%s > %s)"
            % (n_samples, n_population)
        )


# adapted from https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/_random.pyx
def sample_without_replacement(
    n_population: int, n_samples: int, method=None, random_state=None
) -> Any:
    """Sample integers without replacement.

    Select n_samples integers from the set [0, n_population) without
    replacement.


    Parameters
    ----------
    n_population : int
        The size of the set to sample from.

    n_samples : int
        The number of integer to sample.

    random_state : int, RandomState instance or None, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    out : ndarray of shape (n_samples,)
        The sampled subsets of integer. The subset of selected integer might
        not be randomized, see the method argument.
    """
    _sample_without_replacement_check_input(n_population, n_samples)

    out = np.empty((n_samples, ), dtype=np.int64)
    rng = check_random_state(random_state)
    rng_randint = rng.randint
    # The following line of code are heavily inspired from python core,
    # more precisely of random.sample.
    selected = set()
    for i in range(n_samples):
        j = rng_randint(n_population, dtype=np.int64)
        while j in selected:
            j = rng_randint(n_population, dtype=np.int64)
        selected.add(j)
        out[i] = j
    return out
