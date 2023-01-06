import functools
import inspect
import warnings
from collections.abc import Mapping
from copy import deepcopy
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import pandas.io.formats.style
from scipy import sparse
from sklearn.metrics import get_scorer
from sklearn.metrics._scorer import _PredictScorer
from sklearn.model_selection import BaseCrossValidator, KFold, StratifiedKFold
from sklearn.model_selection._split import _BaseKFold

import pycaret.containers
from pycaret.internal.logging import get_logger
from pycaret.internal.validation import (
    is_sklearn_cv_generator,
    is_sklearn_pipeline,
    supports_partial_fit,
)
from pycaret.utils._dependencies import _check_soft_dependencies

if TYPE_CHECKING:
    from pycaret.internal.pycaret_experiment.pycaret_experiment import (
        _PyCaretExperiment,
    )


class MLUsecase(Enum):
    CLASSIFICATION = auto()
    REGRESSION = auto()
    CLUSTERING = auto()
    ANOMALY = auto()
    TIME_SERIES = auto()


def get_ml_task(y):
    c1 = y.dtype == "int64"
    c2 = y.nunique() <= 20
    c3 = y.dtype.name in ["object", "bool", "category"]
    if (c1 & c2) | c3:
        ml_usecase = MLUsecase.CLASSIFICATION
    else:
        ml_usecase = MLUsecase.REGRESSION
    return ml_usecase


def highlight_setup(column):
    return [
        "background-color: lightgreen" if v is True or v == "Yes" else ""
        for v in column
    ]


def get_classification_task(y):
    """Return if the target column is binary or multiclass."""
    return "Binary" if y.nunique() == 2 else "Multiclass"


def to_df(data, index=None, columns=None, dtypes=None):
    """Convert a dataset to pd.Dataframe.

    Parameters
    ----------
    data: list, tuple, dict, np.array, sp.matrix, pd.DataFrame or None
        Dataset to convert to a dataframe.  If None or already a
        dataframe, return unchanged.

    index: sequence or pd.Index
        Values for the dataframe's index.

    columns: sequence or None, optional (default=None)
        Name of the columns. Use None for automatic naming.

    dtypes: str, dict, dtype or None, optional (default=None)
        Data types for the output columns. If None, the types are
        inferred from the data.

    Returns
    -------
    df: pd.DataFrame or None
        Transformed dataframe.

    """
    # Get number of columns (list/tuple have no shape and sp.matrix has no index)
    n_cols = lambda data: data.shape[1] if hasattr(data, "shape") else len(data[0])

    if data is not None:
        if not isinstance(data, pd.DataFrame):
            # Assign default column names (dict already has column names)
            if not isinstance(data, dict) and columns is None:
                columns = [f"feature_{str(i)}" for i in range(1, n_cols(data) + 1)]

            # Create dataframe from sparse matrix or directly from data
            if sparse.issparse(data):
                data = pd.DataFrame.sparse.from_spmatrix(data, index, columns)
            else:
                data = pd.DataFrame(data, index, columns)

            if dtypes is not None:
                data = data.astype(dtypes)

        # Convert all column names to str
        data = data.rename(columns=lambda col: str(col))

    return data


def to_series(data, index=None, name="target"):
    """Convert a column to pd.Series.

    Parameters
    ----------
    data: sequence or None
        Data to convert. If None, return unchanged.

    index: sequence or Index, optional (default=None)
        Values for the indices.

    name: string, optional (default="target")
        Name of the target column.

    Returns
    -------
    series: pd.Series or None
        Transformed series.

    """
    if data is not None and not isinstance(data, pd.Series):
        if isinstance(data, pd.DataFrame):
            try:
                data = data[name]
            except Exception:
                data = data.squeeze()
        elif isinstance(data, np.ndarray) and data.ndim > 1:
            data = data.flatten()
        data = pd.Series(data, index=index, name=name)

    return data


def check_features_exist(features: List[str], X: pd.DataFrame):
    """Raise an error if the features are not in the feature dataframe X.

    Parameters
    ----------
    features : List[str]
        List of features to check
    X : pd.DataFrame
        Dataframe of features

    Raises
    ------
    ValueError
        If any feature is not present in the feature dataframe
    """
    missing_features = []
    for fx in features:
        if fx not in X.columns:
            missing_features.append(fx)

    if len(missing_features) != 0:
        raise ValueError(
            f"\n\nColumn(s): {missing_features} not found in the feature dataset!"
            "\nThey are either missing from the features or you have specified "
            "a target column as a feature. Available feature columns are:"
            f"\n{X.columns.to_list()}"
        )


def id_or_display_name(metric, input_ml_usecase, target_ml_usecase):
    """
    Get id or display_name attribute from metric. In time series experiment
    the pull() method retrieves the metrics id to name the columns of the results
    """

    if input_ml_usecase == target_ml_usecase:
        output = metric.id
    else:
        output = metric.display_name

    return output


def variable_return(X, y):
    """Return one or two arguments depending on which is None."""
    if y is None:
        return X
    elif X is None:
        return y
    else:
        return X, y


def get_config(variable: str, globals_d: dict):

    """
    This function is used to access global environment variables.

    Example
    -------
    >>> X_train = get_config('X_train')

    This will return training features.

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

    if variable not in globals_d["pycaret_globals"]:
        raise ValueError(
            f"Variable {variable} not found. Possible variables are: {globals_d['pycaret_globals']}"
        )

    global_var = globals_d[variable]

    logger.info(f"Global variable: {variable} returned as {global_var}")
    logger.info(
        "get_config() successfully completed......................................"
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

    if variable not in globals_d["pycaret_globals"] or variable == "pycaret_globals":
        raise ValueError(
            f"Variable {variable} not found. Possible variables are: {globals_d['pycaret_globals']}"
        )

    globals_d[variable] = value

    # special case
    if not globals_d["gpu_param"] and variable == "n_jobs_param":
        globals_d["gpu_n_jobs_param"] = value

    logger.info(f"Global variable: {variable} updated to {value}")
    logger.info(
        "set_config() successfully completed......................................"
    )


def save_config(file_name: str, globals_d: dict):
    """
    This function is used to save all environment variables to file,
    allowing to later resume modeling without rerunning setup().

    Example
    -------
    >>> save_config('myvars.pkl')

    This will save all environment variables to 'myvars.pkl'.

    """

    function_params_str = ", ".join(
        [f"{k}={v}" for k, v in locals().items() if not k == "globals_d"]
    )

    logger = get_logger()

    logger.info("Initializing save_config()")
    logger.info(f"save_config({function_params_str})")

    globals_to_ignore = {
        "_all_models",
        "_all_models_internal",
        "_all_metrics",
        "_master_model_container",
        "_display_container",
    }

    globals_to_dump = {
        k: v
        for k, v in globals_d.items()
        if k in globals_d["pycaret_globals"] and k not in globals_to_ignore
    }

    import joblib

    joblib.dump(globals_to_dump, file_name)

    logger.info(f"Global variables dumped to {file_name}")
    logger.info(
        "save_config() successfully completed......................................"
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
        "load_config() successfully completed......................................"
    )


def color_df(
    df: pd.DataFrame, color: str, names: list, axis: int = 1
) -> pandas.io.formats.style.Styler:
    return df.style.apply(
        lambda x: [f"background: {color}" if (x.name in names) else "" for _ in x],
        axis=axis,
    )


def get_model_id(
    e, all_models: Dict[str, "pycaret.containers.models.ModelContainer"]
) -> str:
    from pycaret.internal.meta_estimators import get_estimator_from_meta_estimator

    return next(
        (
            k
            for k, v in all_models.items()
            if v.is_estimator_equal(get_estimator_from_meta_estimator(e))
        ),
        None,
    )


def get_model_name(
    e,
    all_models: Dict[str, "pycaret.containers.models.ModelContainer"],
    deep: bool = True,
) -> str:
    all_models = all_models or {}
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
        except Exception:
            name = str(e).split("(")[0]

    return name


def is_special_model(
    e, all_models: Dict[str, "pycaret.containers.models.ModelContainer"]
) -> bool:
    try:
        return all_models[get_model_id(e, all_models)].is_special
    except Exception:
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
            if not isinstance(v, np.ndarray):
                v = list(v)
            param_grid[k] = v
    return param_grid


def np_list_arange(
    start: float, stop: float, step: float, inclusive: bool = False
) -> List[float]:
    """
    Numpy arange returned as list with floating point conversion
    failsafes.
    """
    convert_to_float = (
        isinstance(start, float) or isinstance(stop, float) or isinstance(step, float)
    )
    if convert_to_float:
        stop = float(stop)
        start = float(start)
        step = float(step)
    stop = stop + (step if inclusive else 0)
    range_ = list(np.arange(start, stop, step))
    range_ = [
        start
        if x < start
        else stop
        if x > stop
        else float(round(x, 15))
        if isinstance(x, float)
        else x
        for x in range_
    ]
    range_[0] = start
    range_[-1] = stop - step
    return range_


def calculate_unsupervised_metrics(
    metrics: Dict[str, "pycaret.containers.metrics.MetricContainer"],
    X,
    labels,
    ground_truth: Optional[Any] = None,
    score_dict: Optional[Dict[str, np.array]] = None,
) -> Dict[str, np.array]:

    score_dict = []

    for k, v in metrics.items():
        score_dict.append(
            _calculate_unsupervised_metric(
                v, v.score_func, v.display_name, X, labels, ground_truth
            )
        )

    score_dict = dict([x for x in score_dict if x is not None])
    return score_dict


def _calculate_unsupervised_metric(
    container,
    score_func,
    display_name,
    X,
    labels,
    ground_truth,
):
    if not score_func:
        return None
    target = ground_truth if container.needs_ground_truth else X
    if target is None:
        calculated_metric = 0
    else:
        try:
            calculated_metric = score_func(target, labels, **container.args)
        except Exception:
            calculated_metric = 0

    return (display_name, calculated_metric)


def get_function_params(function: Callable) -> Set[str]:
    return inspect.signature(function).parameters


def calculate_metrics(
    metrics: Dict[str, "pycaret.containers.metrics.MetricContainer"],
    y_test,
    pred,
    pred_proba: Optional[float] = None,
    score_dict: Optional[Dict[str, np.array]] = None,
    weights: Optional[list] = None,
    **additional_kwargs,
) -> Dict[str, np.array]:

    score_dict = []

    for k, v in metrics.items():
        score_dict.append(
            _calculate_metric(
                v,
                v.score_func,
                v.display_name,
                y_test,
                pred,
                pred_proba,
                weights,
                **additional_kwargs,
            )
        )

    score_dict = dict([x for x in score_dict if x is not None])
    return score_dict


def _calculate_metric(
    container, score_func, display_name, y_test, pred_, pred_proba, weights, **kwargs
):
    if not score_func:
        return None
    # get all kwargs in additional_kwargs
    # that correspond to parameters in function signature
    kwargs = {
        **{k: v for k, v in kwargs.items() if k in get_function_params(score_func)},
        **container.args,
    }
    target = pred_proba if container.target == "pred_proba" else pred_
    try:
        calculated_metric = score_func(y_test, target, sample_weight=weights, **kwargs)
    except Exception:
        try:
            calculated_metric = score_func(y_test, target, **kwargs)
        except Exception:
            calculated_metric = 0

    return display_name, calculated_metric


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
            transformers = ("custom_step", transformers)
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
        (
            hasattr(actual_transformer, "fit")
            and hasattr(actual_transformer, "transform")
            and hasattr(actual_transformer, "fit_transform")
        )
        or (
            hasattr(actual_transformer, "fit")
            and hasattr(actual_transformer, "fit_resample")
        )
    ):
        raise TypeError(
            "Transformer must be an object implementing methods 'fit', 'transform' and 'fit_transform'/'fit_resample'."
        )


def get_cv_splitter(
    fold: Optional[Union[int, BaseCrossValidator]],
    default: BaseCrossValidator,
    seed: int,
    shuffle: bool,
    int_default: str = "kfold",
) -> BaseCrossValidator:
    """Returns the cross validator object used to perform cross validation.

    Parameters
    ----------
    fold : Optional[Union[int, BaseCrossValidator]]
        [description]
    default : BaseCrossValidator
        [description]
    seed : int
        [description]
    shuffle : bool
        [description]
    int_default : str, optional
        [description], by default "kfold"

    Returns
    -------
    BaseCrossValidator
        [description]

    Raises
    ------
    ValueError
        [description]
    ValueError
        [description]
    ValueError
        [description]
    TypeError
        [description]
    """
    if not fold:
        return default
    if is_sklearn_cv_generator(fold):
        return fold
    if type(fold) is int:
        if default is not None:
            if isinstance(default, _BaseKFold) and fold <= 1:
                raise ValueError(
                    "k-fold cross-validation requires at least one"
                    " train/test split by setting n_splits=2 or more,"
                    f" got n_splits={fold}."
                )
            try:
                default_copy = deepcopy(default)
                default_copy.n_splits = fold
                return default_copy
            except Exception:
                raise ValueError(f"Couldn't set 'n_splits' to {fold} for {default}.")
        else:
            fold_seed = seed if shuffle else None
            if int_default == "kfold":
                return KFold(fold, random_state=fold_seed, shuffle=shuffle)
            elif int_default == "stratifiedkfold":
                return StratifiedKFold(fold, random_state=fold_seed, shuffle=shuffle)
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
    """Returns the number of folds to use for cross validation

    Parameters
    ----------
    fold : Optional[Union[int, BaseCrossValidator]]
        [description]
    default : [type]
        [description]
    X : [type]
        [description]
    y : [type], optional
        [description], by default None
    groups : [type], optional
        [description], by default None

    Returns
    -------
    int
        [description]
    """
    if not fold:
        fold = default
    if isinstance(fold, int):
        return fold
    else:
        return fold.get_n_splits(X, y=y, groups=groups)


class set_n_jobs(object):
    """
    Context which sets `n_jobs` or `thread_count` to None for passed model.
    """

    def __init__(self, model, n_jobs=None):
        self.params = {}
        self.model = model
        self.n_jobs = n_jobs
        try:
            self.params = {
                k: v
                for k, v in self.model.get_params().items()
                if k.endswith("n_jobs") or k.endswith("thread_count")
            }
        except Exception:
            pass

    def __enter__(self):
        if self.params:
            self.model.set_params(**{k: self.n_jobs for k, v in self.params.items()})

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
        except Exception:
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
    groups: Union[str, pd.DataFrame],
    X_train: pd.DataFrame,
    default: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    if groups is None:
        if default is None:
            return default
        else:
            # Select rows from X_train that match the index from default (all rows)
            return default.loc[X_train.index]
    elif isinstance(groups, str):
        if groups not in X_train.columns:
            raise ValueError(
                f"Column {groups} used for groups is not present in the dataset."
            )
        groups = X_train[groups]
    else:
        groups = groups.loc[X_train.index]
        if groups.shape[0] != X_train.shape[0]:
            raise ValueError(
                f"groups has length {groups.shape[0]} which doesn't match X_train "
                f"length of {len(X_train)}."
            )

    return groups


def get_all_object_vars_and_properties(object):
    """
    Gets all class, static and dynamic attributes from an object.

    Calling ``vars()`` would only return static attributes.

    https://stackoverflow.com/a/59769926

    # TODO: Do both:
    # Option 1: Set fh before calling any model
    "C:\ProgramData\Anaconda3\envs\pycaret_dev\lib\site-packages\sktime\forecasting\base\_sktime.py", line 187
    def _set_fh(self, fh):
    # Option 2: Ignore the exceptions
    """
    d = {}
    for k in object.__dir__():
        try:
            if k[:2] != "__" and type(getattr(object, k, "")).__name__ != "method":
                d[k] = getattr(object, k, "")
        except Exception:
            pass
    return d


def is_fit_var(key):
    return key and (
        (key.endswith("_") and not key.startswith("_")) or (key in ["n_clusters"])
    )


def can_early_stop(
    estimator,
    consider_partial_fit,
    consider_warm_start,
    consider_xgboost,
    params,
):
    """
    From https://github.com/ray-project/tune-sklearn/blob/master/tune_sklearn/tune_basesearch.py.

    Helper method to determine if it is possible to do early stopping.
    Only sklearn estimators with ``partial_fit`` or ``warm_start`` can be early
    stopped. warm_start works by picking up training from the previous
    call to ``fit``.

    Returns
    -------
        bool
            if the estimator can early stop
    """

    logger = get_logger()

    from sklearn.ensemble import BaseEnsemble
    from sklearn.tree import BaseDecisionTree

    try:
        base_estimator = estimator.steps[-1][1]
    except Exception:
        base_estimator = estimator

    if consider_partial_fit:
        can_partial_fit = supports_partial_fit(base_estimator, params=params)
    else:
        can_partial_fit = False

    if consider_warm_start:
        is_not_tree_subclass = not issubclass(type(base_estimator), BaseDecisionTree)
        is_ensemble_subclass = issubclass(type(base_estimator), BaseEnsemble)
        can_warm_start = hasattr(base_estimator, "warm_start") and (
            (
                hasattr(base_estimator, "max_iter")
                and is_not_tree_subclass
                and not is_ensemble_subclass
            )
            or (is_ensemble_subclass and hasattr(base_estimator, "n_estimators"))
        )
    else:
        can_warm_start = False

    is_xgboost = False

    if _check_soft_dependencies("xgboost", extra="models", severity="warning"):
        if consider_xgboost:
            from xgboost.sklearn import XGBModel

            is_xgboost = isinstance(base_estimator, XGBModel)

    logger.info(
        f"can_partial_fit: {can_partial_fit}, can_warm_start: {can_warm_start}, is_xgboost: {is_xgboost}"
    )

    return can_partial_fit or can_warm_start or is_xgboost


def infer_ml_usecase(y: pd.Series) -> Tuple[str, str]:
    c1 = "int" in y.dtype.name
    c2 = y.nunique() <= 20
    c3 = y.dtype.name in ["object", "bool", "category"]

    if (c1 and c2) or c3:
        ml_usecase = "classification"
    else:
        ml_usecase = "regression"

    if y.nunique() > 2 and ml_usecase != "regression":
        subcase = "multi"
    else:
        subcase = "binary"
    return ml_usecase, subcase


def get_columns_to_stratify_by(
    X: pd.DataFrame, y: pd.DataFrame, stratify: Union[bool, List[str]]
) -> pd.DataFrame:
    if not stratify:
        stratify = None
    else:
        if isinstance(stratify, list):
            data = pd.concat([X, y], axis=1)
            if not all(col in data.columns for col in stratify):
                raise ValueError("Column to stratify by does not exist in the dataset.")
            stratify = data[stratify]
        else:
            stratify = y
    return stratify


def check_if_global_is_not_none(globals_d: dict, global_names: dict):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for name, message in global_names.items():
                if globals_d[name] is None:
                    raise ValueError(message)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def df_shrink_dtypes(df, skip=[], obj2cat=True, int2uint=False):
    """Shrink a dataframe.

    Return any possible smaller data types for DataFrame columns.
    Allows `object`->`category`, `int`->`uint`, and exclusion.
    From: https://github.com/fastai/fastai/blob/master/fastai/tabular/core.py

    """

    # 1: Build column filter and typemap
    excl_types, skip = {"category", "datetime64[ns]", "bool"}, set(skip)

    # no int16 as orjson in plotly doesn't support it
    typemap = {
        "int": [
            (np.dtype(x), np.iinfo(x).min, np.iinfo(x).max)
            for x in (np.int8, np.int32, np.int64)
        ],
        "uint": [
            (np.dtype(x), np.iinfo(x).min, np.iinfo(x).max)
            for x in (np.uint8, np.uint32, np.uint64)
        ],
        "float": [
            (np.dtype(x), np.finfo(x).min, np.finfo(x).max)
            for x in (np.float32, np.float64, np.longdouble)
        ],
    }

    if obj2cat:
        # User wants to categorify dtype('Object'), which may not always save space
        typemap["object"] = "category"
    else:
        excl_types.add("object")

    new_dtypes = {}
    exclude = lambda dt: dt[1].name not in excl_types and dt[0] not in skip

    for c, old_t in filter(exclude, df.dtypes.items()):
        t = next((v for k, v in typemap.items() if old_t.name.startswith(k)), None)

        if isinstance(t, list):  # Find the smallest type that fits
            if int2uint and t == typemap["int"] and df[c].min() >= 0:
                t = typemap["uint"]
            new_t = next(
                (r[0] for r in t if r[1] <= df[c].min() and r[2] >= df[c].max()), None
            )
            if new_t and new_t == old_t:
                new_t = None
        else:
            new_t = t if isinstance(t, str) else None

        if new_t:
            new_dtypes[c] = new_t

    return df.astype(new_dtypes)


def get_label_encoder(pipeline):
    """Return the label encoder in the pipeline if any."""
    try:
        encoder = next(
            step[1] for step in pipeline.steps if step[0] == "label_encoding"
        )
        return encoder.transformer
    except StopIteration:
        return


def mlflow_remove_bad_chars(string: str) -> str:
    """Leaves only alphanumeric, spaces _, -, ., / in a string"""
    return "".join(c for c in string if c.isalpha() or c in ("_", "-", ".", " ", "/"))


def deep_clone(estimator: Any) -> Any:
    """Does a deep clone of a model/estimator.

    NOTE: A simple clone does not copy the fitted model (only model hyperparameters)
    # In some cases when we need to copy the fitted parameters, and internal
    # attributes as well. Deep Clone will do this per https://stackoverflow.com/a/33576345/8925915.
    This will copy both model hyperparameters as well as all fitted properties.

    Parameters
    ----------
    estimator : Any
        Estimator to be copied

    Returns
    -------
    Any
        Cloned estimator
    """
    estimator_ = deepcopy(estimator)
    return estimator_


def check_metric(
    actual: pd.Series,
    prediction: pd.Series,
    metric: str,
    round: int = 4,
    train: Optional[pd.Series] = None,
):

    """
    Function to evaluate classification, regression and timeseries metrics.


    actual : pandas.Series
        Actual values of the target variable.


    prediction : pandas.Series
        Predicted values of the target variable.


    train: pandas.Series
        Train values of the target variable.


    metric : str
        Metric to use.


    round: integer, default = 4
        Number of decimal places the metrics will be rounded to.


    Returns:
        float

    """
    from pycaret.containers.metrics import (
        get_all_class_metric_containers,
        get_all_reg_metric_containers,
        get_all_ts_metric_containers,
    )

    globals_dict = {"y": prediction}
    metric_containers = {
        **get_all_class_metric_containers(globals_dict),
        **get_all_reg_metric_containers(globals_dict),
        **get_all_ts_metric_containers(globals_dict),
    }
    metrics = {
        v.name: functools.partial(v.score_func, **(v.args or {}))
        for k, v in metric_containers.items()
    }

    if isinstance(train, pd.Series):
        input_params = [actual, prediction, train]
    else:
        input_params = [actual, prediction]

    # metric calculation starts here

    if metric in metrics:
        try:
            result = metrics[metric](*input_params)
        except Exception:
            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()
            actual = le.fit_transform(actual)
            prediction = le.transform(prediction)
            result = metrics[metric](actual, prediction)
        result = np.around(result, round)
        return float(result)
    else:
        raise ValueError(
            f"Couldn't find metric '{metric}' Possible metrics are: {', '.join(metrics.keys())}."
        )


def _get_metrics_dict(
    metrics_dict: Dict[str, Union[str, _PredictScorer]]
) -> Dict[str, _PredictScorer]:
    """Returns a metrics dictionary in which all values are callables
    of type _PredictScorer

    Parameters
    ----------
    metrics_dict : A metrics dictionary in which some values can be strings.
        If the value is a string, the corresponding callable metric is returned
        e.g. Dictionary Value of 'neg_mean_absolute_error' will return
        make_scorer(mean_absolute_error, greater_is_better=False)
    """
    return_metrics_dict = {}
    for k, v in metrics_dict.items():
        if isinstance(v, str):
            return_metrics_dict[k] = get_scorer(v)
        else:
            return_metrics_dict[k] = v
    return return_metrics_dict


def enable_colab():
    # TODO: Remove with pycaret v3.2.0
    warnings.warn(
        "This function is no longer necessary in pycaret>=3.0 "
        "and will be removed with release 3.2.0",
        DeprecationWarning,
    )


def get_system_logs():

    """
    Read and print 'logs.log' file from current active directory
    """

    with open("logs.log", "r") as file:
        lines = file.read().splitlines()

    for line in lines:
        if not line:
            continue

        columns = [col.strip() for col in line.split(":") if col]
        print(columns)


def _coerce_empty_dataframe_to_none(
    data: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    """Returns None if the data is an empty dataframe or None,
    else return the dataframe as is.

    Parameters
    ----------
    data : Optional[pd.DataFrame]
        Dataframe to be checked or None

    Returns
    -------
    Optional[pd.DataFrame]
        Returned Dataframe OR None (if dataframe is empty or None)
    """
    if isinstance(data, pd.DataFrame) and data.empty:
        return None
    else:
        return data


def _resolve_dict_keys(
    dict_: Dict[str, Any], key: str, defaults: Dict[str, Any]
) -> Any:
    """Returns the value of "key" from `dict`. If key is not present, then the
    value is picked from the `defaults` dictionary. Note that `defaults` must
    contain the `key` else this will give an error.

    Parameters
    ----------
    dict : Dict[str, Any]
        The dictionary from which the "key"'s value must be obtained
    key : str
        The "key" whose value must be obtained
    defaults : Dict[str, Any]
        The dictionary containing the default value of the "key"

    Returns
    -------
    Any
        The value of the "key"

    Raises
    ------
    KeyError
        If the `defaults` dictionary does not contain the `key`
    """
    if key not in defaults:
        raise KeyError(f"Key '{key}' not present in Defaults dictionary.")
    return dict_.get(key, defaults[key])


def get_allowed_engines(
    estimator: str, all_allowed_engines: Dict[str, List[str]]
) -> Optional[List[str]]:
    """Get all the allowed engines for the specified estimator

    Parameters
    ----------
    estimator : str
        Identifier for the model for which the engines should be retrieved,
        e.g. "auto_arima"
    all_allowed_engines : Dict[str, List[str]]
        All allowed engines for models of this experiment class to which the
        model belongs

    Returns
    -------
    Optional[List[str]]
        The allowed engines for the model. If the model only supports the
        default engine, then it return `None`.
    """
    allowed_engines = all_allowed_engines.get(estimator, None)
    return allowed_engines


class LazyExperimentMapping(Mapping):
    """
    This class provides a dict-like interface while calling properties lazily.

    This improves performance if those properties are not accessed.
    """

    def __init__(self, experiment: "_PyCaretExperiment"):
        self._experiment = experiment
        self._keys = self._experiment._variable_keys.union(
            self._experiment._property_keys
        )
        if "variables" in self._keys:
            self._keys.remove("variables")
        self._cache = {}

    def __getitem__(self, key):
        if key in self._cache:
            return self._cache[key]
        if key in self._keys:
            item = getattr(self._experiment, key, None)
            self._cache[key] = item
            return item
        else:
            raise KeyError(key)

    def __len__(self):
        return len(self._keys)

    def __iter__(self):
        return iter(self._keys)
