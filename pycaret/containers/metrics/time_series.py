# Module: containers.metrics.time_series
# Author: Antoni Baum (Yard1) <antoni.baum@protonmail.com> and Miguel Trejo <amtrema@hotmail.com>
# License: MIT

# The purpose of this module is to serve as a central repository of time series metrics. The `time_series` module will
# call `get_all_metrics_containers()`, which will return instances of all classes in this module that have `TimeSeriesMetricContainer`
# as a base (but not `TimeSeriesMetricContainer` itself). In order to add a new model, you only need to create a new class that has
# `TimeSeriesMetricContainer` as a base, set all of the required parameters in the `__init__` and then call `super().__init__`
# to complete the process. Refer to the existing classes for examples.

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn import metrics  # type: ignore
from sklearn.metrics._scorer import _BaseScorer  # type: ignore
from sktime.performance_metrics.forecasting._functions import (  # type: ignore
    mean_absolute_percentage_error,
    mean_absolute_scaled_error,
    mean_squared_scaled_error,
)

import pycaret.internal.metrics
from pycaret.containers.metrics.base_metric import MetricContainer


class TimeSeriesMetricContainer(MetricContainer):
    """
    Base time series metric container class, for easier definition of containers.
    Ensures consistent format before being turned into a dataframe row.
    Parameters
    ----------
    id : str
        ID used as index.
    name : str
        Full name.
    score_func : type
        The callable used for the score function, eg. sklearn.metrics.accuracy_score.
    scorer : str or callable, default = None
        The scorer passed to models. Can be a string representing a built-in sklearn scorer,
        a sklearn Scorer object, or None, in which case a Scorer object will be created from
        score_func and args.
    target : str, default = 'pred'
        The target of the score function. Only 'pred' is supported for regression.
    args : dict, default = {} (empty dict)
        The arguments to always pass to constructor when initializing score_func of class_def class.
    display_name : str, default = None
        Display name (shorter than name). Used in display dataframe header. If None or empty, will use name.
    greater_is_better: bool, default = True
        Whether score_func is a score function (default), meaning high is good,
        or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the score_func.
    is_custom : bool, default = False
        Is the metric custom. Should be False for all metrics defined in PyCaret.
    Attributes
    ----------
    id : str
        ID used as index.
    name : str
        Full name.
    score_func : type
        The callable used for the score function, eg. metrics.accuracy_score.
    scorer : str or callable
        The scorer passed to models. Can be a string representing a built-in sklearn scorer,
        a sklearn Scorer object, or None, in which case a Scorer object will be created from
        score_func and args.
    target : str
        The target of the score function.
        - 'pred' for the prediction table
    args : dict
        The arguments to always pass to constructor when initializing score_func of class_def class.
    display_name : str
        Display name (shorter than name). Used in display dataframe header.
    greater_is_better: bool
        Whether score_func is a score function (default), meaning high is good,
        or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the score_func.
    is_custom : bool
        Is the metric custom. Should be False for all metrics defined in PyCaret.
    """

    def __init__(
        self,
        id: str,
        name: str,
        score_func: type,
        scorer: Optional[Union[str, _BaseScorer]] = None,
        target: str = "pred",
        args: Dict[str, Any] = None,
        display_name: Optional[str] = None,
        greater_is_better: bool = True,
        is_custom: bool = False,
    ) -> None:

        allowed_targets = ["pred"]
        if target not in allowed_targets:
            raise ValueError(f"Target must be one of {', '.join(allowed_targets)}.")

        if not args:
            args = {}
        if not isinstance(args, dict):
            raise TypeError("args needs to be a dictionary.")

        scorer = (
            scorer
            if scorer
            else pycaret.internal.metrics.make_scorer_with_error_score(
                score_func,
                greater_is_better=greater_is_better,
                error_score=0.0,
                **args,
            )
        )

        super().__init__(
            id=id,
            name=name,
            score_func=score_func,
            scorer=scorer,
            args=args,
            display_name=display_name,
            greater_is_better=greater_is_better,
            is_custom=is_custom,
        )

        self.target = target

    def get_dict(self, internal: bool = True) -> Dict[str, Any]:
        """
        Returns a dictionary of the model properties, to
        be turned into a pandas DataFrame row.
        Parameters
        ----------
        internal : bool, default = True
            If True, will return all properties. If False, will only
            return properties intended for the user to see.
        Returns
        -------
        dict of str : Any
        """
        d = {
            "ID": self.id,
            "Name": self.name,
            "Display Name": self.display_name,
            "Score Function": self.score_func,
            "Scorer": self.scorer,
            "Target": self.target,
            "Args": self.args,
            "Greater is Better": self.greater_is_better,
            "Custom": self.is_custom,
        }

        return d


def _smape_loss(y_true, y_pred, **kwargs):
    """Wrapper for sktime metrics"""
    y_true = _check_series(y_true)
    y_pred = _check_series(y_pred)
    return mean_absolute_percentage_error(
        y_true=y_true, y_pred=y_pred, symmetric=True, **kwargs
    )


def mape(y_true, y_pred, **kwargs):
    """Wrapper for sktime metrics"""
    y_true = _check_series(y_true)
    y_pred = _check_series(y_pred)
    return mean_absolute_percentage_error(
        y_true=y_true, y_pred=y_pred, symmetric=False, **kwargs
    )


def mase(y_true, y_pred, y_train, sp):
    """Wrapper for sktime metrics"""
    return mean_absolute_scaled_error(
        y_true=_check_series(y_true),
        y_pred=_check_series(y_pred),
        sp=sp,
        y_train=_check_series(y_train),
    )


def rmsse(y_true, y_pred, y_train, sp):
    """Wrapper for sktime metrics"""
    return mean_squared_scaled_error(
        y_true=_check_series(y_true),
        y_pred=_check_series(y_pred),
        sp=sp,
        y_train=_check_series(y_train),
        square_root=True,
    )


def coverage(y_true, y_pred, lower: pd.Series, upper: pd.Series):
    """Returns the percentage of actual values that are within the
    prediction interval. Higher score is better.
    NOTE: If lower and upper have NAN values, it returns np.nan
    """
    y_true = _check_series(y_true)
    y_pred = _check_series(y_pred)
    lower = _check_series(lower)
    upper = _check_series(upper)

    # First combine the true, upper and lower values and keep only those values
    # that match the y_true indices. Then if any of the values have NAN in them,
    # return NAN, else proceed to calculating metric.

    # NAN's can occur due to following reasons:
    # (1) Indices match between y_true and (lower or upper) but lower or upper
    # values have NAN (i.e. forecaster does not support prediction intervals)
    # (2) y_true is NAN (i.e. model has been finalized)
    # (3) Indices do not match up between y_true and (lower or upper) - failsafe

    combined = pd.concat([y_true, lower, upper], axis=1)
    combined.columns = ["y_true", "lower", "upper"]
    combined.dropna(subset=["y_true"], inplace=True)

    # Override lower and upper to only those indices that match y_true indices
    lower = combined["lower"]
    upper = combined["upper"]

    if y_true.isna().any() or lower.isna().any() or upper.isna().any():
        return np.nan

    in_limits = np.logical_and(y_true > lower, y_true < upper)
    return sum(in_limits) / len(in_limits)


def _check_series(y):
    """
    Check whether y is pandas.Series. Pycaret Experiment
    internally converts data to pandas.DataFrame.
    """
    if isinstance(y, pd.Series):
        return y
    elif isinstance(y, pd.DataFrame):
        return _set_y_as_series(y)


def _set_y_as_series(y):
    """Set first column of a DataFrame as pandas.Series"""
    return pd.Series(y.iloc[:, 0])


class MASEMetricContainer(TimeSeriesMetricContainer):
    def __init__(self, globals_dict: dict) -> None:
        super().__init__(
            id="mase", name="MASE", score_func=mase, greater_is_better=False
        )


class RMSSEMetricContainer(TimeSeriesMetricContainer):
    def __init__(self, globals_dict: dict) -> None:
        super().__init__(
            id="rmsse", name="RMSSE", score_func=rmsse, greater_is_better=False
        )


class MAEMetricContainer(TimeSeriesMetricContainer):
    def __init__(self, globals_dict: dict) -> None:
        super().__init__(
            id="mae",
            name="MAE",
            score_func=metrics.mean_absolute_error,
            greater_is_better=False,
            scorer="neg_mean_absolute_error",
        )


class RMSEMetricContainer(TimeSeriesMetricContainer):
    def __init__(self, globals_dict: dict) -> None:

        super().__init__(
            id="rmse",
            name="RMSE",
            score_func=metrics.mean_squared_error,
            greater_is_better=False,
            args={"squared": False},
            scorer="neg_root_mean_squared_error",
        )


class MAPEMetricContainer(TimeSeriesMetricContainer):
    def __init__(self, globals_dict: dict) -> None:
        super().__init__(
            id="mape", name="MAPE", score_func=mape, greater_is_better=False
        )


class SMAPEMetricContainer(TimeSeriesMetricContainer):
    def __init__(self, globals_dict: dict) -> None:
        super().__init__(
            id="smape", name="SMAPE", score_func=_smape_loss, greater_is_better=False
        )


class R2MetricContainer(TimeSeriesMetricContainer):
    def __init__(self, globals_dict: dict) -> None:

        super().__init__(
            id="r2",
            name="R2",
            score_func=metrics.r2_score,
            greater_is_better=True,
            scorer="r2",
        )


class CovProbMetricContainer(TimeSeriesMetricContainer):
    def __init__(self, globals_dict: dict) -> None:
        super().__init__(
            id="coverage", name="COVERAGE", score_func=coverage, greater_is_better=True
        )


def get_all_metric_containers(
    globals_dict: dict, raise_errors: bool = True
) -> Dict[str, TimeSeriesMetricContainer]:
    return pycaret.containers.base_container.get_all_containers(
        globals(), globals_dict, TimeSeriesMetricContainer, raise_errors
    )
