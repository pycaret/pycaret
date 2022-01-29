"""Module containing utility functions for time series analysis"""

import warnings
from typing import Optional, List, Tuple, Any

import pandas as pd

from sktime.transformations.series.difference import Differencer


def _reconcile_order_and_lags(
    order_list: Optional[List[Any]] = None, lags_list: Optional[List[Any]] = None
) -> Tuple[List[int], List[str]]:
    """Reconciles the differences to lags and returns the names
    If order_list is provided, it is converted to lags_list.
    If lags_list is provided, it is uses as is.
    If none are provided, assumes order = [1]
    If both are provided, returns empty lists

    Parameters
    ----------
    order_list : Optional[List[Any]], optional
        order of the differences, by default None
    lags_list : Optional[List[Any]], optional
        lags of the differences, by default None

    Returns
    -------
    Tuple[List[int], List[str]]
        (1) Reconciled lags_list AND
        (2) Names corresponding to the difference lags
    """

    return_lags = []
    return_names = []

    if order_list is not None and lags_list is not None:
        msg = "ERROR: Can not specify both 'order_list' and 'lags_list'. Please specify only one."
        warnings.warn(msg)  # print on screen
        return return_lags, return_names
    elif order_list is not None:
        for order in order_list:
            return_lags.append([1] * order)
            return_names.append(f"Order={order}")
    elif lags_list is not None:
        return_lags = lags_list
        for lags in lags_list:
            return_names.append("Lags=" + str(lags))
    else:
        # order_list is None and lags_list is None
        # Only perform first difference by default
        return_lags.append([1])
        return_names.append("Order=1")

    return return_lags, return_names


def _get_diffs(data: pd.Series, lags_list: List[Any]) -> List[pd.Series]:
    """Returns the requested differences of the provided `data`

    Parameters
    ----------
    data : pd.Series
        Data whose differences have to be computed
    lags_list : List[Any]
        lags of the differences

    Returns
    -------
    List[pd.Series]
        List of differences per the lags_list
    """

    diffs = [Differencer(lags=lags).fit_transform(data) for lags in lags_list]
    return diffs


def get_diffs(
    data: pd.Series,
    order_list: Optional[List[Any]] = None,
    lags_list: Optional[List[Any]] = None,
) -> Tuple[List[pd.Series], List[str]]:
    """Returns the requested differences of the provided `data`
    Either `order_list` or `lags_list` can be provided but not both.

    Refer to the following for more details:
    https://www.sktime.org/en/latest/api_reference/auto_generated/sktime.transformations.series.difference.Differencer.html
    Note: order = 2 is equivalent to lags = [1, 1]

    Parameters
    ----------
    data : pd.Series
        Data whose differences have to be computed
    order_list : Optional[List[Any]], optional
        order of the differences, by default None
    lags_list : Optional[List[Any]], optional
        lags of the differences, by default None

    Returns
    -------
    Tuple[List[pd.Series], List[str]]
        (1) List of differences per the order_list or lags_list AND
        (2) Names corresponding to the differences
    """

    lags_list_, names = _reconcile_order_and_lags(
        order_list=order_list, lags_list=lags_list
    )
    diffs = _get_diffs(data=data, lags_list=lags_list_)
    return diffs, names


def _get_diff_name_list(data, model_name, data_kwargs):
    # TODO: Fix this
    # model_name = None

    data_kwargs = data_kwargs or {}
    order_list = data_kwargs.get("order_list", None)
    lags_list = data_kwargs.get("lags_list", None)

    diff_list = []
    name_list = []
    if order_list or lags_list:
        diff_list, name_list = get_diffs(
            data=data, order_list=order_list, lags_list=lags_list
        )

    if len(diff_list) != 0:
        diff_list = [data] + diff_list
        name_list = ["Actual" if model_name is None else "Residuals"] + name_list
    else:
        # Issue with reconciliation of orders and diffs
        diff_list = [data]
        name_list = ["Actual" if model_name is None else "Residuals"]

    return diff_list, name_list
