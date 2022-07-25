"""Module containing utility functions for time series analysis"""

import math
import re
import warnings
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Tuple

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
    # Default na_handling changed in sktime 0.13.0 (compared to 0.11.x).
    # https://www.sktime.org/en/latest/changelog.html#id3. Hence hard coding.
    diffs = [
        Differencer(lags=lags, na_handling="drop_na").fit_transform(data)
        for lags in lags_list
    ]
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


def _get_diff_name_list(
    data: pd.Series, data_name: Optional[str] = None, data_kwargs: Optional[Dict] = None
) -> Tuple[List[pd.Series], List[str]]:
    """Returns the data along with any differences that are requested
    If no differences are requested, only the original data is returned.

    Parameters
    ----------
    data : pd.Series
        Data whose differences have to be (potentially) computed
    data_name : Optional[str], optional
        Name of the data, by default None
    data_kwargs : Optional[Dict], optional
        Can (optionally) contain keywords 'order_list' or 'lags_list' corresponding
        to the difference order or lags that need to be used for differencing.
        Can not specify both 'order_list' and 'lags_list', by default None

    Returns
    -------
    Tuple[List[pd.Series], List[str]]
        (1) Original Data + (optionally) List of differences per the order_list or
            lags_list AND
        (2) Names corresponding to the original data and the differences
    """
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
        name_list = [data_name] + name_list
    else:
        # Issue with reconciliation of orders and diffs
        diff_list = [data]
        name_list = [data_name]

    return diff_list, name_list


def get_sp_from_str(str_freq: str) -> int:
    """Takes the seasonal period as string detects if it is alphanumeric and returns its integer equivalent.
        For example -
        input - '30W'
        output - 26
        explanation - we take the lcm of 30 and 52 ( as W = 52) which in this case is 780.
        And the output is ( lcm / prefix). Here, 780 / 30 = 26.

    Parameters
    ----------
    str_freq : str
        frequency of the dataset passed as a string

    Returns
    -------
    int
        integer equivalent of the string frequency

    Raises
    ------
    ValueError
        If the frequency suffix does not correspond to any of the values in the
        class SeasonalPeriod then the error is thrown.
    """
    str_freq = str_freq.split("-")[0] or str_freq
    # Checking whether the index_freq contains both digit and alphabet
    if bool(re.search(r"\d", str_freq)):
        temp = re.compile("([0-9]+)([a-zA-Z]+)")
        res = temp.match(str_freq).groups()
        # separating the digits and alphabets
        if res[1] in SeasonalPeriod.__members__:
            prefix = int(res[0])
            value = SeasonalPeriod[res[1]].value
            lcm = abs(value * prefix) // math.gcd(value, prefix)
            seasonal_period = int(lcm / prefix)
            return seasonal_period
        else:
            raise ValueError(
                f"Unsupported Period frequency: {str_freq}, valid Period frequency "
                f"suffixes are: {', '.join(SeasonalPeriod.__members__.keys())}"
            )
    else:

        if str_freq in SeasonalPeriod.__members__:
            seasonal_period = SeasonalPeriod[str_freq].value
            return seasonal_period
        else:
            raise ValueError(
                f"Unsupported Period frequency: {str_freq}, valid Period frequency "
                f"suffixes are: {', '.join(SeasonalPeriod.__members__.keys())}"
            )


class SeasonalPeriod(IntEnum):
    """ENUM corresponding to Seasonal Periods

    Currently supports a subset of these. Eventually try to support all.
    https://stackoverflow.com/questions/35339139/what-values-are-valid-in-pandas-freq-tags/35339226#35339226
    B        business day frequency
    C        custom business day frequency
    D        calendar day frequency
    W        weekly frequency
    M        month end frequency
    SM       semi-month end frequency (15th and end of month)
    BM       business month end frequency
    CBM      custom business month end frequency
    MS       month start frequency
    SMS      semi-month start frequency (1st and 15th)
    BMS      business month start frequency
    CBMS     custom business month start frequency
    Q        quarter end frequency
    BQ       business quarter end frequency
    QS       quarter start frequency
    BQS      business quarter start frequency
    A, Y     year end frequency
    BA, BY   business year end frequency
    AS, YS   year start frequency
    BAS, BYS business year start frequency
    BH       business hour frequency
    H        hourly frequency
    T, min   minutely frequency
    S        secondly frequency
    L, ms    milliseconds
    U, us    microseconds
    N        nanoseconds
    """

    B = 5
    C = 5
    D = 7
    W = 52
    M = 12
    SM = 24
    BM = 12
    CBM = 12
    MS = 12
    SMS = 24
    BMS = 12
    CBMS = 12
    Q = 4
    BQ = 4
    QS = 4
    BQS = 4
    A = 1
    Y = 1
    BA = 1
    BY = 1
    AS = 1
    YS = 1
    BAS = 1
    BYS = 1
    # BH = ??
    H = 24
    T = 60
    min = 60
    S = 60
    # L = ??
    # ms = ??
    # U = ??
    # us = ??
    # N = ??


class TSModelTypes(Enum):
    BASELINE = "baseline"
    CLASSICAL = "classical"
    LINEAR = "linear"
    NEIGHBORS = "neighbors"
    TREE = "tree"


class TSExogenousPresent(Enum):
    YES = "Present"
    NO = "Not Present"


class TSApproachTypes(Enum):
    UNI = "Univariate"
    MULTI = "Multivariate"


class TSAllowedPlotDataTypes(Enum):
    ORIGINAL = "original"
    TRANSFORMED = "transformed"
    IMPUTED = "imputed"
