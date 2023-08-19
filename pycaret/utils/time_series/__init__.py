"""Module containing utility functions for time series analysis"""

import math
import re
import warnings
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pmdarima.arima.utils import ndiffs
from sktime.param_est.seasonality import SeasonalityACF
from sktime.transformations.series.difference import Differencer
from sktime.utils.plotting import plot_series


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

    return_lags: List = []
    return_names: List = []

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
) -> Tuple[List[pd.Series], List[Optional[str]]]:
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

    diff_list: List = []
    name_list: List = []
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


def auto_detect_sp(
    y: pd.Series, verbose: bool = False, plot: bool = False
) -> Tuple[int, list, int]:
    """Automatically detects the seasonal period of the time series.
    The time series is internally differenced before the seasonality is detected
    using ACF.

    Parameters
    ----------
    y : pd.Series
        Time series whose seasonal period has to be detected
    verbose : bool, optional
        Whether to print intermediate values , by default False
    plot : bool, optional
        Whether to plot original and differenced data, by default False

    Returns
    -------
    Tuple[int, list, int]
        (1) Primary Seasonal Period
        (2) List of all significant seasonal periods
        (3) The number of lags used to detected seasonality
    """

    yt = y.copy()
    for i in np.arange(ndiffs(yt)):
        if verbose:
            print(f"Differencing: {i+1}")
        differencer = Differencer()
        yt = differencer.fit_transform(yt)

    if plot:
        _ = plot_series(y, yt, labels=["original", "differenced"])

    nobs = len(yt)
    # Increasing lags otherwise high sp values are not detected since they are
    # limited by internal nlags calculation in SeasonalityACF
    # lags_to_use = min(10 * np.log10(nobs), nobs - 1)
    # lags_to_use = max(lags_to_use, nobs/3)
    lags_to_use = int((nobs - 1) / 2)
    sp_est = SeasonalityACF(nlags=lags_to_use)
    sp_est.fit(yt)

    primary_sp = sp_est.get_fitted_params().get("sp")
    significant_sps = sp_est.get_fitted_params().get("sp_significant")
    if isinstance(significant_sps, np.ndarray):
        significant_sps = significant_sps.tolist()

    if verbose:
        print(f"\tLags used for seasonal detection: {lags_to_use}")
        print(f"\tDetected Significant SP: {significant_sps}")
        print(f"\tDetected Primary SP: {primary_sp}")

    return primary_sp, significant_sps, lags_to_use


def remove_harmonics_from_sp(
    significant_sps: list, harmonic_order_method: str = "raw_strength"
) -> list:
    """Remove harmonics from the list provided. Similar to Kats - Ref:
    https://github.com/facebookresearch/Kats/blob/v0.2.0/kats/detectors/seasonality.py#L311-L321

    Parameters
    ----------
    significant_sps : list
        The list of significant seasonal periods (ordered by significance)
    harmonic_order_method: str, default = "harmonic_strength"
        This determines how the harmonics are replaced.
        Allowed values are "harmonic_strength", "harmonic_max" or "raw_strength.
        - If set to  "harmonic_strength", then lower seasonal period is replaced by its
        highest strength harmonic seasonal period in same position as the lower seasonal period.
        - If set to  "harmonic_max", then lower seasonal period is replaced by its
        highest harmonic seasonal period in same position as the lower seasonal period.
        - If set to  "raw_strength", then lower seasonal periods is removed and the
        higher harmonic seasonal periods is retained in its original position
        based on its seasonal strength.

        e.g. Assuming detected seasonal periods in strength order are [2, 3, 4, 50]
        and remove_harmonics = True, then:
        - If harmonic_order_method = "harmonic_strength", result = [4, 3, 50]
        - If harmonic_order_method = "harmonic_max", result = [50, 3, 4]
        - If harmonic_order_method = "raw_strength", result = [3, 4, 50]

    Returns
    -------
    list
        The list of significant seasonal periods with harmonics removed
    """
    # Convert period to frequency for harmonic removal
    significant_freqs = [1 / sp for sp in significant_sps]

    if len(significant_freqs) > 1:
        # Sort from lowest freq to highest
        significant_freqs = sorted(significant_freqs)
        # Start from highest freq and remove it if it is a multiple of a lower freq
        # i.e if it is a harmonic of a lower frequency
        for i in range(len(significant_freqs) - 1, 0, -1):
            for j in range(i - 1, -1, -1):
                fraction = (significant_freqs[i] / significant_freqs[j]) % 1
                if fraction < 0.001 or fraction > 0.999:
                    significant_freqs.pop(i)
                    break

    # Convert frequency back to period
    # Rounding, else there is precision issues
    filtered_sps = [round(1 / freq, 4) for freq in significant_freqs]

    if harmonic_order_method == "raw_strength":
        # Keep order of significance
        final_filtered_sps = [sp for sp in significant_sps if sp in filtered_sps]
    else:
        # Replace higher strength sp with lower strength harmonic sp
        retained = [True if sp in filtered_sps else False for sp in significant_sps]
        final_filtered_sps = []
        for i, sp_iter in enumerate(significant_sps):
            if retained[i] is False:
                div = [sp / sp_iter for sp in significant_sps]
                div_int = [round(elem) for elem in div]
                equal = [True if a == b else False for a, b in zip(div, div_int)]
                replacement_candidates = [
                    sp for sp, eq in zip(significant_sps, equal) if eq
                ]
                if harmonic_order_method == "harmonic_max":
                    replacement_sp = max(replacement_candidates)
                elif harmonic_order_method == "harmonic_strength":
                    replacement_sp = replacement_candidates[
                        [
                            i
                            for i, candidate in enumerate(replacement_candidates)
                            if candidate != sp_iter
                        ][0]
                    ]
                final_filtered_sps.append(replacement_sp)
            else:
                final_filtered_sps.append(sp_iter)
        # Replacement for ordered set: https://stackoverflow.com/a/53657523/8925915
        final_filtered_sps = list(dict.fromkeys(final_filtered_sps))

    return final_filtered_sps


def clean_time_index(
    data: pd.DataFrame, freq: str, index_col: Optional[str] = None
) -> pd.DataFrame:
    """Cleans and sets the index of the dataframe in a pycaret compliant format.

    Allowed index in pycaret can be of type Int64Index, DatetimeIndex, or PeriodIndex.
    Steps followed by this function (in order) are as follows:
    1. If column is provided and is of type string, it is converted to PerodIndex.
    2. If column is provided, it is set as index
    3. If Index is DataTimeIndex, it is converted to PeriodIndex (IntIndex is left as is)
    2. If Index is PeriodIndex, then missing index values are added and filled with np.nan.

    Parameters
    ----------
    data : pd.DataFrame
        Data that needs to have its index cleaned
    freq : str
        The frequency of the data. Valid values are the ones that can be provided
        to pd.period_range, pd.PeriodIndex, pd.to_period. Examples: "H", "D", "M".
    index_col : Optional[str], optional
        If index values are in a column, then this argument is the column name,
        by default None which assumes that the index has already been set.

    Returns
    -------
    pd.DataFrame
        Cleaned data with index = PeriodIndex or IntIndex and with missing index
        values filled with np.nan (if PeriodIndex only).
    """
    data_ = data.copy()

    # Step 1: Set the index if not already set
    if index_col is not None:
        # If column has string values, convert to PeriodIndex since pycaret
        # works best with PeriodIndex. For all other index types (DatetimeIndex,
        # PeriodIndex, Int64Index), leave as is.
        if isinstance(data_[index_col][0], str):
            data_[index_col] = pd.PeriodIndex(data_[index_col], freq=freq)
        data_.set_index(index_col, inplace=True)

    # Step 2: Convert DateTimeIndex to PeriodIndex (IntIndex is left as is)
    if isinstance(data_.index, pd.DatetimeIndex):
        try:
            data_.index = data_.index.to_period(freq=freq)
        except AttributeError:
            raise AttributeError(
                f"You are using a frequency of '{freq}' along with a DatetimeIndex. "
                "PyCaret internally converts DateTimeIndex to PeriodIndex and this "
                "frequency is not supported by PeriodIndex.\nAs an alternative, you "
                "can convert the index to Int64Index by using the following code "
                "and pass to pycaret:\n"
                ">>> data.index = data.index.astype('int64') "
                "\nLater, when you get the results back from "
                "pycaret, you can convert them back to DatetimeIndex by using:\n"
                ">>> data.index = pd.to_datetime(data.index)"
            )

    # Step 3: Fill missing index values (only if index is PeriodIndex, not for IntIndex)
    if isinstance(data_.index, pd.PeriodIndex):
        idx = pd.period_range(min(data_.index), max(data_.index), freq=freq)
        data_ = data_.reindex(idx, fill_value=np.nan)

    return data_


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
