from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import statsmodels.api as sm
from pmdarima.arima.utils import ndiffs, nsdiffs
from statsmodels.tools.sm_exceptions import MissingDataError as SmMissingDataError
from statsmodels.tsa.api import kpss
from statsmodels.tsa.stattools import adfuller

from pycaret.internal.logging import get_logger
from pycaret.internal.tests import _format_test_results
from pycaret.internal.tests.stats import _is_gaussian, _summary_stats
from pycaret.utils.time_series import _get_diff_name_list
from pycaret.utils.time_series.exceptions import MissingDataError

logger = get_logger()


def run_test(
    data: pd.Series,
    test: str,
    data_name: Optional[str] = None,
    alpha: float = 0.05,
    data_kwargs: Optional[Dict] = None,
    *kwargs,
) -> pd.DataFrame:
    """Performs the specified test on the data.

    Parameters
    ----------
    data : pd.Series
        Time Series data on which the test needs to be performed
    test : str
        Test to run on the data. Allowed tests are
        - "summary": Summary Statistics
        - "white_noise": Ljung-Box Test for white noise
        - "adf": ADF test for difference stationarity
        - "kpss": KPSS test for trend stationarity
        - "stationarity": ADF and KPSS test
        - "normality": Shapiro Test for Normality
        - "all": All of the above tests
    alpha : float, optional
        Significance Level, by default 0.05
    data_kwargs : Optional[Dict], optional
        Users can specify `lags list` or `order_list` to run the test for the
        data as well as for its lagged versions, by default None

        >>> run_test(data=data, test="white_noise", data_kwargs={"order_list": [1, 2]})
        >>> run_test(data=data, test="white_noise", data_kwargs={"lags_list": [1, [1, 12]]})

    Returns
    -------
    pd.DataFrame
        Detailed results dataframe

    Raises
    ------
    ValueError
        Wrong test name provided
    """
    if test == "all":
        results = _test_all(
            data=data, data_name=data_name, alpha=alpha, data_kwargs=data_kwargs
        )
    elif test == "summary":
        results = _summary_stats(
            data=data, data_name=data_name, data_kwargs=data_kwargs
        )
    elif test == "white_noise":
        results = _is_white_noise(
            data=data,
            data_name=data_name,
            alpha=alpha,
            verbose=True,
            data_kwargs=data_kwargs,
            *kwargs,
        )[1]
    elif test == "stationarity":
        results = _is_stationary(
            data=data, data_name=data_name, alpha=alpha, data_kwargs=data_kwargs
        )
    elif test == "adf":
        results = _is_stationary_adf(
            data=data,
            data_name=data_name,
            alpha=alpha,
            verbose=True,
            data_kwargs=data_kwargs,
        )[1]
    elif test == "kpss":
        results = _is_stationary_kpss(
            data=data,
            data_name=data_name,
            alpha=alpha,
            verbose=True,
            data_kwargs=data_kwargs,
        )[1]
    elif test == "normality":
        results = _is_gaussian(
            data=data,
            data_name=data_name,
            alpha=alpha,
            verbose=True,
            data_kwargs=data_kwargs,
        )[1]
    else:
        raise ValueError(f"Tests: '{test}' is not supported.")
    return results


########################
# Combined Tests ####
########################
def _test_all(
    data: pd.Series,
    data_name: Optional[str] = None,
    alpha: float = 0.05,
    data_kwargs: Optional[Dict] = None,
) -> pd.DataFrame:
    """Performs several tests on on the time series data

    - Summary Statistics
    - Ljung-Box Test for white noise
    - ADF test for difference stationarity
    - KPSS test for trend stationarity
    - Shapiro Test for Normality

    Parameters
    ----------
    data : pd.Series
        Time Series data on which the test needs to be performed
    alpha : float, optional
        Significance Level, by default 0.05
    data_kwargs : Optional[Dict], optional
        Users can specify `lags list` or `order_list` to run the test for the
        data as well as for its lagged versions, by default None

        >>> _test_all(data=data, data_kwargs={"order_list": [1, 2]})
        >>> _test_all(data=data, data_kwargs={"lags_list": [1, [1, 12]]})

    Returns
    -------
    pd.DataFrame
        Detailed results dataframe
    """
    result_summary_stats = _summary_stats(
        data=data, data_name=data_name, data_kwargs=data_kwargs
    )
    result_wn = _is_white_noise(
        data, data_name=data_name, alpha=alpha, verbose=True, data_kwargs=data_kwargs
    )
    result_adf = _is_stationary_adf(
        data, data_name=data_name, alpha=alpha, verbose=True, data_kwargs=data_kwargs
    )
    result_kpss = _is_stationary_kpss(
        data, data_name=data_name, alpha=alpha, verbose=True, data_kwargs=data_kwargs
    )
    result_normality = _is_gaussian(
        data, data_name=data_name, alpha=alpha, verbose=True, data_kwargs=data_kwargs
    )

    all_dfs = [
        result_summary_stats,
        result_wn[1],
        result_adf[1],
        result_kpss[1],
        result_normality[1],
    ]
    final = pd.concat(all_dfs)
    return final


def _is_stationary(
    data: pd.Series,
    data_name: Optional[str] = None,
    alpha: float = 0.05,
    data_kwargs: Optional[Dict] = None,
) -> pd.DataFrame:
    """Performs Stationarity tests on time series data

    - ADF test for difference stationarity
    - KPSS test for trend stationarity

    Parameters
    ----------
    data : pd.Series
        Time Series data on which the test needs to be performed
    alpha : float, optional
        Significance Level, by default 0.05
    data_kwargs : Optional[Dict], optional
        Users can specify `lags list` or `order_list` to run the test for the
        data as well as for its lagged versions, by default None

        >>> _is_stationary(data=data, data_kwargs={"order_list": [1, 2]})
        >>> _is_stationary(data=data, data_kwargs={"lags_list": [1, [1, 12]]})

    Returns
    -------
    pd.DataFrame
        Detailed results dataframe
    """
    result_adf = _is_stationary_adf(
        data, data_name=data_name, alpha=alpha, verbose=True, data_kwargs=data_kwargs
    )
    result_kpss = _is_stationary_kpss(
        data, data_name=data_name, alpha=alpha, verbose=True, data_kwargs=data_kwargs
    )

    all_dfs = [
        result_adf[1],
        result_kpss[1],
    ]
    final = pd.concat(all_dfs)
    return final


##########################
# Individual Tests ####
##########################


def _is_stationary_adf(
    data: pd.Series,
    data_name: Optional[str] = None,
    alpha: float = 0.05,
    verbose: bool = False,
    data_kwargs: Optional[Dict] = None,
) -> Tuple[Union[bool, List[bool]], Optional[pd.DataFrame]]:
    """Checks Difference Stationarity (ADF test)

    H0: The time series is not stationary (has a unit root)
    H1: The time series is stationary

    If not  stationary, then try differencing.

    Parameters
    ----------
    data : pd.Series
        Time Series data on which the test needs to be performed
    alpha : float, optional
        Significance Level, by default 0.05
    verbose : bool, optional
        If False, returns boolean value(s) for whether the data is stationary
        or not. If True, then it returns the detailed results dataframe along
        with the boolean value(s), by default False
    data_kwargs : Optional[Dict], optional
        Users can specify `lags list` or `order_list` to run the test for the
        data as well as for its lagged versions, by default None

        >>> _is_stationary_adf(data=data, data_kwargs={"order_list": [1, 2]})
        >>> _is_stationary_adf(data=data, data_kwargs={"lags_list": [1, [1, 12]]})

    Returns
    -------
    Tuple[Union[bool, List[bool]], Optional[pd.DataFrame]]
        If verbose=False, returns boolean value(s) for whether the data is stationary
        or not. If test for lags/orders are not requested, then returns a single
        boolean value corresponding to the data. If tests are requested for
        lags/orders (using kwargs), then returns a list of boolean values
        corresponding to the data and the lags/order specified by user (in that
        order). If verbose=True, then it returns the detailed results dataframe
        along with the boolean value(s).
    """

    test_category = "Stationarity"

    # Step 1: Get list of all data that needs to be tested ----
    diff_list, name_list = _get_diff_name_list(
        data=data, data_name=data_name, data_kwargs=data_kwargs
    )

    # Step 2: Test all data ----
    results_list = []
    is_stationary_list = []
    for data_, name_ in zip(diff_list, name_list):
        # Step 2A: Validate inputs and adjust as needed ----
        if len(data_) == 0:
            # Differencing led to no remaining data, hence skip it
            continue

        # Step 2B: Get Test Results ----
        try:
            results_ = adfuller(data_, autolag="AIC", maxlag=None)
        except SmMissingDataError as exception:
            logger.warning(exception)
            raise MissingDataError(
                "ADF test can not be run on data with missing values. "
                "Please check input data type."
            )

        test_statistic = results_[0]
        critical_values = results_[4]
        p_value = results_[1]
        is_stationary = True if p_value < alpha else False

        # Step 2C: Create Result DataFrame ----
        results = {
            "Stationarity": is_stationary,
            "p-value": p_value,
            "Test Statistic": test_statistic,
        }
        critical_values = {
            f"Critical Value {key}": value for key, value in critical_values.items()
        }
        results.update(critical_values)
        results = pd.DataFrame(results, index=["Value"]).T.reset_index()
        results["Data"] = name_

        # Step 2D: Update list of all results ----
        results_list.append(results)
        is_stationary_list.append(is_stationary)

    # Step 3: Combine all results ----
    results = pd.concat(results_list)
    results.reset_index(inplace=True)

    # Step 4: Add Settings & Format Results ----
    def add_and_format_settings(row):
        row["Setting"] = {"alpha": alpha}
        return row

    results = results.apply(add_and_format_settings, axis=1)
    results = _format_test_results(results, test_category, "ADF")

    # Step 5: Return values ----
    if len(is_stationary_list) == 1:
        is_stationary_list = is_stationary_list[0]
    if verbose:
        return is_stationary_list, results
    else:
        return is_stationary_list


def _is_stationary_kpss(
    data: pd.Series,
    data_name: Optional[str] = None,
    alpha: float = 0.05,
    verbose: bool = False,
    data_kwargs: Optional[Dict] = None,
) -> Tuple[Union[bool, List[bool]], Optional[pd.DataFrame]]:
    """Checks Stationarity around a deterministic trend (KPSS Test)

    H0: The time series is trend stationary
    H1: The time series is not trend stationary

    If not trend stationary, then try removing trend.

    Parameters
    ----------
    data : pd.Series
        Time Series data on which the test needs to be performed
    alpha : float, optional
        Significance Level, by default 0.05
    verbose : bool, optional
        If False, returns boolean value(s) for whether the data is stationary
        or not. If True, then it returns the detailed results dataframe along
        with the boolean value(s), by default False
    data_kwargs : Optional[Dict], optional
        Users can specify `lags list` or `order_list` to run the test for the
        data as well as for its lagged versions, by default None

        >>> _is_stationary_kpss(data=data, data_kwargs={"order_list": [1, 2]})
        >>> _is_stationary_kpss(data=data, data_kwargs={"lags_list": [1, [1, 12]]})

    Returns
    -------
    Tuple[Union[bool, List[bool]], Optional[pd.DataFrame]]
        If verbose=False, returns boolean value(s) for whether the data is stationary
        or not. If test for lags/orders are not requested, then returns a single
        boolean value corresponding to the data. If tests are requested for
        lags/orders (using kwargs), then returns a list of boolean values
        corresponding to the data and the lags/order specified by user (in that
        order). If verbose=True, then it returns the detailed results dataframe
        along with the boolean value(s).
    """

    test_category = "Stationarity"

    # Step 1: Get list of all data that needs to be tested ----
    diff_list, name_list = _get_diff_name_list(
        data=data, data_name=data_name, data_kwargs=data_kwargs
    )

    # Step 2: Test all data ----
    results_list = []
    is_stationary_list = []
    for data_, name_ in zip(diff_list, name_list):
        # Step 2A: Validate inputs and adjust as needed ----
        if len(data_) == 0:
            # Differencing led to no remaining data, hence skip it
            continue

        # Step 2B: Get Test Results ----
        try:
            results_ = kpss(data_, regression="ct", nlags="auto")
        except ValueError as exception:
            logger.warning(exception)
            if data_.isna().sum() > 0:
                raise MissingDataError(
                    "KPSS test can not be run on data with missing values. "
                    "Please check input data type."
                )
            else:
                raise ValueError()
        test_statistic = results_[0]
        p_value = results_[1]
        critical_values = results_[3]
        is_stationary = False if p_value < alpha else True

        # Step 2C: Create Result DataFrame ----
        results = {
            "Trend Stationarity": is_stationary,
            "p-value": p_value,
            "Test Statistic": test_statistic,
        }
        critical_values = {
            f"Critical Value {key}": value for key, value in critical_values.items()
        }
        results.update(critical_values)
        results = pd.DataFrame(results, index=["Value"]).T.reset_index()
        results["Data"] = name_

        # Step 2D: Update list of all results ----
        results_list.append(results)
        is_stationary_list.append(is_stationary)

    # Step 3: Combine all results ----
    results = pd.concat(results_list)
    results.reset_index(inplace=True)

    # Step 4: Add Settings & Format Results ----
    def add_and_format_settings(row):
        row["Setting"] = {"alpha": alpha}
        return row

    results = results.apply(add_and_format_settings, axis=1)
    results = _format_test_results(results, test_category, "KPSS")

    # Step 5: Return values ----
    if len(is_stationary_list) == 1:
        is_stationary_list = is_stationary_list[0]
    if verbose:
        return is_stationary_list, results
    else:
        return is_stationary_list


def _is_white_noise(
    data: pd.Series,
    data_name: Optional[str] = None,
    lags: List[int] = [24, 48],
    alpha: float = 0.05,
    verbose: bool = False,
    data_kwargs: Optional[Dict] = None,
) -> Tuple[Union[bool, List[bool]], Optional[pd.DataFrame]]:
    """Performs the Ljung-Box test for testing if a time series is White Noise

    H0: The data is consistent with white noise
    Ha: The data is not consistent with white noise.

    Parameters
    ----------
    data : pd.Series
        Time Series data on which the test needs to be performed
    lags : List[int], optional
        The lags used to test the autocorelation for white noise, by default [24, 48]
    alpha : float, optional
        Significance Level, by default 0.05
    verbose : bool, optional
        If False, returns boolean value(s) for whether the data is white noise
        or not. If True, then it returns the detailed results dataframe along
        with the boolean value(s), by default False
    data_kwargs : Optional[Dict], optional
        Users can specify `lags list` or `order_list` to run the test for the
        data as well as for its lagged versions, by default None

        >>> _is_white_noise(data=data, data_kwargs={"order_list": [1, 2]})
        >>> _is_white_noise(data=data, data_kwargs={"lags_list": [1, [1, 12]]})

    Returns
    -------
    Tuple[Union[bool, List[bool]], Optional[pd.DataFrame]]
        If verbose=False, returns boolean value(s) for whether the data is white
        noise or not. If test for lags/orders are not requested, then returns a
        single boolean value corresponding to the data. If tests are requested
        for lags/orders (using kwargs), then returns a list of boolean values
        corresponding to the data and the lags/order specified by user (in that
        order). If verbose=True, then it returns the detailed results dataframe
        along with the boolean value(s).
    """

    test_category = "White Noise"

    # Step 1: Get list of all data that needs to be tested ----
    diff_list, name_list = _get_diff_name_list(
        data=data, data_name=data_name, data_kwargs=data_kwargs
    )

    # Step 2: Test all data ----
    results_list = []
    is_white_noise_list = []
    for data_, name_ in zip(diff_list, name_list):
        # Step 2A: Validate inputs and adjust as needed ----
        if len(data_) == 0:
            # Differencing led to no remaining data, hence skip it
            continue

        lags_ = [lag for lag in lags if lag < len(data_)]
        lags_ = None if len(lags_) == 0 else lags_

        # Step 2B: Run test ----
        if data_.isna().sum() == 0:
            results = sm.stats.acorr_ljungbox(data_, lags=lags_, return_df=True)
        else:
            raise MissingDataError(
                "White Noise Test (Ljung-Box) can not be run on data with missing "
                "values. Please check input data type."
            )

        # Step 2C: Cleanup results ----
        results[test_category] = results["lb_pvalue"] > alpha
        is_white_noise = False if results[test_category].all() is False else True
        results.rename(
            columns={"lb_stat": "Test Statictic", "lb_pvalue": "p-value"},
            inplace=True,
        )
        results["Data"] = name_

        # Long Format
        results.reset_index(inplace=True)
        results.rename(columns={"index": "Setting"}, inplace=True)
        results = pd.melt(
            results, id_vars=["Setting", "Data"], var_name="index", value_name="Value"
        )

        # Step 2D: Update list of all results ----
        results_list.append(results)
        is_white_noise_list.append(is_white_noise)

    # Step 3: Combine all results ----
    results = pd.concat(results_list)
    results.reset_index(inplace=True)

    # Step 4: Add Settings & Format Results ----
    def add_and_format_settings(row):
        row["Setting"] = {"alpha": alpha, "K": row["Setting"]}
        return row

    results = results.apply(add_and_format_settings, axis=1)
    results = _format_test_results(results, test_category, "Ljung-Box")

    # Step 5: Return values ----
    if len(is_white_noise_list) == 1:
        is_white_noise_list = is_white_noise_list[0]
    if verbose:
        return is_white_noise_list, results
    else:
        return is_white_noise_list


def is_trending():
    """TBD"""
    pass


def is_seasonal():
    """TBD"""
    pass


#########################
# Recommendations ####
#########################


def recommend_lowercase_d(data: pd.Series, **kwargs) -> int:
    """Returns the recommended value of differencing order 'd' to use

    Parameters
    ----------
    data : pd.Series
        The data for which the differencing order needs to be calculated

    *kwargs: Keyword arguments that can be passed to the difference test.
        Values are:
            alpha : float, optional
                Significance Value, by default 0.05
            test : str, optional
                The test to use to test the order of differencing, by default 'kpss'
            max_d : int, optional
                maximum differencing order to try, by default 2

    Returns
    -------
    int
        The differencing order to use
    """
    recommended_lowercase_d = ndiffs(data, **kwargs)
    return recommended_lowercase_d


def recommend_uppercase_d(data: pd.Series, sp: int, **kwargs) -> int:
    """Returns the recommended value of differencing order 'D' to use

    Parameters
    ----------
    data : pd.Series
        The data for which the differencing order needs to be calculated

    sp : int
        The number of seasonal periods (i.e., frequency of the time series)

    *kwargs: Keyword arguments that can be passed to the difference test.
        Values are:
            alpha : float, optional
                Significance Value, by default 0.05
            test : str, optional
                Type of unit root test of seasonality to use in order to detect
                seasonal periodicity. Valid tests include (“ocsb”, “ch”).
                Note that the CHTest is very slow for large data.
            max_D : int, optional
                maximum seasonal differencing order to try, by default 2

    Returns
    -------
    int
        The differencing order to use
    """
    recommended_uppercase_d = nsdiffs(data, m=sp, **kwargs)
    return recommended_uppercase_d


def recommend_seasonal_period():
    """TBD"""
    pass
