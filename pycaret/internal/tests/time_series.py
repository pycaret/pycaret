from typing import Optional, Dict, List, Tuple, Union
import pandas as pd

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import kpss

from pmdarima.arima.utils import ndiffs, nsdiffs

from pycaret.internal.tests.stats import summary_stats, is_gaussian
from pycaret.internal.tests import _format_test_results

from pycaret.utils.time_series import get_diffs, _get_diff_name_list


#############################################################
###################### TESTS STARTS HERE ####################
#############################################################

#################
#### Helpers ####
#################


def _test(
    data, test: str, alpha: float = 0.05, data_kwargs: Optional[Dict] = None, *kwargs
):
    if test == "all":
        results = test_all(data=data, alpha=alpha, data_kwargs=data_kwargs)
    elif test == "summary":
        results = summary_stats(data=data, data_kwargs=data_kwargs)
    elif test == "white_noise":
        results = is_white_noise(
            data=data, alpha=alpha, verbose=True, data_kwargs=data_kwargs, *kwargs
        )[1]
    elif test == "stationarity":
        results = is_stationary(data=data, alpha=alpha, data_kwargs=data_kwargs)
    elif test == "adf":
        results = is_stationary_adf(
            data=data, alpha=alpha, verbose=True, data_kwargs=data_kwargs
        )[1]
    elif test == "kpss":
        results = is_stationary_kpss(
            data=data, alpha=alpha, verbose=True, data_kwargs=data_kwargs
        )[1]
    elif test == "normality":
        results = is_gaussian(
            data=data, alpha=alpha, verbose=True, data_kwargs=data_kwargs
        )[1]
    else:
        raise ValueError(f"Tests: '{test}' is not supported.")
    return results


########################
#### Combined Tests ####
########################
def test_all(data, alpha: float = 0.05, data_kwargs: Optional[Dict] = None):
    result_summary_stats = summary_stats(data=data, data_kwargs=data_kwargs)
    result_wn = is_white_noise(data, alpha=alpha, verbose=True, data_kwargs=data_kwargs)
    result_adf = is_stationary_adf(
        data, alpha=alpha, verbose=True, data_kwargs=data_kwargs
    )
    result_kpss = is_stationary_kpss(
        data, alpha=alpha, verbose=True, data_kwargs=data_kwargs
    )
    result_normality = is_gaussian(
        data, alpha=alpha, verbose=True, data_kwargs=data_kwargs
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


def is_stationary(
    data, alpha: float = 0.05, data_kwargs: Optional[Dict] = None,
):
    result_adf = is_stationary_adf(
        data, alpha=alpha, verbose=True, data_kwargs=data_kwargs
    )
    result_kpss = is_stationary_kpss(
        data, alpha=alpha, verbose=True, data_kwargs=data_kwargs
    )

    all_dfs = [
        result_adf[1],
        result_kpss[1],
    ]
    final = pd.concat(all_dfs)
    return final


##########################
#### Individual Tests ####
##########################


def is_stationary_adf(
    data: pd.Series,
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

        >>> is_stationary_adf(test=data, data_kwargs={"order_list": [1, 2]})
        >>> is_stationary_adf(test=data, data_kwargs={"lags_list": [1, [1, 12]]})

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
    # TODO: Fix this
    model_name = None
    diff_list, name_list = _get_diff_name_list(
        data=data, model_name=model_name, data_kwargs=data_kwargs
    )

    #### Step 2: Test all data ----
    results_list = []
    is_stationary_list = []
    for data_, name_ in zip(diff_list, name_list):
        # Step 2A: Get Test Results ----
        results_ = adfuller(data_, autolag="AIC", maxlag=None)
        test_statistic = results_[0]
        critical_values = results_[4]
        p_value = results_[1]
        is_stationary_ = True if p_value < alpha else False

        #### Step 2B: Create Result DataFrame ----
        results = {
            "Stationarity": is_stationary_,
            "p-value": p_value,
            "Test Statistic": test_statistic,
        }
        critical_values = {
            f"Critical Value {key}": value for key, value in critical_values.items()
        }
        results.update(critical_values)
        results = pd.DataFrame(results, index=["Value"]).T.reset_index()
        results["Data"] = name_

        #### Step 2C: Update list of all results ----
        results_list.append(results)
        is_stationary_list.append(is_stationary_)

    #### Step 3: Combine all results ----
    results = pd.concat(results_list)
    results.reset_index(inplace=True)

    #### Step 4: Add Settings & Format Results ----
    def add_and_format_settings(row):
        row["Setting"] = {"alpha": alpha}
        return row

    results = results.apply(add_and_format_settings, axis=1)
    results = _format_test_results(results, test_category, "ADF")

    #### Step 5: Return values ----
    if len(is_stationary_list) == 1:
        is_stationary_list = is_stationary_list[0]
    if verbose:
        return is_stationary_list, results
    else:
        return is_stationary_list


def is_stationary_kpss(
    data: pd.Series,
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

        >>> is_stationary_kpss(test=data, data_kwargs={"order_list": [1, 2]})
        >>> is_stationary_kpss(test=data, data_kwargs={"lags_list": [1, [1, 12]]})

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
    # TODO: Fix this
    model_name = None
    diff_list, name_list = _get_diff_name_list(
        data=data, model_name=model_name, data_kwargs=data_kwargs
    )

    #### Step 2: Test all data ----
    results_list = []
    is_stationary_list = []
    for data_, name_ in zip(diff_list, name_list):
        # Step 2A: Get Test Results ----
        results_ = kpss(data_, regression="ct", nlags="auto")
        test_statistic = results_[0]
        p_value = results_[1]
        critical_values = results_[3]
        is_stationary_ = False if p_value < alpha else True

        #### Step 2B: Create Result DataFrame ----
        results = {
            "Trend Stationarity": is_stationary_,
            "p-value": p_value,
            "Test Statistic": test_statistic,
        }
        critical_values = {
            f"Critical Value {key}": value for key, value in critical_values.items()
        }
        results.update(critical_values)
        results = pd.DataFrame(results, index=["Value"]).T.reset_index()
        results["Data"] = name_

        #### Step 2C: Update list of all results ----
        results_list.append(results)
        is_stationary_list.append(is_stationary_)

    #### Step 3: Combine all results ----
    results = pd.concat(results_list)
    results.reset_index(inplace=True)

    #### Step 4: Add Settings & Format Results ----
    def add_and_format_settings(row):
        row["Setting"] = {"alpha": alpha}
        return row

    results = results.apply(add_and_format_settings, axis=1)
    results = _format_test_results(results, test_category, "KPSS")

    #### Step 5: Return values ----
    if len(is_stationary_list) == 1:
        is_stationary_list = is_stationary_list[0]
    if verbose:
        return is_stationary_list, results
    else:
        return is_stationary_list


def is_white_noise(
    data: pd.Series,
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

        >>> is_white_noise(test=data, data_kwargs={"order_list": [1, 2]})
        >>> is_white_noise(test=data, data_kwargs={"lags_list": [1, [1, 12]]})

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
    # TODO: Fix this
    model_name = None
    diff_list, name_list = _get_diff_name_list(
        data=data, model_name=model_name, data_kwargs=data_kwargs
    )

    #### Step 2: Test all data ----
    results_list = []
    is_white_noise_list = []
    for data_, name_ in zip(diff_list, name_list):
        #### Step 2A: Validate inputs and adjust as needed ----
        lags = [lag for lag in lags if lag < len(data_)]
        lags = None if len(lags) == 0 else lags

        #### Step 2B: Run test ----
        results = sm.stats.acorr_ljungbox(data_, lags=lags, return_df=True)

        #### Step 2C: Cleanup results ----
        results[test_category] = results["lb_pvalue"] > alpha
        is_white_noise_ = False if results[test_category].all() == False else True
        results.rename(
            columns={"lb_stat": "Test Statictic", "lb_pvalue": "p-value"}, inplace=True,
        )
        results["Data"] = name_

        # Long Format
        results.reset_index(inplace=True)
        results.rename(columns={"index": "Setting"}, inplace=True)
        results = pd.melt(
            results, id_vars=["Setting", "Data"], var_name="index", value_name="Value"
        )

        #### Step 2D: Update list of all results ----
        results_list.append(results)
        is_white_noise_list.append(is_white_noise_)

    #### Step 3: Combine all results ----
    results = pd.concat(results_list)
    results.reset_index(inplace=True)

    #### Step 4: Add Settings & Format Results ----
    def add_and_format_settings(row):
        row["Setting"] = {"alpha": alpha, "K": row["Setting"]}
        return row

    results = results.apply(add_and_format_settings, axis=1)
    results = _format_test_results(results, test_category, "Ljung-Box")

    #### Step 5: Return values ----
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
#### Recommendations ####
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
