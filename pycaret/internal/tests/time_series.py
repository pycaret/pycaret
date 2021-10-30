import pandas as pd

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import kpss

from pmdarima.arima.utils import ndiffs, nsdiffs

from pycaret.internal.tests.stats import (
    summary_statistics,
    is_gaussian,
    is_white_noise,
)
from pycaret.internal.tests import _format_test_results


#############################################################
###################### TESTS STARTS HERE ####################
#############################################################

#################
#### Helpers ####
#################


def test_(data, test: str, alpha: float = 0.05, *kwargs):
    if test == "all":
        results = test_all(data=data, alpha=alpha)
    elif test == "summary":
        results = summary_statistics(data=data)
    elif test == "white_noise":
        results = is_white_noise(data=data, alpha=alpha, verbose=True, *kwargs)[1]
    elif test == "stationarity":
        results = is_stationary(data=data, alpha=alpha)
    elif test == "adf":
        results = is_stationary_adf(data=data, alpha=alpha, verbose=True)[1]
    elif test == "kpss":
        results = is_stationary_kpss(data=data, alpha=alpha, verbose=True)[1]
    elif test == "normality":
        results = is_gaussian(data=data, alpha=alpha, verbose=True)[1]
    else:
        raise ValueError(f"Tests: '{test}' is not supported.")
    return results


########################
#### Combined Tests ####
########################
def test_all(data, alpha: float = 0.05):
    result_summary_stats = summary_statistics(data=data)
    result_wn = is_white_noise(data, alpha=alpha, verbose=True)
    result_adf = is_stationary_adf(data, alpha=alpha, verbose=True)
    result_kpss = is_stationary_kpss(data, alpha=alpha, verbose=True)
    result_normality = is_gaussian(data, alpha=alpha, verbose=True)

    all_dfs = [
        result_summary_stats,
        result_wn[1],
        result_adf[1],
        result_kpss[1],
        result_normality[1],
    ]
    final = pd.concat(all_dfs)
    return final


def is_stationary(data, alpha: float = 0.05):
    result_adf = is_stationary_adf(data, alpha=alpha, verbose=True)
    result_kpss = is_stationary_kpss(data, alpha=alpha, verbose=True)

    all_dfs = [
        result_adf[1],
        result_kpss[1],
    ]
    final = pd.concat(all_dfs)
    return final


##########################
#### Individual Tests ####
##########################


def is_stationary_adf(data: pd.Series, alpha: float = 0.05, verbose: bool = False):
    """Checks Difference Stationarity

    H0: The time series is not stationary (has a unit root)
    H1: The time series is stationary

    If not stationary, try differencing

    Parameters
    ----------
    data : pd.Series
        Time Series to be tested
    """
    results = adfuller(data, autolag="AIC", maxlag=None)
    test_statistic = results[0]
    critical_values = results[4]
    p_value = results[1]
    stationarity = True if p_value < alpha else False
    details = {
        "Stationarity": stationarity,
        "p-value": p_value,
        "Test Statistic": test_statistic,
    }
    critical_values = {
        f"Critical Value {key}": value for key, value in critical_values.items()
    }
    details.update(critical_values)

    details = pd.DataFrame(details, index=["Value"]).T.reset_index()
    details["Setting"] = [{"alpha": alpha}] * len(details)
    details = _format_test_results(details, "Stationarity", "ADF")

    if verbose:
        return stationarity, details
    else:
        return stationarity


def is_stationary_kpss(data: pd.Series, alpha: float = 0.05, verbose: bool = False):
    """Checks Stationarity around a deterministic trend

    H0: The time series is trend stationary
    H1: The time series is not trend stationary

    If not trend stationary, then try removing trend.

    Parameters
    ----------
    data : pd.Series
        Time Series to be tested
    """
    results = kpss(data, regression="ct", nlags="auto")
    test_statistic = results[0]
    p_value = results[1]
    critical_values = results[3]
    stationarity = False if p_value < alpha else True
    details = {
        "Trend Stationarity": stationarity,
        "p-value": p_value,
        "Test Statistic": test_statistic,
    }
    critical_values = {
        f"Critical Value {key}": value for key, value in critical_values.items()
    }
    details.update(critical_values)

    details = pd.DataFrame(details, index=["Value"]).T.reset_index()
    details["Setting"] = [{"alpha": alpha}] * len(details)
    details = _format_test_results(details, "Stationarity", "KPSS")

    if verbose:
        return stationarity, details
    else:
        return stationarity


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
