from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from scipy.stats import shapiro

from pycaret.internal.tests import _format_test_results
from pycaret.utils.time_series import _get_diff_name_list

##########################
# Individual Tests ####
##########################


def _summary_stats(
    data: pd.Series,
    data_name: Optional[str] = None,
    data_kwargs: Optional[Dict] = None,
) -> pd.DataFrame:
    """Provides Summary Statistics for the data

    Parameters
    ----------
    data : pd.Series
        Data whose summary statistics need to be calculated
    data_kwargs : Optional[Dict], optional
        Useful for time series analysis. Users can specify `lags list` and
        `order_list` to get summary statistics for the data as well as for its
        lagged versions, by default None

        >>> summary_statistics(data=data, data_kwargs={"order_list": [1, 2]})
        >>> summary_statistics(data=data, data_kwargs={"lags_list": [1, [1, 12]]})

    Returns
    -------
    pd.DataFrame
        Dataframe containing the summary statistics
    """
    test_category = "Summary"

    # Step 1: Get list of all data that needs to be tested ----
    diff_list, name_list = _get_diff_name_list(
        data=data, data_name=data_name, data_kwargs=data_kwargs
    )

    # Step 2: Test all data ----
    results_list = []
    for data_, name_ in zip(diff_list, name_list):
        # Step 2A: Get Test Results ----
        distinct_counts = dict(data_.value_counts(normalize=True))
        results = {
            "Length": len(data_),
            "# Missing Values": data_.isna().sum(),
            "Mean": data_.mean(),
            "Median": data_.median(),
            "Standard Deviation": data_.std(),
            "Variance": data_.var(),
            "Kurtosis": data_.kurt(),
            "Skewness": data_.skew(),
            "# Distinct Values": len(distinct_counts),
        }

        # Step 2B: Create Result DataFrame ----
        results = pd.DataFrame(results, index=["Value"]).T.reset_index()
        results["Data"] = name_

        # Step 2C: Update list of all results ----
        results_list.append(results)

    # Step 3: Combine all results ----
    results = pd.concat(results_list)
    results.reset_index(inplace=True)

    # Step 4: Format Results ----
    results = _format_test_results(results, test_category, "Statistics")

    # Step 5: Return values ----
    return results


def _is_gaussian(
    data: pd.Series,
    data_name: Optional[str] = None,
    alpha: float = 0.05,
    verbose: bool = False,
    data_kwargs: Optional[Dict] = None,
) -> Tuple[Union[bool, List[bool]], Optional[pd.DataFrame]]:
    """Performs the shapiro test to check for normality of data

    Parameters
    ----------
    data : pd.Series
        Data on which the test needs to be performed
    alpha : float, optional
        Significance Level, by default 0.05
    verbose : bool, optional
        If False, returns boolean value(s) for whether the data is normal or not.
        If True, then it returns the detailed results dataframe along with the
        boolean value(s), by default False
    data_kwargs : Optional[Dict], optional
        Useful for time series analysis. Users can specify `lags list` or
        `order_list` to run the test for the data as well as for its lagged
        versions, by default None

        >>> _is_gaussian(data=data, data_kwargs={"order_list": [1, 2]})
        >>> _is_gaussian(data=data, data_kwargs={"lags_list": [1, [1, 12]]})

    Returns
    -------
    Tuple[Union[bool, List[bool]], Optional[pd.DataFrame]]
        If verbose=False, returns boolean value(s) for whether the data is normal
        or not. If test for lags/orders are not requested, then returns a single
        boolean value corresponding to the data. If tests are requested for lags
        /orders (using kwargs), then returns a list of boolean values corresponding
        to the data and the lags/order specified by user (in that order).
        If verbose=True, then it returns the detailed results dataframe along with
        the boolean value(s).
    """

    test_category = "Normality"

    # Step 1: Get list of all data that needs to be tested ----
    diff_list, name_list = _get_diff_name_list(
        data=data, data_name=data_name, data_kwargs=data_kwargs
    )

    # Step 2: Test all data ----
    results_list = []
    is_gaussian_list = []
    for data_, name_ in zip(diff_list, name_list):
        # Step 2A: Validate inputs and adjust as needed ----
        if len(data_) == 0:
            # Differencing led to no remaining data, hence skip it
            continue

        # Step 2B: Get Test Results ----
        p_value = shapiro(data_.values.squeeze())[1]
        is_gaussian = True if p_value > alpha else False

        # Step 2C: Create Result DataFrame ----
        results = {
            "Normality": is_gaussian,
            "p-value": p_value,
        }
        results = pd.DataFrame(results, index=["Value"]).T.reset_index()
        results["Data"] = name_

        # Step 2D: Update list of all results ----
        results_list.append(results)
        is_gaussian_list.append(is_gaussian)

    # Step 3: Combine all results ----
    results = pd.concat(results_list)
    results.reset_index(inplace=True)

    # Step 4: Add Settings & Format Results ----
    def add_and_format_settings(row):
        row["Setting"] = {"alpha": alpha}
        return row

    results = results.apply(add_and_format_settings, axis=1)
    results = _format_test_results(results, test_category, "Shapiro")

    # Step 5: Return values ----
    if len(is_gaussian_list) == 1:
        is_gaussian_list = is_gaussian_list[0]
    if verbose:
        return is_gaussian_list, results
    else:
        return is_gaussian_list
