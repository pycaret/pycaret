from typing import List

import pandas as pd
import statsmodels.api as sm

from pycaret.internal.tests import _format_test_results


##########################
#### Individual Tests ####
##########################


def summary_statistics(data: pd.Series):
    distinct_counts = dict(data.value_counts(normalize=True))
    result = {
        "Length": len(data),
        "Mean": data.mean(),
        "Median": data.median(),
        "Standard Deviation": data.std(),
        "Variance": data.var(),
        "Kurtosis": data.kurt(),
        "Skewness": data.skew(),
        "# Distinct Values": len(distinct_counts),
    }

    result = pd.DataFrame(result, index=["Value"]).T.reset_index()
    result = _format_test_results(result, "Summary", "Statistics")

    return result


def is_gaussian(data: pd.Series, alpha: float = 0.05, verbose: bool = False):
    """Performs the shapiro test to check for normality of data
    """
    from scipy.stats import shapiro, norm

    p_value = shapiro(data.values.squeeze())[1]
    normality = True if p_value > alpha else False
    details = {
        "Normality": normality,
        "p-value": p_value,
    }
    details = pd.DataFrame(details, index=["Value"]).T.reset_index()
    details["Setting"] = [{"alpha": alpha}] * len(details)
    details = _format_test_results(details, "Normality", "Shapiro")

    if verbose:
        return normality, details
    else:
        return normality


def is_white_noise(
    data: pd.Series,
    lags: List[int] = [24, 48],
    alpha: float = 0.05,
    verbose: bool = False,
):
    """Performs the Ljung-Box test for testing if a time series is White Noise

    H0: The data is consistent with white noise
    Ha: The data is not consistent with white noise.

    Parameters
    ----------
    data : pd.Series
        The time series that has to be tested
    lags : List[int], optional
        The lags used to test the autocorelation for white noise, by default [24, 48]
    """
    test_category = "White Noise"

    #### Step 1: Validate inputs and adjust as needed ----
    lags = [lag for lag in lags if lag < len(data)]
    lags = None if len(lags) == 0 else lags

    #### Step 2: Run test ----
    results = sm.stats.acorr_ljungbox(data, lags=lags, return_df=True)

    #### Step 3: Cleanup results ----
    results[test_category] = results["lb_pvalue"] > alpha
    is_white_noise = False if results[test_category].all() == False else True
    results.rename(
        columns={"lb_stat": "Test Statictic", "lb_pvalue": "p-value"}, inplace=True,
    )

    results.reset_index(inplace=True)
    results.rename(columns={"index": "Setting"}, inplace=True)

    def add_and_format_settings(row):
        row["Setting"] = {"alpha": alpha, "K": row["Setting"]}
        return row

    # TODO: Add alpha value to Settings
    results = results.apply(add_and_format_settings, axis=1)
    results = pd.melt(results, id_vars="Setting", var_name="index", value_name="Value")
    results = _format_test_results(results, test_category, "Ljung-Box")

    if verbose:
        return is_white_noise, results
    else:
        return is_white_noise
