"""Module to test time_series functionality
"""
import pytest

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from pycaret.containers.metrics.time_series import coverage
from pycaret.time_series import TSForecastingExperiment


pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


##########################
#### Tests Start Here ####
##########################


def test_cov_prob_loss():
    """Tests inpi_loss"""

    ##############################
    #### Regular Calculations ####
    ##############################
    y_pred = None
    lower = pd.Series([0.5, 1.5, 2.5, 3.5])
    upper = pd.Series([1.5, 2.5, 3.5, 4.5])

    # All pass
    y_true = pd.Series([1, 2, 3, 4])
    loss = coverage(y_true=y_true, y_pred=y_pred, lower=lower, upper=upper)
    assert loss == 1.00

    # Upper Limit breached
    y_true = pd.Series([1, 2, 4, 4])
    loss = coverage(y_true=y_true, y_pred=y_pred, lower=lower, upper=upper)
    assert loss == 0.75

    # Lower Limit breached
    y_true = pd.Series([1, 1, 3, 4])
    loss = coverage(y_true=y_true, y_pred=y_pred, lower=lower, upper=upper)
    assert loss == 0.75

    # Both Limits breached
    y_true = pd.Series([1, 1, 4, 4])
    loss = coverage(y_true=y_true, y_pred=y_pred, lower=lower, upper=upper)
    assert loss == 0.50

    ##################################
    #### Check for NANs in limits ####
    ##################################
    lower = pd.Series([np.nan] * 4)
    upper = pd.Series([np.nan] * 4)
    y_true = pd.Series([1, 2, 3, 4])
    loss = coverage(y_true=y_true, y_pred=y_pred, lower=lower, upper=upper)
    assert loss is np.nan

    ##################################
    #### Check for NANs in y_true ####
    ##################################
    lower = pd.Series([0.5, 1.5, 2.5, 3.5])
    upper = pd.Series([1.5, 2.5, 3.5, 4.5])
    y_true = pd.Series([1, 2, np.nan, 4])
    loss = coverage(y_true=y_true, y_pred=y_pred, lower=lower, upper=upper)
    assert loss is np.nan

    ######################################
    #### Check for mismatched indices ####
    ######################################
    lower = pd.Series([0.5, 1.5, 2.5, 3.5], index=[0, 1, 2, 3])
    upper = pd.Series([1.5, 2.5, 3.5, 4.5], index=[0, 1, 2, 3])
    y_true = pd.Series([1, 2, 3, 4], index=[0, 1, 2, 4])
    loss = coverage(y_true=y_true, y_pred=y_pred, lower=lower, upper=upper)
    assert loss is np.nan


def test_add_custom_metric(load_pos_data):
    """Tests addition of custom metrics"""
    exp = TSForecastingExperiment()
    data = load_pos_data
    FH = 12

    exp.setup(data=data, fh=FH, session_id=42)

    def abs_bias(y_true, y_pred, norm=True):
        """Measures the bias in the predictions (aka Cumulative Forecast Error (CFE)
        Absolute value returned so it can be used in scoring

        Ref: https://medium.com/towards-data-science/forecast-error-measures-intermittent-demand-22617a733c9e
        """
        from pycaret.containers.metrics.time_series import _check_series

        y_true = _check_series(y_true)
        y_pred = _check_series(y_pred)

        abs_bias = np.abs(np.sum(y_pred - y_true))
        if norm:
            abs_bias = abs_bias / len(y_true)
        print(f"abs_bias: {abs_bias}")
        return abs_bias

    # Add two custom metrics with kwargs
    exp.add_metric(
        "abs_bias_norm", "ABS_BIAS_NORM", abs_bias, greater_is_better=False, norm=True
    )
    exp.add_metric(
        "abs_bias_cum", "ABS_BIAS_CUM", abs_bias, greater_is_better=False, norm=False
    )

    _ = exp.create_model("arima")
    metrics = exp.pull()

    # test that columns got added properly
    assert "ABS_BIAS_NORM" in metrics.columns
    assert "ABS_BIAS_CUM" in metrics.columns

    # test that kwargs works
    assert (
        (metrics["ABS_BIAS_CUM"] / FH).values.round(4)
        == metrics["ABS_BIAS_NORM"].values.round(4)
    ).all()
