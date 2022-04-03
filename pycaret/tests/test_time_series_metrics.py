"""Module to test time_series functionality
"""
import pytest

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from pycaret.containers.metrics.time_series import coverage


pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


##########################
#### Tests Start Here ####
##########################


def test_cov_prob_loss():
    """Tests inpi_loss
    """

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
