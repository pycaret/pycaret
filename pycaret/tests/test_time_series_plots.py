"""Module to test time_series functionality
"""
import os
from random import uniform
import pytest

import numpy as np  # type: ignore
import pandas as pd  # type: ignore


from pycaret.time_series import TSForecastingExperiment
from pycaret.internal.ensemble import _ENSEMBLE_METHODS


from .time_series_test_utils import (
    _return_data_with_without_period_index,
    _return_model_names_for_plots_stats,
    _ALL_PLOTS_DATA,
    _ALL_PLOTS_ESTIMATOR,
    _ALL_PLOTS_ESTIMATOR_NOT_DATA,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


##############################
#### Functions Start Here ####
##############################

# NOTE: Fixtures can not be used to parameterize tests
# https://stackoverflow.com/questions/52764279/pytest-how-to-parametrize-a-test-with-a-list-that-is-returned-from-a-fixture
# Hence, we have to create functions and create the parameterized list first
# (must happen during collect phase) before passing it to mark.parameterize.


_data_with_without_period_index = _return_data_with_without_period_index()
_model_names_for_plots = _return_model_names_for_plots_stats()

############################
#### Functions End Here ####
############################


##########################
#### Tests Start Here ####
##########################


@pytest.mark.parametrize("data", _data_with_without_period_index)
@pytest.mark.parametrize("plot", _ALL_PLOTS_DATA)
def test_plot_model_data(data, plot):
    """Tests the plot_model functionality on original dataset
    NOTE: Want to show multiplicative plot here so can not take data with negative values
    """
    exp = TSForecastingExperiment()
    fh = np.arange(1, 13)
    fold = 2
    sp = 1 if isinstance(data.index, pd.RangeIndex) else None

    ######################
    #### OOP Approach ####
    ######################

    exp.setup(
        data=data,
        fh=fh,
        fold=fold,
        fold_strategy="sliding",
        verbose=False,
        session_id=42,
        seasonal_period=sp,
    )

    exp.plot_model(plot=plot, system=False)

    ########################
    #### Functional API ####
    ########################
    from pycaret.time_series import setup, plot_model

    os.environ["PYCARET_TESTING"] = "1"

    _ = setup(
        data=data,
        fh=fh,
        fold=fold,
        fold_strategy="expanding",
        session_id=42,
        n_jobs=-1,
        seasonal_period=sp,
    )
    plot_model(plot=plot)


@pytest.mark.parametrize("model_name", _model_names_for_plots)
@pytest.mark.parametrize("data", _data_with_without_period_index)
@pytest.mark.parametrize("plot", _ALL_PLOTS_ESTIMATOR)
def test_plot_model_estimator(model_name, data, plot):
    """Tests the plot_model functionality on estimators
    NOTE: Want to show multiplicative plot here so can not take data with negative values
    """
    exp = TSForecastingExperiment()

    fh = np.arange(1, 13)
    fold = 2

    sp = 1 if isinstance(data.index, pd.RangeIndex) else None

    ######################
    #### OOP Approach ####
    ######################

    exp.setup(
        data=data,
        fh=fh,
        fold=fold,
        fold_strategy="sliding",
        verbose=False,
        session_id=42,
        seasonal_period=sp,
    )

    model = exp.create_model(model_name)
    exp.plot_model(estimator=model, plot=plot, system=False)

    ########################
    #### Functional API ####
    ########################
    from pycaret.time_series import setup, create_model, plot_model

    os.environ["PYCARET_TESTING"] = "1"

    _ = setup(
        data=data,
        fh=fh,
        fold=fold,
        fold_strategy="expanding",
        session_id=42,
        n_jobs=-1,
        seasonal_period=sp,
    )
    model = create_model(model_name)
    plot_model(estimator=model, plot=plot)


@pytest.mark.parametrize("plot", _ALL_PLOTS_ESTIMATOR_NOT_DATA)
def test_plot_model_data_raises(load_pos_and_neg_data, plot):
    """Tests the plot_model functionality when it raises an exception
    on data plots (i.e. estimator is not passed)
    """
    exp = TSForecastingExperiment()
    fh = np.arange(1, 13)
    fold = 2

    ######################
    #### OOP Approach ####
    ######################

    exp.setup(
        data=load_pos_and_neg_data,
        fh=fh,
        fold=fold,
        fold_strategy="sliding",
        verbose=False,
        session_id=42,
    )

    with pytest.raises(ValueError) as errmsg:
        # Some code that produces a value error
        exp.plot_model(plot=plot, system=False)

    # Capture Error message
    exceptionmsg = errmsg.value.args[0]

    # Check exact error received
    assert (
        f"Plot type '{plot}' is not supported when estimator is not provided"
        in exceptionmsg
    )


@pytest.mark.parametrize("data", _data_with_without_period_index)
def test_plot_model_customization(data):
    """Tests the customization of plot_model
    NOTE: Want to show multiplicative plot here so can not take data with negative values
    """
    exp = TSForecastingExperiment()

    fh = np.arange(1, 13)
    fold = 2

    sp = 1 if isinstance(data.index, pd.RangeIndex) else None

    exp.setup(
        data=data,
        fh=fh,
        fold=fold,
        fold_strategy="sliding",
        verbose=False,
        session_id=42,
        seasonal_period=sp,
    )

    model = exp.create_model("naive")

    #######################
    #### Customization ####
    #######################

    print("\n\n==== Testing Customization ON DATA ====")
    exp.plot_model(
        plot="pacf",
        data_kwargs={
            "nlags": 36,
        },
        fig_kwargs={"fig_size": [800, 500], "fig_template": "simple_white"},
        system=False,
    )
    exp.plot_model(
        plot="decomp_classical", data_kwargs={"type": "multiplicative"}, system=False
    )

    print("\n\n====  Testing Customization ON ESTIMATOR ====")
    exp.plot_model(
        estimator=model, plot="forecast", data_kwargs={"fh": 24}, system=False
    )


@pytest.mark.parametrize("data", _data_with_without_period_index)
@pytest.mark.parametrize("plot", _ALL_PLOTS_DATA)
def test_plot_model_return_data_original_data(data, plot):
    """Tests whether the return_data parameter of the plot_model function works
    properly or not for the original data
    """
    exp = TSForecastingExperiment()

    fh = np.arange(1, 13)
    fold = 2

    sp = 1 if isinstance(data.index, pd.RangeIndex) else None

    exp.setup(
        data=data,
        fh=fh,
        fold=fold,
        fold_strategy="sliding",
        verbose=False,
        session_id=42,
        seasonal_period=sp,
    )

    plot_data = exp.plot_model(plot=plot, return_data=True, system=False)
    # If plot is successful, it will return a dictionary
    # If plot is not possible (e.g. decomposition without index), then it will return None
    assert isinstance(plot_data, dict) or plot_data is None


@pytest.mark.parametrize("data", _data_with_without_period_index)
@pytest.mark.parametrize("model_name", _model_names_for_plots)
@pytest.mark.parametrize("plot", _ALL_PLOTS_ESTIMATOR)
def test_plot_model_return_data_estimator(data, model_name, plot):
    """Tests whether the return_data parameter of the plot_model function works
    properly or not for the estimator
    """
    exp = TSForecastingExperiment()

    fh = np.arange(1, 13)
    fold = 2

    sp = 1 if isinstance(data.index, pd.RangeIndex) else None

    exp.setup(
        data=data,
        fh=fh,
        fold=fold,
        fold_strategy="sliding",
        verbose=False,
        session_id=42,
        seasonal_period=sp,
    )

    model = exp.create_model(model_name)

    plot_data = exp.plot_model(
        estimator=model, plot=plot, return_data=True, system=False
    )
    # If plot is successful, it will return a dictionary
    # If plot is not possible (e.g. decomposition without index), then it will return None
    assert isinstance(plot_data, dict) or plot_data is None
