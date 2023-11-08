"""Module to test time_series plotting functionality
"""
import os
import sys

import numpy as np  # type: ignore
import pytest
from time_series_test_utils import (
    _ALL_PLOTS_DATA,
    _ALL_PLOTS_ESTIMATOR,
    _ALL_PLOTS_ESTIMATOR_NOT_DATA,
    _return_all_plots_estimator_ts_results,
    _return_data_with_without_period_index,
    _return_model_names_for_plots_stats,
)

from pycaret.datasets import get_data
from pycaret.time_series import TSForecastingExperiment

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")
os.environ["PYCARET_TESTING"] = "1"

if sys.platform == "win32":
    pytest.skip("Skipping test module on Windows", allow_module_level=True)


##############################
# Functions Start Here ####
##############################

# NOTE: Fixtures can not be used to parameterize tests
# https://stackoverflow.com/questions/52764279/pytest-how-to-parametrize-a-test-with-a-list-that-is-returned-from-a-fixture
# Hence, we have to create functions and create the parameterized list first
# (must happen during collect phase) before passing it to mark.parameterize.


_data_with_without_period_index = _return_data_with_without_period_index()
_model_names_for_plots = _return_model_names_for_plots_stats()
_all_plots_estimator_ts_results = _return_all_plots_estimator_ts_results()


############################
# Functions End Here ####
############################


##########################
# Tests Start Here ####
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

    ######################
    # OOP Approach ####
    ######################

    exp.setup(
        data=data,
        fh=fh,
        fold=fold,
        fold_strategy="sliding",
        verbose=False,
        session_id=42,
    )

    exp.plot_model(plot=plot)

    ########################
    # Functional API ####
    ########################
    from pycaret.time_series import plot_model, setup

    _ = setup(
        data=data,
        fh=fh,
        fold=fold,
        fold_strategy="expanding",
        session_id=42,
        n_jobs=-1,
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

    ######################
    # OOP Approach ####
    ######################

    exp.setup(
        data=data,
        fh=fh,
        fold=fold,
        fold_strategy="sliding",
        verbose=False,
        session_id=42,
    )

    model = exp.create_model(model_name)
    exp.plot_model(estimator=model, plot=plot)

    ########################
    # Functional API ####
    ########################
    from pycaret.time_series import create_model, plot_model, setup

    _ = setup(
        data=data,
        fh=fh,
        fold=fold,
        fold_strategy="expanding",
        session_id=42,
        n_jobs=-1,
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
    # OOP Approach ####
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
        exp.plot_model(plot=plot)

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

    exp.setup(
        data=data,
        fh=fh,
        fold=fold,
        fold_strategy="sliding",
        verbose=False,
        session_id=42,
    )

    model = exp.create_model("naive")

    #######################
    # Customization ####
    #######################

    print("\n\n==== Testing Customization ON DATA ====")
    exp.plot_model(
        plot="pacf",
        data_kwargs={
            "nlags": 36,
        },
        fig_kwargs={"fig_size": [800, 500], "fig_template": "simple_white"},
    )
    exp.plot_model(plot="decomp_classical", data_kwargs={"type": "multiplicative"})

    print("\n\n====  Testing Customization ON ESTIMATOR ====")
    exp.plot_model(estimator=model, plot="forecast", data_kwargs={"fh": 24})


@pytest.mark.parametrize("data", _data_with_without_period_index)
@pytest.mark.parametrize("plot", _ALL_PLOTS_DATA)
def test_plot_model_return_data_original_data(data, plot):
    """Tests whether the return_data parameter of the plot_model function works
    properly or not for the original data
    """
    exp = TSForecastingExperiment()

    fh = np.arange(1, 13)
    fold = 2

    exp.setup(
        data=data,
        fh=fh,
        fold=fold,
        fold_strategy="sliding",
        verbose=False,
        session_id=42,
    )

    plot_data = exp.plot_model(plot=plot, return_data=True)
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

    exp.setup(
        data=data,
        fh=fh,
        fold=fold,
        fold_strategy="sliding",
        verbose=False,
        session_id=42,
    )

    model = exp.create_model(model_name)

    plot_data = exp.plot_model(estimator=model, plot=plot, return_data=True)
    # If plot is successful, it will return a dictionary
    # If plot is not possible (e.g. decomposition without index), then it will return None
    assert isinstance(plot_data, dict) or plot_data is None


@pytest.mark.parametrize("plot, all_models_supported", _all_plots_estimator_ts_results)
def test_plot_multiple_model_overlays(
    load_pos_and_neg_data, plot, all_models_supported
):
    """Tests the plot_model functionality on estimators where the results from
    multiple models get overlaid (time series plots)

    Checks:
        (1) Plots are correct even when the multiple models are of the same type
        (2) Plot labels are correct when user provides custom labels
        (3) When some models do not support certain plots, they are dropped appropriately
        (4) When some models do not support certain plots, they are dropped appropriately
            even when user provides custom labels
        (5) When user provides custom labels, the number of labels must match number of models
    """
    data = load_pos_and_neg_data

    exp = TSForecastingExperiment()
    fh = 12
    fold = 2
    exp.setup(data=data, fh=fh, fold=fold, fold_strategy="sliding")

    # Model that produces insample predictions
    m1 = exp.create_model("exp_smooth")

    # Check 1: Even if same model type is passed, the plot should make overlays ----
    models = [m1, m1]
    fig_data = exp.plot_model(models, plot=plot, return_data=True)
    assert fig_data.get("overlay_data").shape[1] == len(models)

    # Check 2: User specified labels are used in plots
    labels = ["Model 1", "Model 2"]
    fig_data = exp.plot_model(
        models,
        plot=plot,
        data_kwargs={"labels": labels},
        return_data=True,
    )
    assert fig_data.get("overlay_data").shape[1] == len(models)
    assert np.all(fig_data.get("overlay_data").columns.to_list() == labels)

    if not all_models_supported:
        # Model that does not produce insample predictions
        m2 = exp.create_model("lr_cds_dt")

        # Check 3: If Model does not produce insample predictions, it should be excluded
        models = [m1, m2, m1]
        fig_data = exp.plot_model(models, plot=plot, return_data=True)
        assert fig_data.get("overlay_data").shape[1] == len(models) - 1

        # Check 4: If Model does not produce insample predictions, custom labels should exclude it.
        labels = ["Model 1", "Model 2", "Model 3"]
        fig_data = exp.plot_model(
            models,
            plot=plot,
            data_kwargs={"labels": labels},
            return_data=True,
        )
        assert fig_data.get("overlay_data").shape[1] == len(models) - 1
        labels.remove("Model 2")
        assert np.all(fig_data.get("overlay_data").columns.to_list() == labels)

    # Check 5: When user provides custom labels, the number of labels must match
    # number of models

    models = [m1, m1]

    # (A) Less labels than models ----
    labels = ["Model 1"]
    with pytest.raises(ValueError) as errmsg:
        fig_data = exp.plot_model(models, plot=plot, data_kwargs={"labels": labels})

    # Capture Error message
    exceptionmsg = errmsg.value.args[0]

    # Check exact error received
    assert (
        "Please provide a label corresponding to each model to proceed." in exceptionmsg
    )

    # (B) More labels than models ----
    labels = ["Model 1", "Model 2", "Model 3"]
    with pytest.raises(ValueError) as errmsg:
        fig_data = exp.plot_model(models, plot=plot, data_kwargs={"labels": labels})

    # Capture Error message
    exceptionmsg = errmsg.value.args[0]

    # Check exact error received
    assert (
        "Please provide a label corresponding to each model to proceed." in exceptionmsg
    )


def test_plot_final_model_exo():
    """Tests running plot model after running finalize_model when exogenous
    variables are present. Fix for https://github.com/pycaret/pycaret/issues/3565
    """
    data = get_data("uschange")
    target = "Consumption"
    FH = 3
    train = data.iloc[: int(len(data) - FH)]
    test = data.iloc[int(len(data)) - FH :]
    test = test.drop(columns=[target], axis=1)

    exp = TSForecastingExperiment()
    exp.setup(data=train, target=target, fh=FH, session_id=42)
    model = exp.create_model("arima")
    final_model = exp.finalize_model(model)

    # Previous issue coming from renderer resolution due to X

    # This should not give an error (passing X explicitly)
    exp.plot_model(final_model, data_kwargs={"X": test})

    # Also, plotting without explicit passing X should also pass
    exp.plot_model()
