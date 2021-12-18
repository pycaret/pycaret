"""Module to test time_series models
"""
import pytest
import pandas as pd  # type: ignore

from pycaret.internal.pycaret_experiment import TimeSeriesExperiment


##########################
#### Tests Start Here ####
##########################


def test_naive_models(load_pos_and_neg_data):
    """Tests enabling and disabling of naive models"""

    exp = TimeSeriesExperiment()
    data = load_pos_and_neg_data

    #### Seasonal Period != 1 ----
    #### All naive models should be enabled here
    exp.setup(data=data, verbose=False)
    expected = ["naive", "grand_means", "snaive"]
    for model in expected:
        assert model in exp.models().index

    #### Seasonal Period == 1 ----
    #### snaive should be disabled here
    exp.setup(data=data, seasonal_period=1, verbose=False)
    expected = ["naive", "grand_means"]
    for model in expected:
        assert model in exp.models().index
    not_expected = ["snaive"]
    for model in not_expected:
        assert model not in exp.models().index


def test_custom_models(load_pos_data):
    """Tests working with custom models"""

    exp = TimeSeriesExperiment()
    data = load_pos_data

    exp.setup(
        data=data, fh=12, session_id=42,
    )

    #### Create a sktime pipeline with preprocessing ----
    from sktime.transformations.series.impute import Imputer
    from sktime.transformations.series.boxcox import LogTransformer
    from sktime.forecasting.compose import TransformedTargetForecaster
    from sktime.forecasting.arima import ARIMA

    forecaster = TransformedTargetForecaster(
        [
            ("impute", Imputer()),
            ("log", LogTransformer()),
            ("model", ARIMA(seasonal_order=(0, 1, 0, 12))),
        ]
    )

    ##################################
    #### Test Create Custom Model ----
    ##################################
    my_custom_model = exp.create_model(forecaster)
    assert type(my_custom_model) == type(forecaster)

    ################################
    #### Test Tune Custom Model ----
    ################################
    impute_values = ["drift", "bfill", "ffill"]
    my_grid = {"impute__method": impute_values}
    tuned_model, tuner = exp.tune_model(
        my_custom_model, custom_grid=my_grid, return_tuner=True,
    )
    assert type(tuned_model) == type(forecaster)
    assert "param_impute__method" in pd.DataFrame(tuner.cv_results_)
    for index, method in enumerate(tuner.cv_results_.get("param_impute__method")):
        assert method == impute_values[index]

    ############################
    #### Test Tuning raises ----
    ############################
    # No custom grid passed when tuning custom model
    with pytest.raises(ValueError) as errmsg:
        _ = exp.tune_model(my_custom_model)

    # Capture Error message
    exceptionmsg = errmsg.value.args[0]

    # Check exact error received
    assert (
        f"When passing a model not in PyCaret's model library, the custom_grid parameter must be provided."
        in exceptionmsg
    )

