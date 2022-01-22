"""Module to test time_series forecasting - univariate with exogenous variables
"""
import pytest
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from pycaret.datasets import get_data
from pycaret.internal.pycaret_experiment import TimeSeriesExperiment


pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


##############################
#### Functions Start Here ####
##############################


############################
#### Functions End Here ####
############################


##########################
#### Tests Start Here ####
##########################


def test_create_tune_predict_finalize_model(load_uni_exo_data_target):
    """test create_model, tune_model, predict_model and finalize_model
    functionality using exogenous variables
    """
    data, target = load_uni_exo_data_target

    fh = 12
    data_for_modeling = data.iloc[:-12]
    future_data = data.iloc[-12:]
    future_exog = future_data.drop(columns=target)

    exp = TimeSeriesExperiment()
    exp.setup(
        data=data_for_modeling, target=target, fh=fh, seasonal_period=4, session_id=42
    )

    #######################
    ## Test Create Model ##
    #######################
    model = exp.create_model("arima")

    #########################
    #### Expected Values ####
    #########################
    expected_period_index = data_for_modeling.iloc[-fh:].index
    final_expected_period_index = future_exog.index

    ########################
    ## Test Predict Model ##
    ########################
    # Default prediction
    y_pred = exp.predict_model(model)
    assert isinstance(y_pred, pd.Series)
    assert np.all(y_pred.index == expected_period_index)

    #####################
    ## Test Tune Model ##
    #####################
    tuned_model = exp.tune_model(model)

    ########################
    ## Test Predict Model ##
    ########################
    # Default prediction
    y_pred = exp.predict_model(tuned_model)
    assert isinstance(y_pred, pd.Series)
    assert np.all(y_pred.index == expected_period_index)

    #########################
    ## Test Finalize Model ##
    #########################

    final_model = exp.finalize_model(tuned_model)
    y_pred = exp.predict_model(final_model, X=future_exog)

    assert np.all(y_pred.index == final_expected_period_index)


def test_blend_models(load_uni_exo_data_target, load_models_uni_mix_exo_noexo):
    """test blending functionality
    """
    data, target = load_uni_exo_data_target

    fh = 12
    data_for_modeling = data.iloc[:-12]
    future_data = data.iloc[-12:]
    future_exog = future_data.drop(columns=target)

    #########################
    #### Expected Values ####
    #########################
    expected_period_index = data_for_modeling.iloc[-fh:].index
    final_expected_period_index = future_exog.index

    exp = TimeSeriesExperiment()
    exp.setup(
        data=data_for_modeling, target=target, fh=fh, seasonal_period=4, session_id=42
    )

    models_to_include = load_models_uni_mix_exo_noexo
    best_models = exp.compare_models(include=models_to_include, n_select=3)

    blender = exp.blend_models(best_models)
    y_pred = exp.predict_model(blender)
    assert isinstance(y_pred, pd.Series)
    assert np.all(y_pred.index == expected_period_index)

    #########################
    ## Test Finalize Model ##
    #########################

    final_model = exp.finalize_model(blender)
    y_pred = exp.predict_model(final_model, X=future_exog)

    assert np.all(y_pred.index == final_expected_period_index)
