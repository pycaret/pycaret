"""Module to test time_series functionality
"""
import pytest

from random import choice, uniform, randint
from pycaret.internal.ensemble import _ENSEMBLE_METHODS
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from pycaret.datasets import get_data
from pycaret.internal.pycaret_experiment import TimeSeriesExperiment
from pycaret.containers.models.time_series import get_all_model_containers

#############################
#### Fixtures Start Here ####
#############################

# TODO: Eventually remove this and use automatically sourced model names
_all_models = ["naive", "poly_trend", "arima", "exp_smooth", "theta" , "auto_ets", "rf_dts"]

@pytest.fixture(scope="session", name="load_data")
def load_data():
    """Load Pycaret Airline dataset."""
    return get_data("airline")


@pytest.fixture(scope="session", name="load_setup")
def load_setup(load_data):
    """Create a TimeSeriesExperiment to test module functionalities"""
    exp = TimeSeriesExperiment()

    fh = np.arange(1, 13)
    fold = 3

    return exp.setup(
        data=load_data, fh=fh, fold=fold, fold_strategy="expandingwindow", verbose=False
    )


@pytest.fixture(scope="session", name="model_names")
def model_names():
    """Return all model names."""
    globals_dict = {"seed": 0, "n_jobs_param": -1}
    model_containers = get_all_model_containers(globals_dict)

    model_names_ = []
    for model_name in model_containers.keys():
        if not model_name.startswith(("ensemble")):
            model_names_.append(model_name)

    return model_names_


@pytest.fixture(scope="session", name="model_parameters")
def model_parameters(model_names):
    """Parameterize individual models.
    Returns the model names and the corresponding forecast horizons.
    Horizons are alternately picked to be either integers or numpy arrays
    """
    parameters = [
        (name, np.arange(1, randint(6, 24)) if i%2==0  else randint(6, 24))
        for i, name in enumerate(model_names)
    ]
    return parameters


@pytest.fixture(scope="session", name="load_models")
def load_ts_models(load_setup):
    """Load all time series module models"""
    globals_dict = {"seed": 0, "n_jobs_param": -1}
    ts_models = get_all_model_containers(globals_dict)
    ts_experiment = load_setup
    ts_estimators = []

    for key in ts_models.keys():
        if not key.startswith(("ensemble")): # , "poly")):
            ts_estimators.append(ts_experiment.create_model(key))

    return ts_estimators


###########################
#### Fixtures End Here ####
###########################

##########################
#### Tests Start Here ####
##########################

def test_set_up_valid_seasonality(load_data):

    seasonal_parameter = 30
    fh = np.arange(1,13)
    fold = 3
    
    exp = TimeSeriesExperiment()
    
    with pytest.raises(ValueError) as errmsg:
        _ = exp.setup(
            data=load_data, 
            fh=fh, 
            fold=fold, 
            fold_strategy='slidingwindow', 
            seasonal_parameter=seasonal_parameter
        )

    exceptionmsg = errmsg.value.args[0]

    assert exceptionmsg == f"Autocorrelation Seasonality test failed: Invalid Seasonality Period {seasonal_parameter}"


def test_create_model(load_setup, model_parameters, load_data):
    """test create_model functionality
    """
    exp = TimeSeriesExperiment()
    for name, fh in model_parameters:
        # Need to create individual setup for each model since the `fh` will be different for all models
        exp.setup(
            data=load_data, fold=3, fh=fh, fold_strategy="expandingwindow", verbose=False
        )
        model_obj = exp.create_model(name)

        y_pred = model_obj.predict()
        assert isinstance(y_pred, pd.Series)

        fh_index = fh if isinstance(fh, int) else fh[-1]
        expected_period_index = load_data.iloc[-fh_index:].index
        assert np.all(y_pred.index == expected_period_index)


@pytest.mark.parametrize("method", _ENSEMBLE_METHODS)
def test_blend_model(load_setup, load_models, method):

    from pycaret.internal.ensemble import _EnsembleForecasterWithVoting

    ts_experiment = load_setup
    ts_models = load_models
    ts_weights = [uniform(0, 1) for _ in range(len(load_models))]

    blender = ts_experiment.blend_models(
        ts_models, method=method, weights=ts_weights, verbose=False
    )

    assert isinstance(blender, _EnsembleForecasterWithVoting)

    # Test input models are available
    blender_forecasters = blender.forecasters_
    blender_forecasters_class = [f.__class__ for f in blender_forecasters]
    ts_models_class = [f.__class__ for f in ts_models]
    assert blender_forecasters_class == ts_models_class


def test_blend_model_predict(load_setup, load_models):

    ts_experiment = load_setup
    ts_models = load_models
    ts_weights = [uniform(0, 1) for _ in range(len(load_models))]
    fh = ts_experiment.fh

    mean_blender = ts_experiment.blend_models(ts_models, method='mean') #, optimize='MAPE')
    median_blender = ts_experiment.blend_models(ts_models, method='median') #), optimize='MAPE')
    voting_blender = ts_experiment.blend_models(ts_models, method='voting', weights=ts_weights) #, optimize='MAPE')

    mean_blender_pred = mean_blender.predict(fh=fh)
    median_blender_pred = median_blender.predict(fh=fh)
    voting_blender_pred = voting_blender.predict(fh=fh)

    mean_median_equal = np.array_equal(mean_blender_pred, median_blender_pred)
    mean_voting_equal = np.array_equal(mean_blender_pred, voting_blender_pred)
    median_voting_equal = np.array_equal(median_blender_pred, voting_blender_pred)

    assert mean_median_equal == False
    assert mean_voting_equal == False
    assert median_voting_equal == False


@pytest.mark.parametrize("model", _all_models)
def test_tune_model_grid(model, load_data):
    exp = TimeSeriesExperiment()
    fh = 12
    fold = 3

    exp.setup(
        data=load_data,
        fold=fold,
        fh=fh,
        fold_strategy="expandingwindow"
    )

    model_obj = exp.create_model(model)
    tuned_model_obj = exp.tune_model(model_obj)
    y_pred = tuned_model_obj.predict()
    assert isinstance(y_pred, pd.Series)

    expected_period_index = load_data.iloc[-fh:].index
    assert np.all(y_pred.index == expected_period_index)


@pytest.mark.parametrize("model", _all_models)
def test_tune_model_random(model, load_data):
    exp = TimeSeriesExperiment()
    fh = 12
    fold = 3

    exp.setup(
        data=load_data,
        fold=fold,
        fh=fh,
        fold_strategy="expandingwindow"
    )

    model_obj = exp.create_model(model)
    tuned_model_obj = exp.tune_model(model_obj, search_algorithm="random")
    y_pred = tuned_model_obj.predict()
    assert isinstance(y_pred, pd.Series)

    expected_period_index = load_data.iloc[-fh:].index
    assert np.all(y_pred.index == expected_period_index)
