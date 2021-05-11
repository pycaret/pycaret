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

pytestmark = pytest.mark.filterwarnings('ignore::UserWarning')

#############################
#### Fixtures Start Here ####
#############################


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
        data=load_data,
        fh=fh,
        fold=fold,
        fold_strategy="expandingwindow",
        verbose=False,
        session_id=42,
    )


@pytest.fixture(scope="session", name="load_models")
def load_ts_models(load_setup):
    """Load all time series module models"""
    globals_dict = {
        "seed": 0,
        "n_jobs_param": -1,
        "gpu_param": False,
        "X_train": pd.DataFrame(get_data("airline")),
    }
    ts_models = get_all_model_containers(globals_dict)
    ts_experiment = load_setup
    ts_estimators = []

    for key in ts_models.keys():
        if not key.startswith(("ensemble")):
            ts_estimators.append(ts_experiment.create_model(key))

    return ts_estimators


###########################
#### Fixtures End Here ####
###########################

##############################
#### Functions Start Here ####
##############################

# NOTE: Fixtures can not be used to parameterize tests
# https://stackoverflow.com/questions/52764279/pytest-how-to-parametrize-a-test-with-a-list-that-is-returned-from-a-fixture
# Hence, we have to create functions and create the parameterized list first
# (must happen during collect phase) before passing it to mark.parameterize.


def _get_seasonal_values():
    from pycaret.internal.utils import SeasonalPeriod

    return [(k, v.value) for k, v in SeasonalPeriod.__members__.items()]


def _return_model_names():
    """Return all model names."""
    globals_dict = {
        "seed": 0,
        "n_jobs_param": -1,
        "gpu_param": False,
        "X_train": pd.DataFrame(get_data("airline")),
    }
    model_containers = get_all_model_containers(globals_dict)

    model_names_ = []
    for model_name in model_containers.keys():
        if not model_name.startswith(("ensemble")):
            model_names_.append(model_name)

    return model_names_


def _return_model_parameters():
    """Parameterize individual models.
    Returns the model names and the corresponding forecast horizons.
    Horizons are alternately picked to be either integers or numpy arrays
    """
    model_names = _return_model_names()
    parameters = [
        (name, np.arange(1, randint(6, 24)) if i % 2 == 0 else randint(6, 24))
        for i, name in enumerate(model_names)
    ]

    return parameters


_model_names = _return_model_names()
_model_parameters = _return_model_parameters()

############################
#### Functions End Here ####
############################


##########################
#### Tests Start Here ####
##########################


@pytest.mark.parametrize("seasonal_period, seasonal_value", _get_seasonal_values())
def test_setup_seasonal_period_str(load_data, seasonal_period, seasonal_value):

    exp = TimeSeriesExperiment()

    fh = np.arange(1, 13)
    fold = 3
    data = load_data

    exp.setup(
        data=data,
        fh=fh,
        fold=fold,
        fold_strategy="expandingwindow",
        verbose=False,
        session_id=42,
        seasonal_period=seasonal_period,
    )

    assert exp.seasonal_period == seasonal_value


@pytest.mark.parametrize("seasonal_key, seasonal_value", _get_seasonal_values())
def test_setup_seasonal_period_int(load_data, seasonal_key, seasonal_value):

    exp = TimeSeriesExperiment()

    fh = np.arange(1, 13)
    fold = 3
    data = load_data

    exp.setup(
        data=data,
        fh=fh,
        fold=fold,
        fold_strategy="expandingwindow",
        verbose=False,
        seasonal_period=seasonal_value,
    )

    assert exp.seasonal_period == seasonal_value


@pytest.mark.parametrize("name, fh", _model_parameters)
def test_create_model(name, fh, load_data):
    """test create_model functionality"""
    exp = TimeSeriesExperiment()
    exp.setup(
        data=load_data,
        fold=3,
        fh=fh,
        fold_strategy="expandingwindow",
        verbose=False,
    )
    model_obj = exp.create_model(name)

    y_pred = model_obj.predict()
    assert isinstance(y_pred, pd.Series)

    fh_index = fh if isinstance(fh, int) else fh[-1]
    expected_period_index = load_data.iloc[-fh_index:].index
    assert np.all(y_pred.index == expected_period_index)

@pytest.mark.filterwarnings('ignore::statsmodels.tools.sm_exceptions.ConvergenceWarning')
@pytest.mark.parametrize("method", _ENSEMBLE_METHODS)
def test_blend_model(load_setup, load_models, method):

    from pycaret.internal.ensemble import _EnsembleForecasterWithVoting

    ts_experiment = load_setup
    ts_models = list(np.random.choice(load_models, 10))
    ts_weights = [uniform(0, 1) for _ in range(len(ts_models))]

    blender = ts_experiment.blend_models(
        ts_models, method=method, weights=ts_weights, verbose=False
    )

    assert isinstance(blender, _EnsembleForecasterWithVoting)

    # Test input models are available
    blender_forecasters = blender.forecasters_
    blender_forecasters_class = [f.__class__ for f in blender_forecasters]
    ts_models_class = [f.__class__ for f in ts_models]
    assert blender_forecasters_class == ts_models_class

@pytest.mark.filterwarnings('ignore::statsmodels.tools.sm_exceptions.ConvergenceWarning')
def test_blend_model_predict(load_setup, load_models):

    ts_experiment = load_setup
    ts_models = list(np.random.choice(load_models, 5))
    ts_weights = [uniform(0, 1) for _ in range(len(ts_models))]
    fh = ts_experiment.fh

    mean_blender = ts_experiment.blend_models(
        ts_models, method="mean"
    )  # , optimize='MAPE')
    median_blender = ts_experiment.blend_models(
        ts_models, method="median"
    )  # ), optimize='MAPE')
    voting_blender = ts_experiment.blend_models(
        ts_models, method="voting", weights=ts_weights
    )  # , optimize='MAPE')

    mean_blender_pred = mean_blender.predict(fh=fh)
    median_blender_pred = median_blender.predict(fh=fh)
    voting_blender_pred = voting_blender.predict(fh=fh)

    mean_median_equal = np.array_equal(mean_blender_pred, median_blender_pred)
    mean_voting_equal = np.array_equal(mean_blender_pred, voting_blender_pred)
    median_voting_equal = np.array_equal(median_blender_pred, voting_blender_pred)

    assert mean_median_equal == False
    assert mean_voting_equal == False
    assert median_voting_equal == False


@pytest.mark.parametrize("model", _model_names)
def test_tune_model_grid(model, load_data):
    exp = TimeSeriesExperiment()
    fh = 12
    fold = 3

    exp.setup(data=load_data, fold=fold, fh=fh, fold_strategy="expandingwindow")

    model_obj = exp.create_model(model)
    tuned_model_obj = exp.tune_model(model_obj)
    y_pred = tuned_model_obj.predict()
    assert isinstance(y_pred, pd.Series)

    expected_period_index = load_data.iloc[-fh:].index
    assert np.all(y_pred.index == expected_period_index)


@pytest.mark.parametrize("model", _model_names)
def test_tune_model_random(model, load_data):
    exp = TimeSeriesExperiment()
    fh = 12
    fold = 3

    exp.setup(data=load_data, fold=fold, fh=fh, fold_strategy="expandingwindow")

    model_obj = exp.create_model(model)
    tuned_model_obj = exp.tune_model(model_obj, search_algorithm="random")
    y_pred = tuned_model_obj.predict()
    assert isinstance(y_pred, pd.Series)

    expected_period_index = load_data.iloc[-fh:].index
    assert np.all(y_pred.index == expected_period_index)
