import pytest

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from pycaret.datasets import get_data
from pycaret.time_series import TSForecastingExperiment
from pycaret.containers.models.time_series import get_all_model_containers
from pycaret.utils.time_series import TSExogenousPresent

from .time_series_test_utils import _BLEND_TEST_MODELS

#############################
#### Fixtures Start Here ####
#############################


@pytest.fixture(scope="session", name="load_pos_and_neg_data")
def load_pos_and_neg_data():
    """Load Pycaret Airline dataset (with some negative values)."""
    data = get_data("airline")
    data = data - 400  # simulate negative values
    return data


@pytest.fixture(scope="session", name="load_uni_exo_data_target")
def load_uni_exo_data_target():
    """Load Pycaret Univariate data with exogenous variables."""
    data = get_data("uschange")
    target = "Consumption"
    return data, target


@pytest.fixture(scope="session", name="load_models_uni_exo")
def load_models_uni_exo():
    """Load models that support univariate date with exogenous variables."""
    # TODO: Later, get this dynamically from sktime
    models = ["arima", "auto_arima"]
    return models


@pytest.fixture(scope="session", name="load_models_uni_mix_exo_noexo")
def load_models_uni_mix_exo_noexo():
    """Load a sample mix of models that support univariate date with
    exogenous variables and those that do not."""
    # TODO: Later, get this dynamically from sktime
    models = ["naive", "ets", "arima"]
    return models


@pytest.fixture(scope="session", name="load_pos_data")
def load_pos_data():
    """Load Pycaret Airline dataset."""
    data = get_data("airline")
    return data


@pytest.fixture(scope="session", name="load_setup")
def load_setup(load_pos_and_neg_data):
    """Create a TSForecastingExperiment to test module functionalities"""
    exp = TSForecastingExperiment()

    fh = np.arange(1, 13)
    fold = 2

    return exp.setup(
        data=load_pos_and_neg_data,
        fh=fh,
        fold=fold,
        fold_strategy="sliding",
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
        "enforce_pi": False,
        "enforce_exogenous": True,
        "exogenous_present": TSExogenousPresent.NO,
        "sp_to_use": 12,
    }
    ts_models = get_all_model_containers(globals_dict)
    ts_experiment = load_setup
    ts_estimators = [
        ts_experiment.create_model(key)
        for key in ts_models.keys()
        if key in _BLEND_TEST_MODELS
    ]

    return ts_estimators


###########################
#### Fixtures End Here ####
###########################
