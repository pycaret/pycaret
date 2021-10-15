import pytest

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from pycaret.datasets import get_data
from pycaret.internal.pycaret_experiment import TimeSeriesExperiment
from pycaret.containers.models.time_series import get_all_model_containers

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


@pytest.fixture(scope="session", name="load_pos_data")
def load_pos_data():
    """Load Pycaret Airline dataset."""
    data = get_data("airline")
    return data


@pytest.fixture(scope="session", name="load_setup")
def load_setup(load_pos_and_neg_data):
    """Create a TimeSeriesExperiment to test module functionalities"""
    exp = TimeSeriesExperiment()

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
