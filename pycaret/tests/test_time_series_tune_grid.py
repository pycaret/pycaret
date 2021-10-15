"""Module to test time_series functionality
"""
import pytest

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from pycaret.datasets import get_data
from pycaret.internal.pycaret_experiment import TimeSeriesExperiment
from pycaret.containers.models.time_series import get_all_model_containers

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


#############################
#### Fixtures Start Here ####
#############################


@pytest.fixture(scope="session", name="load_data")
def load_data():
    """Load Pycaret Airline dataset."""
    data = get_data("airline")
    data = data - 400  # simulate negative values
    return data


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


def _check_windows():
    """Check if the system is Windows."""
    import sys

    platform = sys.platform
    is_windows = True if platform.startswith("win") else False

    return is_windows


def _return_model_names():
    """Return all model names."""
    globals_dict = {
        "seed": 0,
        "n_jobs_param": -1,
        "gpu_param": False,
        "X_train": pd.DataFrame(get_data("airline")),
        "enforce_pi": False,
    }
    model_containers = get_all_model_containers(globals_dict)

    models_to_ignore = (
        ["prophet", "ensemble_forecaster"]
        if _check_windows()
        else ["ensemble_forecaster"]
    )

    model_names_ = []
    for model_name in model_containers.keys():

        if model_name not in models_to_ignore:
            model_names_.append(model_name)

    return model_names_


_model_names = _return_model_names()


############################
#### Functions End Here ####
############################


##########################
#### Tests Start Here ####
##########################


@pytest.mark.parametrize("model", _model_names)
def test_tune_model_grid(model, load_data):
    exp = TimeSeriesExperiment()
    fh = 12
    fold = 2
    data = load_data

    exp.setup(data=data, fold=fold, fh=fh, fold_strategy="sliding")

    model_obj = exp.create_model(model)
    tuned_model_obj = exp.tune_model(model_obj, search_algorithm="grid")
    y_pred = exp.predict_model(tuned_model_obj)
    assert isinstance(y_pred, pd.Series)

    expected_period_index = data.iloc[-fh:].index
    assert np.all(y_pred.index == expected_period_index)

