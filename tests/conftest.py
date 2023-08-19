import numpy as np
import pytest

import pycaret.anomaly.functional
import pycaret.classification.functional
import pycaret.clustering.functional
import pycaret.regression.functional
import pycaret.time_series.forecasting.functional
from pycaret.containers.models.time_series import get_all_model_containers
from pycaret.datasets import get_data
from pycaret.time_series import TSForecastingExperiment

#############################
# Fixtures Start Here ####
#############################


@pytest.fixture(name="change_test_dir", autouse=True)
def change_test_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def reset_experiments():
    yield
    pycaret.classification.functional._CURRENT_EXPERIMENT = None
    pycaret.regression.functional._CURRENT_EXPERIMENT = None
    pycaret.anomaly.functional._CURRENT_EXPERIMENT = None
    pycaret.clustering.functional._CURRENT_EXPERIMENT = None
    pycaret.time_series.forecasting.functional._CURRENT_EXPERIMENT = None


@pytest.fixture(scope="session", name="load_pos_data")
def load_pos_data():
    """Load Pycaret Airline dataset."""
    data = get_data("airline")
    return data


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


@pytest.fixture(scope="session", name="load_uni_exo_data_target_positive")
def load_uni_exo_data_target_positive():
    """Load Pycaret Univariate data with exogenous variables (strictly positive)."""
    data = get_data("uschange")
    data = data.clip(lower=0.1)
    target = "Consumption"
    return data, target


@pytest.fixture(scope="session", name="load_pos_data_missing")
def load_pos_data_missing():
    """Load Pycaret Airline dataset (with missing values)."""
    data = get_data("airline")
    remove_n = int(0.4 * len(data))
    np.random.seed(42)
    na_indices = np.random.choice(data.index, remove_n, replace=False)
    data[na_indices] = np.nan
    return data


@pytest.fixture(scope="session", name="load_pos_and_neg_data_missing")
def load_pos_and_neg_data_missing():
    """Load Pycaret Airline dataset (with some negative & missing values)."""
    data = get_data("airline")
    data = data - 400  # simulate negative values
    data[10:20] = np.nan  # In train with FH = 12
    data[-5:-2] = np.nan  # In test with FH = 12
    return data


@pytest.fixture(scope="session", name="load_uni_exo_data_target_missing")
def load_uni_exo_data_target_missing():
    """Load Pycaret Univariate data with exogenous variables & missing values."""
    data = get_data("uschange")
    data[10:20] = np.nan  # In train with FH = 12
    data[-5:-2] = np.nan  # In test with FH = 12
    target = "Consumption"
    return data, target


@pytest.fixture(scope="session", name="load_models_uni_exo")
def load_models_uni_exo():
    """Load models that support univariate date with exogenous variables."""
    # TODO: Later, get this dynamically from sktime
    models = ["arima", "lr_cds_dt"]
    return models


@pytest.fixture(scope="session", name="load_models_uni_mix_exo_noexo")
def load_models_uni_mix_exo_noexo():
    """Load a sample mix of models that support univariate date with
    exogenous variables and those that do not."""
    # TODO: Later, get this dynamically from sktime
    models = ["naive", "ets", "arima", "lr_cds_dt"]
    return models


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
    exp = load_setup
    model_containers = get_all_model_containers(exp)

    from time_series_test_utils import (  # TODO Put it back once preprocessing supports series as X
        _BLEND_TEST_MODELS,
    )

    ts_estimators = [
        exp.create_model(key)
        for key in model_containers.keys()
        if key in _BLEND_TEST_MODELS
    ]

    return ts_estimators


###########################
# Fixtures End Here ####
###########################
