"""Helper functions for time series tests
"""
from random import choice, uniform, randint

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from pycaret.internal.pycaret_experiment import TimeSeriesExperiment
from pycaret.datasets import get_data
from pycaret.containers.models.time_series import get_all_model_containers

_BLEND_TEST_MODELS = [
    "naive",
    "poly_trend",
    "arima",
    "auto_ets",
    "lr_cds_dt",
    "en_cds_dt",
    "knn_cds_dt",
    "dt_cds_dt",
    "lightgbm_cds_dt",
]  # Test blend model functionality only in these models

_ALL_STATS_TESTS = [
    "summary",
    "white_noise",
    "stationarity",
    "adf",
    "kpss",
    "normality",
]


def _get_all_plots():
    exp = TimeSeriesExperiment()
    data = get_data("airline")
    exp.setup(data=data)
    all_plots = list(exp._available_plots.keys())
    all_plots = [None] + all_plots
    return all_plots


def _get_all_plots_data():
    exp = TimeSeriesExperiment()
    data = get_data("airline")
    exp.setup(data=data)
    all_plots = exp._available_plots_data_keys
    all_plots = [None] + all_plots
    return all_plots


def _get_all_plots_estimator():
    exp = TimeSeriesExperiment()
    data = get_data("airline")
    exp.setup(data=data)
    all_plots = exp._available_plots_estimator_keys
    all_plots = [None] + all_plots
    return all_plots


_ALL_PLOTS = _get_all_plots()
_ALL_PLOTS_DATA = _get_all_plots_data()
_ALL_PLOTS_ESTIMATOR = _get_all_plots_estimator()
_ALL_PLOTS_ESTIMATOR_NOT_DATA = list(set(_ALL_PLOTS_ESTIMATOR) - set(_ALL_PLOTS_DATA))


def _get_all_metrics():
    exp = TimeSeriesExperiment()
    data = get_data("airline")
    exp.setup(data=data)
    all_metrics = exp.get_metrics()["Name"].to_list()
    return all_metrics


_ALL_METRICS = _get_all_metrics()


def _get_seasonal_values():
    from pycaret.internal.utils import SeasonalPeriod

    return [(k, v.value) for k, v in SeasonalPeriod.__members__.items()]


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


def _return_splitter_args():
    """[summary]
    """
    parametrize_list = [
        (randint(2, 5), randint(5, 10), "expanding"),
        (randint(2, 5), randint(5, 10), "rolling"),
        (randint(2, 5), randint(5, 10), "sliding"),
    ]
    return parametrize_list


def _return_compare_model_args():
    """Returns cross_validation, log_experiment parameters respectively"""
    parametrize_list = [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ]
    return parametrize_list


def _return_setup_args_raises():
    """
    """
    setup_raises_list = [
        (randint(50, 100), randint(10, 20), "expanding"),
        (randint(50, 100), randint(10, 20), "rolling"),
        (randint(50, 100), randint(10, 20), "sliding"),
    ]
    return setup_raises_list


def _return_data_with_without_period_index():
    """Returns one dataset with period index and one with int index"""
    datasets = [
        get_data("airline"),
        get_data("10", folder="time_series/white_noise"),
    ]
    return datasets


def _return_model_names_for_plots():
    """Returns models to be used for testing plots. Needs
        - 1 model that has prediction interval ("theta")
        - 1 model that does not have prediction interval ("lr_cds_dt")
        - 1 model that has in-sample forecasts ("theta")
        - 1 model that does not have in-sample forecasts ("lr_cds_dt")
    """
    model_names = ["theta", "lr_cds_dt"]
    return model_names


def _return_data_big_small():
    """Returns one dataset with 144 data points and one with < 12 data points"""
    data = get_data("airline")
    data = data - 400
    data_small = data[:12]  # 11 data points
    datasets = [data, data_small]

    return datasets


# def _check_data_for_prophet(mdl_name, data):
#     """Convert data index to DatetimeIndex"""
#     if mdl_name == "prophet":
#         data = data.to_timestamp(freq="M")
#     return data
