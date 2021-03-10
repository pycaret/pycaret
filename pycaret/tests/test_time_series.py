"""Module to test time_series functionality
"""
import pytest

from random import choice, uniform
from pycaret.internal.ensemble import _ENSEMBLE_METHODS
import numpy as np
import pandas as pd


@pytest.fixture(scope="session", name="load_data")
def load_data():
    """Load Pycaret Airline dataset."""
    from pycaret.datasets import get_data

    return get_data("airline")


@pytest.fixture(scope="session", name="load_setup")
def load_setup(load_data):
    """Create a TimeSeriesExperiment to test module functionalities"""
    from pycaret.internal.PycaretExperiment import TimeSeriesExperiment
    import numpy as np

    ts_experiment = TimeSeriesExperiment()

    fh = np.arange(1, 13)
    fold = 3

    return ts_experiment.setup(
        data=load_data, fh=fh, fold=fold, fold_strategy="expandingwindow", verbose=False
    )


@pytest.fixture(scope="session", name="load_models")
def load_ts_models(load_setup):
    """Load all time series module models"""
    from pycaret.containers.models.time_series import get_all_model_containers

    globals_dict = {"seed": 0, "n_jobs_param": -1}
    ts_models = get_all_model_containers(globals_dict)
    ts_experiment = load_setup
    ts_estimators = []

    for key in ts_models.keys():
        if not key.startswith(("ensemble", "poly")):
            ts_estimators.append(ts_experiment.create_model(key))

    return ts_estimators


models = ["naive", "poly_trend", "arima", "exp_smooth", "theta"]
parametrize_list = [(choice(models))]


@pytest.mark.parametrize("model", parametrize_list)
def test_create_model(model, load_data):

    from pycaret.internal.PycaretExperiment import TimeSeriesExperiment

    exp = TimeSeriesExperiment()
    exp.setup(
        data=load_data, fold=3, fh=12, fold_strategy="expandingwindow", verbose=False
    )

    model_obj = exp.create_model(model)
    y_pred = model_obj.predict()
    assert isinstance(y_pred, pd.Series)
    expected = pd.core.indexes.period.PeriodIndex(
        [
            "1957-05",
            "1957-06",
            "1957-07",
            "1957-08",
            "1957-09",
            "1957-10",
            "1957-11",
            "1957-12",
            "1958-01",
            "1958-02",
            "1958-03",
            "1958-04",
        ],
        dtype="period[M]",
        freq="M",
    )
    assert np.all(y_pred.index == expected)


@pytest.mark.parametrize("method", _ENSEMBLE_METHODS)
def test_blend_model(load_setup, load_models, method):

    from pycaret.internal.ensemble import _EnsembleForecasterWithVoting

    ts_experiment = load_setup
    ts_models = load_models
    ts_weights = [uniform(0, 1) for _ in range(len(load_models))]

    blender = ts_experiment.blend_models(
        ts_models, method=method, weights=ts_weights, optimize="MAPE_ts", verbose=False
    )

    assert isinstance(blender, _EnsembleForecasterWithVoting)

    # Test input models are available
    blender_forecasters = blender.forecasters_
    blender_forecasters_class = [f.__class__ for f in blender_forecasters]
    ts_models_class = [f.__class__ for f in ts_models]
    assert blender_forecasters_class == ts_models_class
