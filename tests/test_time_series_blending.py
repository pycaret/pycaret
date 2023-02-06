"""Module to test time_series `blend_model` functionality
"""
from random import uniform

import numpy as np  # type: ignore
import pytest

from pycaret.internal.ensemble import _ENSEMBLE_METHODS
from pycaret.time_series import TSForecastingExperiment

##########################
# Tests Start Here ####
##########################


@pytest.mark.filterwarnings(
    "ignore::statsmodels.tools.sm_exceptions.ConvergenceWarning:statsmodels"
)
@pytest.mark.parametrize("method", _ENSEMBLE_METHODS)
def test_blend_model(load_setup, load_models, method):

    from pycaret.internal.ensemble import _EnsembleForecasterWithVoting

    ts_experiment = load_setup
    ts_models = load_models
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


@pytest.mark.filterwarnings(
    "ignore::statsmodels.tools.sm_exceptions.ConvergenceWarning:statsmodels"
)
def test_blend_model_predict(load_setup, load_models):

    ts_experiment = load_setup
    ts_models = load_models
    ts_weights = [uniform(0, 1) for _ in range(len(ts_models))]

    mean_blender = ts_experiment.blend_models(ts_models, method="mean")
    median_blender = ts_experiment.blend_models(ts_models, method="median")
    voting_blender = ts_experiment.blend_models(
        ts_models, method="voting", weights=ts_weights
    )

    mean_blender_pred = ts_experiment.predict_model(mean_blender)
    median_blender_pred = ts_experiment.predict_model(median_blender)
    voting_blender_pred = ts_experiment.predict_model(voting_blender)

    mean_median_equal = np.array_equal(mean_blender_pred, median_blender_pred)
    mean_voting_equal = np.array_equal(mean_blender_pred, voting_blender_pred)
    median_voting_equal = np.array_equal(median_blender_pred, voting_blender_pred)

    assert mean_median_equal is False
    assert mean_voting_equal is False
    assert median_voting_equal is False


def test_blend_model_custom_folds(load_pos_and_neg_data):
    """test custom folds in blend_model"""
    exp = TSForecastingExperiment()
    setup_fold = 3
    exp.setup(
        data=load_pos_and_neg_data,
        fold=setup_fold,
        fh=12,
        fold_strategy="sliding",
        verbose=False,
    )

    #######################################
    # Test Tune Model with custom folds ##
    #######################################
    model = exp.create_model("naive")
    _ = exp.blend_models([model, model, model])
    metrics1 = exp.pull()

    custom_fold = 5
    _ = exp.blend_models([model, model, model], fold=custom_fold)
    metrics2 = exp.pull()

    assert len(metrics1) == setup_fold + 2  # + 2 for Mean and SD
    assert len(metrics2) == custom_fold + 2  # + 2 for Mean and SD
