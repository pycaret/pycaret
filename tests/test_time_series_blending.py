"""Module to test time_series `blend_model` functionality
"""
import random

import numpy as np
import pandas as pd
import pytest

from pycaret.datasets import get_data
from pycaret.time_series import TSForecastingExperiment

##########################
# Tests Start Here ####
##########################


@pytest.mark.filterwarnings(
    "ignore::statsmodels.tools.sm_exceptions.ConvergenceWarning:statsmodels"
)
@pytest.mark.parametrize("method", ["mean", "median", "min", "max", "gmean"])
def test_blend_model_basic(load_setup, load_models, method):
    """Tests basic blender functionality for all methods"""
    from sktime.forecasting.compose import EnsembleForecaster

    exp = load_setup
    models = load_models
    weights = [random.uniform(0, 1) for _ in range(len(models))]
    blender = exp.blend_models(models, method=method, weights=weights, verbose=False)
    assert isinstance(blender, EnsembleForecaster)

    # Test input models are available
    blender_forecasters = blender.forecasters_
    blender_forecasters_class = [f.__class__ for f in blender_forecasters]
    ts_models_class = [f.__class__ for f in models]
    assert blender_forecasters_class == ts_models_class


def test_blend_models_tuning():
    """Test the tuning of blended models."""
    data = get_data("airline", verbose=False)

    exp = TSForecastingExperiment()
    exp.setup(data=data, fh=12, fold=2, session_id=42)
    model1 = exp.create_model("naive")
    model2 = exp.create_model("ets")
    model3 = exp.create_model("lr_cds_dt")
    blender = exp.blend_models([model1, model2, model3])
    _, tuner = exp.tune_model(blender, return_tuner=True)

    assert len(pd.DataFrame(tuner.cv_results_)) > 1


@pytest.mark.filterwarnings(
    "ignore::statsmodels.tools.sm_exceptions.ConvergenceWarning:statsmodels"
)
def test_blend_model_predict(load_setup, load_models):
    """Test to make sure that blending predictions are different when they need
    to be and same when they need to be (depending on the hyperparameters).
    """
    exp = load_setup
    models = load_models
    random.seed(42)
    weights = [random.uniform(0, 1) for _ in range(len(models))]

    # -------------------------------------------------------------------------#
    # Prediction should be different for different methods
    # -------------------------------------------------------------------------#
    mean_blender = exp.blend_models(models, method="mean")
    gmean_blender = exp.blend_models(models, method="gmean")
    median_blender = exp.blend_models(models, method="median")
    min_blender = exp.blend_models(models, method="min")
    max_blender = exp.blend_models(models, method="max")

    mean_blender_w_wts = exp.blend_models(models, method="mean", weights=weights)
    gmean_blender_w_wts = exp.blend_models(models, method="gmean", weights=weights)
    median_blender_w_wts = exp.blend_models(models, method="median", weights=weights)
    min_blender_w_wts = exp.blend_models(models, method="min", weights=weights)
    max_blender_w_wts = exp.blend_models(models, method="max", weights=weights)

    mean_blender_pred = exp.predict_model(mean_blender)
    gmean_blender_pred = exp.predict_model(gmean_blender)
    median_blender_pred = exp.predict_model(median_blender)
    min_blender_pred = exp.predict_model(min_blender)
    max_blender_pred = exp.predict_model(max_blender)

    mean_blender_w_wts_pred = exp.predict_model(mean_blender_w_wts)
    gmean_blender_w_wts_pred = exp.predict_model(gmean_blender_w_wts)
    median_blender_w_wts_pred = exp.predict_model(median_blender_w_wts)
    min_blender_w_wts_pred = exp.predict_model(min_blender_w_wts)
    max_blender_w_wts_pred = exp.predict_model(max_blender_w_wts)

    different_preds = [
        mean_blender_pred,
        gmean_blender_pred,
        median_blender_pred,
        min_blender_pred,
        max_blender_pred,
        mean_blender_w_wts_pred,
        gmean_blender_w_wts_pred,
        median_blender_w_wts_pred,
    ]

    for i, _ in enumerate(different_preds):
        for j in range(i + 1, len(different_preds)):
            assert not np.array_equal(different_preds[i], different_preds[j])

    # -------------------------------------------------------------------------#
    # Prediction for some methods should not be impacted by weights
    # e.g. min, max
    # -------------------------------------------------------------------------#
    assert np.array_equal(
        min_blender_pred, min_blender_w_wts_pred
    ), "min blender predictions with and without weights are not the same"
    assert np.array_equal(
        max_blender_pred, max_blender_w_wts_pred
    ), "max blender predictions with and without weights are not the same"


def test_blend_model_custom_folds(load_pos_and_neg_data):
    """Test custom folds in blend_model"""
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


def test_blend_with_larger_predict_fh():
    """Test to make sure that blending predictions work when the forecast horizon
    used in predictions is larger than the one used for training
    Ref: https://github.com/pycaret/pycaret/issues/2329
    """
    data = get_data("airline", verbose=False)

    exp = TSForecastingExperiment()
    exp.setup(data=data, fh=12, fold=2, session_id=42)
    model1 = exp.create_model("naive")
    model2 = exp.create_model("ets")
    model3 = exp.create_model("lr_cds_dt")
    blender = exp.blend_models([model1, model2, model3])

    # Check that forecasts can be created for FH greater than the one used for training
    FHs = [12, 24]
    for fh in FHs:
        preds = exp.predict_model(blender, fh=fh)
        assert len(preds) == fh


def test_error_conditions(load_setup, load_models):
    """Tests error conditions for blend_models"""
    exp = load_setup
    models = load_models
    random.seed(42)
    weights = [random.uniform(0, 1) for _ in range(len(models))]

    with pytest.raises(ValueError) as err_msg:
        _ = exp.blend_models(models, method="voting", weights=weights)

    exception_msg = err_msg.value.args[0]
    assert "method 'voting' is not supported from pycaret 3.0.1" in exception_msg
