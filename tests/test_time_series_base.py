"""Module to test time_series functionality
"""
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sktime.forecasting.compose import ForecastingPipeline
from time_series_test_utils import _return_compare_model_args, _return_model_parameters

from pycaret.time_series import TSForecastingExperiment

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


##############################
# Functions Start Here ####
##############################

# NOTE: Fixtures can not be used to parameterize tests
# https://stackoverflow.com/questions/52764279/pytest-how-to-parametrize-a-test-with-a-list-that-is-returned-from-a-fixture
# Hence, we have to create functions and create the parameterized list first
# (must happen during collect phase) before passing it to mark.parameterize.

_model_parameters = _return_model_parameters()
_compare_model_args = _return_compare_model_args()

############################
# Functions End Here ####
############################


##########################
# Tests Start Here ####
##########################


@pytest.mark.parametrize("name, fh", _model_parameters)
def test_create_predict_finalize_model(name, fh, load_pos_and_neg_data):
    """test create_model, predict_model and finalize_model functionality
    Combined to save run time
    """
    exp = TSForecastingExperiment()
    data = load_pos_and_neg_data

    exp.setup(
        data=data,
        fold=2,
        fh=fh,
        fold_strategy="sliding",
        verbose=False,
    )
    #######################
    # Test Create Model ##
    #######################
    model = exp.create_model(name)
    assert not isinstance(model, ForecastingPipeline)

    #########################
    # Expected Values ####
    #########################
    # Only forcasted values
    fh_index = fh if isinstance(fh, int) else len(fh)
    # Full forecasting window
    fh_max_window = fh if isinstance(fh, int) else max(fh)

    expected_period_index = load_pos_and_neg_data.iloc[-fh_index:].index
    final_expected_period_index = expected_period_index.shift(fh_max_window)

    ########################
    # Test Predict Model ##
    ########################
    # Default prediction
    y_pred = exp.predict_model(model)
    assert isinstance(y_pred, pd.DataFrame)
    assert np.all(y_pred.index == expected_period_index)

    # With Prediction Interval (default coverage = 0.9)
    y_pred = exp.predict_model(model, return_pred_int=True)
    assert isinstance(y_pred, pd.DataFrame)
    assert np.all(y_pred.columns == ["y_pred", "lower", "upper"])
    assert np.all(y_pred.index == expected_period_index)

    # With Prediction Interval (coverage float = 0.8)
    y_pred2 = exp.predict_model(model, return_pred_int=True, coverage=0.8)
    assert isinstance(y_pred2, pd.DataFrame)
    assert np.all(y_pred2.columns == ["y_pred", "lower", "upper"])
    assert np.all(y_pred2.index == expected_period_index)

    # With Prediction Interval (coverage List = [0.1, 0.9])
    y_pred3 = exp.predict_model(model, return_pred_int=True, coverage=[0.1, 0.9])
    assert_frame_equal(y_pred2, y_pred3)  # check_exact=False

    # Increased forecast horizon to 2 years instead of the original 1 year
    y_pred = exp.predict_model(model, fh=np.arange(1, 25))
    assert len(y_pred) == 24

    #########################
    # Test Finalize Model ##
    #########################

    final_pipeline = exp.finalize_model(model)
    assert isinstance(final_pipeline, ForecastingPipeline)

    y_pred = exp.predict_model(final_pipeline)
    assert np.all(y_pred.index == final_expected_period_index)


def test_predict_model_metrics_displayed(load_pos_and_neg_data):
    """Tests different cases in predict_model when metrics should and should not
    be displayed"""
    exp = TSForecastingExperiment()
    FH = 12
    exp.setup(
        data=load_pos_and_neg_data,
        fold=2,
        fh=FH,
        fold_strategy="sliding",
        verbose=False,
    )

    model = exp.create_model("naive")

    ######################################
    # Test before finalizing model ####
    ######################################
    # Default (Correct comparison to test set)
    _ = exp.predict_model(model)
    expected = exp.pull()

    # Metrics are returned ----
    # (1) User provides fh resulting in prediction whose indices are same as y_test
    _ = exp.predict_model(model, fh=FH)
    metrics = exp.pull()
    assert metrics.equals(expected)

    # No metrics returned ----
    # All values are 0
    expected.iloc[0, 1:] = 0
    cols = expected.select_dtypes(include=["float"])
    for col in cols:
        expected[col] = expected[col].astype(np.int64)

    # (2) User provides fh resulting in prediction whose indices are less than y_test
    _ = exp.predict_model(model, fh=FH - 1)
    metrics = exp.pull()
    assert metrics.equals(expected)

    # (3) User provides fh resulting in prediction whose indices are more than y_test
    _ = exp.predict_model(model, fh=FH + 1)
    metrics = exp.pull()
    assert metrics.equals(expected)


def test_create_model_custom_folds(load_pos_and_neg_data):
    """test custom fold in create_model"""
    exp = TSForecastingExperiment()
    setup_fold = 3
    exp.setup(
        data=load_pos_and_neg_data,
        fold=setup_fold,
        fh=12,
        fold_strategy="sliding",
        verbose=False,
    )

    #########################################
    # Test Create Model with custom folds ##
    #########################################
    _ = exp.create_model("naive")
    metrics1 = exp.pull()

    custom_fold = 5
    _ = exp.create_model("naive", fold=custom_fold)
    metrics2 = exp.pull()

    assert len(metrics1) == setup_fold + 2  # + 2 for Mean and SD
    assert len(metrics2) == custom_fold + 2  # + 2 for Mean and SD


def test_create_model_no_cv(load_pos_and_neg_data):
    """test create_model without cross validation"""
    exp = TSForecastingExperiment()
    exp.setup(
        data=load_pos_and_neg_data,
        fh=12,
        fold_strategy="sliding",
        verbose=False,
    )

    ##################################
    # Test Create Model without cv ##
    ##################################
    model = exp.create_model("naive", cross_validation=False)
    assert not isinstance(model, ForecastingPipeline)
    metrics = exp.pull()

    # Should return only 1 row for the test set (since no CV)
    assert len(metrics) == 1


def test_prediction_interval_na(load_pos_and_neg_data):
    """Tests predict model when interval is NA"""

    exp = TSForecastingExperiment()

    fh = 12
    fold = 2
    data = load_pos_and_neg_data

    exp.setup(
        data=data,
        fh=fh,
        fold=fold,
        fold_strategy="expanding",
        verbose=False,
        session_id=42,
    )

    # For models that do not produce a prediction interval --> returns NA values
    model = exp.create_model("lr_cds_dt")
    y_pred = exp.predict_model(model, return_pred_int=True)
    assert y_pred["lower"].isnull().all()
    assert y_pred["upper"].isnull().all()


@pytest.mark.parametrize("cross_validation, log_experiment", _compare_model_args)
def test_compare_models(cross_validation, log_experiment, load_pos_and_neg_data):
    """tests compare_models functionality"""
    exp = TSForecastingExperiment()

    fh = 12
    fold = 2
    data = load_pos_and_neg_data

    exp.setup(
        data=data,
        fh=fh,
        fold=fold,
        fold_strategy="expanding",
        verbose=False,
        session_id=42,
        log_experiment=log_experiment,
        log_plots=log_experiment,
    )

    best_baseline_models = exp.compare_models(
        n_select=3, cross_validation=cross_validation
    )
    assert len(best_baseline_models) == 3
    for best in best_baseline_models:
        assert not isinstance(best, ForecastingPipeline)


def test_save_load_model_no_setup(load_pos_and_neg_data):
    """Tests the save_model and load_model functionality without setup.
    Applicable when user saves the entire pipeline.
    """

    fh = np.arange(1, 13)
    fold = 2
    data = load_pos_and_neg_data

    ######################
    # OOP Approach ####
    ######################
    exp = TSForecastingExperiment()
    exp.setup(
        data=data,
        fh=fh,
        fold=fold,
        fold_strategy="sliding",
        verbose=False,
        session_id=42,
    )

    model = exp.create_model("ets")
    expected_predictions = exp.predict_model(model)
    exp.save_model(model, "model_unit_test_oop_nosetup")

    # Mimic loading in another session - Predictions without setup
    exp_loaded = TSForecastingExperiment()
    loaded_model = exp_loaded.load_model("model_unit_test_oop_nosetup")
    loaded_predictions = exp_loaded.predict_model(loaded_model)

    assert np.all(loaded_predictions == expected_predictions)

    ########################
    # Functional API ####
    ########################
    from pycaret.time_series import (
        create_model,
        load_model,
        predict_model,
        save_model,
        setup,
    )

    _ = setup(
        data=data, fh=fh, fold=fold, fold_strategy="expanding", session_id=42, n_jobs=-1
    )
    model = create_model("naive")
    expected_predictions = predict_model(model)
    save_model(model, "model_unit_test_func_nosetup")

    # Mimic loading in another session - Predictions without setup
    loaded_model = load_model("model_unit_test_func_nosetup")
    loaded_predictions = predict_model(loaded_model)

    assert np.all(loaded_predictions == expected_predictions)


def test_save_load_model_setup(load_pos_and_neg_data):
    """Tests the save_model and load_model functionality with setup.
    Applicable when user saves the model (without pipeline), then loads the model,
    runs setup and uses this model.
    """

    fh = np.arange(1, 13)
    fold = 2
    data = load_pos_and_neg_data

    ######################
    # OOP Approach ####
    ######################
    exp = TSForecastingExperiment()
    exp.setup(
        data=data,
        fh=fh,
        fold=fold,
        fold_strategy="sliding",
        verbose=False,
        session_id=42,
    )

    model = exp.create_model("ets")
    expected_predictions = exp.predict_model(model)
    exp.save_model(model, "model_unit_test_oop_setup", model_only=True)

    # Mimic loading in another session - Predictions with setup
    exp_loaded = TSForecastingExperiment()
    exp_loaded.setup(
        data=data,
        fh=fh,
        fold=fold,
        fold_strategy="sliding",
        verbose=False,
        session_id=42,
    )
    loaded_model = exp_loaded.load_model("model_unit_test_oop_setup")
    loaded_predictions = exp_loaded.predict_model(loaded_model)

    assert np.all(loaded_predictions == expected_predictions)

    ########################
    # Functional API ####
    ########################
    from pycaret.time_series import (
        create_model,
        load_model,
        predict_model,
        save_model,
        setup,
    )

    _ = setup(
        data=data, fh=fh, fold=fold, fold_strategy="expanding", session_id=42, n_jobs=-1
    )
    model = create_model("naive")
    expected_predictions = predict_model(model)
    save_model(model, "model_unit_test_func_setup", model_only=True)

    # Mimic loading in another session - Predictions with setup
    setup(
        data=data,
        fh=fh,
        fold=fold,
        fold_strategy="sliding",
        verbose=False,
        session_id=42,
    )
    loaded_model = load_model("model_unit_test_func_setup")
    loaded_predictions = predict_model(loaded_model)

    assert np.all(loaded_predictions == expected_predictions)


def test_save_load_raises(load_pos_and_neg_data):
    """Tests the save_model and load_model that raises an exception. i.e. when
    only model is saved (without pipeline) and after loading, setup is not run.
    """

    fh = np.arange(1, 13)
    fold = 2
    data = load_pos_and_neg_data

    exp = TSForecastingExperiment()
    exp.setup(
        data=data,
        fh=fh,
        fold=fold,
        fold_strategy="sliding",
        verbose=False,
        session_id=42,
    )

    model = exp.create_model("ets")
    exp.save_model(model, "model_unit_test_oop_raises", model_only=True)

    # Mimic loading in another session
    exp_loaded = TSForecastingExperiment()
    loaded_model = exp_loaded.load_model("model_unit_test_oop_raises")

    # Setup not run and only passing a estimator without pipeline ----
    with pytest.raises(ValueError) as errmsg:
        _ = exp_loaded.predict_model(loaded_model)
    exceptionmsg = errmsg.value.args[0]
    assert (
        "Setup has not been run and you have provided a estimator without the pipeline"
        in exceptionmsg
    )
