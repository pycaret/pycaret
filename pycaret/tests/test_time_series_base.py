"""Module to test time_series functionality
"""
import pytest
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from pycaret.time_series import TSForecastingExperiment

from .time_series_test_utils import (
    _return_model_parameters,
    _return_compare_model_args,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


##############################
#### Functions Start Here ####
##############################

# NOTE: Fixtures can not be used to parameterize tests
# https://stackoverflow.com/questions/52764279/pytest-how-to-parametrize-a-test-with-a-list-that-is-returned-from-a-fixture
# Hence, we have to create functions and create the parameterized list first
# (must happen during collect phase) before passing it to mark.parameterize.

_model_parameters = _return_model_parameters()
_compare_model_args = _return_compare_model_args()

############################
#### Functions End Here ####
############################


##########################
#### Tests Start Here ####
##########################


@pytest.mark.parametrize("name, fh", _model_parameters)
def test_create_predict_finalize_model(name, fh, load_pos_and_neg_data):
    """test create_model, predict_model and finalize_model functionality
    Combined to save run time
    """
    exp = TSForecastingExperiment()
    data = load_pos_and_neg_data  # _check_data_for_prophet(name, load_pos_and_neg_data)

    exp.setup(
        data=data,
        fold=2,
        fh=fh,
        fold_strategy="sliding",
        verbose=False,
    )
    #######################
    ## Test Create Model ##
    #######################
    model = exp.create_model(name)

    #########################
    #### Expected Values ####
    #########################
    # Only forcasted values
    fh_index = fh if isinstance(fh, int) else len(fh)
    # Full forecasting window
    fh_max_window = fh if isinstance(fh, int) else max(fh)

    expected_period_index = load_pos_and_neg_data.iloc[-fh_index:].index
    final_expected_period_index = expected_period_index.shift(fh_max_window)

    ########################
    ## Test Predict Model ##
    ########################
    # Default prediction
    y_pred = exp.predict_model(model)
    assert isinstance(y_pred, pd.Series)
    assert np.all(y_pred.index == expected_period_index)

    # With Prediction Interval (default alpha = 0.05)
    y_pred = exp.predict_model(model, return_pred_int=True)
    assert isinstance(y_pred, pd.DataFrame)
    assert np.all(y_pred.columns == ["y_pred", "lower", "upper"])
    assert np.all(y_pred.index == expected_period_index)

    # With Prediction Interval (alpha = 0.2)
    y_pred2 = exp.predict_model(model, return_pred_int=True, alpha=0.2)
    assert isinstance(y_pred2, pd.DataFrame)
    assert np.all(y_pred2.columns == ["y_pred", "lower", "upper"])
    assert np.all(y_pred2.index == expected_period_index)

    # Increased forecast horizon to 2 years instead of the original 1 year
    y_pred = exp.predict_model(model, fh=np.arange(1, 25))
    assert len(y_pred) == 24

    #########################
    ## Test Finalize Model ##
    #########################

    final_model = exp.finalize_model(model)
    y_pred = exp.predict_model(final_model)

    assert np.all(y_pred.index == final_expected_period_index)


def test_predict_model_warnings(load_pos_and_neg_data):
    """test predict_model warnings cases"""
    exp = TSForecastingExperiment()
    exp.setup(
        data=load_pos_and_neg_data,
        fold=2,
        fh=12,
        fold_strategy="sliding",
        verbose=False,
    )

    model = exp.create_model("naive")

    ######################################
    #### Test before finalizing model ####
    ######################################
    # Default (Correct comparison to test set)
    _ = exp.predict_model(model)
    expected = exp.pull()

    # Prediction horizon larger than test set --> Metrics limited to common indices
    _ = exp.predict_model(model, fh=np.arange(1, 24))
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
    ## Test Create Model with custom folds ##
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
    ## Test Create Model without cv ##
    ##################################
    _ = exp.create_model("naive", cross_validation=False)
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


def test_save_load_model(load_pos_and_neg_data):
    """Tests the save_model and load_model functionality"""

    fh = np.arange(1, 13)
    fold = 2
    data = load_pos_and_neg_data

    ######################
    #### OOP Approach ####
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
    exp.save_model(model, "model_unit_test_oop")

    # Mimic loading in another session
    exp_loaded = TSForecastingExperiment()
    loaded_model = exp_loaded.load_model("model_unit_test_oop")
    loaded_predictions = exp_loaded.predict_model(loaded_model)

    assert np.all(loaded_predictions == expected_predictions)

    ########################
    #### Functional API ####
    ########################
    from pycaret.time_series import (
        setup,
        create_model,
        predict_model,
        save_model,
        load_model,
    )

    _ = setup(
        data=data, fh=fh, fold=fold, fold_strategy="expanding", session_id=42, n_jobs=-1
    )
    model = create_model("naive")
    expected_predictions = predict_model(model)
    save_model(model, "model_unit_test_func")

    # Mimic loading in another session
    loaded_model = load_model("model_unit_test_func")
    loaded_predictions = predict_model(loaded_model)

    assert np.all(loaded_predictions == expected_predictions)
