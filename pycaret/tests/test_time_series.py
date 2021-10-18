"""Module to test time_series functionality
"""
import os
from random import uniform
import pytest

import numpy as np  # type: ignore
import pandas as pd  # type: ignore


from pycaret.internal.pycaret_experiment import TimeSeriesExperiment
from pycaret.internal.ensemble import _ENSEMBLE_METHODS


from .time_series_test_utils import (
    _get_seasonal_values,
    _return_model_parameters,
    _return_splitter_args,
    _return_compare_model_args,
    _return_setup_args_raises,
    _return_data_with_without_period_index,
    _ALL_METRICS,
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
_splitter_args = _return_splitter_args()
_setup_args_raises = _return_setup_args_raises()
_data_with_without_period_index = _return_data_with_without_period_index()
_compare_model_args = _return_compare_model_args()

############################
#### Functions End Here ####
############################


##########################
#### Tests Start Here ####
##########################


@pytest.mark.parametrize("fold, fh, fold_strategy", _splitter_args)
def test_splitter_using_fold_and_fh(fold, fh, fold_strategy, load_pos_and_neg_data):
    """Tests the splitter creation using fold, fh and a string value for fold_strategy."""

    from pycaret.time_series import setup
    from sktime.forecasting.model_selection._split import (
        ExpandingWindowSplitter,
        SlidingWindowSplitter,
    )

    exp_name = setup(
        data=load_pos_and_neg_data, fold=fold, fh=fh, fold_strategy=fold_strategy,
    )

    allowed_fold_strategies = ["expanding", "rolling", "sliding"]
    if fold_strategy in allowed_fold_strategies:
        if (fold_strategy == "expanding") or (fold_strategy == "rolling"):
            assert isinstance(exp_name.fold_generator, ExpandingWindowSplitter)
        elif fold_strategy == "sliding":
            assert isinstance(exp_name.fold_generator, SlidingWindowSplitter)

        assert np.all(exp_name.fold_generator.fh == np.arange(1, fh + 1))
        assert exp_name.fold_generator.step_length == fh  # Since fh is an int


def test_splitter_pass_cv_object(load_pos_and_neg_data):
    """Tests the passing of a cv splitter to fold_strategy"""

    from pycaret.time_series import setup
    from sktime.forecasting.model_selection._split import (
        ExpandingWindowSplitter,
        SlidingWindowSplitter,
    )

    fold = 3
    fh = np.arange(1, 13)  # regular horizon of 12 months
    fh_extended = np.arange(1, 25)  # extended horizon of 24 months
    fold_strategy = ExpandingWindowSplitter(
        initial_window=72,
        step_length=12,
        # window_length=12,
        fh=fh,
        start_with_window=True,
    )

    exp_name = setup(
        data=load_pos_and_neg_data,
        fold=fold,  # should be ignored since we are passing explicit fold_strategy
        fh=fh_extended,  # should be ignored since we are passing explicit fold_strategy
        fold_strategy=fold_strategy,
    )

    assert exp_name.fold_generator.initial_window == fold_strategy.initial_window
    assert np.all(exp_name.fold_generator.fh == fold_strategy.fh)
    assert exp_name.fold_generator.step_length == fold_strategy.step_length
    num_folds = exp_name.get_config("fold_param")
    y_train = exp_name.get_config("y_train")

    expected = fold_strategy.get_n_splits(y=y_train)
    assert num_folds == expected


@pytest.mark.parametrize("fold, fh, fold_strategy", _setup_args_raises)
def test_setup_raises(fold, fh, fold_strategy, load_pos_and_neg_data):
    """Tests conditions that raise an error due to lack of data"""

    from pycaret.time_series import setup

    with pytest.raises(ValueError) as errmsg:
        _ = setup(
            data=load_pos_and_neg_data, fold=fold, fh=fh, fold_strategy=fold_strategy,
        )

    exceptionmsg = errmsg.value.args[0]

    assert exceptionmsg == "Not Enough Data Points, set a lower number of folds or fh"


def test_save_load_model(load_pos_and_neg_data):
    """Tests the save_model and load_model functionality"""

    fh = np.arange(1, 13)
    fold = 2
    data = load_pos_and_neg_data

    ######################
    #### OOP Approach ####
    ######################
    exp = TimeSeriesExperiment()
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
    exp_loaded = TimeSeriesExperiment()
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


def test_check_stats(load_pos_and_neg_data):
    """Tests the check_stats functionality"""

    exp = TimeSeriesExperiment()

    fh = np.arange(1, 13)
    fold = 2
    data = load_pos_and_neg_data

    exp.setup(
        data=data,
        fh=fh,
        fold=fold,
        fold_strategy="sliding",
        verbose=False,
        session_id=42,
    )

    expected = ["Test", "Test Name", "Property", "Setting", "Value"]
    # expected_small = ["Test", "Test Name", "Property"]

    results = exp.check_stats()
    column_names = list(results.columns)
    for i, name in enumerate(expected):
        assert column_names[i] == name

    # Individual Tests
    tests = ["summary", "white_noise", "stationarity", "adf", "kpss", "normality"]
    for test in tests:
        results = exp.check_stats(test=test)
        column_names = list(results.columns)
        for i, name in enumerate(expected):
            assert column_names[i] == name

    alpha = 0.2
    results = exp.check_stats(alpha=alpha)
    assert (
        results.query("Test == 'White Noise'").iloc[0]["Setting"].get("alpha") == alpha
    )
    assert (
        results.query("Test == 'Stationarity'").iloc[0]["Setting"].get("alpha") == alpha
    )
    assert results.query("Test == 'Normality'").iloc[0]["Setting"].get("alpha") == alpha


@pytest.mark.parametrize("data", _data_with_without_period_index)
def test_plot_model(data):
    """Tests the plot_model functionality
    NOTE: Want to show multiplicative plot here so can not take data with negative values
    """
    exp = TimeSeriesExperiment()

    fh = np.arange(1, 13)
    fold = 2

    sp = 1 if isinstance(data.index, pd.RangeIndex) else None

    ######################
    #### OOP Approach ####
    ######################

    exp.setup(
        data=data,
        fh=fh,
        fold=fold,
        fold_strategy="sliding",
        verbose=False,
        session_id=42,
        seasonal_period=sp,
    )

    model = exp.create_model("naive")

    print("\n\n==== ON DATA (using OOP) ====")
    exp.plot_model(system=False)
    exp.plot_model(plot="ts", system=False)
    exp.plot_model(plot="train_test_split", system=False)
    exp.plot_model(plot="cv", system=False)
    exp.plot_model(plot="acf", system=False)
    exp.plot_model(plot="pacf", system=False)
    exp.plot_model(plot="diagnostics", system=False)
    exp.plot_model(plot="decomp_classical", system=False)
    exp.plot_model(plot="decomp_stl", system=False)

    print("\n\n==== ON ESTIMATOR (using OOP) ====")
    exp.plot_model(estimator=model, system=False)
    exp.plot_model(estimator=model, plot="ts", system=False)
    exp.plot_model(estimator=model, plot="train_test_split", system=False)
    exp.plot_model(estimator=model, plot="cv", system=False)
    exp.plot_model(estimator=model, plot="acf", system=False)
    exp.plot_model(estimator=model, plot="pacf", system=False)
    exp.plot_model(estimator=model, plot="diagnostics", system=False)
    exp.plot_model(estimator=model, plot="decomp_classical", system=False)
    exp.plot_model(estimator=model, plot="decomp_stl", system=False)
    exp.plot_model(estimator=model, plot="forecast", system=False)
    exp.plot_model(estimator=model, plot="residuals", system=False)

    ########################
    #### Functional API ####
    ########################
    from pycaret.time_series import setup, create_model, plot_model

    _ = setup(
        data=data,
        fh=fh,
        fold=fold,
        fold_strategy="expanding",
        session_id=42,
        n_jobs=-1,
        seasonal_period=sp,
    )
    model = create_model("naive")

    os.environ["PYCARET_TESTING"] = "1"

    print("\n\n==== ON DATA (using Functional API) ====")
    plot_model()
    plot_model(plot="ts")
    plot_model(plot="train_test_split")
    plot_model(plot="cv")
    plot_model(plot="acf")
    plot_model(plot="pacf")
    plot_model(plot="diagnostics")
    plot_model(plot="decomp_classical")
    plot_model(plot="decomp_stl")

    print("\n\n==== ON ESTIMATOR (using Functional API) ====")
    plot_model(estimator=model)
    plot_model(estimator=model, plot="ts")
    plot_model(estimator=model, plot="train_test_split")
    plot_model(estimator=model, plot="cv")
    plot_model(estimator=model, plot="acf")
    plot_model(estimator=model, plot="pacf")
    plot_model(estimator=model, plot="diagnostics")
    plot_model(estimator=model, plot="decomp_classical")
    plot_model(estimator=model, plot="decomp_stl")
    plot_model(estimator=model, plot="forecast")
    plot_model(estimator=model, plot="residuals")

    #######################
    #### Customization ####
    #######################

    print("\n\n==== Testing Customization ON DATA ====")
    exp.plot_model(
        plot="pacf",
        data_kwargs={"nlags": 36,},
        fig_kwargs={"fig_size": [800, 500], "fig_template": "simple_white"},
        system=False,
    )
    exp.plot_model(
        plot="decomp_classical", data_kwargs={"type": "multiplicative"}, system=False
    )

    print("\n\n====  Testing Customization ON ESTIMATOR ====")
    exp.plot_model(
        estimator=model, plot="forecast", data_kwargs={"fh": 24}, system=False
    )


@pytest.mark.parametrize("seasonal_period, seasonal_value", _get_seasonal_values())
def test_setup_seasonal_period_str(
    load_pos_and_neg_data, seasonal_period, seasonal_value
):

    exp = TimeSeriesExperiment()

    fh = np.arange(1, 13)
    fold = 2
    data = load_pos_and_neg_data

    exp.setup(
        data=data,
        fh=fh,
        fold=fold,
        fold_strategy="sliding",
        verbose=False,
        session_id=42,
        seasonal_period=seasonal_period,
    )

    assert exp.seasonal_period == seasonal_value


@pytest.mark.parametrize("seasonal_key, seasonal_value", _get_seasonal_values())
def test_setup_seasonal_period_int(load_pos_and_neg_data, seasonal_key, seasonal_value):

    exp = TimeSeriesExperiment()

    fh = np.arange(1, 13)
    fold = 2
    data = load_pos_and_neg_data

    exp.setup(
        data=data,
        fh=fh,
        fold=fold,
        fold_strategy="sliding",
        verbose=False,
        seasonal_period=seasonal_value,
    )

    assert exp.seasonal_period == seasonal_value


def test_enforce_pi(load_pos_and_neg_data):
    """Tests the enforcement of prediction interval"""
    data = load_pos_and_neg_data

    exp1 = TimeSeriesExperiment()
    exp1.setup(data=data, enforce_pi=True)
    num_models1 = len(exp1.models())

    exp2 = TimeSeriesExperiment()
    exp2.setup(data=data, enforce_pi=False)
    num_models2 = len(exp2.models())

    # We know that some models do not offer PI capability to the following
    # check is valid for now.
    assert num_models1 < num_models2


@pytest.mark.parametrize("name, fh", _model_parameters)
def test_create_predict_finalize_model(name, fh, load_pos_and_neg_data):
    """test create_model, predict_model and finalize_model functionality
    Combined to save run time
    """
    exp = TimeSeriesExperiment()
    data = load_pos_and_neg_data  # _check_data_for_prophet(name, load_pos_and_neg_data)

    exp.setup(
        data=data, fold=2, fh=fh, fold_strategy="sliding", verbose=False,
    )
    #######################
    ## Test Create Model ##
    #######################
    model = exp.create_model(name)

    ########################
    ## Test Predict Model ##
    ########################
    fh_index = fh if isinstance(fh, int) else fh[-1]
    expected_period_index = load_pos_and_neg_data.iloc[-fh_index:].index

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

    final_expected_period_index = expected_period_index.shift(fh_index)
    assert np.all(y_pred.index == final_expected_period_index)


def test_predict_model_warnings(load_pos_and_neg_data):
    """test predict_model warnings cases"""
    exp = TimeSeriesExperiment()
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
    exp = TimeSeriesExperiment()
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
    exp = TimeSeriesExperiment()
    exp.setup(
        data=load_pos_and_neg_data, fh=12, fold_strategy="sliding", verbose=False,
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

    exp = TimeSeriesExperiment()

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
    exp = TimeSeriesExperiment()

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
    )

    best_baseline_models = exp.compare_models(
        n_select=3, cross_validation=cross_validation
    )
    assert len(best_baseline_models) == 3


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

    assert mean_median_equal == False
    assert mean_voting_equal == False
    assert median_voting_equal == False


def test_blend_model_custom_folds(load_pos_and_neg_data):
    """test custom folds in blend_model"""
    exp = TimeSeriesExperiment()
    setup_fold = 3
    exp.setup(
        data=load_pos_and_neg_data,
        fold=setup_fold,
        fh=12,
        fold_strategy="sliding",
        verbose=False,
    )

    #######################################
    ## Test Tune Model with custom folds ##
    #######################################
    model = exp.create_model("naive")
    _ = exp.blend_models([model, model, model])
    metrics1 = exp.pull()

    custom_fold = 5
    _ = exp.blend_models([model, model, model], fold=custom_fold)
    metrics2 = exp.pull()

    assert len(metrics1) == setup_fold + 2  # + 2 for Mean and SD
    assert len(metrics2) == custom_fold + 2  # + 2 for Mean and SD


def test_tune_custom_grid_and_choose_better(load_pos_and_neg_data):
    """Tests
    (1) passing a custom grid to tune_model, and
    (2) choose_better=True
    """

    exp = TimeSeriesExperiment()

    fh = np.arange(1, 13)
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

    model = exp.create_model("naive")

    # Custom Grid
    only_strategy = "mean"
    custom_grid = {"strategy": [only_strategy]}
    tuned_model1 = exp.tune_model(model, custom_grid=custom_grid)

    # Choose Better
    tuned_model2 = exp.tune_model(model, custom_grid=custom_grid, choose_better=True)

    # Different strategy should be picked since grid is limited (verified manually)
    assert tuned_model1.strategy != model.strategy
    # should pick only value in custom grid
    assert tuned_model1.strategy == only_strategy
    # tuned model does improve score (verified manually), so pick original
    assert tuned_model2.strategy == model.strategy


def test_tune_model_custom_folds(load_pos_and_neg_data):
    """test custom folds in tune_model"""
    exp = TimeSeriesExperiment()
    setup_fold = 3
    exp.setup(
        data=load_pos_and_neg_data,
        fold=setup_fold,
        fh=12,
        fold_strategy="sliding",
        verbose=False,
    )

    #######################################
    ## Test Tune Model with custom folds ##
    #######################################
    model = exp.create_model("naive")
    _ = exp.tune_model(model)
    metrics1 = exp.pull()

    custom_fold = 5
    _ = exp.tune_model(model, fold=5)
    metrics2 = exp.pull()

    assert len(metrics1) == setup_fold + 2  # + 2 for Mean and SD
    assert len(metrics2) == custom_fold + 2  # + 2 for Mean and SD


@pytest.mark.parametrize("metric", _ALL_METRICS)
def test_tune_model_alternate_metric(load_pos_and_neg_data, metric):
    """tests model selection using non default metric"""
    exp = TimeSeriesExperiment()
    fh = 12
    fold = 2

    exp.setup(data=load_pos_and_neg_data, fold=fold, fh=fh, fold_strategy="sliding")

    model_obj = exp.create_model("naive")

    tuned_model_obj = exp.tune_model(model_obj, optimize=metric)
    y_pred = exp.predict_model(tuned_model_obj)
    assert isinstance(y_pred, pd.Series)

    expected_period_index = load_pos_and_neg_data.iloc[-fh:].index
    assert np.all(y_pred.index == expected_period_index)


def test_tune_model_raises(load_pos_and_neg_data):
    """Tests conditions that raise an error due to lack of data"""

    exp = TimeSeriesExperiment()

    fh = np.arange(1, 13)
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

    model = exp.create_model("naive")
    with pytest.raises(ValueError) as errmsg:
        search_algorithm = "wrong_algorithm"
        _ = exp.tune_model(model, search_algorithm=search_algorithm)

    exceptionmsg = errmsg.value.args[0]

    assert (
        exceptionmsg
        == f"`search_algorithm` must be one of 'None, random, grid'. You passed '{search_algorithm}'."
    )
