"""Module to test time_series functionality
"""
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
    _return_data_big_small,
    _ALL_METRICS,
    _ALL_STATS_TESTS,
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
_compare_model_args = _return_compare_model_args()
_data_big_small = _return_data_big_small()

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


@pytest.mark.parametrize("test", _ALL_STATS_TESTS)
@pytest.mark.parametrize("data", _data_big_small)
def test_check_stats(data, test):
    """Tests the check_stats functionality"""

    exp = TimeSeriesExperiment()

    # Reduced fh since we are testing with small dataset as well
    fh = 1
    fold = 2

    exp.setup(
        data=data,
        fh=fh,
        fold=fold,
        fold_strategy="sliding",
        verbose=False,
        session_id=42,
    )

    # Individual Tests
    results = exp.check_stats(test=test)
    expected_order = ["Test", "Test Name", "Property", "Setting", "Value"]
    column_names = list(results.columns)
    for i, name in enumerate(expected_order):
        assert column_names[i] == name


def test_check_stats_combined(load_pos_and_neg_data):
    """Tests the check_stats functionality combined test"""

    exp = TimeSeriesExperiment()

    fh = 12
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

    expected_order = ["Test", "Test Name", "Property", "Setting", "Value"]

    results = exp.check_stats()
    column_names = list(results.columns)
    for i, name in enumerate(expected_order):
        assert column_names[i] == name


def test_check_stats_alpha(load_pos_and_neg_data):
    """Tests the check_stats functionality with different alpha"""

    exp = TimeSeriesExperiment()

    fh = 12
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

    alpha = 0.2
    results = exp.check_stats(alpha=alpha)
    assert (
        results.query("Test == 'White Noise'").iloc[0]["Setting"].get("alpha") == alpha
    )
    assert (
        results.query("Test == 'Stationarity'").iloc[0]["Setting"].get("alpha") == alpha
    )
    assert results.query("Test == 'Normality'").iloc[0]["Setting"].get("alpha") == alpha


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


def test_mlflow_logging(load_pos_and_neg_data):
    """Tests the logging of MLFlow experiment"""
    data = load_pos_and_neg_data

    exp = TimeSeriesExperiment()
    exp.setup(
        data=data,
        fh=12,
        session_id=42,
        log_experiment=True,
        experiment_name="ts_unit_test",
        log_plots=True,
    )

    model = exp.create_model("naive")
    _ = exp.tune_model(model)
    _ = exp.compare_models(include=["naive", "ets"])

    mlflow_logs = exp.get_logs()

    # When running locally, there can be multiple experiments with the same name
    # Just get he last one so that the asserts work (otherwise, the count of the
    # various function calls will not match)
    last_start = mlflow_logs["start_time"].max()
    last_experiment_usi = mlflow_logs.query("start_time == @last_start")[
        "tags.USI"
    ].unique()[0]

    num_create_models = len(
        mlflow_logs.query(
            "`tags.USI` == @last_experiment_usi & `tags.Source` == 'create_model'"
        )
    )
    num_tune_models = len(
        mlflow_logs.query(
            "`tags.USI` == @last_experiment_usi &`tags.Source` == 'tune_model'"
        )
    )
    num_compare_models = len(
        mlflow_logs.query(
            "`tags.USI` == @last_experiment_usi &`tags.Source` == 'compare_models'"
        )
    )

    assert num_create_models == 1
    assert num_tune_models == 1
    assert num_compare_models == 2


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
        log_plots=log_experiment,
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

    # By default choose_better = True
    tuned_model1 = exp.tune_model(model, custom_grid=custom_grid)

    # Choose Better = False
    tuned_model2 = exp.tune_model(model, custom_grid=custom_grid, choose_better=False)

    # Same strategy should be chosen since choose_better = True by default
    assert tuned_model1.strategy == model.strategy
    # should pick only value in custom grid
    assert tuned_model2.strategy == only_strategy
    # tuned model does improve score (verified manually), and choose_better
    # set to False. So pick worse value itself.
    assert tuned_model2.strategy != model.strategy


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
