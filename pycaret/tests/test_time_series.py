"""Module to test time_series functionality
"""
import pytest

from random import choice, uniform, randint

from pycaret.internal.ensemble import _ENSEMBLE_METHODS
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from pycaret.datasets import get_data
from pycaret.internal.pycaret_experiment import TimeSeriesExperiment
from pycaret.containers.models.time_series import get_all_model_containers

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

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

#############################
#### Fixtures Start Here ####
#############################


@pytest.fixture(scope="session", name="load_data")
def load_data():
    """Load Pycaret Airline dataset."""
    data = get_data("airline")
    data = data - 400  # simulate negative values
    return data


@pytest.fixture(scope="session", name="load_setup")
def load_setup(load_data):
    """Create a TimeSeriesExperiment to test module functionalities"""
    exp = TimeSeriesExperiment()

    fh = np.arange(1, 13)
    fold = 2

    return exp.setup(
        data=load_data,
        fh=fh,
        fold=fold,
        fold_strategy="sliding",
        verbose=False,
        session_id=42,
    )


@pytest.fixture(scope="session", name="load_models")
def load_ts_models(load_setup):
    """Load all time series module models"""
    globals_dict = {
        "seed": 0,
        "n_jobs_param": -1,
        "gpu_param": False,
        "X_train": pd.DataFrame(get_data("airline")),
    }
    ts_models = get_all_model_containers(globals_dict)
    ts_experiment = load_setup
    ts_estimators = [
        ts_experiment.create_model(key)
        for key in ts_models.keys()
        if key in _BLEND_TEST_MODELS
    ]

    return ts_estimators


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


# def _check_data_for_prophet(mdl_name, data):
#     """Convert data index to DatetimeIndex"""
#     if mdl_name == "prophet":
#         data = data.to_timestamp(freq="M")
#     return data


_model_names = _return_model_names()
_model_parameters = _return_model_parameters()

############################
#### Functions End Here ####
############################


##########################
#### Tests Start Here ####
##########################


def test_check_stats(load_data):
    """Tests the check_stats functionality"""

    exp = TimeSeriesExperiment()

    fh = np.arange(1, 13)
    fold = 2
    data = load_data

    exp.setup(
        data=data,
        fh=fh,
        fold=fold,
        fold_strategy="sliding",
        verbose=False,
        session_id=42,
    )

    expected = ["Test", "Test Name", "Property", "Setting"]
    expected_small = ["Test", "Test Name", "Property"]

    results = exp.check_stats()
    index_names = list(results.index.names)
    for i, name in enumerate(expected):
        assert index_names[i] == name

    # Individual Tests
    tests = ["white_noise", "stationarity", "adf", "kpss", "normality"]
    for test in tests:
        results = exp.check_stats(test=test)
        index_names = list(results.index.names)
        for i, name in enumerate(expected):
            assert index_names[i] == name

    results = exp.check_stats(test="stat_summary")
    index_names = list(results.index.names)
    for i, name in enumerate(expected_small):
        assert index_names[i] == name

    alpha = 0.2
    results = exp.check_stats(alpha=alpha)
    assert alpha in results.index.get_level_values("Setting")


def test_plot_model(load_data):
    """Tests the plot_model functionality"""
    exp = TimeSeriesExperiment()

    fh = np.arange(1, 13)
    fold = 2
    data = load_data

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
    )

    model = exp.create_model("naive")

    print("\n\n==== ON DATA (using OOP) ====")
    exp.plot_model(system=False)
    exp.plot_model(plot="ts", system=False)
    exp.plot_model(plot="splits-tt", system=False)
    exp.plot_model(plot="splits_cv", system=False)
    exp.plot_model(plot="acf", system=False)
    exp.plot_model(plot="pacf", system=False)

    print("\n\n==== ON ESTIMATOR (using OOP) ====")
    exp.plot_model(estimator=model, system=False)
    exp.plot_model(estimator=model, plot="ts", system=False)
    exp.plot_model(estimator=model, plot="splits-tt", system=False)
    exp.plot_model(estimator=model, plot="splits_cv", system=False)
    exp.plot_model(estimator=model, plot="predictions", system=False)

    ## Not Implemented on Residuals yet
    # exp.plot_model(estimator=model, plot="acf")
    # exp.plot_model(estimator=model, plot="pacf")
    # exp.plot_model(estimator=model, plot="residuals")

    ########################
    #### Functional API ####
    ########################
    from pycaret.time_series import setup, create_model, plot_model

    _ = setup(
        data=data, fh=fh, fold=fold, fold_strategy="expanding", session_id=42, n_jobs=-1
    )
    model = create_model("naive")

    print("\n\n==== ON DATA (using Functional API) ====")
    plot_model(system=False)
    plot_model(plot="ts", system=False)
    plot_model(plot="splits-tt", system=False)
    plot_model(plot="splits_cv", system=False)
    plot_model(plot="acf", system=False)
    plot_model(plot="pacf", system=False)

    print("\n\n==== ON ESTIMATOR (using Functional API) ====")
    plot_model(estimator=model, system=False)
    plot_model(estimator=model, plot="ts", system=False)
    plot_model(estimator=model, plot="splits-tt", system=False)
    plot_model(estimator=model, plot="splits_cv", system=False)
    plot_model(estimator=model, plot="predictions", system=False)

    ## Not Implemented on Residuals yet
    # plot_model(estimator=model, plot="acf")
    # plot_model(estimator=model, plot="pacf")
    # plot_model(estimator=model, plot="residuals")


@pytest.mark.parametrize("seasonal_period, seasonal_value", _get_seasonal_values())
def test_setup_seasonal_period_str(load_data, seasonal_period, seasonal_value):

    exp = TimeSeriesExperiment()

    fh = np.arange(1, 13)
    fold = 2
    data = load_data

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
def test_setup_seasonal_period_int(load_data, seasonal_key, seasonal_value):

    exp = TimeSeriesExperiment()

    fh = np.arange(1, 13)
    fold = 2
    data = load_data

    exp.setup(
        data=data,
        fh=fh,
        fold=fold,
        fold_strategy="sliding",
        verbose=False,
        seasonal_period=seasonal_value,
    )

    assert exp.seasonal_period == seasonal_value


@pytest.mark.parametrize("name, fh", _model_parameters)
def test_create_predict_finalize_model(name, fh, load_data):
    """test create_model, predict_model and finalize_model functionality
    Combined to save run time
    """
    exp = TimeSeriesExperiment()
    data = load_data  # _check_data_for_prophet(name, load_data)

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

    ########################
    ## Test Predict Model ##
    ########################
    fh_index = fh if isinstance(fh, int) else fh[-1]
    expected_period_index = load_data.iloc[-fh_index:].index

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


def test_predict_model_warnings(load_data):
    """test predict_model warnings cases"""
    exp = TimeSeriesExperiment()
    exp.setup(
        data=load_data,
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

    #####################################
    #### Test after finalizing model ####
    #####################################
    final_model = exp.finalize_model(model)

    # Expect to get all NaN values in metrics since no indices match
    model_col = expected["Model"]
    expected = pd.DataFrame(np.nan, index=expected.index, columns=expected.columns)
    expected["Model"] = model_col  # Replace Model column with correct value

    # Expect to get all NaN values in metrics since no indices match
    _ = exp.predict_model(final_model)
    metrics = exp.pull()
    assert metrics.equals(expected)

    # Expect to get all NaN values in metrics since no indices match
    _ = exp.predict_model(final_model, fh=np.arange(1, 24))
    metrics = exp.pull()
    assert metrics.equals(expected)


def test_create_model_custom_folds(load_data):
    """test custom fold in create_model"""
    exp = TimeSeriesExperiment()
    setup_fold = 3
    exp.setup(
        data=load_data,
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


def test_prediction_interval_na(load_data):
    """Tests predict model when interval is NA"""

    exp = TimeSeriesExperiment()

    fh = 12
    fold = 2
    data = load_data

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


def test_compare_models(load_data):
    """tests compare_models functionality"""
    exp = TimeSeriesExperiment()

    fh = 12
    fold = 2
    data = load_data

    exp.setup(
        data=data,
        fh=fh,
        fold=fold,
        fold_strategy="expanding",
        verbose=False,
        session_id=42,
    )

    best_baseline_models = exp.compare_models(n_select=3)
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


def test_blend_model_custom_folds(load_data):
    """test custom folds in blend_model"""
    exp = TimeSeriesExperiment()
    setup_fold = 3
    exp.setup(
        data=load_data,
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


@pytest.mark.parametrize("model", _model_names)
def test_tune_model_random(model, load_data):
    exp = TimeSeriesExperiment()
    fh = 12
    fold = 2
    data = load_data

    exp.setup(data=data, fold=fold, fh=fh, fold_strategy="sliding")

    model_obj = exp.create_model(model)
    tuned_model_obj = exp.tune_model(model_obj)  # default search_algorithm = "random"
    y_pred = exp.predict_model(tuned_model_obj)
    assert isinstance(y_pred, pd.Series)

    expected_period_index = data.iloc[-fh:].index
    assert np.all(y_pred.index == expected_period_index)


def test_tune_custom_grid_and_choose_better(load_data):
    """Tests
    (1) passing a custom grid to tune_model, and
    (2) choose_better=True
    """

    exp = TimeSeriesExperiment()

    fh = np.arange(1, 13)
    fold = 2
    data = load_data

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


def test_tune_model_custom_folds(load_data):
    """test custom folds in tune_model"""
    exp = TimeSeriesExperiment()
    setup_fold = 3
    exp.setup(
        data=load_data,
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


def test_tune_model_alternate_metric(load_data):
    """tests model selection using non default metric"""
    exp = TimeSeriesExperiment()
    fh = 12
    fold = 2

    exp.setup(data=load_data, fold=fold, fh=fh, fold_strategy="sliding")

    model_obj = exp.create_model("naive")
    tuned_model_obj = exp.tune_model(model_obj, optimize="MAE")
    y_pred = exp.predict_model(tuned_model_obj)
    assert isinstance(y_pred, pd.Series)

    expected_period_index = load_data.iloc[-fh:].index
    assert np.all(y_pred.index == expected_period_index)


def test_tune_model_raises(load_data):
    """Tests conditions that raise an error due to lack of data"""

    exp = TimeSeriesExperiment()

    fh = np.arange(1, 13)
    fold = 2
    data = load_data

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
