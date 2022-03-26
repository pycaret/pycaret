"""Module to test time_series "setup" functionality
"""
import math
import pytest
import numpy as np  # type: ignore

from pycaret.datasets import get_data
from pycaret.time_series import TSForecastingExperiment

from .time_series_test_utils import (
    _return_splitter_args,
    _return_setup_args_raises,
    _get_seasonal_values,
    _get_seasonal_values_alphanumeric,
)


##############################
#### Functions Start Here ####
##############################

# NOTE: Fixtures can not be used to parameterize tests
# https://stackoverflow.com/questions/52764279/pytest-how-to-parametrize-a-test-with-a-list-that-is-returned-from-a-fixture
# Hence, we have to create functions and create the parameterized list first
# (must happen during collect phase) before passing it to mark.parameterize.

_splitter_args = _return_splitter_args()
_setup_args_raises = _return_setup_args_raises()


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
        data=load_pos_and_neg_data,
        fold=fold,
        fh=fh,
        fold_strategy=fold_strategy,
    )

    allowed_fold_strategies = ["expanding", "rolling", "sliding"]
    if fold_strategy in allowed_fold_strategies:
        if (fold_strategy == "expanding") or (fold_strategy == "rolling"):
            assert isinstance(exp_name.fold_generator, ExpandingWindowSplitter)
        elif fold_strategy == "sliding":
            assert isinstance(exp_name.fold_generator, SlidingWindowSplitter)

        if isinstance(fh, int):
            # Since fh is an int, we can check as follows ----
            assert np.all(exp_name.fold_generator.fh == np.arange(1, fh + 1))
            assert exp_name.fold_generator.step_length == fh
        else:
            assert np.all(exp_name.fold_generator.fh == fh)

            # When fh has np gaps: e.g. fh = np.arange(1, 37), step length = 36
            # When fh has gaps, e.g. fh = np.arange(25, 37), step length = 12
            assert exp_name.fold_generator.step_length == len(fh)


def test_splitter_pass_cv_object(load_pos_and_neg_data):
    """Tests the passing of a `sktime` cv splitter to fold_strategy"""

    from pycaret.time_series import setup
    from sktime.forecasting.model_selection._split import ExpandingWindowSplitter

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
            data=load_pos_and_neg_data,
            fold=fold,
            fh=fh,
            fold_strategy=fold_strategy,
        )

    exceptionmsg = errmsg.value.args[0]

    assert exceptionmsg == "Not Enough Data Points, set a lower number of folds or fh"


def test_enforce_pi(load_pos_and_neg_data):
    """Tests the enforcement of prediction interval"""
    data = load_pos_and_neg_data

    exp1 = TSForecastingExperiment()
    exp1.setup(data=data, enforce_pi=True)
    num_models1 = len(exp1.models())

    exp2 = TSForecastingExperiment()
    exp2.setup(data=data, enforce_pi=False)
    num_models2 = len(exp2.models())

    # We know that some models do not offer PI capability, so the following
    # check is valid for now.
    assert num_models1 < num_models2


def test_enforce_exogenous_no_exo_data(load_pos_and_neg_data):
    """Tests the enforcement of exogenous variable support in models when
    univariate data without exogenous variables is passed."""
    data = load_pos_and_neg_data

    exp1 = TSForecastingExperiment()
    exp1.setup(data=data, enforce_exogenous=True)
    num_models1 = len(exp1.models())

    exp2 = TSForecastingExperiment()
    exp2.setup(data=data, enforce_exogenous=False)
    num_models2 = len(exp2.models())

    # Irrespective of the enforce_exogenous flag, all models are enabled when
    # the data does not contain exogenous variables.
    assert num_models1 == num_models2


def test_enforce_exogenous_exo_data(load_uni_exo_data_target):
    """Tests the enforcement of exogenous variable support in models when
    univariate data with exogenous variables is passed."""
    data, target = load_uni_exo_data_target

    exp1 = TSForecastingExperiment()
    exp1.setup(data=data, target=target, seasonal_period=4, enforce_exogenous=True)
    num_models1 = len(exp1.models())

    exp2 = TSForecastingExperiment()
    exp2.setup(data=data, target=target, seasonal_period=4, enforce_exogenous=False)
    num_models2 = len(exp2.models())

    # We know that some models do not offer exogenous variables support, so the
    # following check is valid for now.
    assert num_models1 < num_models2


def test_seasonal_period_to_use():

    exp = TSForecastingExperiment()
    fh = 12

    # Airline Data with seasonality of 12
    data = get_data("airline", verbose=False)
    exp.setup(
        data=data,
        fh=fh,
        verbose=False,
        session_id=42,
    )
    assert exp.seasonal_period == 12
    assert exp.all_sp_values == [12]
    assert exp.primary_sp_to_use == 12

    # Airline Data with seasonality of M (12), 6
    data = get_data("airline", verbose=False)
    exp.setup(data=data, fh=fh, verbose=False, session_id=42, seasonal_period=["M", 6])
    assert exp.seasonal_period == [12, 6]
    assert exp.all_sp_values == [12, 6]
    assert exp.primary_sp_to_use == 12

    # White noise Data with seasonality of 12
    data = get_data("1", folder="time_series/white_noise", verbose=False)
    exp.setup(
        data=data,
        fh=fh,
        seasonal_period=12,
        verbose=False,
        session_id=42,
    )

    # Should get 1 even though we passed 12
    assert exp.seasonal_period == 12
    assert exp.all_sp_values == [1]
    assert exp.primary_sp_to_use == 1


@pytest.mark.parametrize("seasonal_key, seasonal_value", _get_seasonal_values())
def test_setup_seasonal_period_int(load_pos_and_neg_data, seasonal_key, seasonal_value):

    exp = TSForecastingExperiment()

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


@pytest.mark.parametrize("seasonal_period, seasonal_value", _get_seasonal_values())
def test_setup_seasonal_period_str(
    load_pos_and_neg_data, seasonal_period, seasonal_value
):
    exp = TSForecastingExperiment()

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


@pytest.mark.parametrize(
    "prefix, seasonal_period, seasonal_value", _get_seasonal_values_alphanumeric()
)
def test_setup_seasonal_period_alphanumeric(
    load_pos_and_neg_data, prefix, seasonal_period, seasonal_value
):
    """Tests the get_sp_from_str function with different values of frequency"""

    seasonal_period = prefix + seasonal_period
    prefix = int(prefix)
    lcm = abs(seasonal_value * prefix) // math.gcd(seasonal_value, prefix)
    expected_seasonal_value = int(lcm / prefix)

    exp = TSForecastingExperiment()

    fh = np.arange(1, 13)
    fold = 2
    data = load_pos_and_neg_data

    exp.setup(
        data=data,
        fh=fh,
        fold=fold,
        fold_strategy="sliding",
        verbose=False,
        seasonal_period=seasonal_period,
    )

    assert exp.seasonal_period == expected_seasonal_value


def test_train_test_split(load_pos_and_neg_data):
    """Tests the enforcement of prediction interval"""
    data = load_pos_and_neg_data

    ####################################
    #### Continuous fh without Gaps ####
    ####################################

    #### Integer fh ----
    exp = TSForecastingExperiment()
    fh = 12
    exp.setup(data=data, fh=fh, session_id=42)
    y_test = exp.get_config("y_test")
    assert len(y_test) == fh

    #### Numpy fh ----
    exp = TSForecastingExperiment()
    fh = np.arange(1, 10)  # 9 values
    exp.setup(data=data, fh=fh, session_id=42)
    y_test = exp.get_config("y_test")
    assert len(y_test) == len(fh)

    #### List fh ----
    exp = TSForecastingExperiment()
    fh = [1, 2, 3, 4, 5, 6]
    exp.setup(data=data, fh=fh, session_id=42)
    y_test = exp.get_config("y_test")
    assert len(y_test) == len(fh)

    #################################
    #### Continuous fh with Gaps ####
    #################################

    #### Numpy fh ----
    exp = TSForecastingExperiment()
    fh = np.arange(7, 13)  # 6 values
    exp.setup(data=data, fh=fh, session_id=42)
    y_test = exp.get_config("y_test")
    assert len(y_test) == len(fh)

    #### List fh ----
    exp = TSForecastingExperiment()
    fh = [4, 5, 6]
    exp.setup(data=data, fh=fh, session_id=42)
    y_test = exp.get_config("y_test")
    assert len(y_test) == len(fh)

    ####################################
    #### Discontinuous fh with Gaps ####
    ####################################

    #### Numpy fh ----
    exp = TSForecastingExperiment()
    fh = np.array([4, 5, 6, 10, 11, 12])  # 6 values
    exp.setup(data=data, fh=fh, session_id=42)
    y_test = exp.get_config("y_test")
    assert len(y_test) == len(fh)

    #### List fh ----
    exp = TSForecastingExperiment()
    fh = [4, 5, 6, 10, 11, 12]
    exp.setup(data=data, fh=fh, session_id=42)
    y_test = exp.get_config("y_test")
    assert len(y_test) == len(fh)
