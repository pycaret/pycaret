"""Module to test time_series "setup" functionality
"""
import math

import numpy as np
import pandas as pd
import pytest
from time_series_test_utils import (
    _get_seasonal_values,
    _get_seasonal_values_alphanumeric,
    _return_data_seasonal_types_strictly_pos,
    _return_setup_args_raises,
    _return_splitter_args,
)

from pycaret.datasets import get_data
from pycaret.time_series import TSForecastingExperiment

##############################
# Functions Start Here ####
##############################

# NOTE: Fixtures can not be used to parameterize tests
# https://stackoverflow.com/questions/52764279/pytest-how-to-parametrize-a-test-with-a-list-that-is-returned-from-a-fixture
# Hence, we have to create functions and create the parameterized list first
# (must happen during collect phase) before passing it to mark.parameterize.

_splitter_args = _return_splitter_args()
_setup_args_raises = _return_setup_args_raises()
_data_seasonal_types_strictly_pos = _return_data_seasonal_types_strictly_pos()


############################
# Functions End Here ####
############################


##########################
# Tests Start Here ####
##########################


@pytest.mark.parametrize("fold, fh, fold_strategy", _splitter_args)
def test_splitter_using_fold_and_fh(fold, fh, fold_strategy, load_pos_and_neg_data):
    """Tests the splitter creation using fold, fh and a string value for fold_strategy."""

    from sktime.forecasting.model_selection._split import (
        ExpandingWindowSplitter,
        SlidingWindowSplitter,
    )

    from pycaret.time_series import setup

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

    from sktime.forecasting.model_selection._split import ExpandingWindowSplitter

    from pycaret.time_series import setup

    fold = 3
    fh = np.arange(1, 13)  # regular horizon of 12 months
    fh_extended = np.arange(1, 25)  # extended horizon of 24 months
    fold_strategy = ExpandingWindowSplitter(
        initial_window=72,
        step_length=12,
        # window_length=12,
        fh=fh,
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

    # With enforcement ----
    exp1 = TSForecastingExperiment()
    exp1.setup(data=data, point_alpha=0.5)
    num_models1 = len(exp1.models())

    # Without enforcement ----
    exp2 = TSForecastingExperiment()
    exp2.setup(data=data, point_alpha=None)
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
    exp1.setup(data=data, target=target, enforce_exogenous=True)
    num_models1 = len(exp1.models())

    exp2 = TSForecastingExperiment()
    exp2.setup(data=data, target=target, enforce_exogenous=False)
    num_models2 = len(exp2.models())

    # We know that some models do not offer exogenous variables support, so the
    # following check is valid for now.
    assert num_models1 < num_models2


def test_sp_to_use_using_index_and_user_def():
    """Seasonal Period detection using Indices (used before 3.0.0rc5). Also
    tests the user defined seasonal periods when used in conjunction with "index".
    """

    exp = TSForecastingExperiment()
    data = get_data("airline", verbose=False)

    # 1.1 Airline Data with seasonality of 12
    exp.setup(
        data=data,
        sp_detection="index",
        verbose=False,
        session_id=42,
    )
    assert exp.seasonal_period is None
    assert exp.sp_detection == "index"
    assert exp.ignore_seasonality_test is False
    assert exp.candidate_sps == [12]
    assert exp.significant_sps == [12]
    assert exp.significant_sps_no_harmonics == [12]
    assert exp.all_sps_to_use == [12]
    assert exp.primary_sp_to_use == 12

    # 1.2 Airline Data with seasonality of M (12), 6
    exp.setup(
        data=data,
        sp_detection="index",
        verbose=False,
        session_id=42,
        seasonal_period=["M", 6],
        num_sps_to_use=-1,
    )
    assert exp.seasonal_period == ["M", 6]
    # overridden to user_defined even through we pass "index"
    assert exp.sp_detection == "user_defined"
    assert exp.ignore_seasonality_test is False
    assert exp.candidate_sps == [12, 6]
    assert exp.significant_sps == [12, 6]
    assert exp.significant_sps_no_harmonics == [12]
    assert exp.all_sps_to_use == [12, 6]
    assert exp.primary_sp_to_use == 12

    # 1.3 White noise Data with seasonality of 12
    data = get_data("1", folder="time_series/white_noise", verbose=False)
    exp.setup(
        data=data,
        sp_detection="index",
        seasonal_period=12,
        verbose=False,
        session_id=42,
    )

    # Should get 1 even though we passed 12
    assert exp.seasonal_period == 12
    # overridden to user_defined even through we pass "index"
    assert exp.sp_detection == "user_defined"
    assert exp.ignore_seasonality_test is False
    assert exp.candidate_sps == [12]
    assert exp.significant_sps == [1]
    assert exp.significant_sps_no_harmonics == [1]
    assert exp.all_sps_to_use == [1]
    assert exp.primary_sp_to_use == 1

    # 1.4 White noise Data with seasonality of 12 and ignore_seasonality_test = True
    data = get_data("1", folder="time_series/white_noise", verbose=False)
    exp.setup(
        data=data,
        sp_detection="index",
        seasonal_period=12,
        ignore_seasonality_test=True,
        verbose=False,
        session_id=42,
    )

    # Should get 1 even though we passed 12
    assert exp.seasonal_period == 12
    # overridden to user_defined even through we pass "index"
    assert exp.sp_detection == "user_defined"
    assert exp.ignore_seasonality_test is True
    assert exp.candidate_sps == [12]
    assert exp.significant_sps == [12]
    assert exp.significant_sps_no_harmonics == [12]
    assert exp.all_sps_to_use == [12]
    assert exp.primary_sp_to_use == 12


def test_sp_to_use_using_auto_and_user_def():
    """Seasonal Period detection using Statistical tests (used on and after 3.0.0rc5).
    Also tests the user defined seasonal periods when used in conjunction with "auto".
    """

    exp = TSForecastingExperiment()
    data = get_data("airline", verbose=False)

    # 1.1 Auto Detection of Seasonal Period ----
    exp.setup(
        data=data,
        sp_detection="auto",
        verbose=False,
        session_id=42,
    )
    assert exp.candidate_sps == [12, 24, 36, 11, 48]
    assert exp.significant_sps == [12, 24, 36, 11, 48]
    assert exp.significant_sps_no_harmonics == [48, 36, 11]
    assert exp.all_sps_to_use == [12]
    assert exp.primary_sp_to_use == 12

    # 1.2 Auto Detection with multiple values allowed ----
    # 1.2.1 Multiple Seasonalities < tested and detected ----
    exp.setup(
        data=data,
        sp_detection="auto",
        num_sps_to_use=2,
        verbose=False,
        session_id=42,
    )
    assert exp.candidate_sps == [12, 24, 36, 11, 48]
    assert exp.significant_sps == [12, 24, 36, 11, 48]
    assert exp.significant_sps_no_harmonics == [48, 36, 11]
    assert exp.all_sps_to_use == [12, 24]
    assert exp.primary_sp_to_use == 12

    # 1.2.2 Multiple Seasonalities > tested and detected ----
    exp.setup(
        data=data,
        sp_detection="auto",
        num_sps_to_use=100,
        verbose=False,
        session_id=42,
    )
    assert exp.candidate_sps == [12, 24, 36, 11, 48]
    assert exp.significant_sps == [12, 24, 36, 11, 48]
    assert exp.significant_sps_no_harmonics == [48, 36, 11]
    assert exp.all_sps_to_use == [12, 24, 36, 11, 48]
    assert exp.primary_sp_to_use == 12

    # 2.0 Auto Detection based on length of data ----
    # 2.1 Length barely enough to detect seasonality (2*sp + 1)
    np.random.seed(42)
    sp = 60
    data = np.random.randint(0, 100, size=sp)
    data = pd.DataFrame(np.concatenate((np.tile(data, 2), [data[0]])))
    exp = TSForecastingExperiment()
    exp.setup(data=data)
    assert exp.primary_sp_to_use == sp

    # 2.2 Length just below threshold to detect seasonality (2*sp)
    exp = TSForecastingExperiment()
    exp.setup(data=data.iloc[: 2 * sp])
    assert exp.primary_sp_to_use < sp

    # 3.0 Overwritten by user defined seasonal period ----
    sp = 19
    # 3.1 ignore_seasonality_test = False (default)
    exp.setup(
        data=data,
        seasonal_period=sp,
        # ignore_seasonality_test=False,  # default
        sp_detection="auto",
        verbose=False,
        session_id=42,
    )
    assert exp.seasonal_period == sp
    # overridden to user_defined even through we pass "auto"
    assert exp.sp_detection == "user_defined"
    assert exp.ignore_seasonality_test is False
    assert exp.candidate_sps == [sp]
    assert exp.significant_sps == [1]
    assert exp.significant_sps_no_harmonics == [1]
    assert exp.all_sps_to_use == [1]
    assert exp.primary_sp_to_use == 1

    # 3.2 ignore_seasonality_test = True
    exp.setup(
        data=data,
        seasonal_period=sp,
        ignore_seasonality_test=True,
        sp_detection="auto",
        verbose=False,
        session_id=42,
    )
    assert exp.seasonal_period == sp
    # overridden to user_defined even through we pass "auto"
    assert exp.sp_detection == "user_defined"
    assert exp.ignore_seasonality_test is True
    assert exp.candidate_sps == [sp]
    assert exp.significant_sps == [sp]
    assert exp.significant_sps_no_harmonics == [sp]
    assert exp.all_sps_to_use == [sp]
    assert exp.primary_sp_to_use == sp


def test_sp_to_use_upto_max_sp():
    """Seasonal Period detection upto a max seasonal period provided by user."""
    data = get_data("airline", verbose=False)

    # 1.0 Max SP not specified ----
    exp = TSForecastingExperiment()
    exp.setup(
        data=data, fh=12, session_id=42, remove_harmonics=False, max_sp_to_consider=None
    )
    assert exp.candidate_sps == [12, 24, 36, 11, 48]
    assert exp.significant_sps == [12, 24, 36, 11, 48]
    assert exp.significant_sps_no_harmonics == [48, 36, 11]
    assert exp.all_sps_to_use == [12]
    assert exp.primary_sp_to_use == 12

    # 2.0 Max SP more than at least some detected values ----
    # 2.1 Without removing harmonics
    exp = TSForecastingExperiment()
    exp.setup(
        data=data, fh=12, session_id=42, remove_harmonics=False, max_sp_to_consider=24
    )
    assert exp.candidate_sps == [12, 24, 11]
    assert exp.significant_sps == [12, 24, 11]
    assert exp.significant_sps_no_harmonics == [24, 11]
    assert exp.all_sps_to_use == [12]
    assert exp.primary_sp_to_use == 12

    # 2.2 Removing harmonics
    exp = TSForecastingExperiment()
    exp.setup(
        data=data, fh=12, session_id=42, remove_harmonics=True, max_sp_to_consider=24
    )
    assert exp.candidate_sps == [12, 24, 11]
    assert exp.significant_sps == [12, 24, 11]
    assert exp.significant_sps_no_harmonics == [24, 11]
    assert exp.all_sps_to_use == [24]
    assert exp.primary_sp_to_use == 24

    # 3.0 Max SP less than all detected values ----
    # 3.1 Without removing harmonics
    exp = TSForecastingExperiment()
    exp.setup(
        data=data, fh=12, session_id=42, remove_harmonics=False, max_sp_to_consider=2
    )
    assert exp.candidate_sps == []
    assert exp.significant_sps == [1]
    assert exp.significant_sps_no_harmonics == [1]
    assert exp.all_sps_to_use == [1]
    assert exp.primary_sp_to_use == 1

    # 3.2 Removing harmonics
    exp = TSForecastingExperiment()
    exp.setup(
        data=data, fh=12, session_id=42, remove_harmonics=True, max_sp_to_consider=2
    )
    assert exp.candidate_sps == []
    assert exp.significant_sps == [1]
    assert exp.significant_sps_no_harmonics == [1]
    assert exp.all_sps_to_use == [1]
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

    assert exp.candidate_sps == [seasonal_value]


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

    assert exp.candidate_sps == [seasonal_value]


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
    expected_candidate_sps = [int(lcm / prefix)]

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

    assert exp.candidate_sps == expected_candidate_sps


def test_train_test_split_uni_no_exo(load_pos_and_neg_data):
    """Tests the train-test splits for univariate time series without exogenous variables"""
    data = load_pos_and_neg_data

    ####################################
    # Continuous fh without Gaps ####
    ####################################

    # Integer fh ----
    exp = TSForecastingExperiment()
    fh = 12
    exp.setup(data=data, fh=fh, session_id=42)
    assert np.all(exp.dataset.index == data.index)
    assert np.all(exp.train.index == data.iloc[: (len(data) - fh)].index)
    assert np.all(exp.test.index == data.iloc[-fh:].index)
    assert exp.X is None
    assert np.all(exp.y.index == data.index)
    assert exp.X_train is None
    assert exp.X_test is None
    assert np.all(exp.y_train.index == data.iloc[: (len(data) - fh)].index)
    assert np.all(exp.y_test.index == data.iloc[-fh:].index)
    assert np.all(exp.dataset_transformed.index == data.index)
    assert np.all(exp.train_transformed.index == data.iloc[: (len(data) - fh)].index)
    assert np.all(exp.test_transformed.index == data.iloc[-fh:].index)
    assert exp.X_transformed is None
    assert np.all(exp.y_transformed.index == data.index)
    assert exp.X_train_transformed is None
    assert exp.X_test_transformed is None
    assert np.all(exp.y_train_transformed.index == data.iloc[: (len(data) - fh)].index)
    assert np.all(exp.y_test_transformed.index == data.iloc[-fh:].index)

    # Numpy fh ----
    exp = TSForecastingExperiment()
    fh = np.arange(1, 10)  # 9 values
    exp.setup(data=data, fh=fh, session_id=42)
    assert np.all(exp.dataset.index == data.index)
    assert np.all(exp.train.index == data.iloc[: (len(data) - max(fh))].index)
    assert np.all(exp.test.index == data.iloc[-len(fh) :].index)
    assert exp.X is None
    assert np.all(exp.y.index == data.index)
    assert exp.X_train is None
    assert exp.X_test is None
    assert np.all(exp.y_train.index == data.iloc[: (len(data) - max(fh))].index)
    assert np.all(exp.y_test.index == data.iloc[-len(fh) :].index)
    assert np.all(exp.dataset_transformed.index == data.index)
    assert np.all(
        exp.train_transformed.index == data.iloc[: (len(data) - max(fh))].index
    )
    assert np.all(exp.test_transformed.index == data.iloc[-len(fh) :].index)
    assert exp.X_transformed is None
    assert np.all(exp.y_transformed.index == data.index)
    assert exp.X_train_transformed is None
    assert exp.X_test_transformed is None
    assert np.all(
        exp.y_train_transformed.index == data.iloc[: (len(data) - max(fh))].index
    )
    assert np.all(exp.y_test_transformed.index == data.iloc[-len(fh) :].index)

    # List fh ----
    exp = TSForecastingExperiment()
    fh = [1, 2, 3, 4, 5, 6]
    exp.setup(data=data, fh=fh, session_id=42)
    assert np.all(exp.dataset.index == data.index)
    assert np.all(exp.train.index == data.iloc[: (len(data) - max(fh))].index)
    assert np.all(exp.test.index == data.iloc[-len(fh) :].index)
    assert exp.X is None
    assert np.all(exp.y.index == data.index)
    assert exp.X_train is None
    assert exp.X_test is None
    assert np.all(exp.y_train.index == data.iloc[: (len(data) - max(fh))].index)
    assert np.all(exp.y_test.index == data.iloc[-len(fh) :].index)
    assert np.all(exp.dataset_transformed.index == data.index)
    assert np.all(
        exp.train_transformed.index == data.iloc[: (len(data) - max(fh))].index
    )
    assert np.all(exp.test_transformed.index == data.iloc[-len(fh) :].index)
    assert exp.X_transformed is None
    assert np.all(exp.y_transformed.index == data.index)
    assert exp.X_train_transformed is None
    assert exp.X_test_transformed is None
    assert np.all(
        exp.y_train_transformed.index == data.iloc[: (len(data) - max(fh))].index
    )
    assert np.all(exp.y_test_transformed.index == data.iloc[-len(fh) :].index)

    #################################
    # Continuous fh with Gaps ####
    #################################

    # Numpy fh ----
    exp = TSForecastingExperiment()
    fh = np.arange(7, 13)  # 6 values
    exp.setup(data=data, fh=fh, session_id=42)
    assert np.all(exp.dataset.index == data.index)
    assert np.all(exp.train.index == data.iloc[: (len(data) - max(fh))].index)
    assert len(exp.test) == len(fh)
    assert exp.X is None
    assert np.all(exp.y.index == data.index)
    assert exp.X_train is None
    assert exp.X_test is None
    assert np.all(exp.y_train.index == data.iloc[: (len(data) - max(fh))].index)
    assert len(exp.y_test) == len(fh)
    assert np.all(exp.dataset_transformed.index == data.index)
    assert np.all(
        exp.train_transformed.index == data.iloc[: (len(data) - max(fh))].index
    )
    assert len(exp.test_transformed) == len(fh)
    assert exp.X_transformed is None
    assert np.all(exp.y_transformed.index == data.index)
    assert exp.X_train_transformed is None
    assert exp.X_test_transformed is None
    assert np.all(
        exp.y_train_transformed.index == data.iloc[: (len(data) - max(fh))].index
    )
    assert len(exp.y_test_transformed) == len(fh)

    # List fh ----
    exp = TSForecastingExperiment()
    fh = [4, 5, 6]
    exp.setup(data=data, fh=fh, session_id=42)
    assert np.all(exp.dataset.index == data.index)
    assert np.all(exp.train.index == data.iloc[: (len(data) - max(fh))].index)
    assert len(exp.test) == len(fh)
    assert exp.X is None
    assert np.all(exp.y.index == data.index)
    assert exp.X_train is None
    assert exp.X_test is None
    assert np.all(exp.y_train.index == data.iloc[: (len(data) - max(fh))].index)
    assert len(exp.y_test) == len(fh)
    assert np.all(exp.dataset_transformed.index == data.index)
    assert np.all(
        exp.train_transformed.index == data.iloc[: (len(data) - max(fh))].index
    )
    assert len(exp.test_transformed) == len(fh)
    assert exp.X_transformed is None
    assert np.all(exp.y_transformed.index == data.index)
    assert exp.X_train_transformed is None
    assert exp.X_test_transformed is None
    assert np.all(
        exp.y_train_transformed.index == data.iloc[: (len(data) - max(fh))].index
    )
    assert len(exp.y_test_transformed) == len(fh)

    ####################################
    # Discontinuous fh with Gaps ####
    ####################################

    # Numpy fh ----
    exp = TSForecastingExperiment()
    fh = np.array([4, 5, 6, 10, 11, 12])  # 6 values
    exp.setup(data=data, fh=fh, session_id=42)
    assert np.all(exp.dataset.index == data.index)
    assert np.all(exp.train.index == data.iloc[: (len(data) - max(fh))].index)
    assert len(exp.test) == len(fh)
    assert exp.X is None
    assert np.all(exp.y.index == data.index)
    assert exp.X_train is None
    assert exp.X_test is None
    assert np.all(exp.y_train.index == data.iloc[: (len(data) - max(fh))].index)
    assert len(exp.y_test) == len(fh)
    assert np.all(exp.dataset_transformed.index == data.index)
    assert np.all(
        exp.train_transformed.index == data.iloc[: (len(data) - max(fh))].index
    )
    assert len(exp.test_transformed) == len(fh)
    assert exp.X_transformed is None
    assert np.all(exp.y_transformed.index == data.index)
    assert exp.X_train_transformed is None
    assert exp.X_test_transformed is None
    assert np.all(
        exp.y_train_transformed.index == data.iloc[: (len(data) - max(fh))].index
    )
    assert len(exp.y_test_transformed) == len(fh)

    # List fh ----
    exp = TSForecastingExperiment()
    fh = [4, 5, 6, 10, 11, 12]
    exp.setup(data=data, fh=fh, session_id=42)
    assert np.all(exp.dataset.index == data.index)
    assert np.all(exp.train.index == data.iloc[: (len(data) - max(fh))].index)
    assert len(exp.test) == len(fh)
    assert exp.X is None
    assert np.all(exp.y.index == data.index)
    assert exp.X_train is None
    assert exp.X_test is None
    assert np.all(exp.y_train.index == data.iloc[: (len(data) - max(fh))].index)
    assert len(exp.y_test) == len(fh)
    assert np.all(exp.dataset_transformed.index == data.index)
    assert np.all(
        exp.train_transformed.index == data.iloc[: (len(data) - max(fh))].index
    )
    assert len(exp.test_transformed) == len(fh)
    assert exp.X_transformed is None
    assert np.all(exp.y_transformed.index == data.index)
    assert exp.X_train_transformed is None
    assert exp.X_test_transformed is None
    assert np.all(
        exp.y_train_transformed.index == data.iloc[: (len(data) - max(fh))].index
    )
    assert len(exp.y_test_transformed) == len(fh)


def test_train_test_split_uni_exo(load_uni_exo_data_target):
    """Tests the train-test splits for univariate time series with exogenous variables"""
    data, target = load_uni_exo_data_target

    ####################################
    # Continuous fh without Gaps ####
    ####################################

    # Integer fh ----
    exp = TSForecastingExperiment()
    fh = 12
    exp.setup(data=data, target=target, fh=fh, session_id=42)
    assert np.all(exp.dataset.index == data.index)
    assert np.all(exp.train.index == data.iloc[: (len(data) - fh)].index)
    assert np.all(exp.test.index == data.iloc[-fh:].index)
    assert np.all(exp.X.index == data.index)
    assert np.all(exp.y.index == data.index)
    assert np.all(exp.X_train.index == data.iloc[: (len(data) - fh)].index)
    assert np.all(exp.X_test.index == data.iloc[-fh:].index)
    assert np.all(exp.y_train.index == data.iloc[: (len(data) - fh)].index)
    assert np.all(exp.y_test.index == data.iloc[-fh:].index)
    assert np.all(exp.dataset_transformed.index == data.index)
    assert np.all(exp.train_transformed.index == data.iloc[: (len(data) - fh)].index)
    assert np.all(exp.test_transformed.index == data.iloc[-fh:].index)
    assert np.all(exp.X_transformed.index == data.index)
    assert np.all(exp.y_transformed.index == data.index)
    assert np.all(exp.X_train_transformed.index == data.iloc[: (len(data) - fh)].index)
    assert np.all(exp.X_test_transformed.index == data.iloc[-fh:].index)
    assert np.all(exp.y_train_transformed.index == data.iloc[: (len(data) - fh)].index)
    assert np.all(exp.y_test_transformed.index == data.iloc[-fh:].index)

    # Numpy fh ----
    exp = TSForecastingExperiment()
    fh = np.arange(1, 10)  # 9 values
    exp.setup(data=data, target=target, fh=fh, session_id=42)
    assert np.all(exp.dataset.index == data.index)
    assert np.all(exp.train.index == data.iloc[: (len(data) - max(fh))].index)
    assert np.all(exp.test.index == data.iloc[-len(fh) :].index)
    assert np.all(exp.X.index == data.index)
    assert np.all(exp.y.index == data.index)
    assert np.all(exp.X_train.index == data.iloc[: (len(data) - max(fh))].index)
    assert np.all(exp.X_test.index == data.iloc[-len(fh) :].index)
    assert np.all(exp.y_train.index == data.iloc[: (len(data) - max(fh))].index)
    assert np.all(exp.y_test.index == data.iloc[-len(fh) :].index)
    assert np.all(exp.dataset_transformed.index == data.index)
    assert np.all(
        exp.train_transformed.index == data.iloc[: (len(data) - max(fh))].index
    )
    assert np.all(exp.test_transformed.index == data.iloc[-len(fh) :].index)
    assert np.all(exp.X_transformed.index == data.index)
    assert np.all(exp.y_transformed.index == data.index)
    assert np.all(
        exp.X_train_transformed.index == data.iloc[: (len(data) - max(fh))].index
    )
    assert np.all(exp.X_test_transformed.index == data.iloc[-len(fh) :].index)
    assert np.all(
        exp.y_train_transformed.index == data.iloc[: (len(data) - max(fh))].index
    )
    assert np.all(exp.y_test_transformed.index == data.iloc[-len(fh) :].index)

    # List fh ----
    exp = TSForecastingExperiment()
    fh = [1, 2, 3, 4, 5, 6]
    exp.setup(data=data, target=target, fh=fh, session_id=42)
    assert np.all(exp.dataset.index == data.index)
    assert np.all(exp.train.index == data.iloc[: (len(data) - max(fh))].index)
    assert np.all(exp.test.index == data.iloc[-len(fh) :].index)
    assert np.all(exp.X.index == data.index)
    assert np.all(exp.y.index == data.index)
    assert np.all(exp.X_train.index == data.iloc[: (len(data) - max(fh))].index)
    assert np.all(exp.X_test.index == data.iloc[-len(fh) :].index)
    assert np.all(exp.y_train.index == data.iloc[: (len(data) - max(fh))].index)
    assert np.all(exp.y_test.index == data.iloc[-len(fh) :].index)
    assert np.all(exp.dataset_transformed.index == data.index)
    assert np.all(
        exp.train_transformed.index == data.iloc[: (len(data) - max(fh))].index
    )
    assert np.all(exp.test_transformed.index == data.iloc[-len(fh) :].index)
    assert np.all(exp.X_transformed.index == data.index)
    assert np.all(exp.y_transformed.index == data.index)
    assert np.all(
        exp.X_train_transformed.index == data.iloc[: (len(data) - max(fh))].index
    )
    assert np.all(exp.X_test_transformed.index == data.iloc[-len(fh) :].index)
    assert np.all(
        exp.y_train_transformed.index == data.iloc[: (len(data) - max(fh))].index
    )
    assert np.all(exp.y_test_transformed.index == data.iloc[-len(fh) :].index)

    #################################
    # Continuous fh with Gaps ####
    #################################

    # Numpy fh ----
    exp = TSForecastingExperiment()
    fh = np.arange(7, 13)  # 6 values
    exp.setup(data=data, target=target, fh=fh, session_id=42)
    assert np.all(exp.dataset.index == data.index)
    assert np.all(exp.train.index == data.iloc[: (len(data) - max(fh))].index)
    # `test`` call still refers to y_test indices and not X_test indices
    assert len(exp.test) == len(fh)
    assert np.all(exp.X.index == data.index)
    assert np.all(exp.y.index == data.index)
    assert np.all(exp.X_train.index == data.iloc[: (len(data) - max(fh))].index)
    # Exogenous variables will not have any gaps (only target has gaps)
    assert np.all(exp.X_test.index == data.iloc[-max(fh) :].index)
    assert np.all(exp.y_train.index == data.iloc[: (len(data) - max(fh))].index)
    assert len(exp.y_test) == len(fh)
    assert np.all(exp.dataset_transformed.index == data.index)
    assert np.all(
        exp.train_transformed.index == data.iloc[: (len(data) - max(fh))].index
    )
    assert len(exp.test_transformed) == len(fh)
    assert np.all(exp.X_transformed.index == data.index)
    assert np.all(exp.y_transformed.index == data.index)

    # List fh ----
    exp = TSForecastingExperiment()
    fh = [4, 5, 6]
    exp.setup(data=data, target=target, fh=fh, session_id=42)
    assert np.all(exp.dataset.index == data.index)
    assert np.all(exp.train.index == data.iloc[: (len(data) - max(fh))].index)
    # `test`` call still refers to y_test indices and not X_test indices
    assert len(exp.test) == len(fh)
    assert np.all(exp.X.index == data.index)
    assert np.all(exp.y.index == data.index)
    assert np.all(exp.X_train.index == data.iloc[: (len(data) - max(fh))].index)
    # Exogenous variables will not have any gaps (only target has gaps)
    assert np.all(exp.X_test.index == data.iloc[-max(fh) :].index)
    assert np.all(exp.y_train.index == data.iloc[: (len(data) - max(fh))].index)
    assert len(exp.y_test) == len(fh)
    assert np.all(exp.dataset_transformed.index == data.index)
    assert np.all(
        exp.train_transformed.index == data.iloc[: (len(data) - max(fh))].index
    )
    assert len(exp.test_transformed) == len(fh)
    assert np.all(exp.X_transformed.index == data.index)
    assert np.all(exp.y_transformed.index == data.index)

    ####################################
    # Discontinuous fh with Gaps ####
    ####################################

    # Numpy fh ----
    exp = TSForecastingExperiment()
    fh = np.array([4, 5, 6, 10, 11, 12])  # 6 values
    exp.setup(data=data, target=target, fh=fh, session_id=42)
    assert np.all(exp.dataset.index == data.index)
    assert np.all(exp.train.index == data.iloc[: (len(data) - max(fh))].index)
    # `test`` call still refers to y_test indices and not X_test indices
    assert len(exp.test) == len(fh)
    assert np.all(exp.X.index == data.index)
    assert np.all(exp.y.index == data.index)
    assert np.all(exp.X_train.index == data.iloc[: (len(data) - max(fh))].index)
    # Exogenous variables will not have any gaps (only target has gaps)
    assert np.all(exp.X_test.index == data.iloc[-max(fh) :].index)
    assert np.all(exp.y_train.index == data.iloc[: (len(data) - max(fh))].index)
    assert len(exp.y_test) == len(fh)
    assert np.all(exp.dataset_transformed.index == data.index)
    assert np.all(
        exp.train_transformed.index == data.iloc[: (len(data) - max(fh))].index
    )
    assert len(exp.test_transformed) == len(fh)
    assert np.all(exp.X_transformed.index == data.index)
    assert np.all(exp.y_transformed.index == data.index)
    assert np.all(
        exp.X_train_transformed.index == data.iloc[: (len(data) - max(fh))].index
    )
    # Exogenous variables will not have any gaps (only target has gaps)
    assert np.all(exp.X_test_transformed.index == data.iloc[-max(fh) :].index)
    assert np.all(
        exp.y_train_transformed.index == data.iloc[: (len(data) - max(fh))].index
    )
    assert len(exp.y_test_transformed) == len(fh)

    # List fh ----
    exp = TSForecastingExperiment()
    fh = [4, 5, 6, 10, 11, 12]
    exp.setup(data=data, target=target, fh=fh, session_id=42)
    assert np.all(exp.dataset.index == data.index)
    assert np.all(exp.train.index == data.iloc[: (len(data) - max(fh))].index)
    # `test`` call still refers to y_test indices and not X_test indices
    assert len(exp.test) == len(fh)
    assert np.all(exp.X.index == data.index)
    assert np.all(exp.y.index == data.index)
    assert np.all(exp.X_train.index == data.iloc[: (len(data) - max(fh))].index)
    # Exogenous variables will not have any gaps (only target has gaps)
    assert np.all(exp.X_test.index == data.iloc[-max(fh) :].index)
    assert np.all(exp.y_train.index == data.iloc[: (len(data) - max(fh))].index)
    assert len(exp.y_test) == len(fh)
    assert np.all(exp.dataset_transformed.index == data.index)
    assert np.all(
        exp.train_transformed.index == data.iloc[: (len(data) - max(fh))].index
    )
    assert len(exp.test_transformed) == len(fh)
    assert np.all(exp.X_transformed.index == data.index)
    assert np.all(exp.y_transformed.index == data.index)
    assert np.all(
        exp.X_train_transformed.index == data.iloc[: (len(data) - max(fh))].index
    )
    # Exogenous variables will not have any gaps (only target has gaps)
    assert np.all(exp.X_test_transformed.index == data.iloc[-max(fh) :].index)
    assert np.all(
        exp.y_train_transformed.index == data.iloc[: (len(data) - max(fh))].index
    )
    assert len(exp.y_test_transformed) == len(fh)


def test_missing_indices():
    """Tests setup when data has missing indices"""

    data = pd.read_csv(
        "https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv"
    )
    data["ds"] = pd.to_datetime(data["ds"])
    data.set_index("ds", inplace=True)
    data.index = data.index.to_period("D")
    data.info()

    exp = TSForecastingExperiment()

    with pytest.raises(ValueError) as errmsg:
        exp.setup(data=data, fh=365, session_id=42)
    exceptionmsg = errmsg.value.args[0]

    assert "Data has missing indices!" in exceptionmsg


def test_hyperparameter_splits():
    """Tests the splits to use to determine the hyperparameters"""

    # 1.0 Recommended d, white noise, seasonal_period ----
    data = get_data("airline")

    FOLD = 1
    FH = 60
    TRAIN_SIZE = len(data) - FH
    # Train set
    data[:TRAIN_SIZE] = 1
    print("Experiment 1 ----")
    exp1 = TSForecastingExperiment()
    exp1.setup(data=data, fh=FH, fold=FOLD)

    print("Experiment 2 ----")
    exp2 = TSForecastingExperiment()
    exp2.setup(data=data, hyperparameter_split="train", fh=FH, fold=FOLD)

    assert exp1.primary_sp_to_use != exp2.primary_sp_to_use
    assert exp1.lowercase_d != exp2.lowercase_d
    assert exp1.white_noise != exp2.white_noise
    # uppercase_d turns out to be the same, hence tested separately
    assert exp1.uppercase_d == exp2.uppercase_d

    # 2.0 Recommended Seasonal D
    data = get_data("airline")

    FOLD = 1
    FH = 36
    TRAIN_SIZE = len(data) - FH

    np.random.seed(42)
    indices = np.random.randint(1, int(TRAIN_SIZE / 2), 12)
    data.iloc[indices] = 200

    exp1 = TSForecastingExperiment()
    exp1.setup(data=data, fh=FH, fold=FOLD)

    exp2 = TSForecastingExperiment()
    exp2.setup(data=data, hyperparameter_split="train", fh=FH, fold=FOLD)

    assert exp1.uppercase_d != exp2.uppercase_d


@pytest.mark.parametrize("index", ["RangeIndex", "DatetimeIndex"])
@pytest.mark.parametrize("seasonality_type", ["mul", "add", "auto"])
def test_seasonality_type_no_season(index: str, seasonality_type: str):
    """Tests the detection of the seasonality type with data that has no seasonality.

    Parameters
    ----------
    index : str
        Type of index. Options are: "RangeIndex" and "DatetimeIndex"
    seasonality_type : str
        The seasonality type to pass to setup
    """
    # Create base data without seasonality
    N = 100
    y = pd.Series(np.arange(100, 100 + N))  # No negative values when creating final y

    # RangeIndex is default index
    if index == "DatetimeIndex":
        dates = pd.date_range(start="2020-01-01", periods=N, freq="MS")
        y.index = dates

    err_msg = "Expected seasonality_type = None, but got something else."
    exp = TSForecastingExperiment()
    exp.setup(data=y, seasonality_type=seasonality_type, session_id=42)
    assert exp.seasonality_type is None, err_msg


@pytest.mark.parametrize("index", ["RangeIndex", "DatetimeIndex"])
@pytest.mark.parametrize("seasonality_type", ["mul", "add", "auto"])
@pytest.mark.parametrize(
    "y", _data_seasonal_types_strictly_pos, ids=["data_add", "data_mul"]
)
def test_seasonality_type_with_season_not_stricly_positive(
    index: str, seasonality_type: str, y: pd.Series
):
    """Tests the detection of the seasonality type with user defined type and
    data that has seasonality and is not strictly positive.

    Parameters
    ----------
    index : str
        Type of index. Options are: "RangeIndex" and "DatetimeIndex"
    seasonality_type : str
        The seasonality type to pass to setup
    y : pd.Series
        Dataset to use
    """
    # Make data not strictly positive
    y = y - y.max()

    # RangeIndex is default index
    if index == "DatetimeIndex":
        dates = pd.date_range(start="2020-01-01", periods=len(y), freq="MS")
        y.index = dates

    err_msg = "Expected 'additive' seasonality, got something else"
    exp = TSForecastingExperiment()
    exp.setup(data=y, seasonality_type=seasonality_type, session_id=42)
    assert exp.seasonality_type == "add", err_msg


@pytest.mark.parametrize("index", ["RangeIndex", "DatetimeIndex"])
@pytest.mark.parametrize("seasonality_type", ["mul", "add"])
@pytest.mark.parametrize(
    "y", _data_seasonal_types_strictly_pos, ids=["data_add", "data_mul"]
)
def test_seasonality_type_user_def_with_season_strictly_pos(
    index: str, seasonality_type: str, y: pd.Series
):
    """Tests the detection of the seasonality type with user defined type and
    data that has seasonality and is strictly positive.

    Parameters
    ----------
    index : str
        Type of index. Options are: "RangeIndex" and "DatetimeIndex"
    seasonality_type : str
        The seasonality type to pass to setup
    y : pd.Series
        Dataset to use
    """
    # RangeIndex is default index
    if index == "DatetimeIndex":
        dates = pd.date_range(start="2020-01-01", periods=len(y), freq="MS")
        y.index = dates

    err_msg = f"Expected '{seasonality_type}' seasonality, got something else"
    exp = TSForecastingExperiment()
    exp.setup(data=y, seasonality_type=seasonality_type, session_id=42)
    assert exp.seasonality_type == seasonality_type, err_msg


@pytest.mark.parametrize("index", ["RangeIndex", "DatetimeIndex"])
@pytest.mark.parametrize("seasonality_type", ["auto"])
def test_seasonality_type_auto_with_season_strictly_pos(
    index: str, seasonality_type: str
):
    """Tests the detection of the seasonality type using the internal auto algorithm
    when data that has seasonality and is strictly positive.

    Tests various index types and tests for both additive and multiplicative
    seasonality.

    Parameters
    ----------
    index : str
        Type of index. Options are: "RangeIndex" and "DatetimeIndex"
    seasonality_type : str
        The seasonality type to pass to setup
    """
    # Create base data
    N = 100
    y_trend = np.arange(100, 100 + N)
    y_season = 100 * (1 + np.sin(y_trend))  # No negative values when creating final y
    y = pd.Series(y_trend + y_season)

    # RangeIndex is default index
    if index == "DatetimeIndex":
        dates = pd.date_range(start="2020-01-01", periods=N, freq="MS")
        y.index = dates

    # -------------------------------------------------------------------------#
    # Test 1: Additive Seasonality
    # -------------------------------------------------------------------------#
    err_msg = "Expected additive seasonality, got multiplicative"
    exp = TSForecastingExperiment()
    exp.setup(data=y, seasonality_type=seasonality_type, session_id=42)
    assert exp.seasonality_type == "add", err_msg

    # # -------------------------------------------------------------------------#
    # # Test 2A: Multiplicative Seasonality (1)
    # # -------------------------------------------------------------------------#
    # y = pd.Series(y_trend * y_season)
    # y.index = dates

    # err_msg = "Expected multiplicative seasonality, got additive (1)"
    # exp = TSForecastingExperiment()
    # exp.setup(data=y, session_id=42)
    # assert exp.seasonality_type == "mul", err_msg

    # -------------------------------------------------------------------------#
    # Test 2B: Multiplicative Seasonality (2)
    # -------------------------------------------------------------------------#
    y = get_data("airline", verbose=False)

    err_msg = "Expected multiplicative seasonality, got additive (2)"
    exp = TSForecastingExperiment()
    exp.setup(data=y, seasonality_type=seasonality_type, session_id=42)
    assert exp.seasonality_type == "mul", err_msg
