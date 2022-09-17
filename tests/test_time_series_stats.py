import pytest
from time_series_test_utils import (
    _ALL_DATA_TYPES,
    _ALL_STATS_TESTS,
    _ALL_STATS_TESTS_MISSING_DATA,
    _return_data_big_small,
    _return_model_names_for_plots_stats,
)

from pycaret.time_series import TSForecastingExperiment
from pycaret.utils.time_series.exceptions import MissingDataError

##############################
# Functions Start Here ####
##############################

# NOTE: Fixtures can not be used to parameterize tests
# https://stackoverflow.com/questions/52764279/pytest-how-to-parametrize-a-test-with-a-list-that-is-returned-from-a-fixture
# Hence, we have to create functions and create the parameterized list first
# (must happen during collect phase) before passing it to mark.parameterize.


_data_big_small = _return_data_big_small()
_model_names_for_stats = _return_model_names_for_plots_stats()

############################
# Functions End Here ####
############################


##########################
# Tests Start Here ####
##########################


@pytest.mark.parametrize("data_type", _ALL_DATA_TYPES)
@pytest.mark.parametrize("test", _ALL_STATS_TESTS)
@pytest.mark.parametrize("data", _data_big_small)
def test_check_stats_data(data, test, data_type):
    """Tests the check_stats functionality on the data"""

    exp = TSForecastingExperiment()

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

    expected_column_order = [
        "Test",
        "Test Name",
        "Data",
        "Property",
        "Setting",
        "Value",
    ]

    ##############################################
    # Individual Tests (with all defaults) ####
    ##############################################
    # Column Order ----
    results = exp.check_stats(test=test)
    column_names = list(results.columns)
    for i, name in enumerate(expected_column_order):
        assert column_names[i] == name

    # Data Names ----
    # Default data type should be "Transformed"
    expected_data_names = ["Transformed"]
    data_names = results["Data"].unique().tolist()
    for i, _ in enumerate(data_names):
        assert data_names[i] in expected_data_names

    ######################################################
    # Individual Default with different Data Types ####
    ######################################################
    results = exp.check_stats(test=test, data_type=data_type)
    column_names = list(results.columns)
    for i, name in enumerate(expected_column_order):
        assert column_names[i] == name

    # Data Names ----
    expected_data_names = [data_type.capitalize()]
    data_names = results["Data"].unique().tolist()
    for i, _ in enumerate(data_names):
        assert data_names[i] in expected_data_names

    ###################################################
    # Individual Tests with "order" differences ####
    ###################################################
    # Column Order ----
    results = exp.check_stats(
        test=test, data_type=data_type, data_kwargs={"order_list": [1, 2]}
    )
    column_names = list(results.columns)
    for i, name in enumerate(expected_column_order):
        assert column_names[i] == name

    # Data Names ----
    expected_data_names = [data_type.capitalize(), "Order=1", "Order=2"]
    data_names = results["Data"].unique().tolist()
    for i, expected_name in enumerate(data_names):
        assert data_names[i] in expected_data_names

    ##################################################
    # Individual Tests with "lags" differences ####
    ##################################################
    # Column Order ----
    results = exp.check_stats(
        test=test, data_type=data_type, data_kwargs={"lags_list": [1, [1, 12]]}
    )
    column_names = list(results.columns)
    for i, name in enumerate(expected_column_order):
        assert column_names[i] == name

    # Data Names ----
    expected_data_names = [data_type.capitalize(), "Lags=1", "Lags=[1, 12]"]
    data_names = results["Data"].unique().tolist()
    for i, expected_name in enumerate(data_names):
        assert data_names[i] in expected_data_names


@pytest.mark.parametrize("model_name", _model_names_for_stats)
@pytest.mark.parametrize("test", _ALL_STATS_TESTS)
@pytest.mark.parametrize("data", _data_big_small)
def test_check_stats_estimator(model_name, data, test):
    """Tests the check_stats functionality on the data"""

    exp = TSForecastingExperiment()

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
    model = exp.create_model(model_name)

    expected_column_order = [
        "Test",
        "Test Name",
        "Data",
        "Property",
        "Setting",
        "Value",
    ]

    ##########################
    # Individual Tests ####
    ##########################
    # Column Order ----
    results = exp.check_stats(model, test=test)
    if results is not None:
        # Results will be none if residuals can not be computed
        column_names = list(results.columns)
        for i, name in enumerate(expected_column_order):
            assert column_names[i] == name

        # Data Names ----
        expected_data_names = ["Residual"]
        data_names = results["Data"].unique().tolist()
        for i, expected_name in enumerate(data_names):
            assert data_names[i] in expected_data_names

    ###################################################
    # Individual Tests with "order" differences ####
    ###################################################
    # Column Order ----
    results = exp.check_stats(model, test=test, data_kwargs={"order_list": [1, 2]})
    if results is not None:
        # Results will be none if residuals can not be computed
        column_names = list(results.columns)
        for i, name in enumerate(expected_column_order):
            assert column_names[i] == name

        # Data Names ----
        expected_data_names = ["Residual", "Order=1", "Order=2"]
        data_names = results["Data"].unique().tolist()
        for i, expected_name in enumerate(data_names):
            assert data_names[i] in expected_data_names

    ##################################################
    # Individual Tests with "lags" differences ####
    ##################################################
    # Column Order ----
    results = exp.check_stats(model, test=test, data_kwargs={"lags_list": [1, [1, 12]]})
    if results is not None:
        # Results will be none if residuals can not be computed
        column_names = list(results.columns)
        for i, name in enumerate(expected_column_order):
            assert column_names[i] == name

        # Data Names ----
        expected_data_names = ["Residual", "Lags=1", "Lags=[1, 12]"]
        data_names = results["Data"].unique().tolist()
        for i, expected_name in enumerate(data_names):
            assert data_names[i] in expected_data_names


def test_check_stats_alpha(load_pos_and_neg_data):
    """Tests the check_stats functionality with different alpha"""

    exp = TSForecastingExperiment()

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


@pytest.mark.parametrize("test, supports_missing", _ALL_STATS_TESTS_MISSING_DATA)
def test_check_stats_data_raises(load_pos_data_missing, test, supports_missing):
    """Tests the check_stats functionality on the data with missing values.
    Not all tests support this and this checks that these tests flag this appropriately.
    """

    exp = TSForecastingExperiment()
    data = load_pos_data_missing

    # Reduced fh since we are testing with small dataset as well
    fh = 1
    fold = 2

    exp.setup(
        data=data,
        fh=fh,
        fold=fold,
        fold_strategy="sliding",
        numeric_imputation_target="drift",
        verbose=False,
        session_id=42,
    )

    # raise MissingValueError if test does not support it.
    if not supports_missing:
        with pytest.raises(MissingDataError) as errmsg:
            _ = exp.check_stats(test=test, data_type="original")

        # Capture Error message
        exceptionmsg = errmsg.value.args[0]

        # Check exact error received
        assert (
            "can not be run on data with missing values. Please check input data type."
            in exceptionmsg
        )
