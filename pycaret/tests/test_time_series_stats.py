import pytest

from pycaret.time_series import TSForecastingExperiment

from .time_series_test_utils import (
    _return_data_big_small,
    _return_model_names_for_plots_stats,
    _ALL_STATS_TESTS,
)


##############################
#### Functions Start Here ####
##############################

# NOTE: Fixtures can not be used to parameterize tests
# https://stackoverflow.com/questions/52764279/pytest-how-to-parametrize-a-test-with-a-list-that-is-returned-from-a-fixture
# Hence, we have to create functions and create the parameterized list first
# (must happen during collect phase) before passing it to mark.parameterize.


_data_big_small = _return_data_big_small()
_model_names_for_stats = _return_model_names_for_plots_stats()

############################
#### Functions End Here ####
############################


##########################
#### Tests Start Here ####
##########################


@pytest.mark.parametrize("test", _ALL_STATS_TESTS)
@pytest.mark.parametrize("data", _data_big_small)
def test_check_stats_data(data, test):
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

    ##########################
    #### Individual Tests ####
    ##########################
    # Column Order ----
    results = exp.check_stats(test=test)
    column_names = list(results.columns)
    for i, name in enumerate(expected_column_order):
        assert column_names[i] == name

    # Data Names ----
    expected_data_names = ["Actual"]
    data_names = results["Data"].unique().tolist()
    for i, expected_name in enumerate(data_names):
        assert data_names[i] in expected_data_names

    ###################################################
    #### Individual Tests with "order" differences ####
    ###################################################
    # Column Order ----
    results = exp.check_stats(test=test, data_kwargs={"order_list": [1, 2]})
    column_names = list(results.columns)
    for i, name in enumerate(expected_column_order):
        assert column_names[i] == name

    # Data Names ----
    expected_data_names = ["Actual", "Order=1", "Order=2"]
    data_names = results["Data"].unique().tolist()
    for i, expected_name in enumerate(data_names):
        assert data_names[i] in expected_data_names

    ##################################################
    #### Individual Tests with "lags" differences ####
    ##################################################
    # Column Order ----
    results = exp.check_stats(test=test, data_kwargs={"lags_list": [1, [1, 12]]})
    column_names = list(results.columns)
    for i, name in enumerate(expected_column_order):
        assert column_names[i] == name

    # Data Names ----
    expected_data_names = ["Actual", "Lags=1", "Lags=[1, 12]"]
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
        seasonal_period=1,  # TODO: Remove after models start using `primary_sp_to_use`
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
    #### Individual Tests ####
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
    #### Individual Tests with "order" differences ####
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
    #### Individual Tests with "lags" differences ####
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
