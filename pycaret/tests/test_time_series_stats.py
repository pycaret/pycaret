import pytest

from pycaret.internal.pycaret_experiment import TimeSeriesExperiment

from .time_series_test_utils import _return_data_big_small, _ALL_STATS_TESTS

##############################
#### Functions Start Here ####
##############################

# NOTE: Fixtures can not be used to parameterize tests
# https://stackoverflow.com/questions/52764279/pytest-how-to-parametrize-a-test-with-a-list-that-is-returned-from-a-fixture
# Hence, we have to create functions and create the parameterized list first
# (must happen during collect phase) before passing it to mark.parameterize.


_data_big_small = _return_data_big_small()

############################
#### Functions End Here ####
############################


##########################
#### Tests Start Here ####
##########################


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
