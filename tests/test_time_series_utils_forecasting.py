"""Module to test time_series forecasting utils
"""
import pytest

from pycaret.utils.time_series.forecasting import _check_and_clean_coverage

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

##############################
# Functions Start Here ####
##############################

# NOTE: Fixtures can not be used to parameterize tests
# https://stackoverflow.com/questions/52764279/pytest-how-to-parametrize-a-test-with-a-list-that-is-returned-from-a-fixture
# Hence, we have to create functions and create the parameterized list first
# (must happen during collect phase) before passing it to mark.parameterize.

############################
# Functions End Here ####
############################


##########################
# Tests Start Here ####
##########################


def test_check_and_clean_coverage():
    """Tests _check_and_clean_coverage"""

    # Tests floating point value ----
    coverage = 0.9
    coverage = _check_and_clean_coverage(coverage=coverage)
    coverage = [round(value, 2) for value in coverage]
    assert isinstance(coverage, list)
    assert coverage == [0.05, 0.95]

    # Tests List values (sorted) ----
    coverage_expected = [0.1, 0.9]
    coverage = _check_and_clean_coverage(coverage=coverage_expected)
    assert isinstance(coverage, list)
    assert coverage == coverage_expected

    # Tests List values (unsorted) ----
    coverage = [0.9, 0.1]
    coverage = _check_and_clean_coverage(coverage=coverage)
    assert isinstance(coverage, list)
    assert coverage == coverage_expected

    # Tests List values (incorrect length 1) ----
    with pytest.raises(ValueError) as errmsg:
        coverage = [0.1]
        coverage = _check_and_clean_coverage(coverage=coverage)
    exceptionmsg = errmsg.value.args[0]
    assert (
        "When coverage is a list, it must be of length 2 corresponding to"
        in exceptionmsg
    )

    # Tests List values (incorrect length 2) ----
    with pytest.raises(ValueError) as errmsg:
        coverage = [0.1, 0.5, 0.9]
        coverage = _check_and_clean_coverage(coverage=coverage)
    exceptionmsg = errmsg.value.args[0]
    assert (
        "When coverage is a list, it must be of length 2 corresponding to"
        in exceptionmsg
    )

    # Tests incorrect types ----
    with pytest.raises(TypeError) as errmsg:
        coverage = None
        coverage = _check_and_clean_coverage(coverage=coverage)
    exceptionmsg = errmsg.value.args[0]
    assert (
        "'coverage' must be of type float or a List of floats of length 2."
        in exceptionmsg
    )
