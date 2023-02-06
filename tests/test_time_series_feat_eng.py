"""Module to test time_series functionality
"""
import numpy as np
import pytest
from pandas.testing import assert_frame_equal
from sktime.transformations.series.date import DateTimeFeatures
from sktime.transformations.series.summarize import WindowSummarizer
from time_series_test_utils import assert_frame_not_equal

from pycaret.time_series import TSForecastingExperiment

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


def test_fe_target(load_pos_and_neg_data):
    """Test custom feature engineering for target (applicable to reduced
    regression models only)
    """
    data = load_pos_and_neg_data

    kwargs = {"lag_feature": {"lag": [36, 24, 13, 12, 11, 9, 6, 3, 2, 1]}}
    fe_target_rr = [WindowSummarizer(n_jobs=1, truncate="bfill", **kwargs)]

    # Baseline
    exp = TSForecastingExperiment()
    exp.setup(data=data, fh=12, fold=3, session_id=42)
    exp.create_model("lr_cds_dt")
    metrics1 = exp.pull()

    # With Feature Engineering
    exp = TSForecastingExperiment()
    exp.setup(data=data, fh=12, fold=3, fe_target_rr=fe_target_rr, session_id=42)
    exp.create_model("lr_cds_dt")
    metrics2 = exp.pull()

    assert_frame_not_equal(metrics1, metrics2)


@pytest.mark.parametrize(
    "model, expected_equal", [("arima", False), ("lr_cds_dt", False), ("naive", True)]
)
def test_fe_exogenous(load_uni_exo_data_target, model, expected_equal):
    """Test custom feature engineering for exogenous variables (applicable to all models)."""
    data, target = load_uni_exo_data_target
    fh = 12

    # Example: function num_above_thresh to count how many observations lie above
    # the threshold within a window of length 2, lagged by 0 periods.
    def num_above_thresh(x):
        """Count how many observations lie above threshold."""
        return np.sum((x > 0.7)[::-1])

    kwargs1 = {"lag_feature": {"lag": [0, 1], "mean": [[0, 4]]}}
    kwargs2 = {
        "lag_feature": {
            "lag": [0, 1],
            num_above_thresh: [[0, 2]],
            "mean": [[0, 4]],
            "std": [[0, 4]],
        }
    }
    fe_exogenous = [
        (
            "a",
            WindowSummarizer(
                n_jobs=1, target_cols=["Income"], truncate="bfill", **kwargs1
            ),
        ),
        (
            "b",
            WindowSummarizer(
                n_jobs=1,
                target_cols=["Unemployment", "Production"],
                truncate="bfill",
                **kwargs2,
            ),
        ),
    ]

    # Baseline
    exp = TSForecastingExperiment()
    exp.setup(data=data, target=target, fh=fh, session_id=42)
    features1 = exp.get_config("X_transformed").columns
    _ = exp.create_model(model)
    metrics1 = exp.pull()

    # With Feature Engineering
    exp = TSForecastingExperiment()
    exp.setup(
        data=data,
        target=target,
        fh=fh,
        fe_exogenous=fe_exogenous,
        session_id=42,
    )
    features2 = exp.get_config("X_transformed").columns
    _ = exp.create_model(model)
    metrics2 = exp.pull()

    assert len(features1) != len(features2)
    if expected_equal:
        assert_frame_equal(metrics1, metrics2)
    else:
        assert_frame_not_equal(metrics1, metrics2)


def test_fe_exog_data_no_exo(load_pos_and_neg_data):
    """Test custom feature engineering for target and exogenous when data does
    not have any exogenous variables. e.g. extracting DateTimeFeatures from Index.
    """
    data = load_pos_and_neg_data
    kwargs = {"lag_feature": {"lag": [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]}}
    fe_target_rr = [WindowSummarizer(n_jobs=1, truncate="bfill", **kwargs)]

    # Baseline
    exp = TSForecastingExperiment()
    exp.setup(data=data, fh=12, session_id=42)
    _ = exp.create_model("lr_cds_dt")
    metrics1 = exp.pull()

    # With Feature Engineering (1) - replicate the default
    exp = TSForecastingExperiment()
    exp.setup(data=data, fh=12, fe_target_rr=fe_target_rr, session_id=42)
    _ = exp.create_model("lr_cds_dt")
    metrics2 = exp.pull()
    assert_frame_equal(metrics1, metrics2)

    # With Feature Engineering (2) - Date Time features created in y
    fe_target2 = fe_target_rr + [DateTimeFeatures(ts_freq="M")]
    exp = TSForecastingExperiment()
    exp.setup(data=data, fh=12, fe_target_rr=fe_target2, session_id=42)
    _ = exp.create_model("lr_cds_dt")
    metrics3 = exp.pull()
    assert_frame_not_equal(metrics1, metrics3)

    # With Feature Engineering (3) - Date Time features created in X
    fe_exogenous = [DateTimeFeatures(ts_freq="M")]
    exp = TSForecastingExperiment()
    exp.setup(
        data=data,
        fh=12,
        fe_target_rr=fe_target_rr,
        fe_exogenous=fe_exogenous,
        session_id=42,
    )
    _ = exp.create_model("lr_cds_dt")
    metrics4 = exp.pull()
    assert_frame_not_equal(metrics1, metrics4)

    # TODO: Not sure why these are different. Needs investigation.
    # assert_frame_equal(metrics3, metrics4)
