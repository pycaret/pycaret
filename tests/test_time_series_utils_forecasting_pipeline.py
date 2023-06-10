"""Module to test time_series forecasting pipeline utils
"""
import numpy as np
import pytest
from sktime.forecasting.naive import NaiveForecaster

from pycaret.time_series import TSForecastingExperiment
from pycaret.utils.time_series.forecasting.models import DummyForecaster
from pycaret.utils.time_series.forecasting.pipeline import (
    _add_model_to_pipeline,
    _are_pipeline_tansformations_empty,
    _get_imputed_data,
    _transformations_present_X,
    _transformations_present_y,
)

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


def test_get_imputed_data_noexo(load_pos_data_missing):
    """Tests _get_imputed_data WITHOUT exogenous variables"""
    y = load_pos_data_missing

    exp = TSForecastingExperiment()
    FH = 12

    ###################################
    # 1: Missing Values Present ####
    ###################################
    # Due to imputation, the imputed values will not be same as original

    # 1A: Missing Values Present: Only Imputation Steps in Pipeline ----
    exp.setup(data=y, fh=FH, numeric_imputation_target="drift")
    y_imputed, X_imputed = _get_imputed_data(pipeline=exp.pipeline, y=y)
    assert not np.array_equal(y_imputed, y)
    assert X_imputed is None

    # 1B: Missing Values Present: Imputation + Other Steps in Pipeline ----
    y_imputed_expected = y_imputed.copy()
    exp.setup(
        data=y,
        fh=FH,
        numeric_imputation_target="drift",
        transform_target="exp",
    )
    y_imputed, X_imputed = _get_imputed_data(pipeline=exp.pipeline, y=y)
    assert not np.array_equal(y_imputed, y)
    assert np.array_equal(y_imputed, y_imputed_expected)
    assert X_imputed is None

    #######################################
    # 2: Missing Values not Present ####
    #######################################
    # There are no missing values, so imputation should return original values

    y_no_miss = y.copy()
    y_no_miss.fillna(10, inplace=True)

    # 2A: Missing Values not Present: No imputation step in Pipeline ----
    exp.setup(data=y_no_miss, fh=FH)
    y_imputed, X_imputed = _get_imputed_data(pipeline=exp.pipeline, y=y_no_miss)
    assert np.array_equal(y_imputed, y_no_miss)
    assert X_imputed is None

    # 2B: Missing Values not Present: Only Imputation Steps in Pipeline ----
    exp.setup(data=y_no_miss, fh=FH, numeric_imputation_target="drift")
    y_imputed, X_imputed = _get_imputed_data(pipeline=exp.pipeline, y=y_no_miss)
    assert np.array_equal(y_imputed, y_no_miss)
    assert X_imputed is None

    # 2C: Missing Values not Present: Imputation + Other Steps in Pipeline ----
    exp.setup(
        data=y_no_miss,
        fh=FH,
        numeric_imputation_target="drift",
        transform_target="exp",
    )
    y_imputed, X_imputed = _get_imputed_data(pipeline=exp.pipeline, y=y_no_miss)
    assert np.array_equal(y_imputed, y_no_miss)
    assert X_imputed is None


def test_get_imputed_data_exo(load_uni_exo_data_target_missing):
    """Tests _get_imputed_data WITH exogenous variables"""
    data, target = load_uni_exo_data_target_missing
    y = data[target]
    X = data.drop(columns=target)

    exp = TSForecastingExperiment()
    FH = 12

    ###################################
    # 1: Missing Values Present ####
    ###################################
    # Due to imputation, the imputed values will not be same as original

    # 1A: Missing Values Present: Only Imputation Steps in Pipeline ----
    exp.setup(
        data=data,
        target=target,
        fh=FH,
        numeric_imputation_target="drift",
        numeric_imputation_exogenous="drift",
    )
    y_imputed, X_imputed = _get_imputed_data(pipeline=exp.pipeline, y=y, X=X)
    assert not np.array_equal(y_imputed, y)
    assert not X_imputed.equals(X)

    # 1B: Missing Values Present: Imputation + Other Steps in Pipeline ----
    y_imputed_expected = y_imputed.copy()
    exp.setup(
        data=data,
        target=target,
        fh=FH,
        numeric_imputation_target="drift",
        numeric_imputation_exogenous="drift",
        transform_target="exp",
        transform_exogenous="exp",
    )
    y_imputed, X_imputed = _get_imputed_data(pipeline=exp.pipeline, y=y, X=X)
    assert not np.array_equal(y_imputed, y)
    assert np.array_equal(y_imputed, y_imputed_expected)
    assert not X_imputed.equals(X)

    #######################################
    # 2: Missing Values not Present ####
    #######################################
    # There are no missing values, so imputation should return original values

    data_no_miss = data.copy()
    data_no_miss.fillna(10, inplace=True)
    y_no_miss = data_no_miss[target]
    X_no_miss = data_no_miss.drop(columns=target)

    # 2A: Missing Values not Present: No imputation step in Pipeline ----
    exp.setup(data=data_no_miss, target=target, fh=FH)
    y_imputed, X_imputed = _get_imputed_data(
        pipeline=exp.pipeline, y=y_no_miss, X=X_no_miss
    )
    assert np.array_equal(y_imputed, y_no_miss)
    assert X_imputed.equals(X_no_miss)

    # 2B: Missing Values not Present: Only Imputation Steps in Pipeline ----
    exp.setup(
        data=data_no_miss,
        target=target,
        fh=FH,
        numeric_imputation_target="drift",
        numeric_imputation_exogenous="drift",
    )
    y_imputed, X_imputed = _get_imputed_data(
        pipeline=exp.pipeline, y=y_no_miss, X=X_no_miss
    )
    assert np.array_equal(y_imputed, y_no_miss)
    assert X_imputed.equals(X_no_miss)

    # 2C: Missing Values not Present: Imputation + Other Steps in Pipeline ----
    exp.setup(
        data=data_no_miss,
        target=target,
        fh=FH,
        numeric_imputation_target="drift",
        numeric_imputation_exogenous="drift",
        transform_target="exp",
        transform_exogenous="exp",
    )
    y_imputed, X_imputed = _get_imputed_data(
        pipeline=exp.pipeline, y=y_no_miss, X=X_no_miss
    )
    assert np.array_equal(y_imputed, y_no_miss)
    assert X_imputed.equals(X_no_miss)


def test_are_pipeline_tansformations_empty_noexo(load_pos_data_missing):
    """Tests _are_pipeline_tansformations_empty, _transformations_present_X, and
    _transformations_present_y WITHOUT exogenous variables"""
    y = load_pos_data_missing

    y_no_miss = y.copy()
    y_no_miss.fillna(10, inplace=True)

    exp = TSForecastingExperiment()
    FH = 12

    ###############################
    # 1: Not Empty Pipeline ####
    ###############################

    # 1A: Data has missing values ----
    exp.setup(data=y, fh=FH, numeric_imputation_target="drift")
    assert not _transformations_present_X(pipeline=exp.pipeline)
    assert _transformations_present_y(pipeline=exp.pipeline)
    assert not _are_pipeline_tansformations_empty(pipeline=exp.pipeline)

    # 1B: Data has no missing values, but y impute step added ----
    # Even though data has no missing values, imputation step is added as user has requested
    exp.setup(data=y_no_miss, fh=FH, numeric_imputation_target="drift")
    assert not _are_pipeline_tansformations_empty(pipeline=exp.pipeline)

    ###########################
    # 2: Empty Pipeline ####
    ###########################

    # 2A: No Imputation in Pipeline ----
    exp.setup(data=y_no_miss, fh=FH)
    assert not _transformations_present_X(pipeline=exp.pipeline)
    assert not _transformations_present_y(pipeline=exp.pipeline)
    assert _are_pipeline_tansformations_empty(pipeline=exp.pipeline)


def test_are_pipeline_tansformations_empty_exo(load_uni_exo_data_target_missing):
    """Tests _are_pipeline_tansformations_empty, _transformations_present_X, and
    _transformations_present_y WITH exogenous variables"""
    data, target = load_uni_exo_data_target_missing
    data_no_miss = data.copy()
    data_no_miss.fillna(10, inplace=True)

    exp = TSForecastingExperiment()
    FH = 12

    ###############################
    # 1: Not Empty Pipeline ####
    ###############################

    # 1A: Both y and X have missing values ----
    exp.setup(
        data=data,
        target=target,
        fh=FH,
        numeric_imputation_target="drift",
        numeric_imputation_exogenous="drift",
    )
    assert _transformations_present_X(pipeline=exp.pipeline)
    assert _transformations_present_y(pipeline=exp.pipeline)
    assert not _are_pipeline_tansformations_empty(pipeline=exp.pipeline)

    # 1B: Data has no missing values, but y impute step added ----
    # Even though data has no missing values, imputation step y is added as user has requested
    exp.setup(
        data=data_no_miss,
        target=target,
        fh=FH,
        numeric_imputation_target="drift",
    )
    assert not _transformations_present_X(pipeline=exp.pipeline)
    assert _transformations_present_y(pipeline=exp.pipeline)
    assert not _are_pipeline_tansformations_empty(pipeline=exp.pipeline)

    # 1C: Data has no missing values, but X impute step added ----
    # Even though data has no missing values, imputation step X is added as user has requested
    exp.setup(
        data=data_no_miss,
        target=target,
        fh=FH,
        numeric_imputation_exogenous="drift",
    )
    assert _transformations_present_X(pipeline=exp.pipeline)
    assert not _transformations_present_y(pipeline=exp.pipeline)
    assert not _are_pipeline_tansformations_empty(pipeline=exp.pipeline)

    ###########################
    # 2: Empty Pipeline ####
    ###########################

    # 2A: No Imputation in Pipeline ----
    exp.setup(data=data_no_miss, target=target, fh=FH)
    assert not _transformations_present_X(pipeline=exp.pipeline)
    assert not _transformations_present_y(pipeline=exp.pipeline)
    assert _are_pipeline_tansformations_empty(pipeline=exp.pipeline)


def test_add_model_to_pipeline_noexo(load_pos_and_neg_data):
    """Tests _add_model_to_pipeline WITHOUT exogenous variables"""
    y = load_pos_and_neg_data

    exp = TSForecastingExperiment()
    FH = 12
    model = NaiveForecaster()

    ###########################
    # 1: Empty Pipeline ####
    ###########################

    exp.setup(data=y, fh=FH)

    # -------------------------------------------------------------------------#
    # A. Final Model
    # -------------------------------------------------------------------------#

    # Check that the final model has changed ----
    assert isinstance(exp.pipeline.steps[-1][1].steps[-1][1], DummyForecaster)
    pipeline = _add_model_to_pipeline(pipeline=exp.pipeline, model=model)
    assert isinstance(pipeline.steps[-1][1].steps[-1][1], NaiveForecaster)
    assert isinstance(pipeline.steps_[-1][1].steps_[-1][1], NaiveForecaster)
    assert isinstance(pipeline.steps[-1][1].steps_[-1][1], NaiveForecaster)
    assert isinstance(pipeline.steps_[-1][1].steps[-1][1], NaiveForecaster)

    # -------------------------------------------------------------------------#
    # B. Forecasting Pipeline
    # -------------------------------------------------------------------------#

    # Check that the length of the Forecasting Pipeline has not changed ----
    assert len(exp.pipeline.steps) == len(pipeline.steps)
    assert len(exp.pipeline.steps_) == len(pipeline.steps_)

    # Check that the steps for X in the Forecasting Pipeline have not changed ----
    for i in np.arange(len(exp.pipeline.steps_)):
        assert exp.pipeline.steps[i][1].__class__ is pipeline.steps[i][1].__class__
        assert exp.pipeline.steps_[i][1].__class__ is pipeline.steps_[i][1].__class__

    # -------------------------------------------------------------------------#
    # C. Transformed Target Forecaster
    # -------------------------------------------------------------------------#

    # Check that the length of the Transformed Target Forecaster has not changed ----
    assert len(exp.pipeline.steps[-1][1].steps) == len(pipeline.steps[-1][1].steps)
    assert len(exp.pipeline.steps_[-1][1].steps_) == len(pipeline.steps_[-1][1].steps_)
    assert len(exp.pipeline.steps[-1][1].steps_) == len(pipeline.steps[-1][1].steps_)
    assert len(exp.pipeline.steps_[-1][1].steps) == len(pipeline.steps_[-1][1].steps)

    # Check that the steps for y in the Forecasting Pipeline have not changed ----
    # Check except last step which has been checked above (Dummy vs Naive)
    for i in np.arange(len(exp.pipeline.steps_[-1][1]) - 1):
        assert (
            exp.pipeline.steps[-1][1].steps[i][1].__class__
            is pipeline.steps[-1][1].steps[i][1].__class__
        )
        assert (
            exp.pipeline.steps_[-1][1].steps_[i][1].__class__
            is pipeline.steps_[-1][1].steps_[i][1].__class__
        )
        assert (
            exp.pipeline.steps[-1][1].steps_[i][1].__class__
            is pipeline.steps[-1][1].steps_[i][1].__class__
        )
        assert (
            exp.pipeline.steps_[-1][1].steps[i][1].__class__
            is pipeline.steps_[-1][1].steps[i][1].__class__
        )

    ###############################
    # 2: Not Empty Pipeline ####
    ###############################

    exp.setup(data=y, fh=FH, numeric_imputation_target="drift")

    # -------------------------------------------------------------------------#
    # A. Final Model
    # -------------------------------------------------------------------------#

    # Check that the final model has changed ----
    assert isinstance(exp.pipeline.steps[-1][1].steps[-1][1], DummyForecaster)
    pipeline = _add_model_to_pipeline(pipeline=exp.pipeline, model=model)
    assert isinstance(pipeline.steps[-1][1].steps[-1][1], NaiveForecaster)
    assert isinstance(pipeline.steps_[-1][1].steps_[-1][1], NaiveForecaster)
    assert isinstance(pipeline.steps[-1][1].steps_[-1][1], NaiveForecaster)
    assert isinstance(pipeline.steps_[-1][1].steps[-1][1], NaiveForecaster)

    # -------------------------------------------------------------------------#
    # B. Forecasting Pipeline
    # -------------------------------------------------------------------------#

    # Check that the length of the Forecasting Pipeline has not changed ----
    assert len(exp.pipeline.steps) == len(pipeline.steps)
    assert len(exp.pipeline.steps_) == len(pipeline.steps_)

    # Check that the steps for X in the Forecasting Pipeline have not changed ----
    for i in np.arange(len(exp.pipeline.steps_)):
        assert exp.pipeline.steps[i][1].__class__ is pipeline.steps[i][1].__class__
        assert exp.pipeline.steps_[i][1].__class__ is pipeline.steps_[i][1].__class__

    # -------------------------------------------------------------------------#
    # C. Transformed Target Forecaster
    # -------------------------------------------------------------------------#

    # Check that the length of the Transformed Target Forecaster has not changed ----
    assert len(exp.pipeline.steps[-1][1].steps) == len(pipeline.steps[-1][1].steps)
    assert len(exp.pipeline.steps_[-1][1].steps_) == len(pipeline.steps_[-1][1].steps_)
    assert len(exp.pipeline.steps[-1][1].steps_) == len(pipeline.steps[-1][1].steps_)
    assert len(exp.pipeline.steps_[-1][1].steps) == len(pipeline.steps_[-1][1].steps)

    # Check that the steps for y in the Forecasting Pipeline have not changed ----
    # Check except last step which has been checked above (Dummy vs Naive)
    for i in np.arange(len(exp.pipeline.steps_[-1][1]) - 1):
        assert (
            exp.pipeline.steps[-1][1].steps[i][1].__class__
            is pipeline.steps[-1][1].steps[i][1].__class__
        )
        assert (
            exp.pipeline.steps_[-1][1].steps_[i][1].__class__
            is pipeline.steps_[-1][1].steps_[i][1].__class__
        )
        assert (
            exp.pipeline.steps[-1][1].steps_[i][1].__class__
            is pipeline.steps[-1][1].steps_[i][1].__class__
        )
        assert (
            exp.pipeline.steps_[-1][1].steps[i][1].__class__
            is pipeline.steps_[-1][1].steps[i][1].__class__
        )


def test_add_model_to_pipeline_exo(load_uni_exo_data_target):
    """Tests _add_model_to_pipeline WITH exogenous variables"""
    data, target = load_uni_exo_data_target

    exp = TSForecastingExperiment()
    FH = 12
    model = NaiveForecaster()

    ###########################
    # 1: Empty Pipeline ####
    ###########################

    exp.setup(data=data, target=target, fh=FH)

    # -------------------------------------------------------------------------#
    # A. Final Model
    # -------------------------------------------------------------------------#

    # Check that the final model has changed ----
    assert isinstance(exp.pipeline.steps[-1][1].steps[-1][1], DummyForecaster)
    pipeline = _add_model_to_pipeline(pipeline=exp.pipeline, model=model)
    assert isinstance(pipeline.steps[-1][1].steps[-1][1], NaiveForecaster)
    assert isinstance(pipeline.steps_[-1][1].steps_[-1][1], NaiveForecaster)
    assert isinstance(pipeline.steps[-1][1].steps_[-1][1], NaiveForecaster)
    assert isinstance(pipeline.steps_[-1][1].steps[-1][1], NaiveForecaster)

    # -------------------------------------------------------------------------#
    # B. Forecasting Pipeline
    # -------------------------------------------------------------------------#

    # Check that the length of the Forecasting Pipeline has not changed ----
    assert len(exp.pipeline.steps) == len(pipeline.steps)
    assert len(exp.pipeline.steps_) == len(pipeline.steps_)

    # Check that the steps for X in the Forecasting Pipeline have not changed ----
    for i in np.arange(len(exp.pipeline.steps_)):
        assert exp.pipeline.steps[i][1].__class__ is pipeline.steps[i][1].__class__
        assert exp.pipeline.steps_[i][1].__class__ is pipeline.steps_[i][1].__class__

    # -------------------------------------------------------------------------#
    # C. Transformed Target Forecaster
    # -------------------------------------------------------------------------#

    # Check that the length of the Transformed Target Forecaster has not changed ----
    assert len(exp.pipeline.steps[-1][1].steps) == len(pipeline.steps[-1][1].steps)
    assert len(exp.pipeline.steps_[-1][1].steps_) == len(pipeline.steps_[-1][1].steps_)
    assert len(exp.pipeline.steps[-1][1].steps_) == len(pipeline.steps[-1][1].steps_)
    assert len(exp.pipeline.steps_[-1][1].steps) == len(pipeline.steps_[-1][1].steps)

    # Check that the steps for y in the Forecasting Pipeline have not changed ----
    # Check except last step which has been checked above (Dummy vs Naive)
    for i in np.arange(len(exp.pipeline.steps_[-1][1]) - 1):
        assert (
            exp.pipeline.steps[-1][1].steps[i][1].__class__
            is pipeline.steps[-1][1].steps[i][1].__class__
        )
        assert (
            exp.pipeline.steps_[-1][1].steps_[i][1].__class__
            is pipeline.steps_[-1][1].steps_[i][1].__class__
        )
        assert (
            exp.pipeline.steps[-1][1].steps_[i][1].__class__
            is pipeline.steps[-1][1].steps_[i][1].__class__
        )
        assert (
            exp.pipeline.steps_[-1][1].steps[i][1].__class__
            is pipeline.steps_[-1][1].steps[i][1].__class__
        )

    ###############################
    # 2: Not Empty Pipeline ####
    ###############################

    exp.setup(
        data=data,
        target=target,
        fh=FH,
        numeric_imputation_target="drift",
        numeric_imputation_exogenous="drift",
    )

    # -------------------------------------------------------------------------#
    # A. Final Model
    # -------------------------------------------------------------------------#

    # Check that the final model has changed ----
    assert isinstance(exp.pipeline.steps[-1][1].steps[-1][1], DummyForecaster)
    pipeline = _add_model_to_pipeline(pipeline=exp.pipeline, model=model)
    assert isinstance(pipeline.steps[-1][1].steps[-1][1], NaiveForecaster)
    assert isinstance(pipeline.steps_[-1][1].steps_[-1][1], NaiveForecaster)
    assert isinstance(pipeline.steps[-1][1].steps_[-1][1], NaiveForecaster)
    assert isinstance(pipeline.steps_[-1][1].steps[-1][1], NaiveForecaster)

    # -------------------------------------------------------------------------#
    # B. Forecasting Pipeline
    # -------------------------------------------------------------------------#

    # Check that the length of the Forecasting Pipeline has not changed ----
    assert len(exp.pipeline.steps) == len(pipeline.steps)
    assert len(exp.pipeline.steps_) == len(pipeline.steps_)

    # Check that the steps for X in the Forecasting Pipeline have not changed ----
    for i in np.arange(len(exp.pipeline.steps_)):
        assert exp.pipeline.steps[i][1].__class__ is pipeline.steps[i][1].__class__
        assert exp.pipeline.steps_[i][1].__class__ is pipeline.steps_[i][1].__class__

    # -------------------------------------------------------------------------#
    # C. Transformed Target Forecaster
    # -------------------------------------------------------------------------#

    # Check that the length of the Transformed Target Forecaster has not changed ----
    assert len(exp.pipeline.steps[-1][1].steps) == len(pipeline.steps[-1][1].steps)
    assert len(exp.pipeline.steps_[-1][1].steps_) == len(pipeline.steps_[-1][1].steps_)
    assert len(exp.pipeline.steps[-1][1].steps_) == len(pipeline.steps[-1][1].steps_)
    assert len(exp.pipeline.steps_[-1][1].steps) == len(pipeline.steps_[-1][1].steps)

    # Check that the steps for y in the Forecasting Pipeline have not changed ----
    # Check except last step which has been checked above (Dummy vs Naive)
    for i in np.arange(len(exp.pipeline.steps_[-1][1]) - 1):
        assert (
            exp.pipeline.steps[-1][1].steps[i][1].__class__
            is pipeline.steps[-1][1].steps[i][1].__class__
        )
        assert (
            exp.pipeline.steps_[-1][1].steps_[i][1].__class__
            is pipeline.steps_[-1][1].steps_[i][1].__class__
        )
        assert (
            exp.pipeline.steps[-1][1].steps_[i][1].__class__
            is pipeline.steps[-1][1].steps_[i][1].__class__
        )
        assert (
            exp.pipeline.steps_[-1][1].steps[i][1].__class__
            is pipeline.steps_[-1][1].steps[i][1].__class__
        )
