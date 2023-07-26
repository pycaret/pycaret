"""Module to test time_series functionality
"""
import os

import numpy as np
import pytest
import sktime
from packaging import version
from sktime.forecasting.compose import ForecastingPipeline, TransformedTargetForecaster
from time_series_test_utils import (
    _IMPUTE_METHODS_STR,
    _SCALE_METHODS,
    _TRANSFORMATION_METHODS,
    _TRANSFORMATION_METHODS_NO_NEG,
    _return_model_names_for_missing_data,
)

from pycaret.time_series import TSForecastingExperiment

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")
os.environ["PYCARET_TESTING"] = "1"

########################################################
# TODO: Test compare_models with missing values ####
########################################################

##############################
# Functions Start Here ####
##############################

# NOTE: Fixtures can not be used to parameterize tests
# https://stackoverflow.com/questions/52764279/pytest-how-to-parametrize-a-test-with-a-list-that-is-returned-from-a-fixture
# Hence, we have to create functions and create the parameterized list first
# (must happen during collect phase) before passing it to mark.parameterize.

_model_names_for_missing_data = _return_model_names_for_missing_data()

############################
# Functions End Here ####
############################


##########################
# Tests Start Here ####
##########################


def test_pipeline_no_exo_but_exo_steps(load_pos_and_neg_data):
    """Tests preprocessing pipeline data types without exogenous variables"""
    data = load_pos_and_neg_data

    exp = TSForecastingExperiment()

    # Make sure that no exogenous steps are added to the pipeline when
    # there is no exogenous data
    exp.setup(data=data, numeric_imputation_exogenous=True)
    assert len(exp.pipeline.steps) == 1

    exp.setup(data=data, numeric_imputation_exogenous=True, transform_exogenous="cos")
    assert len(exp.pipeline.steps) == 1

    exp.setup(data=data, numeric_imputation_exogenous=True, scale_exogenous="min-max")
    assert len(exp.pipeline.steps) == 1

    exp.setup(
        data=data,
        numeric_imputation_exogenous=True,
        transform_exogenous="cos",
        scale_exogenous="min-max",
    )
    assert len(exp.pipeline.steps) == 1


def test_pipeline_types_no_exo(load_pos_and_neg_data):
    """Tests preprocessing pipeline data types without exogenous variables"""
    data = load_pos_and_neg_data

    exp = TSForecastingExperiment()

    # Default
    exp.setup(data=data)
    assert isinstance(exp.pipeline, ForecastingPipeline)
    assert isinstance(exp.pipeline.steps[-1][1], TransformedTargetForecaster)

    # Transform Target only
    exp.setup(data=data, numeric_imputation_target=True)
    assert isinstance(exp.pipeline, ForecastingPipeline)
    assert isinstance(exp.pipeline.steps[-1][1], TransformedTargetForecaster)

    # Transform Exogenous only (but no exogenous present)
    exp.setup(data=data, numeric_imputation_exogenous=True)
    assert isinstance(exp.pipeline, ForecastingPipeline)
    assert isinstance(exp.pipeline.steps[-1][1], TransformedTargetForecaster)

    # Transform Exogenous & Target (but no exogenous present)
    exp.setup(
        data=data,
        numeric_imputation_target=True,
        numeric_imputation_exogenous=True,
    )
    assert isinstance(exp.pipeline, ForecastingPipeline)
    assert isinstance(exp.pipeline.steps[-1][1], TransformedTargetForecaster)

    # No preprocessing (still sets empty pipeline internally)
    exp.setup(data=data)
    assert isinstance(exp.pipeline, ForecastingPipeline)
    assert isinstance(exp.pipeline.steps[-1][1], TransformedTargetForecaster)


def test_pipeline_types_exo(load_uni_exo_data_target):
    """Tests preprocessing pipeline data types with exogenous variables"""

    data, target = load_uni_exo_data_target

    exp = TSForecastingExperiment()

    # Default
    exp.setup(data=data, target=target)
    assert isinstance(exp.pipeline, ForecastingPipeline)
    assert isinstance(exp.pipeline.steps[-1][1], TransformedTargetForecaster)

    # Transform Target only
    exp.setup(
        data=data,
        target=target,
        numeric_imputation_target=True,
    )
    assert isinstance(exp.pipeline, ForecastingPipeline)
    assert isinstance(exp.pipeline.steps[-1][1], TransformedTargetForecaster)

    # Transform Exogenous only
    exp.setup(
        data=data,
        target=target,
        numeric_imputation_exogenous=True,
    )
    assert isinstance(exp.pipeline, ForecastingPipeline)
    assert isinstance(exp.pipeline.steps[-1][1], TransformedTargetForecaster)

    # Transform Exogenous & Target
    exp.setup(
        data=data,
        target=target,
        numeric_imputation_target=True,
        numeric_imputation_exogenous=True,
    )
    assert isinstance(exp.pipeline, ForecastingPipeline)
    assert isinstance(exp.pipeline.steps[-1][1], TransformedTargetForecaster)

    # No preprocessing (still sets empty pipeline internally)
    exp.setup(data=data, target=target)
    assert isinstance(exp.pipeline, ForecastingPipeline)
    assert isinstance(exp.pipeline.steps[-1][1], TransformedTargetForecaster)


def test_preprocess_setup_raises_missing_no_exo(load_pos_and_neg_data_missing):
    """Tests setup conditions that raise errors due to missing data
    Univariate without exogenous variables"""
    data = load_pos_and_neg_data_missing

    exp = TSForecastingExperiment()

    with pytest.raises(ValueError) as errmsg:
        exp.setup(data=data)
    exceptionmsg = errmsg.value.args[0]
    assert "Please enable imputation to proceed" in exceptionmsg

    with pytest.raises(ValueError) as errmsg:
        exp.setup(data=data, numeric_imputation_target=None)
    exceptionmsg = errmsg.value.args[0]
    assert "Please enable imputation to proceed" in exceptionmsg


def test_preprocess_setup_raises_missing_exo(load_uni_exo_data_target_missing):
    """Tests setup conditions that raise errors due to missing data
    Univariate with exogenous variables"""

    data, target = load_uni_exo_data_target_missing

    exp = TSForecastingExperiment()
    with pytest.raises(ValueError) as errmsg:
        exp.setup(data=data, target=target)
    exceptionmsg = errmsg.value.args[0]
    assert "Please enable imputation to proceed" in exceptionmsg

    exp = TSForecastingExperiment()
    with pytest.raises(ValueError) as errmsg:
        exp.setup(
            data=data,
            target=target,
            numeric_imputation_target=None,
        )
    exceptionmsg = errmsg.value.args[0]
    assert "Please enable imputation to proceed" in exceptionmsg

    exp = TSForecastingExperiment()
    with pytest.raises(ValueError) as errmsg:
        exp.setup(
            data=data,
            target=target,
            numeric_imputation_exogenous=None,
        )
    exceptionmsg = errmsg.value.args[0]
    assert "Please enable imputation to proceed" in exceptionmsg


@pytest.mark.parametrize("method", _TRANSFORMATION_METHODS_NO_NEG)
def test_preprocess_setup_raises_negative_no_exo(load_pos_and_neg_data, method):
    """Tests setup conditions that raise errors due to negative values before
    transformatons. Univariate without exogenous variables"""

    continue_ = _continue_negative_value_checks(method=method)
    if continue_:
        data = load_pos_and_neg_data

        exp = TSForecastingExperiment()

        with pytest.raises(ValueError) as errmsg:
            exp.setup(data=data, transform_target=method)
        exceptionmsg = errmsg.value.args[0]
        # The first message is given when then the transformation produced NA values
        # and the underlying model can handle missing data (in this case, the sktime
        # checks pass but pycaret checks fail)
        # The second message is given the transformation produces NA values and the
        # underlying model can not handle missing data (in this case, the sktime checks
        # fail and the pycaret checks are not reached.)
        assert (
            "This can happen when you have negative and/or zero values in the data"
            or "DummyForecaster cannot handle missing data (nans), but y passed contained missing data"
            in exceptionmsg
        )


@pytest.mark.parametrize("method", _TRANSFORMATION_METHODS_NO_NEG)
def test_preprocess_setup_raises_negative_exo(load_uni_exo_data_target, method):
    """Tests setup conditions that raise errors due to negative values before
    transformations. Univariate with exogenous variables"""

    continue_ = _continue_negative_value_checks(method=method)
    if continue_:
        data, target = load_uni_exo_data_target

        exp = TSForecastingExperiment()

        # Transform Target ----
        with pytest.raises(ValueError) as errmsg:
            exp.setup(
                data=data,
                target=target,
                transform_target=method,
            )
        exceptionmsg = errmsg.value.args[0]
        # The first message is given when then the transformation produced NA values
        # and the underlying model can handle missing data (in this case, the sktime
        # checks pass but pycaret checks fail)
        # The second message is given the transformation produces NA values and the
        # underlying model can not handle missing data (in this case, the sktime checks
        # fail and the pycaret checks are not reached.)
        assert (
            "This can happen when you have negative and/or zero values in the data"
            or "DummyForecaster cannot handle missing data (nans), but y passed contained missing data"
            in exceptionmsg
        )

        # Transform Exogenous ----
        with pytest.raises(ValueError) as errmsg:
            exp.setup(
                data=data,
                target=target,
                transform_exogenous=method,
            )
        exceptionmsg = errmsg.value.args[0]
        # The first message is given when then the transformation produced NA values
        # and the underlying model can handle missing data (in this case, the sktime
        # checks pass but pycaret checks fail)
        # The second message is given the transformation produces NA values and the
        # underlying model can not handle missing data (in this case, the sktime checks
        # fail and the pycaret checks are not reached.)
        assert (
            "This can happen when you have negative and/or zero values in the data"
            or "DummyForecaster cannot handle missing data (nans), but y passed contained missing data"
            in exceptionmsg
        )


@pytest.mark.parametrize("model_name", _model_names_for_missing_data)
def test_pipeline_works_no_exo(load_pos_and_neg_data_missing, model_name):
    """Tests that the pipeline works for various operations for Univariate
    forecasting without exogenous variables"""
    data = load_pos_and_neg_data_missing

    exp = TSForecastingExperiment()
    FH = 12
    exp.setup(data=data, fh=FH, numeric_imputation_target="drift")

    assert exp.get_config("y").isna().sum() > 0
    assert exp.get_config("y_transformed").isna().sum() == 0

    model = exp.create_model(model_name)
    preds = exp.predict_model(model)
    assert len(preds) == FH
    plot_data = exp.plot_model(model, return_data=True)
    assert isinstance(plot_data, dict)

    tuned = exp.tune_model(model)
    preds = exp.predict_model(tuned)
    assert len(preds) == FH
    plot_data = exp.plot_model(tuned, return_data=True)
    assert isinstance(plot_data, dict)

    final = exp.finalize_model(tuned)
    preds = exp.predict_model(final)
    assert len(preds) == FH
    plot_data = exp.plot_model(final, return_data=True)
    assert isinstance(plot_data, dict)


@pytest.mark.parametrize("model_name", _model_names_for_missing_data)
def test_pipeline_works_exo(load_uni_exo_data_target_missing, model_name):
    """Tests that the pipeline works for various operations for Univariate
    forecasting with exogenous variables"""
    data, target = load_uni_exo_data_target_missing

    exp = TSForecastingExperiment()
    FH = 12
    exp.setup(
        data=data,
        target=target,
        fh=FH,
        numeric_imputation_target="drift",
        numeric_imputation_exogenous="drift",
        enforce_exogenous=False,
    )

    assert exp.get_config("y").isna().sum() > 0
    assert exp.get_config("X").isna().sum().sum() > 0
    assert exp.get_config("y_transformed").isna().sum() == 0
    assert exp.get_config("X_transformed").isna().sum().sum() == 0

    model = exp.create_model(model_name)
    preds = exp.predict_model(model)
    assert len(preds) == FH
    plot_data = exp.plot_model(model, return_data=True)
    assert isinstance(plot_data, dict)

    tuned = exp.tune_model(model)
    preds = exp.predict_model(tuned)
    assert len(preds) == FH
    plot_data = exp.plot_model(tuned, return_data=True)
    assert isinstance(plot_data, dict)

    _ = exp.finalize_model(tuned)
    # # Exogenous models predictions and plots after finalizing will need future X
    # # values. Hence disabling this test.
    # preds = exp.predict_model(final)
    # assert len(preds) == FH
    # plot_data = exp.plot_model(final, return_data=True)
    # assert isinstance(plot_data, dict)


@pytest.mark.parametrize("method", _IMPUTE_METHODS_STR)
def test_impute_str_no_exo(load_pos_and_neg_data_missing, method):
    """Tests Imputation methods (str) for Univariate forecasting without
    exogenous variables"""
    data = load_pos_and_neg_data_missing

    exp = TSForecastingExperiment()
    FH = 12
    exp.setup(data=data, fh=FH, numeric_imputation_target=method)

    # Due to preprocessing not all 'y' values should be the same
    assert not np.all(exp.y.values == exp.y_transformed.values)
    assert not np.all(exp.y_train.values == exp.y_train_transformed.values)
    assert not np.all(exp.y_test.values == exp.y_test_transformed.values)

    # X None. No preprocessing is applied to it.
    assert exp.X is None
    assert exp.X_transformed is None
    assert exp.X_train is None
    assert exp.X_train_transformed is None
    assert exp.X_test is None
    assert exp.X_test_transformed is None


def test_impute_num_no_exo(load_pos_and_neg_data_missing):
    """Tests Imputation methods (numeric) methods for Univariate forecasting
    without exogenous variables"""
    data = load_pos_and_neg_data_missing
    impute_val = data.max() + 1  # outside the range of data

    exp = TSForecastingExperiment()
    FH = 12
    exp.setup(data=data, fh=FH, numeric_imputation_target=impute_val)

    y_check = [impute_val] * len(exp.y.values)
    y_train_check = [impute_val] * len(exp.y_train.values)
    y_test_check = [impute_val] * len(exp.y_test.values)

    # Due to preprocessing not all 'y' values should be the same
    assert not np.any(exp.y.values == y_check)
    assert np.any(exp.y_transformed.values == y_check)
    assert not np.any(exp.y_train.values == y_train_check)
    assert np.any(exp.y_train_transformed.values == y_train_check)
    assert not np.any(exp.y_test.values == y_test_check)
    assert np.any(exp.y_test_transformed.values == y_test_check)

    # X None. No preprocessing is applied to it.
    assert exp.X is None
    assert exp.X_transformed is None
    assert exp.X_train is None
    assert exp.X_train_transformed is None
    assert exp.X_test is None
    assert exp.X_test_transformed is None


@pytest.mark.parametrize("method", _TRANSFORMATION_METHODS)
def test_transform_no_exo(load_pos_data, method):
    """Tests Transformation methods for Univariate forecasting without exogenous
    variables"""
    data = load_pos_data

    exp = TSForecastingExperiment()
    FH = 12
    exp.setup(data=data, fh=FH, transform_target=method)

    # Due to preprocessing not all 'y' values should be the same
    assert not np.all(exp.y.values == exp.y_transformed.values)
    assert not np.all(exp.y_train.values == exp.y_train_transformed.values)
    assert not np.all(exp.y_test.values == exp.y_test_transformed.values)

    # X None. No preprocessing is applied to it.
    assert exp.X is None
    assert exp.X_transformed is None
    assert exp.X_train is None
    assert exp.X_train_transformed is None
    assert exp.X_test is None
    assert exp.X_test_transformed is None


@pytest.mark.parametrize("method", _SCALE_METHODS)
def test_scale_no_exo(load_pos_data, method):
    """Tests Scaling methods for Univariate forecasting without exogenous
    variables"""
    data = load_pos_data

    exp = TSForecastingExperiment()
    FH = 12
    exp.setup(data=data, fh=FH, scale_target=method)

    # Due to preprocessing not all 'y' values should be the same
    assert not np.all(exp.y.values == exp.y_transformed.values)
    assert not np.all(exp.y_train.values == exp.y_train_transformed.values)
    assert not np.all(exp.y_test.values == exp.y_test_transformed.values)

    # X None. No preprocessing is applied to it.
    assert exp.X is None
    assert exp.X_transformed is None
    assert exp.X_train is None
    assert exp.X_train_transformed is None
    assert exp.X_test is None
    assert exp.X_test_transformed is None


@pytest.mark.parametrize("method", _IMPUTE_METHODS_STR)
def test_impute_str_exo(load_uni_exo_data_target_missing, method):
    """Tests Imputation methods (str) for Univariate forecasting with
    exogenous variables"""
    data, target = load_uni_exo_data_target_missing

    exp = TSForecastingExperiment()
    FH = 12
    exp.setup(
        data=data,
        target=target,
        fh=FH,
        numeric_imputation_target=method,
        numeric_imputation_exogenous=method,
    )

    # Due to preprocessing not all 'y' values should be the same
    assert not np.all(exp.y.values == exp.y_transformed.values)
    assert not np.all(exp.y_train.values == exp.y_train_transformed.values)
    assert not np.all(exp.y_test.values == exp.y_test_transformed.values)

    # Due to preprocessing not all 'X' values should be the same
    assert not np.all(exp.X.values == exp.X_transformed.values)
    assert not np.all(exp.X_train.values == exp.X_train_transformed.values)
    assert not np.all(exp.X_test.values == exp.X_test_transformed.values)


def test_impute_num_exo(load_uni_exo_data_target_missing):
    """Tests Imputation methods (numeric) methods for Univariate forecasting
    with exogenous variables"""
    data, target = load_uni_exo_data_target_missing
    impute_val = data.max().max() + 1  # outside the range of data

    exp = TSForecastingExperiment()
    FH = 12
    exp.setup(
        data=data,
        target=target,
        fh=FH,
        numeric_imputation_target=impute_val,
        numeric_imputation_exogenous=impute_val,
    )

    y_check = [impute_val] * len(exp.y.values)
    y_train_check = [impute_val] * len(exp.y_train.values)
    y_test_check = [impute_val] * len(exp.y_test.values)

    X_check = [([impute_val] * exp.X.shape[1]) for _ in range(len(exp.X))]
    X_train_check = [
        ([impute_val] * exp.X_train.shape[1]) for _ in range(len(exp.X_train))
    ]
    X_test_check = [
        ([impute_val] * exp.X_test.shape[1]) for _ in range(len(exp.X_test))
    ]

    # Due to preprocessing not all 'y' values should be the same
    assert not np.any(exp.y.values == y_check)
    assert np.any(exp.y_transformed.values == y_check)
    assert not np.any(exp.y_train.values == y_train_check)
    assert np.any(exp.y_train_transformed.values == y_train_check)
    assert not np.any(exp.y_test.values == y_test_check)
    assert np.any(exp.y_test_transformed.values == y_test_check)

    # Due to preprocessing not all 'X' values should be the same
    assert not np.any(exp.X.values == X_check)
    assert np.any(exp.X_transformed.values == X_check)
    assert not np.any(exp.X_train.values == X_train_check)
    assert np.any(exp.X_train_transformed.values == X_train_check)
    assert not np.any(exp.X_test.values == X_test_check)
    assert np.any(exp.X_test_transformed.values == X_test_check)


@pytest.mark.parametrize("method", _TRANSFORMATION_METHODS)
def test_transform_exo(load_uni_exo_data_target_positive, method):
    """Tests Transformation methods for Univariate forecasting with exogenous variables"""
    data, target = load_uni_exo_data_target_positive

    exp = TSForecastingExperiment()
    FH = 12
    exp.setup(
        data=data,
        target=target,
        fh=FH,
        transform_target=method,
        transform_exogenous=method,
    )

    # Due to preprocessing not all 'y' values should be the same
    assert not np.all(exp.y.values == exp.y_transformed.values)
    assert not np.all(exp.y_train.values == exp.y_train_transformed.values)
    assert not np.all(exp.y_test.values == exp.y_test_transformed.values)

    # Due to preprocessing not all 'X' values should be the same
    assert not np.all(exp.X.values == exp.X_transformed.values)
    assert not np.all(exp.X_train.values == exp.X_train_transformed.values)
    assert not np.all(exp.X_test.values == exp.X_test_transformed.values)


@pytest.mark.parametrize("method", _SCALE_METHODS)
def test_scale_exo(load_uni_exo_data_target, method):
    """Tests Scaling methods for Univariate forecasting with exogenous variables"""
    data, target = load_uni_exo_data_target

    exp = TSForecastingExperiment()
    FH = 12
    exp.setup(
        data=data,
        target=target,
        fh=FH,
        scale_target=method,
        scale_exogenous=method,
    )

    # Due to preprocessing not all 'y' values should be the same
    assert not np.all(exp.y.values == exp.y_transformed.values)
    assert not np.all(exp.y_train.values == exp.y_train_transformed.values)
    assert not np.all(exp.y_test.values == exp.y_test_transformed.values)

    # Due to preprocessing not all 'X' values should be the same
    assert not np.all(exp.X.values == exp.X_transformed.values)
    assert not np.all(exp.X_train.values == exp.X_train_transformed.values)
    assert not np.all(exp.X_test.values == exp.X_test_transformed.values)


def test_pipeline_after_finalizing(load_pos_and_neg_data_missing):
    """After finalizing the model, the data memory in the Forecasting Pipeline
    must match with the memory in the model used in the pipeline (last step of pipeline)
    """
    data = load_pos_and_neg_data_missing

    exp = TSForecastingExperiment()
    FH = 12
    exp.setup(data=data, fh=FH, numeric_imputation_target="drift")

    model = exp.create_model("exp_smooth")
    final = exp.finalize_model(model)

    exp.save_model(final, "my_model")
    loaded_model = exp.load_model("my_model")

    # Check if pipeline data index (ForecastingPipeline) matches up with
    # the actual model data
    assert len(loaded_model._y.index) == len(
        loaded_model.steps[-1][1].steps[-1][1]._y.index
    )
    assert np.array_equal(
        loaded_model._y.index, loaded_model.steps[-1][1].steps[-1][1]._y.index
    )


def test_no_transform_noexo(load_pos_and_neg_data_missing):
    """
    NOTE: VERY IMPORTANT TEST ----

    Test to make sure that when modeling univariate data WITHOUT exogenous
    variables, if there is no transformation in setup, then

    (1A) y_train_imputed = y_train_transformed
    (1B) X_train_imputed = X_train_transformed = None
    (2A) y_test_imputed = y_test_transformed
    (2B) X_test_imputed = X_test_transformed = None
    (3A) y_imputed = y_transformed
    (2B) X_imputed = X_transformed = None

    Also: When imputing a dataset, only values in the past should be used
    (not any future values). i.e.
        (4) Imputed values in train should not be equal to imputed values in test.
        (5) Imputed values in test should be equal to imputed values in complete
            dataset (train + test)
    """
    data = load_pos_and_neg_data_missing

    exp = TSForecastingExperiment()
    FH = 12
    exp.setup(data=data, fh=FH, numeric_imputation_target="mean")

    # Tests 1A, 2A, and 3A ----
    y_train_imputed = exp._get_y_data(split="train", data_type="imputed")
    y_test_imputed = exp._get_y_data(split="test", data_type="imputed")
    y_imputed = exp._get_y_data(split="all", data_type="imputed")
    assert np.array_equal(y_train_imputed, exp.y_train_transformed)
    assert np.array_equal(y_test_imputed, exp.y_test_transformed)
    assert np.array_equal(y_imputed, exp.y_transformed)

    # Tests 1B, 2B, and 3B ----
    X_train_imputed = exp._get_X_data(split="train", data_type="imputed")
    X_test_imputed = exp._get_X_data(split="test", data_type="imputed")
    X_imputed = exp._get_X_data(split="all", data_type="imputed")
    assert X_train_imputed is None
    assert exp.X_train_transformed is None
    assert X_test_imputed is None
    assert exp.X_test_transformed is None
    assert X_imputed is None
    assert exp.X_transformed is None

    # Tests 4, and 5 ----
    missing_index_train = exp.y_train.index[exp.y_train.isna()]
    missing_index_test = exp.y_test.index[exp.y_test.isna()]

    # Test 4 ----
    missing_imputed_data_train = y_train_imputed.loc[missing_index_train]
    missing_imputed_data_test = y_test_imputed.loc[missing_index_test]
    # Just checking first value
    assert missing_imputed_data_train.iloc[0] != missing_imputed_data_test.iloc[0]

    # Test 5 ----
    missing_imputed_data_all_train = y_imputed.loc[missing_index_train]
    # Just checking first value
    assert missing_imputed_data_test.iloc[0] == missing_imputed_data_all_train.iloc[0]


def test_no_transform_exo(load_uni_exo_data_target_missing):
    """
    NOTE: VERY IMPORTANT TEST ----

    Test to make sure that when modeling univariate data WITH exogenous
    variables, if there is no transformation in setup, then

    (1A) y_train_imputed = y_train_transformed
    (1B) X_train_imputed = X_train_transformed
    (2A) y_test_imputed = y_test_transformed
    (2B) X_test_imputed = X_test_transformed
    (3A) y_imputed = y_transformed
    (2B) X_imputed = X_transformed

    Also: When imputing a dataset, only values in the past should be used
    (not any future values). i.e.
        (4) Imputed values in train should not be equal to imputed values in test.
        (5) Imputed values in test should be equal to imputed values in complete
            dataset (train + test)
    """
    data, target = load_uni_exo_data_target_missing

    exp = TSForecastingExperiment()
    FH = 12
    exp.setup(
        data=data,
        target=target,
        fh=FH,
        numeric_imputation_target="mean",
        numeric_imputation_exogenous="mean",
    )

    # Tests 1A, 2A, and 3A ----
    y_train_imputed = exp._get_y_data(split="train", data_type="imputed")
    y_test_imputed = exp._get_y_data(split="test", data_type="imputed")
    y_imputed = exp._get_y_data(split="all", data_type="imputed")
    assert np.array_equal(y_train_imputed, exp.y_train_transformed)
    assert np.array_equal(y_test_imputed, exp.y_test_transformed)
    assert np.array_equal(y_imputed, exp.y_transformed)

    # Tests 1B, 2B, and 3B ----
    X_train_imputed = exp._get_X_data(split="train", data_type="imputed")
    X_test_imputed = exp._get_X_data(split="test", data_type="imputed")
    X_imputed = exp._get_X_data(split="all", data_type="imputed")

    assert exp.X_train_transformed.equals(X_train_imputed)
    assert exp.X_test_transformed.equals(X_test_imputed)
    assert exp.X_transformed.equals(X_imputed)

    ################################
    # Tests 4, and 5 (for y) ----
    ################################
    missing_index_train = exp.y_train.index[exp.y_train.isna()]
    missing_index_test = exp.y_test.index[exp.y_test.isna()]

    # Test 4 ----
    missing_imputed_data_train = y_train_imputed.loc[missing_index_train]
    missing_imputed_data_test = y_test_imputed.loc[missing_index_test]
    # Just checking first value
    assert missing_imputed_data_train.iloc[0] != missing_imputed_data_test.iloc[0]

    # Test 5 ----
    missing_imputed_data_all_train = y_imputed.loc[missing_index_train]
    # Just checking first value
    assert missing_imputed_data_test.iloc[0] == missing_imputed_data_all_train.iloc[0]

    ################################
    # Tests 4, and 5 (for X) ----
    ################################
    # Input is created such that all values in row will be nan
    missing_index_train = exp.X_train.index[exp.X_train.isna().all(axis=1)]
    missing_index_test = exp.X_test.index[exp.X_test.isna().all(axis=1)]

    # Test 4 ----
    missing_imputed_data_train = X_train_imputed.loc[missing_index_train]
    missing_imputed_data_test = X_test_imputed.loc[missing_index_test]
    # Just checking first row (all values in row would be missing)
    assert not missing_imputed_data_train.iloc[0].equals(
        missing_imputed_data_test.iloc[0]
    )

    # Test 5 ----
    missing_imputed_data_all_train = X_imputed.loc[missing_index_train]
    # Just checking first row (all values in row would be missing)
    assert missing_imputed_data_test.iloc[0].equals(
        missing_imputed_data_all_train.iloc[0]
    )


def _continue_negative_value_checks(method):
    """Checks if the negative value checks should be continued"""
    continue_ = True

    # Negative values are handled in Boc Cox Transformer after sktime 0.20.1
    # https://github.com/sktime/sktime/pull/4770
    if method == "box-cox" and version.parse(sktime.__version__) >= version.parse(
        "0.20.1"
    ):
        continue_ = False

    return continue_
