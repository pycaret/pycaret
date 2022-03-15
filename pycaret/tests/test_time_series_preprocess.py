"""Module to test time_series functionality
"""
import pytest
import numpy as np

# from sktime.forecasting.compose import ForecastingPipeline
from pycaret.utils.time_series.forecasting.pipeline import PyCaretForecastingPipeline
from sktime.forecasting.compose import TransformedTargetForecaster

from .time_series_test_utils import (
    _return_model_names_for_missing_data,
    _IMPUTE_METHODS_STR,
    _TRANSFORMATION_METHODS,
    _TRANSFORMATION_METHODS_NO_NEG,
    _SCALE_METHODS,
)

from pycaret.time_series import TSForecastingExperiment


pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

########################################################
##### TODO: Test compare_models with missing values ####
########################################################

##############################
#### Functions Start Here ####
##############################

# NOTE: Fixtures can not be used to parameterize tests
# https://stackoverflow.com/questions/52764279/pytest-how-to-parametrize-a-test-with-a-list-that-is-returned-from-a-fixture
# Hence, we have to create functions and create the parameterized list first
# (must happen during collect phase) before passing it to mark.parameterize.

_model_names_for_missing_data = _return_model_names_for_missing_data()

############################
#### Functions End Here ####
############################


##########################
#### Tests Start Here ####
##########################


def test_pipeline_types_no_exo(load_pos_and_neg_data):
    """Tests preprocessing pipeline data types without exogenous variables"""
    data = load_pos_and_neg_data  # _check_data_for_prophet(name, load_pos_and_neg_data)

    # # Simulate misssing values
    # data[10:20] = np.nan

    exp = TSForecastingExperiment()

    #### Default
    exp.setup(data=data)
    assert isinstance(exp.pipeline, PyCaretForecastingPipeline)
    assert isinstance(exp.pipeline.steps[-1][1], TransformedTargetForecaster)

    #### Transform Target only
    exp.setup(data=data, preprocess=True, numeric_imputation_target=True)
    assert isinstance(exp.pipeline, PyCaretForecastingPipeline)
    assert isinstance(exp.pipeline.steps[-1][1], TransformedTargetForecaster)

    #### Transform Exogenous only (but no exogenous present)
    exp.setup(data=data, preprocess=True, numeric_imputation_exogenous=True)
    assert isinstance(exp.pipeline, PyCaretForecastingPipeline)
    assert isinstance(exp.pipeline.steps[-1][1], TransformedTargetForecaster)

    #### Transform Exogenous & Target (but no exogenous present)
    exp.setup(
        data=data,
        preprocess=True,
        numeric_imputation_target=True,
        numeric_imputation_exogenous=True,
    )
    assert isinstance(exp.pipeline, PyCaretForecastingPipeline)
    assert isinstance(exp.pipeline.steps[-1][1], TransformedTargetForecaster)

    # No preprocessing (still sets empty pipeline internally)
    exp.setup(data=data, preprocess=False)
    assert isinstance(exp.pipeline, PyCaretForecastingPipeline)
    assert isinstance(exp.pipeline.steps[-1][1], TransformedTargetForecaster)


def test_pipeline_types_exo(load_uni_exo_data_target):
    """Tests preprocessing pipeline data types with exogenous variables"""

    data, target = load_uni_exo_data_target

    exp = TSForecastingExperiment()

    #### Default
    exp.setup(data=data, target=target, seasonal_period=4)
    assert isinstance(exp.pipeline, PyCaretForecastingPipeline)
    assert isinstance(exp.pipeline.steps[-1][1], TransformedTargetForecaster)

    #### Transform Target only
    exp.setup(
        data=data,
        target=target,
        seasonal_period=4,
        preprocess=True,
        numeric_imputation_target=True,
    )
    assert isinstance(exp.pipeline, PyCaretForecastingPipeline)
    assert isinstance(exp.pipeline.steps[-1][1], TransformedTargetForecaster)

    #### Transform Exogenous only
    exp.setup(
        data=data,
        target=target,
        seasonal_period=4,
        preprocess=True,
        numeric_imputation_exogenous=True,
    )
    assert isinstance(exp.pipeline, PyCaretForecastingPipeline)
    assert isinstance(exp.pipeline.steps[-1][1], TransformedTargetForecaster)

    #### Transform Exogenous & Target
    exp.setup(
        data=data,
        target=target,
        seasonal_period=4,
        preprocess=True,
        numeric_imputation_target=True,
        numeric_imputation_exogenous=True,
    )
    assert isinstance(exp.pipeline, PyCaretForecastingPipeline)
    assert isinstance(exp.pipeline.steps[-1][1], TransformedTargetForecaster)

    # No preprocessing (still sets empty pipeline internally)
    exp.setup(data=data, target=target, seasonal_period=4, preprocess=False)
    assert isinstance(exp.pipeline, PyCaretForecastingPipeline)
    assert isinstance(exp.pipeline.steps[-1][1], TransformedTargetForecaster)


def test_preprocess_setup_raises_missing_no_exo(load_pos_and_neg_data_missing):
    """Tests setup conditions that raise errors due to missing data
    Univariate without exogenous variables"""
    data = load_pos_and_neg_data_missing

    exp = TSForecastingExperiment()

    with pytest.raises(ValueError) as errmsg:
        exp.setup(data=data, preprocess=False)
    exceptionmsg = errmsg.value.args[0]
    assert "Please enable imputation to proceed" in exceptionmsg

    with pytest.raises(ValueError) as errmsg:
        exp.setup(data=data, preprocess=True, numeric_imputation_target=None)
    exceptionmsg = errmsg.value.args[0]
    assert "Please enable imputation to proceed" in exceptionmsg


def test_preprocess_setup_raises_missing_exo(load_uni_exo_data_target_missing):
    """Tests setup conditions that raise errors due to missing data
    Univariate with exogenous variables"""

    data, target = load_uni_exo_data_target_missing

    exp = TSForecastingExperiment()
    with pytest.raises(ValueError) as errmsg:
        exp.setup(data=data, target=target, seasonal_period=4, preprocess=False)
    exceptionmsg = errmsg.value.args[0]
    assert "Please enable imputation to proceed" in exceptionmsg

    exp = TSForecastingExperiment()
    with pytest.raises(ValueError) as errmsg:
        exp.setup(
            data=data,
            target=target,
            seasonal_period=4,
            preprocess=True,
            numeric_imputation_target=None,
        )
    exceptionmsg = errmsg.value.args[0]
    assert "Please enable imputation to proceed" in exceptionmsg

    exp = TSForecastingExperiment()
    with pytest.raises(ValueError) as errmsg:
        exp.setup(
            data=data,
            target=target,
            seasonal_period=4,
            preprocess=True,
            numeric_imputation_exogenous=None,
        )
    exceptionmsg = errmsg.value.args[0]
    assert "Please enable imputation to proceed" in exceptionmsg


@pytest.mark.parametrize("method", _TRANSFORMATION_METHODS_NO_NEG)
def test_preprocess_setup_raises_negative_no_exo(load_pos_and_neg_data, method):
    """Tests setup conditions that raise errors due to negative values before
    transformatons. Univariate without exogenous variables"""
    data = load_pos_and_neg_data

    exp = TSForecastingExperiment()

    with pytest.raises(ValueError) as errmsg:
        exp.setup(data=data, transform_target=method)
    exceptionmsg = errmsg.value.args[0]
    assert (
        "This can happen when you have negative and/or zero values in the data"
        in exceptionmsg
    )


@pytest.mark.parametrize("method", _TRANSFORMATION_METHODS_NO_NEG)
def test_preprocess_setup_raises_negative_exo(load_uni_exo_data_target, method):
    """Tests setup conditions that raise errors due to negative values before
    transformatons. Univariate with exogenous variables"""
    data, target = load_uni_exo_data_target

    exp = TSForecastingExperiment()

    with pytest.raises(ValueError) as errmsg:
        exp.setup(
            data=data,
            target=target,
            seasonal_period=4,
            preprocess=True,
            transform_target=method,
        )
    exceptionmsg = errmsg.value.args[0]
    assert (
        "This can happen when you have negative and/or zero values in the data"
        in exceptionmsg
    )

    with pytest.raises(ValueError) as errmsg:
        exp.setup(
            data=data,
            target=target,
            seasonal_period=4,
            preprocess=True,
            transform_exogenous=method,
        )
    exceptionmsg = errmsg.value.args[0]
    assert (
        "This can happen when you have negative and/or zero values in the data"
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
    plot_data = exp.plot_model(model, return_data=True, system=False)
    assert isinstance(plot_data, dict)

    tuned = exp.tune_model(model)
    preds = exp.predict_model(tuned)
    assert len(preds) == FH
    plot_data = exp.plot_model(tuned, return_data=True, system=False)
    assert isinstance(plot_data, dict)

    final = exp.finalize_model(tuned)
    preds = exp.predict_model(final)
    assert len(preds) == FH
    plot_data = exp.plot_model(final, return_data=True, system=False)
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
        seasonal_period=4,
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
    plot_data = exp.plot_model(model, return_data=True, system=False)
    assert isinstance(plot_data, dict)

    tuned = exp.tune_model(model)
    preds = exp.predict_model(tuned)
    assert len(preds) == FH
    plot_data = exp.plot_model(tuned, return_data=True, system=False)
    assert isinstance(plot_data, dict)

    _ = exp.finalize_model(tuned)
    # # Exogenous models predictions and plots after finalizing will need future X
    # # values. Hence disabling this test.
    # preds = exp.predict_model(final)
    # assert len(preds) == FH
    # plot_data = exp.plot_model(final, return_data=True, system=False)
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
        seasonal_period=4,
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
        seasonal_period=4,
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
        seasonal_period=4,
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
        seasonal_period=4,
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
