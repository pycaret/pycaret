"""Module to test time_series functionality
"""
import pytest
import numpy as np  # type: ignore

# from sktime.forecasting.compose import ForecastingPipeline
from pycaret.utils.time_series.forecasting.pipeline import PyCaretForecastingPipeline
from sktime.forecasting.compose import TransformedTargetForecaster

from .time_series_test_utils import _return_model_names_for_missing_data

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


def test_preprocess_no_exo_types(load_pos_and_neg_data):
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


def test_preprocess_exo_types(load_uni_exo_data_target):
    """Tests preprocessing pipeline data types with exogenous variables"""

    data, target = load_uni_exo_data_target

    # # Simulate misssing values
    # data[10:20] = np.nan

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


def test_preprocess_raises_no_exo(load_pos_and_neg_data):
    """Tests conditions that raise errors"""
    data = load_pos_and_neg_data

    # Simulate misssing values
    data[10:20] = np.nan

    exp = TSForecastingExperiment()

    with pytest.raises(ValueError) as errmsg:
        exp.setup(data=data, preprocess=False)
    exceptionmsg = errmsg.value.args[0]
    assert "Please enable imputation to proceed" in exceptionmsg

    with pytest.raises(ValueError) as errmsg:
        exp.setup(data=data, preprocess=True, numeric_imputation_target=None)
    exceptionmsg = errmsg.value.args[0]
    assert "Please enable imputation to proceed" in exceptionmsg


def test_preprocess_raises_exo(load_uni_exo_data_target):
    """Tests conditions that raise errors"""

    data, target = load_uni_exo_data_target

    # Simulate misssing values
    data[10:20] = np.nan

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


@pytest.mark.parametrize("model_name", _model_names_for_missing_data)
def test_preprocess_uni(load_pos_and_neg_data, model_name):
    """Tests normal preprocessing"""
    data = load_pos_and_neg_data

    # Simulate misssing values
    data[10:20] = np.nan

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
def test_preprocess_exo(load_uni_exo_data_target, model_name):
    """Tests normal preprocessing"""
    data, target = load_uni_exo_data_target

    # Simulate misssing values
    data[10:20] = np.nan

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

    final = exp.finalize_model(tuned)
    # # Exogenous models predictions and plots after finalizing will need future X
    # # values. Hence disabling this test.
    # preds = exp.predict_model(final)
    # assert len(preds) == FH
    # plot_data = exp.plot_model(final, return_data=True, system=False)
    # assert isinstance(plot_data, dict)
