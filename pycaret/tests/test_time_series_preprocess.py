"""Module to test time_series functionality
"""
import pytest
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from sktime.forecasting.compose import ForecastingPipeline
from sktime.forecasting.compose import TransformedTargetForecaster

from pycaret.time_series import TSForecastingExperiment


from .time_series_test_utils import (
    _return_model_parameters,
    _return_compare_model_args,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


##############################
#### Functions Start Here ####
##############################

# NOTE: Fixtures can not be used to parameterize tests
# https://stackoverflow.com/questions/52764279/pytest-how-to-parametrize-a-test-with-a-list-that-is-returned-from-a-fixture
# Hence, we have to create functions and create the parameterized list first
# (must happen during collect phase) before passing it to mark.parameterize.

_model_parameters = _return_model_parameters()
_compare_model_args = _return_compare_model_args()

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

    # No preprocessing
    exp.setup(data=data, preprocess=False)
    assert exp.pipeline is None

    #### No preprocessing (Need to explicitly set what steps to add to preprocessing)
    exp.setup(data=data, preprocess=True)
    assert exp.pipeline is None

    #### Transform Target only
    exp.setup(data=data, preprocess=True, numeric_imputation_target=True)
    assert isinstance(exp.pipeline, TransformedTargetForecaster)

    #### Transform Exogenous only (but no exogenous present)
    exp.setup(data=data, preprocess=True, numeric_imputation_exogenous=True)
    assert exp.pipeline is None

    #### Transform Exogenous & Target (but no exogenous present)
    exp.setup(
        data=data,
        preprocess=True,
        numeric_imputation_target=True,
        numeric_imputation_exogenous=True,
    )
    assert isinstance(exp.pipeline, TransformedTargetForecaster)


def test_preprocess_exo_types(load_uni_exo_data_target):
    """Tests preprocessing pipeline data types with exogenous variables"""

    data, target = load_uni_exo_data_target

    # # Simulate misssing values
    # data[10:20] = np.nan

    exp = TSForecastingExperiment()

    # No preprocessing
    exp.setup(data=data, target=target, seasonal_period=4, preprocess=False)
    assert exp.pipeline is None

    #### No preprocessing (Need to explicitly set what steps to add to preprocessing)
    exp.setup(data=data, target=target, seasonal_period=4, preprocess=True)
    assert exp.pipeline is None

    #### Transform Target only
    exp.setup(
        data=data,
        target=target,
        seasonal_period=4,
        preprocess=True,
        numeric_imputation_target=True,
    )
    assert isinstance(exp.pipeline, TransformedTargetForecaster)

    #### Transform Exogenous only
    exp.setup(
        data=data,
        target=target,
        seasonal_period=4,
        preprocess=True,
        numeric_imputation_exogenous=True,
    )
    assert isinstance(exp.pipeline, ForecastingPipeline)
    assert not isinstance(exp.pipeline.steps[-1][1], TransformedTargetForecaster)

    #### Transform Exogenous & Target
    exp.setup(
        data=data,
        target=target,
        seasonal_period=4,
        preprocess=True,
        numeric_imputation_target=True,
        numeric_imputation_exogenous=True,
    )
    assert isinstance(exp.pipeline, ForecastingPipeline)
    assert isinstance(exp.pipeline.steps[-1][1], TransformedTargetForecaster)


# TODO: Test actual model creation, tuning, predict, finalize, compare, etc. with missing values
