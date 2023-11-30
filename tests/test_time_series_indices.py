import os
from typing import Any

import pandas as pd
import pytest

from pycaret.datasets import get_data
from pycaret.time_series import TSForecastingExperiment

os.environ["PYCARET_TESTING"] = "1"


# =============================================================================#
# Data set generation begins here
# =============================================================================#


def _get_univar_noexo_data_with_index_index():
    """Generates multiple datasets of univariate data without exogenous variables
    where the index has been set to various types of indices.
    """
    # PeriodIndex
    data1 = get_data("airline")

    # DatetimeIndex
    data2 = data1.copy()
    data2 = data2.to_timestamp()

    # Int64Index
    data3 = data1.copy()
    data3.reset_index(drop=True, inplace=True)

    ids = ["Period", "Datetime", "Int"]

    # DateTimeIndex is coerced and returned as PeriodIndex in PyCaret
    return [
        [(data1, pd.PeriodIndex), (data2, pd.PeriodIndex), (data3, pd.Index)],
        ids,
    ]


def _get_univar_noexo_data_with_index_column():
    """Generates multiple datasets of univariate data without exogenous variables
    where the index is specified through a column of different.
    """
    # PeriodIndex column
    data1 = pd.DataFrame(get_data("airline"))

    # DatetimeIndex column
    data2 = data1.copy()
    data2 = data2.to_timestamp()

    # Int64Index column
    data3 = data1.copy()
    data3.reset_index(drop=True, inplace=True)

    data1.reset_index(inplace=True)
    data2.reset_index(inplace=True)
    data3.reset_index(inplace=True)

    data1.rename(columns={"Period": "index"}, inplace=True)
    data2.rename(columns={"Period": "index"}, inplace=True)

    # String Index (created from datetime index)
    data4 = data2.copy()
    data4["index"] = data4["index"].dt.strftime("%m/%d/%Y")

    ids = ["Period", "Datetime", "Int", "String"]

    # DateTimeIndex & String index column is coerced and returned as PeriodIndex
    # in PyCaret
    return [
        [
            (data1, pd.PeriodIndex),
            (data2, pd.PeriodIndex),
            (data3, pd.Index),
            (data4, pd.PeriodIndex),
        ],
        ids,
    ]


def _get_univar_exo_data_with_index_index():
    """Generates multiple datasets of univariate data with exogenous variables
    where the index has been set to various types of indices.
    """
    # TODO: Find a better source of data ----
    data1 = pd.read_csv(
        "https://raw.githubusercontent.com/ngupta23/DS6373_TimeSeries/2b40f0071c3b7ec6a05dc0106f64e041f8cbaaef/Projects/gdp_prediction/data/economic_indicators_all_ex_3mo_china_inc_treas3mo.csv"
    )
    data1["date"] = pd.to_datetime(data1["date"].str.replace(" ", "-"))
    data1.set_index("date", inplace=True)

    # PeriodIndex
    data1.index = data1.index.to_period()

    # DatetimeIndex
    data2 = data1.copy()
    data2 = data2.to_timestamp()

    # Int64Index
    data3 = data1.copy()
    data3.reset_index(drop=True, inplace=True)

    ids = ["Period", "Datetime", "Int"]

    # DateTimeIndex is coerced and returned as PeriodIndex in PyCaret
    return [
        [(data1, pd.PeriodIndex), (data2, pd.PeriodIndex), (data3, pd.Index)],
        ids,
    ]


def _get_univar_exo_data_with_index_column():
    """Generates multiple datasets of univariate data with exogenous variables
    where the index is specified through a column of different.
    """
    # TODO: Find a better source of data ----
    data1 = pd.read_csv(
        "https://raw.githubusercontent.com/ngupta23/DS6373_TimeSeries/2b40f0071c3b7ec6a05dc0106f64e041f8cbaaef/Projects/gdp_prediction/data/economic_indicators_all_ex_3mo_china_inc_treas3mo.csv"
    )
    data1["date"] = pd.to_datetime(data1["date"].str.replace(" ", "-"))
    data1.set_index("date", inplace=True)

    # PeriodIndex
    data1.index = data1.index.to_period()

    # DatetimeIndex column
    data2 = data1.copy()
    data2 = data2.to_timestamp()

    # Int64Index column
    data3 = data1.copy()
    data3.reset_index(drop=True, inplace=True)

    data1.reset_index(inplace=True)
    data2.reset_index(inplace=True)
    data3.reset_index(inplace=True)

    data1.rename(columns={"date": "index"}, inplace=True)
    data2.rename(columns={"date": "index"}, inplace=True)

    # String Index (created from datetime index)
    data4 = data2.copy()
    data4["index"] = data4["index"].dt.strftime("%m/%d/%Y")

    ids = ["Period", "Datetime", "Int", "String"]

    # DateTimeIndex & String index column is coerced and returned as PeriodIndex
    # in PyCaret
    return [
        [
            (data1, pd.PeriodIndex),
            (data2, pd.PeriodIndex),
            (data3, pd.Index),
            (data4, pd.PeriodIndex),
        ],
        ids,
    ]


# =============================================================================#
# Checker function(s)
# =============================================================================#
def _check_model_creation_and_indices(
    exp: TSForecastingExperiment, model: str, expected_return_index_type: Any
):
    """Function to create a few trial models that support both univariate and
    multivariate forecasting.

    Parameters
    ----------
    exp : TSForecastingExperiment
        The Time Series experiment object
    model : str
        The model to create using the experiment provided
    expected_return_index_type: Any
        The expected return type of the index of the predictions dataframe
    """
    if model in exp.models().index:
        model = exp.create_model(model)
        preds = exp.predict_model(model)
        assert isinstance(preds.index, expected_return_index_type)
        exp.plot_model(model)


# =============================================================================#
# Tests begin here
# =============================================================================#

# Includes models from statistical family, reduced regression family and prophet
# since prophet has special handling for indices in patched version in pycaret.
# ETS and Exponential Smoothing are included since ETS was failing in manual testing.
models = ["arima", "ets", "exp_smooth", "lr_cds_dt", "prophet"]

# -----------------------------------------------------------------------------#
# Test 1: Univariate No Exogenous Variables with Data Index set to various types
# -----------------------------------------------------------------------------#

(
    univar_noexo_data_with_index_index_plus_return_type,
    ids_univar_noexo_data_with_index_index,
) = _get_univar_noexo_data_with_index_index()


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize(
    "data, expected_return_index_type",
    univar_noexo_data_with_index_index_plus_return_type,
    ids=ids_univar_noexo_data_with_index_index,
)
def test_ts_indices_univar_noexo_index_index(data, model, expected_return_index_type):
    """
    Checks working with various types of indices with univariate data without
    exogenous variables when index is already set in the dataframe.
    """
    exp = TSForecastingExperiment()
    exp.setup(data=data, fh=12, fold=2, session_id=42)
    _check_model_creation_and_indices(
        exp, model=model, expected_return_index_type=expected_return_index_type
    )


# -----------------------------------------------------------------------------#
# Test 2: Univariate No Exogenous Variables with Data Index provided through column
# -----------------------------------------------------------------------------#

(
    univar_noexo_data_with_index_column_plus_return_type,
    ids_univar_noexo_data_with_index_column,
) = _get_univar_noexo_data_with_index_column()


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize(
    "data, expected_return_index_type",
    univar_noexo_data_with_index_column_plus_return_type,
    ids=ids_univar_noexo_data_with_index_column,
)
def test_ts_indices_univar_noexo_index_column(data, model, expected_return_index_type):
    """
    Checks working with various types of indices with univariate data without
    exogenous variables when index is provided through a column.
    """
    exp = TSForecastingExperiment()
    exp.setup(data=data, index="index", fh=12, fold=2, session_id=42)
    _check_model_creation_and_indices(
        exp, model=model, expected_return_index_type=expected_return_index_type
    )


# -----------------------------------------------------------------------------#
# Test 3: Univariate with Exogenous Variables with Data Index set to various types
# -----------------------------------------------------------------------------#

(
    univar_exo_data_with_index_index_plus_return_type,
    ids_univar_exo_data_with_index_index,
) = _get_univar_exo_data_with_index_index()


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize(
    "data, expected_return_index_type",
    univar_exo_data_with_index_index_plus_return_type,
    ids=ids_univar_exo_data_with_index_index,
)
def test_ts_indices_univar_exo_index_index(data, model, expected_return_index_type):
    """
    Checks working with various types of indices with univariate data with exogenous
    variables when index is already set in the dataframe.
    """
    exp = TSForecastingExperiment()
    exp.setup(
        data=data,
        target="gdp_change",
        fh=2,
        fold=2,
        session_id=42,
    )
    _check_model_creation_and_indices(
        exp, model=model, expected_return_index_type=expected_return_index_type
    )


# -----------------------------------------------------------------------------#
# Test 4: Univariate with Exogenous Variables with Data Index provided through column
# -----------------------------------------------------------------------------#

(
    univar_exo_data_with_index_column_plus_return_type,
    ids_univar_exo_data_with_index_column,
) = _get_univar_exo_data_with_index_column()


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize(
    "data, expected_return_index_type",
    univar_exo_data_with_index_column_plus_return_type,
    ids=ids_univar_exo_data_with_index_column,
)
def test_ts_indices_univar_exo_index_column(data, model, expected_return_index_type):
    """
    Checks working with various types of indices with univariate data with exogenous
    variables when index is already set in the dataframe.
    """
    exp = TSForecastingExperiment()
    exp.setup(
        data=data,
        target="gdp_change",
        index="index",
        fh=2,
        fold=2,
        session_id=42,
    )
    _check_model_creation_and_indices(
        exp, model=model, expected_return_index_type=expected_return_index_type
    )
