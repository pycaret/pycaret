"""Module to test the datetime utility functions
"""
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from pycaret.datasets import get_data
from pycaret.utils.datetime import (
    coerce_datetime_to_period_index,
    coerce_period_to_datetime_index,
)


def test_coerce_period_to_datetime_index():
    """Tests coercion of PeriodIndex to DatetimeIndex"""

    # TODO: Get both Series and DataFrame with Period Index later
    data = get_data("airline")
    orig_freq = data.index.freq

    # Basic coercion ----
    new_data = coerce_period_to_datetime_index(data=data)
    assert isinstance(new_data.index, pd.DatetimeIndex)
    assert new_data.index.freq == orig_freq

    # Passing Freq parameter ----
    data_diff_freq = data.copy()
    data_diff_freq = data_diff_freq.asfreq("D")
    new_data = coerce_period_to_datetime_index(data=data_diff_freq, freq=orig_freq)
    assert isinstance(new_data.index, pd.DatetimeIndex)
    assert new_data.index.freq == orig_freq

    # In place coercion ----
    assert isinstance(data.index, pd.PeriodIndex)
    coerce_period_to_datetime_index(data=data, inplace=True)
    assert isinstance(data.index, pd.DatetimeIndex)
    assert data.index.freq == orig_freq

    # No Coercion (numpy - no index) ----
    data_np = np.array(data.values)
    assert isinstance(data_np, np.ndarray)
    data_np_new = coerce_period_to_datetime_index(data=data_np)
    assert isinstance(data_np_new, np.ndarray)

    # No Coercion (Pandas Int Index)
    data = get_data("uschange")
    original_index_type = type(data.index)
    new_data = coerce_period_to_datetime_index(data=data)
    assert isinstance(new_data.index, original_index_type)

    # Corner condition (Q-DEC with only 2 data points)----
    orig_freq = "Q-DEC"
    data = pd.DataFrame(
        [1, 2], index=pd.PeriodIndex(["2018Q2", "2018Q3"], freq=orig_freq)
    )
    new_data = coerce_period_to_datetime_index(data=data)
    assert isinstance(new_data.index, pd.DatetimeIndex)
    assert new_data.index.freq == orig_freq


def test_coerce_datetime_to_period_index():
    """Tests coercion of DatetimeIndex to PeriodIndex
    Note since we are converting from a period to Datetime,
    there is no guarantee of the frequency unless we explicitly
    pass it. e.g. DateTime could be MonthStart, but Period will
    represent Month.
    """
    # TODO: Get both Series and DataFrame with Period Index later
    data = get_data("airline")
    data.index = data.index.to_timestamp()

    # Basic coercion ----
    new_data = coerce_datetime_to_period_index(data=data)
    assert isinstance(new_data.index, pd.PeriodIndex)

    # Passing Freq parameter ----
    # Convert Monthly data to Daily data
    new_data = coerce_datetime_to_period_index(data=data, freq="D")
    assert isinstance(new_data.index, pd.PeriodIndex)
    assert new_data.index.freq == "D"

    # In place coercion ----
    assert isinstance(data.index, pd.DatetimeIndex)
    coerce_datetime_to_period_index(data=data, inplace=True)
    assert isinstance(data.index, pd.PeriodIndex)

    # No Coercion (numpy - no index) ----
    data_np = np.array(data.values)
    assert isinstance(data_np, np.ndarray)
    data_np_new = coerce_datetime_to_period_index(data=data_np)
    assert isinstance(data_np_new, np.ndarray)

    # No Coercion (Pandas Int Index)
    data = get_data("uschange")
    original_index_type = type(data.index)
    new_data = coerce_datetime_to_period_index(data=data)
    assert isinstance(new_data.index, original_index_type)
