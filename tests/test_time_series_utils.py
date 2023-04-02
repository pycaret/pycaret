import numpy as np
import pandas as pd
import pytest

from pycaret.utils.time_series import (
    SeasonalPeriod,
    clean_time_index,
    remove_harmonics_from_sp,
)


def test_harmonic_removal():
    """Tests the removal of harmonics"""

    # 1.0 No harmonics removed ----
    results = remove_harmonics_from_sp([2, 51, 5])
    assert results == [2, 51, 5]

    # 2.0 One base frequency removed ----
    results = remove_harmonics_from_sp([2, 52, 3])
    assert results == [52, 3]

    # 3.0 Remove more than one base period ----
    results = remove_harmonics_from_sp([50, 3, 11, 100, 39])
    assert results == [11, 100, 39]

    # 4.0 Order of replacement ----

    # 4.1 Only one removed
    # 4.1A Ordered by raw strength
    results = remove_harmonics_from_sp([2, 3, 4, 50])
    assert results == [3, 4, 50]
    # 4.1B Ordered by harmonic max
    results = remove_harmonics_from_sp(
        [2, 3, 4, 50], harmonic_order_method="harmonic_max"
    )
    assert results == [50, 3, 4]
    # 4.1C Ordered by harmonic strength
    results = remove_harmonics_from_sp(
        [2, 3, 4, 50], harmonic_order_method="harmonic_strength"
    )
    assert results == [4, 3, 50]

    # 4.2 More than one removed
    # 4.2A Ordered by raw strength
    results = remove_harmonics_from_sp([3, 2, 6, 50])
    assert results == [6, 50]
    # 4.2B Ordered by harmonic max
    results = remove_harmonics_from_sp(
        [3, 2, 6, 50], harmonic_order_method="harmonic_max"
    )
    assert results == [6, 50]
    results = remove_harmonics_from_sp(
        [2, 3, 6, 50], harmonic_order_method="harmonic_max"
    )
    assert results == [50, 6]
    # 4.2C Ordered by harmonic strength
    results = remove_harmonics_from_sp(
        [3, 2, 6, 50], harmonic_order_method="harmonic_strength"
    )
    assert results == [6, 50]
    results = remove_harmonics_from_sp(
        [2, 3, 6, 50], harmonic_order_method="harmonic_strength"
    )
    assert results == [6, 50]

    # 4.2D Other variants
    results = remove_harmonics_from_sp(
        [10, 20, 30, 40, 50, 60], harmonic_order_method="harmonic_strength"
    )
    assert results == [20, 40, 60, 50]
    results = remove_harmonics_from_sp(
        [10, 20, 30, 40, 50, 60], harmonic_order_method="harmonic_max"
    )
    assert results == [60, 40, 50]

    # 5.0 These were giving precision issues earlier. Now fixed by rounding internally. ----

    # 5.1
    results = remove_harmonics_from_sp([50, 100, 150, 49, 200, 51, 23, 27, 10, 250])
    assert results == [150, 49, 200, 51, 23, 27, 250]

    # 5.2
    results = remove_harmonics_from_sp([49, 98, 18])
    assert results == [98, 18]

    # 5.3
    results = remove_harmonics_from_sp([50, 16, 15, 17, 34, 2, 33, 49, 18, 100, 32])
    assert results == [15, 34, 33, 49, 18, 100, 32]


def _get_seasonal_keys():
    return [freq for freq, _ in SeasonalPeriod.__members__.items()]


@pytest.mark.parametrize("freq", _get_seasonal_keys())
@pytest.mark.parametrize("index", [True, False])
def test_clean_time_index_datetime(freq, index):
    """Test clean_time_index utility when index/column is of type DateTime"""
    dates = pd.date_range("2019-01-01", "2022-01-30", freq=freq)

    # At least 3 data points to allow test to insert a missing index in the middle
    # but not so many data points that the code slows down too much.
    if len(dates) > 100:
        dates = dates[:100]
    assert len(dates) >= 3

    data = pd.DataFrame(
        {
            "date": dates,
            "value": np.random.rand(len(dates)),
        }
    )
    if index:
        data.set_index("date", inplace=True)
        index_col = None
    else:
        index_col = "date"

    # -------------------------------------------------------------------------#
    # Test 1: Cleaning without any missing time index
    # -------------------------------------------------------------------------#
    try:
        cleaned = clean_time_index(data=data, index_col=index_col, freq=freq)
    except AttributeError:
        # Unsupported freq conversion from DatetimeIndex to PeriodIndex
        return
    assert len(cleaned) == len(data)

    # -------------------------------------------------------------------------#
    # Test 2: Cleaning with any missing time index
    # -------------------------------------------------------------------------#
    # Drop 2nd row
    data_missing = data.copy()
    data_missing = data_missing.drop(data_missing.index[1])
    cleaned = clean_time_index(data=data_missing, index_col=index_col, freq=freq)
    # Datetime Index missing values are filled in  by clean_time_index
    assert len(cleaned) == len(data)


@pytest.mark.parametrize("freq", _get_seasonal_keys())
@pytest.mark.parametrize("index", [False])
def test_clean_time_index_str_datetime(freq, index):
    """Test clean_time_index utility when index/column is of type str in format
    acceptable to DatetimeIndex

    NOTE: Index can not be string (only column). Code unchanges, just parameter
    restricted to False
    """
    dates = pd.date_range("2019-01-01 00:00:00", "2022-01-30 00:00:00", freq=freq)

    # At least 3 data points to allow test to insert a missing index in the middle
    # but not so many data points that the code slows down too much.
    if len(dates) > 100:
        dates = dates[:100]
    assert len(dates) >= 3
    dates = dates.strftime("%Y-%m-%d %H:%M:%S")

    data = pd.DataFrame(
        {
            "date": dates,
            "value": np.random.rand(len(dates)),
        }
    )
    if index:
        data.set_index("date", inplace=True)
        index_col = None
    else:
        index_col = "date"

    # -------------------------------------------------------------------------#
    # Test 1: Cleaning without any missing time index
    # -------------------------------------------------------------------------#
    try:
        cleaned = clean_time_index(data=data, index_col=index_col, freq=freq)
    except AttributeError:
        # Unsupported freq conversion from DatetimeIndex to PeriodIndex
        return
    assert len(cleaned) == len(data)

    # -------------------------------------------------------------------------#
    # Test 2: Cleaning with any missing time index
    # -------------------------------------------------------------------------#
    # Drop 2nd row
    data_missing = data.copy()
    data_missing = data_missing.drop(data_missing.index[1])
    cleaned = clean_time_index(data=data_missing, index_col=index_col, freq=freq)
    # Datetime Index missing values are filled in  by clean_time_index
    assert len(cleaned) == len(data)


@pytest.mark.parametrize("freq", _get_seasonal_keys())
@pytest.mark.parametrize("index", [True, False])
def test_clean_time_index_period(freq, index):
    """Test clean_time_index utility when index/column is of type Period"""
    try:
        dates = pd.period_range("2019-01-01", "2022-01-30", freq=freq)
    except ValueError:
        # Unsupported freq for PeriodIndex
        return

    # At least 3 data points to allow test to insert a missing index in the middle
    # but not so many data points that the code slows down too much.
    if len(dates) > 100:
        dates = dates[:100]
    assert len(dates) >= 3

    data = pd.DataFrame(
        {
            "date": dates,
            "value": np.random.rand(len(dates)),
        }
    )
    if index:
        data.set_index("date", inplace=True)
        index_col = None
    else:
        index_col = "date"

    # -------------------------------------------------------------------------#
    # Test 1: Cleaning without any missing time index
    # -------------------------------------------------------------------------#
    cleaned = clean_time_index(data=data, index_col=index_col, freq=freq)
    assert len(cleaned) == len(data)

    # -------------------------------------------------------------------------#
    # Test 2: Cleaning with any missing time index
    # -------------------------------------------------------------------------#
    # Drop 2nd row
    data_missing = data.copy()
    data_missing = data_missing.drop(data_missing.index[1])
    cleaned = clean_time_index(data=data_missing, index_col=index_col, freq=freq)
    # Period Index missing values are filled in  by clean_time_index
    assert len(cleaned) == len(data)


@pytest.mark.parametrize("freq", _get_seasonal_keys())
@pytest.mark.parametrize("index", [False])
def test_clean_time_index_str_period(freq, index):
    """Test clean_time_index utility when index/column is of type str in format
    acceptable to PeriodIndex

    NOTE: Index can not be string (only column). Code unchanges, just parameter
    restricted to False
    """
    try:
        dates = pd.period_range("2019-01-01", "2022-01-30", freq=freq)
    except ValueError:
        # Unsupported freq for PeriodIndex
        return

    # At least 3 data points to allow test to insert a missing index in the middle
    # but not so many data points that the code slows down too much.
    if len(dates) > 100:
        dates = dates[:100]
    assert len(dates) >= 3
    dates = dates.astype(str)

    data = pd.DataFrame(
        {
            "date": dates,
            "value": np.random.rand(len(dates)),
        }
    )
    if index:
        data.set_index("date", inplace=True)
        index_col = None
    else:
        index_col = "date"

    # -------------------------------------------------------------------------#
    # Test 1: Cleaning without any missing time index
    # -------------------------------------------------------------------------#
    cleaned = clean_time_index(data=data, index_col=index_col, freq=freq)
    assert len(cleaned) == len(data)

    # -------------------------------------------------------------------------#
    # Test 2: Cleaning with any missing time index
    # -------------------------------------------------------------------------#
    # Drop 2nd row
    data_missing = data.copy()
    data_missing = data_missing.drop(data_missing.index[1])
    cleaned = clean_time_index(data=data_missing, index_col=index_col, freq=freq)
    # Period Index missing values are filled in  by clean_time_index
    assert len(cleaned) == len(data)


@pytest.mark.parametrize("freq", _get_seasonal_keys())
@pytest.mark.parametrize("index", [True, False])
def test_clean_time_index_int(freq, index):
    """Test clean_time_index utility when index/column is of type Int"""
    dates = np.arange(100)

    # At least 3 data points to allow test to insert a missing index in the middle
    # but not so many data points that the code slows down too much.
    if len(dates) > 100:
        dates = dates[:100]
    assert len(dates) >= 3

    data = pd.DataFrame(
        {
            "date": dates,
            "value": np.random.rand(len(dates)),
        }
    )
    if index:
        data.set_index("date", inplace=True)
        index_col = None
    else:
        index_col = "date"

    # -------------------------------------------------------------------------#
    # Test 1: Cleaning without any missing time index
    # -------------------------------------------------------------------------#
    cleaned = clean_time_index(data=data, index_col=index_col, freq=freq)
    assert len(cleaned) == len(data)

    # -------------------------------------------------------------------------#
    # Test 2: Cleaning with any missing time index
    # -------------------------------------------------------------------------#
    # Drop 2nd row
    data_missing = data.copy()
    data_missing = data_missing.drop(data_missing.index[1])
    cleaned = clean_time_index(data=data_missing, index_col=index_col, freq=freq)
    # Int Index is untouched by clean_time_index
    assert len(cleaned) == len(data) - 1
