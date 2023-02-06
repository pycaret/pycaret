"""Module containing utility functions for datatime manipulation"""

from typing import Any, Optional, Union

import pandas as pd


def coerce_period_to_datetime_index(
    data: Optional[Union[pd.Series, pd.DataFrame, Any]],
    freq: Optional[str] = None,
    inplace: bool = False,
) -> Optional[Union[pd.Series, pd.DataFrame, Any]]:
    """Converts a dataframe or series index from PeriodIndex to DatetimeIndex

    Parameters
    ----------
    data : Optional[Union[pd.Series, pd.DataFrame, Any]]
        The data with a PeriodIndex that needs to be converted to DatetimeIndex
        If None, it does nothing.
    freq : Optional[str], optional
        The frequency to be used to convert the index, by default None which
        uses data.index.freq to perform the conversion
    inplace : bool, optional
        If True, convert inplace and do not return anything.
        If False, then do not modify the original object and return a copy with
        converted index, by default False

    Returns
    -------
    Optional[Union[pd.Series, pd.DataFrame, Any]]
        The data with DatetimeIndex. Note: If input is not of type pd.Series or
        pd.DataFrame, then no change is made to the data.
    """
    if isinstance(data, (pd.DataFrame, pd.Series)):
        data_ = data.copy() if not inplace else data

        if isinstance(data_.index, pd.PeriodIndex):
            if freq is None:
                # We can allow it to infer it by default but it can sometimes
                # infer MonthStart as MonthEnd. Passing the frequency to the
                # conversion explicitly makes it more robust.
                freq = data_.index.freq

            data_.index = data_.index.to_timestamp(freq=freq)

            # Corner Case Handling ----
            # When data.index is of type Q-DEC with only 2 points, frequency
            # is not set after conversion (details below). However, this
            # works OK if there are more than 2 data points.
            # Before Conversion: PeriodIndex(['2018Q2', '2018Q3'], dtype='period[Q-DEC]', freq='Q-DEC')
            # After Conversion: DatetimeIndex(['2018-06-30', '2018-09-30'], dtype='datetime64[ns]', freq=None)
            # Hence, setting it manually if the frequency is not set after conversion.
            if data_.index.freq is None:
                data_.index.freq = freq

        if not inplace:
            return data_

    elif not inplace:
        return data


def coerce_datetime_to_period_index(
    data: Optional[Union[pd.Series, pd.DataFrame, Any]],
    freq: Optional[str] = None,
    inplace: bool = False,
) -> Optional[Union[pd.Series, pd.DataFrame, Any]]:
    """Converts a dataframe or series index from DatetimeIndex to PeriodIndex

    Parameters
    ----------
    data : Optional[Union[pd.Series, pd.DataFrame, Any]]
        The data with a DatetimeIndex that needs to be converted to PeriodIndex
        If None, it does nothing.
    freq : Optional[str], optional
        The frequency to be used to convert the index, by default None which
        uses data.index.freq to perform the conversion
    inplace : bool, optional
        If True, convert inplace and do not return anything.
        If False, then do not modify the original object and return a copy with
        converted index, by default False

    Returns
    -------
    Optional[Union[pd.Series, pd.DataFrame, Any]]
        The data with PeriodIndex. Note: If input is not of type pd.Series or
        pd.DataFrame, then no change is made to the data.
    """
    if isinstance(data, (pd.DataFrame, pd.Series)):
        data_ = data.copy() if not inplace else data

        if isinstance(data_.index, pd.DatetimeIndex):
            if freq is None:
                # Inferred by default
                # Do not pass from freq=data_.index.freq. Let it infer by default
                # since the data_ frequency could be MonthBegin which gives error
                # if explicitly passed to `to_period`, but works ok if we allow
                # it to infer automatically.
                data_.index = data_.index.to_period()
            else:
                data_.index = data_.index.to_period(freq=freq)

        if not inplace:
            return data_

    elif not inplace:
        return data
