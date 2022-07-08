import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import numpy as np
from conf.config_core import config
from src.data_processing import process_data


def test_data_processing_func() -> None:
    """
    This function tests if the current data processing function
    is producting the right output.
    """
    # correct template
    template = {
        'pickup_year': np.dtype('int64'),
        'pickup_month': np.dtype('int64'),
        'pickup_dayofyear': np.dtype('int64'),
        'pickup_dayofweek': np.dtype('int64'),
        'pickup_hour': np.dtype('int64'),
        'is_holiday': np.dtype('int64'),
        'passenger_count': np.dtype('int64'),
        'pickup_latitude': np.dtype('float64'),
        'pickup_longitude': np.dtype('float64'),
        'dropoff_latitude': np.dtype('float64'),
        'dropoff_longitude': np.dtype('float64'),
        'trip_distance_km': np.dtype('float64'),
        'fare_amount': np.dtype('float64')
    }

    processed_data = process_data()

    assert processed_data.dtypes.to_dict() == template