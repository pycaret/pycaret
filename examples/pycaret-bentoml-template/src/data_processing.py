import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import holidays
import numpy as np
import pandas as pd
from conf.config_core import config
from math import radians, sqrt, sin, cos, asin


def check_holiday(date: str, country: str = 'US') -> bool:
    """
    This function checks if a given string date
    is a holiday or not.
    """
    country_holidays = holidays.country_holidays(country)
    
    return int(date in country_holidays)


def apply_heversine(latitude1, longitude1, latitude2, longitude2) -> list:
    """
    This function calculates the distance (in km) between 
    two coordinates (lat, long).
    """
    dist = []
    
    for pos in range(len(longitude1)):
        long1,lati1,long2,lati2 = map(radians,[longitude1[pos],latitude1[pos],longitude2[pos],latitude2[pos]])
        dist_long = long2 - long1
        dist_lati = lati2 - lati1
        a = sin(dist_lati/2)**2 + cos(lati1) * cos(lati2) * sin(dist_long/2)**2
        c = 2 * asin(sqrt(a))*6371
        dist.append(c)
       
    return dist


def process_data() -> pd.DataFrame:
    """
    This function takes the raw dataset as an input
    and then make it ready for modeling.
    """
    data = pd.read_csv(config.data_config.raw.path)

    # removing useless columns & converting dtypes
    data = data.drop(columns=['Unnamed: 0', 'key'])
    data.pickup_datetime = pd.to_datetime(data.pickup_datetime)

    # extracting temporal features
    data['pickup_year'] = data.pickup_datetime.map(lambda x: x.year)
    data['pickup_month'] = data.pickup_datetime.map(lambda x: x.month)
    data['pickup_dayofyear'] = data.pickup_datetime.map(lambda x: x.dayofyear)
    data['pickup_dayofweek'] = data.pickup_datetime.map(lambda x: x.dayofweek)
    data['pickup_hour'] = data.pickup_datetime.map(lambda x: x.hour)
    data['is_holiday'] = data.pickup_datetime.map(lambda x: check_holiday(str(x.date())))

    # revoming useless date column
    data = data.drop(columns=['pickup_datetime'])

    # handling nans
    data = data.dropna()
    data = data.drop(index=data.query("pickup_latitude == 0").index)
    data = data.drop(index=data.query("fare_amount == 0").index)
    data.reset_index(drop=True, inplace=True)

    # calculating distance between coordinates
    data['trip_distance_km'] = apply_heversine(
        data.pickup_latitude,
        data.pickup_longitude,
        data.dropoff_latitude,
        data.dropoff_longitude
    )

    # organizing columns
    data = data[
        [
            'pickup_year', 'pickup_month', 'pickup_dayofyear', 
            'pickup_dayofweek', 'pickup_hour', 'is_holiday', 
            'passenger_count', 'pickup_latitude', 'pickup_longitude',
            'dropoff_latitude' ,'dropoff_longitude', 'trip_distance_km', 'fare_amount'
        ]
    ]

    return data


if __name__ == "__main__":
    process_data()