"""Module to test time_series functionality
"""
import pytest

from random import choice, randint
import numpy as np
import pandas as pd

@pytest.fixture(scope="session", name="load_data")
def load_data():
    """Load Pycaret Airline dataset."""
    from pycaret.datasets import get_data
    airline_data = get_data("airline")
    return airline_data


models = ['naive', 'poly_trend', 'arima', 'exp_smooth', 'theta']
parametrize_list = [
    (choice(models))
]


@pytest.mark.parametrize("model", parametrize_list)
def test_create_model(model, load_data):

    from pycaret.internal.PycaretExperiment import TimeSeriesExperiment
    exp = TimeSeriesExperiment()

    exp.setup(
        data=load_data,
        fold=3,
        fh=12,
        fold_strategy="expandingwindow"
    )

    model_obj = exp.create_model(model)
    y_pred = model_obj.predict()
    assert isinstance(y_pred, pd.Series)
    expected = pd.core.indexes.period.PeriodIndex(
        [
            '1957-05', '1957-06', '1957-07', '1957-08', '1957-09', '1957-10',
            '1957-11', '1957-12', '1958-01', '1958-02', '1958-03', '1958-04'
        ],
        dtype='period[M]',
        freq='M'
    )
    assert np.all(y_pred.index == expected)

