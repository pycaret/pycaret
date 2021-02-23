import pytest

from random import choice, randint


@pytest.fixture(scope="session", name="load_data")
def load_data():
    """Load Pycaret Airline dataset."""
    from pycaret.datasets import get_data

    airline_data = get_data("airline")

    airline_data.index = airline_data.index.astype("datetime64[ns]")

    return airline_data


parametrize_list = [
    (randint(2, 5), randint(5, 10), "expandingwindow"),
    (randint(2, 5), randint(5, 10), "slidingwindow"),
]


@pytest.mark.parametrize("fold, forecast_horizon, fold_strategy", parametrize_list)
def test_setup_initialization(fold, forecast_horizon, fold_strategy, load_data):

    from pycaret.time_series import setup
    from sktime.forecasting.model_selection._split import (
        ExpandingWindowSplitter,
        SlidingWindowSplitter,
    )

    exp_name = setup(
        data=load_data,
        fold=fold,
        forecast_horizon=forecast_horizon,
        fold_strategy=fold_strategy,
    )

    if fold_strategy == "expandingwindow":
        assert isinstance(exp_name.fold_generator, ExpandingWindowSplitter)
    elif fold_strategy == "slidingwindow":
        assert isinstance(exp_name.fold_generator, SlidingWindowSplitter)
