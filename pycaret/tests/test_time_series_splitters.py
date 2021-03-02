import pytest

from random import choice, randint
import numpy as np

@pytest.fixture(scope="session", name="load_data")
def load_data():
    """Load Pycaret Airline dataset."""
    from pycaret.datasets import get_data
    airline_data = get_data("airline")
    return airline_data


parametrize_list = [
    (randint(2, 5), randint(5, 10), "expandingwindow"),
    (randint(2, 5), randint(5, 10), "slidingwindow"),
    # (randint(2, 5), randint(5, 10), "timeseries"),  # Not supporting right now
]


@pytest.mark.parametrize("fold, fh, fold_strategy", parametrize_list)
def test_setup_initialization(fold, fh, fold_strategy, load_data):

    from pycaret.time_series import setup
    from sktime.forecasting.model_selection._split import (
        ExpandingWindowSplitter,
        SlidingWindowSplitter,
    )
    from sklearn.model_selection import TimeSeriesSplit

    train_size = 0.7
    exp_name = setup(
        data=load_data,
        fold=fold,
        fh=fh,
        fold_strategy=fold_strategy,
        train_size=train_size
    )

    if (fold_strategy == "expandingwindow") or (fold_strategy == "slidingwindow"):
        if fold_strategy == "expandingwindow":
            assert isinstance(exp_name.fold_generator, ExpandingWindowSplitter)
        elif fold_strategy == "slidingwindow":
            assert isinstance(exp_name.fold_generator, SlidingWindowSplitter)

        expected = int(len(load_data)*train_size) - fold * fh  # Since fh is an int
        assert exp_name.fold_generator.initial_window == expected
        assert np.all(exp_name.fold_generator.fh == np.arange(1, fh+1))
        assert exp_name.fold_generator.step_length == fh  # Since fh is an int
    # elif fold_strategy == "timeseries":
    #     assert isinstance(exp_name.fold_generator, TimeSeriesSplit)


setup_raises_list = [
    (randint(50, 100), randint(10, 20), "expandingwindow"),
    (randint(50, 100), randint(10, 20), "slidingwindow"),
]


@pytest.mark.parametrize("fold, fh, fold_strategy", setup_raises_list)
def test_setup_raises(fold, fh, fold_strategy, load_data):

    from pycaret.time_series import setup

    with pytest.raises(ValueError) as errmsg:
        _ = setup(
            data=load_data,
            fold=fold,
            fh=fh,
            fold_strategy=fold_strategy,
        )

    exceptionmsg = errmsg.value.args[0]

    assert exceptionmsg == "Not Enough Data Points, set a lower number of folds or fh"
