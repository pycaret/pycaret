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
    (randint(2, 5), randint(5, 10), "expanding"),
    (randint(2, 5), randint(5, 10), "rolling"),
    (randint(2, 5), randint(5, 10), "sliding"),
]


@pytest.mark.parametrize("fold, fh, fold_strategy", parametrize_list)
def test_setup_initialization(fold, fh, fold_strategy, load_data):

    from pycaret.time_series import setup
    from sktime.forecasting.model_selection._split import (
        ExpandingWindowSplitter,
        SlidingWindowSplitter,
    )

    exp_name = setup(
        data=load_data,
        fold=fold,
        fh=fh,
        fold_strategy=fold_strategy,
    )

    allowed_fold_strategies = ["expanding", "rolling", "sliding"]
    if fold_strategy in allowed_fold_strategies:
        if (fold_strategy == "expanding") or (fold_strategy == "rolling"):
            assert isinstance(exp_name.fold_generator, ExpandingWindowSplitter)
        elif fold_strategy == "sliding":
            assert isinstance(exp_name.fold_generator, SlidingWindowSplitter)

        # expected = int(len(load_data)*train_size) - fold * fh  # Since fh is an int, if train_size splits original data
        expected = int(len(load_data) - fh) - fold * fh  # if fh splits original data
        assert exp_name.fold_generator.initial_window == expected
        assert np.all(exp_name.fold_generator.fh == np.arange(1, fh + 1))
        assert exp_name.fold_generator.step_length == fh  # Since fh is an int


setup_raises_list = [
    (randint(50, 100), randint(10, 20), "expanding"),
    (randint(50, 100), randint(10, 20), "rolling"),
    (randint(50, 100), randint(10, 20), "sliding"),
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
