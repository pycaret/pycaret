import os, sys

sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import pytest
import pycaret.datasets


def test():
    # loading list of datasets
    data = pycaret.datasets.get_data("index")
    assert isinstance(data, pd.core.frame.DataFrame)
    row, col = data.shape
    assert row > 1
    assert col == 8

    # loading dataset
    credit = pycaret.datasets.get_data("credit")
    assert isinstance(credit, pd.core.frame.DataFrame)
    row, col = credit.shape
    assert row == 24000
    assert col == 24
    assert credit.size == 576000

    assert 1 == 1


if __name__ == "__main__":
    test()
