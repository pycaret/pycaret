import os, sys

sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import numpy as np
import pytest
import pycaret.classification
import pycaret.datasets


def test():

    # loading dataset
    data = pycaret.datasets.get_data("blood")

    # normalize_method = 'minmax'
    clf1 = pycaret.classification.setup(
        data,
        target="Class",
        silent=True,
        html=False,
        normalize=True,
        normalize_method="minmax",
        n_jobs=1,
    )
    assert np.array(pycaret.classification.get_config("X_train")).min().round(2) >= 0
    assert np.array(pycaret.classification.get_config("X_train")).max().round(2) <= 1

    # normalize_method = 'maxabs'
    clf1 = pycaret.classification.setup(
        data,
        target="Class",
        silent=True,
        html=False,
        normalize=True,
        normalize_method="maxabs",
        n_jobs=1,
    )
    assert np.array(pycaret.classification.get_config("X_train")).min().round(2) >= 0
    assert np.array(pycaret.classification.get_config("X_train")).max().round(2) <= 1

    # normalize_method = 'zscore'
    clf1 = pycaret.classification.setup(
        data,
        target="Class",
        silent=True,
        html=False,
        normalize=True,
        normalize_method="zscore",
        n_jobs=1,
    )
    assert np.array(pycaret.classification.get_config("X_train")).min().round(2) >= -10
    assert np.array(pycaret.classification.get_config("X_train")).max().round(2) <= 10

    # normalize_method = 'robust'
    clf1 = pycaret.classification.setup(
        data,
        target="Class",
        silent=True,
        html=False,
        normalize=True,
        normalize_method="robust",
        n_jobs=1,
    )
    # to create assert later on


if __name__ == "__main__":
    test()
