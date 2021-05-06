import os, sys

sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import pytest
import pycaret.classification
import pycaret.datasets


def test():

    # loading dataset
    data = pycaret.datasets.get_data("nba")

    # pca_method = 'linear', pca_components = 5
    clf1 = pycaret.classification.setup(
        data,
        target="TARGET_5Yrs",
        silent=True,
        html=False,
        pca=True,
        pca_method="linear",
        pca_components=5,
        n_jobs=1,
    )
    assert len(pycaret.classification.get_config("X_train").columns) == 5

    # pca_method = 'kernel', pca_components = 6
    clf1 = pycaret.classification.setup(
        data,
        target="TARGET_5Yrs",
        silent=True,
        html=False,
        pca=True,
        pca_method="kernel",
        pca_components=6,
        n_jobs=1,
    )
    assert len(pycaret.classification.get_config("X_train").columns) == 6

    # pca_method = 'incremental', pca_components = 7
    clf1 = pycaret.classification.setup(
        data,
        target="TARGET_5Yrs",
        silent=True,
        html=False,
        pca=True,
        pca_method="incremental",
        pca_components=7,
        n_jobs=1,
    )
    assert len(pycaret.classification.get_config("X_train").columns) == 7


if __name__ == "__main__":
    test()
