import os, sys

sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import numpy as np
import pytest
import pycaret.classification
import pycaret.regression
import pycaret.datasets


def test_classification_create_docker():

    # loading dataset
    data = pycaret.datasets.get_data("blood")

    # initialize setup
    clf1 = pycaret.classification.setup(
        data,
        target="Class",
        silent=True,
        html=False,
        n_jobs=1,
    )

    # train model
    lr = pycaret.classification.create_model("lr")

    # create api
    pycaret.classification.create_api(lr, "blood_api")
    pycaret.classification.create_docker("blood_api")
    assert 1 == 1


def test_regression_create_docker():

    # loading dataset
    data = pycaret.datasets.get_data("boston")

    # initialize setup
    reg1 = pycaret.regression.setup(
        data,
        target="medv",
        silent=True,
        html=False,
        n_jobs=1,
    )

    # train model
    lr = pycaret.regression.create_model("lr")

    # create api
    pycaret.regression.create_api(lr, "boston_api")
    pycaret.regression.create_docker("boston_api")
    assert 1 == 1


if __name__ == "__main__":
    test_classification_create_docker()
    test_regression_create_docker()
