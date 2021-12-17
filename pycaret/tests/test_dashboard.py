import os, sys

sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import numpy as np
import pytest
import pycaret.classification
import pycaret.datasets


def test_classification_dashboard():

    # loading dataset
    data = pycaret.datasets.get_data("blood")

    # setup environment
    clf1 = pycaret.classification.setup(
        data,
        target="Class",
        silent=True,
        html=False,
        n_jobs=1,
    )
   
    # train model
    lr = pycaret.classification.create_model("lr")

    # run dashboard
    pycaret.classification.dashboard(lr, display_format = 'inline')

    # assert statement
    assert 1 == 1

def test_regression_dashboard():

    # loading dataset
    data = pycaret.datasets.get_data("boston")

    # setup environment
    reg1 = pycaret.regression.setup(
        data,
        target="medv",
        silent=True,
        html=False,
        n_jobs=1,
    )
   
    # train model
    dt = pycaret.regression.create_model("dt")

    # run dashboard
    pycaret.regression.dashboard(dt, display_format = 'inline')

    # assert statement
    assert 1 == 1

if __name__ == "__main__":
    test_classification_dashboard()
    test_regression_dashboard()