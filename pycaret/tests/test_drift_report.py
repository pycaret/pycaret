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

    # generate drift report
    predictions = pycaret.classification.predict_model(lr, drift_report=True)
    assert 1 == 1

if __name__ == "__main__":
    test()
