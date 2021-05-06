import os, sys

sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import pytest
import pycaret.classification
import pycaret.datasets


def test():

    # loading dataset
    data = pycaret.datasets.get_data("blood")

    # categorical_feature = Recency
    clf1 = pycaret.classification.setup(
        data,
        target="Class",
        silent=True,
        html=False,
        ignore_features=["Time"],
        n_jobs=1,
    )
    assert "Time" not in list(pycaret.classification.get_config("X_train").columns)


if __name__ == "__main__":
    test()
