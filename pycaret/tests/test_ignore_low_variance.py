import os, sys

sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import pytest
import pycaret.classification
import pycaret.datasets


def test():

    # loading dataset
    data = pycaret.datasets.get_data("blood")
    data["dummyCol"] = "DummyVal"  # create a new low variance column
    clf1 = pycaret.classification.setup(
        data,
        target="Class",
        silent=True,
        html=False,
        ignore_low_variance=True,
        remove_perfect_collinearity=False,
        n_jobs=1,
    )
    assert len(data.columns) - 1 - 1 == len(
        pycaret.classification.get_config("X").columns
    )


if __name__ == "__main__":
    test()
