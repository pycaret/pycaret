import os, sys

sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import numpy as np
import pytest
import pycaret.classification
import pycaret.datasets
from IPython.display import display

@pytest.mark.skip("AutoViz is broken in 0.1.37 due to panel compat issue")
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

    # EDA
    pycaret.classification.eda(display_format = 'svg')

    # assert
    assert 1 == 1

if __name__ == "__main__":
    test()
