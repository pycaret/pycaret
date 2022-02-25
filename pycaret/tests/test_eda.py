import os, sys

sys.path.insert(0, os.path.abspath(".."))

import pycaret.classification
import pycaret.datasets


def test():

    # loading dataset
    data = pycaret.datasets.get_data("blood")

    # initialize setup
    pycaret.classification.setup(
        data,
        target="Class",
        silent=True,
        html=False,
        n_jobs=1,
    )

    # EDA
    pycaret.classification.eda(display_format='svg')

    # assert
    assert 1 == 1
