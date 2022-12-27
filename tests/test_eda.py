import pytest

import pycaret.classification
import pycaret.datasets


@pytest.mark.skip("AutoViz is broken in 0.1.37 due to panel compat issue")
def test_eda():

    # loading dataset
    data = pycaret.datasets.get_data("blood")

    # initialize setup
    pycaret.classification.setup(
        data,
        target="Class",
        html=False,
        n_jobs=1,
    )

    # EDA
    pycaret.classification.eda(display_format="svg")

    # assert
    assert 1 == 1
