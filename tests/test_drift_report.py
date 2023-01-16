import os

import pytest
from sklearn.model_selection import train_test_split

import pycaret.classification
import pycaret.datasets


def test_drift_report(tmpdir):
    # loading dataset
    data = pycaret.datasets.get_data("blood")
    experiment = pycaret.classification.ClassificationExperiment()

    # initialize setup
    experiment.setup(
        data,
        target="Class",
        html=False,
        n_jobs=1,
    )

    # generate drift report
    file = experiment.drift_report()
    assert os.path.exists(file)


def test_drift_report_no_setup(tmpdir):
    # loading dataset
    data = pycaret.datasets.get_data("blood")
    reference_data, current_data = train_test_split(data, test_size=0.2, shuffle=False)
    experiment = pycaret.classification.ClassificationExperiment()

    with pytest.raises(ValueError):
        experiment.drift_report()

    # generate drift report
    file = experiment.drift_report(
        reference_data=reference_data,
        current_data=current_data,
        target="Class",
        categorical_features=["Recency"],
    )
    assert os.path.exists(file)
