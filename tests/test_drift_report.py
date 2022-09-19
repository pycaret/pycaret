import pycaret.classification
import pycaret.datasets


def test_drift_report():

    # loading dataset
    data = pycaret.datasets.get_data("blood")

    # initialize setup
    pycaret.classification.setup(
        data,
        target="Class",
        html=False,
        n_jobs=1,
    )

    # train model
    lr = pycaret.classification.create_model("lr")

    # generate drift report
    pycaret.classification.predict_model(lr, drift_report=True)
    assert 1 == 1
