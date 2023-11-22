import pycaret.classification
import pycaret.datasets
import pycaret.regression


def test_classification_convert_model():
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

    # convert model
    lr_java = pycaret.classification.convert_model(lr, "java")
    assert isinstance(lr_java, str)


def test_regression_convert_model():
    # loading dataset
    data = pycaret.datasets.get_data("boston")

    # initialize setup
    pycaret.regression.setup(
        data,
        target="medv",
        html=False,
        n_jobs=1,
    )

    # train model
    lr = pycaret.regression.create_model("lr")

    # convert model
    lr_java = pycaret.regression.convert_model(lr, "java")
    assert isinstance(lr_java, str)


if __name__ == "__main__":
    test_classification_convert_model()
    test_regression_convert_model()
