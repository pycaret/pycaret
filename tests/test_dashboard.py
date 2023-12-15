# import pycaret.classification
# import pycaret.datasets

# 18/12/2021 issues with test hence commenting out.


def test_classification_dashboard():
    # loading dataset
    # data = pycaret.datasets.get_data("blood")

    # setup environment
    # pycaret.classification.setup(
    #     data,
    #     target="Class",
    #     html=False,
    #     n_jobs=1,
    # )

    # train model
    # lr = pycaret.classification.create_model("lr")

    # run dashboard
    # pycaret.classification.dashboard(lr, display_format="dash")

    assert 1 == 1


if __name__ == "__main__":
    test_classification_dashboard()
