import pandas as pd
import pytest

import pycaret.classification
import pycaret.datasets


@pytest.fixture(scope="module")
def iris_dataframe():
    # loading dataset
    return pycaret.datasets.get_data("iris")


@pytest.mark.parametrize("return_train_score", [True, False])
def test_multiclass(iris_dataframe, return_train_score):
    # loading dataset
    assert isinstance(iris_dataframe, pd.DataFrame)

    # init setup
    pycaret.classification.setup(
        iris_dataframe,
        target="species",
        log_experiment=True,
        html=False,
        session_id=123,
        n_jobs=1,
    )

    # compare models
    top3 = pycaret.classification.compare_models(errors="raise", n_select=100)[:3]
    assert isinstance(top3, list)

    # tune model
    tuned_top3 = [
        pycaret.classification.tune_model(i, return_train_score=return_train_score)
        for i in top3
    ]
    assert isinstance(tuned_top3, list)

    # ensemble model
    bagged_top3 = [
        pycaret.classification.ensemble_model(i, return_train_score=return_train_score)
        for i in tuned_top3
    ]
    assert isinstance(bagged_top3, list)

    # blend models
    pycaret.classification.blend_models(top3, return_train_score=return_train_score)

    # stack models
    stacker = pycaret.classification.stack_models(
        estimator_list=top3, return_train_score=return_train_score
    )
    pycaret.classification.predict_model(stacker)

    # plot model
    lr = pycaret.classification.create_model(
        "lr", return_train_score=return_train_score
    )
    pycaret.classification.plot_model(lr, save=True, scale=5)

    # select best model
    pycaret.classification.automl(optimize="MCC", use_holdout=True)
    best = pycaret.classification.automl(optimize="MCC")

    # hold out predictions
    predict_holdout = pycaret.classification.predict_model(best)
    assert isinstance(predict_holdout, pd.DataFrame)

    # predictions on new dataset
    predict_holdout = pycaret.classification.predict_model(
        best, data=iris_dataframe.drop("species", axis=1)
    )
    assert isinstance(predict_holdout, pd.DataFrame)

    # calibrate model
    pycaret.classification.calibrate_model(best, return_train_score=return_train_score)

    # finalize model
    pycaret.classification.finalize_model(best)

    # save model
    pycaret.classification.save_model(best, "best_model_23122019")

    # load model
    pycaret.classification.load_model("best_model_23122019")

    # returns table of models
    all_models = pycaret.classification.models()
    assert isinstance(all_models, pd.DataFrame)

    # get config
    X_train = pycaret.classification.get_config("X_train")
    X_test = pycaret.classification.get_config("X_test")
    y_train = pycaret.classification.get_config("y_train")
    y_test = pycaret.classification.get_config("y_test")
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

    # set config
    pycaret.classification.set_config("seed", 124)
    seed = pycaret.classification.get_config("seed")
    assert seed == 124

    assert 1 == 1


def test_multiclass_predict_on_unseen(iris_dataframe):
    exp = pycaret.classification.ClassificationExperiment()
    # init setup
    exp.setup(
        iris_dataframe,
        target="species",
        log_experiment=True,
        html=False,
        session_id=123,
        n_jobs=1,
    )
    model = exp.create_model("dt", cross_validation=False)

    # save model
    exp.save_model(model, "best_model_23122019")

    exp = pycaret.classification.ClassificationExperiment()
    # load model
    model = exp.load_model("best_model_23122019")
    exp.predict_model(model, iris_dataframe)


if __name__ == "__main__":
    test_multiclass()
    test_multiclass_predict_on_unseen()
