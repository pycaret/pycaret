import uuid

import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from sklearn.metrics import recall_score

import pycaret.classification
import pycaret.datasets


@pytest.fixture(scope="module")
def juice_dataframe():
    # loading dataset
    return pycaret.datasets.get_data("juice")


@pytest.mark.parametrize("return_train_score", [True, False])
def test_classification(juice_dataframe, return_train_score):
    assert isinstance(juice_dataframe, pd.core.frame.DataFrame)

    # init setup
    pycaret.classification.setup(
        juice_dataframe,
        target="Purchase",
        remove_multicollinearity=True,
        multicollinearity_threshold=0.95,
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
        pycaret.classification.tune_model(
            i, n_iter=3, return_train_score=return_train_score
        )
        for i in top3
    ]
    assert isinstance(tuned_top3, list)

    pycaret.classification.tune_model(
        top3[0], n_iter=3, choose_better=True, return_train_score=return_train_score
    )

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
    predict_holdout = pycaret.classification.predict_model(best, data=juice_dataframe)
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


def test_classification_predict_on_unseen(juice_dataframe):
    exp = pycaret.classification.ClassificationExperiment()
    # init setup
    exp.setup(
        juice_dataframe,
        target="Purchase",
        remove_multicollinearity=True,
        multicollinearity_threshold=0.95,
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
    exp.predict_model(model, juice_dataframe)


def test_classification_custom_metric(juice_dataframe):
    exp = pycaret.classification.ClassificationExperiment()
    # init setup
    exp.setup(
        juice_dataframe,
        target="Purchase",
        log_experiment=True,
        html=False,
        session_id=123,
        n_jobs=1,
    )

    # create a custom function (sklearn >=1.3.0 requires kwargs in func def)
    def specificity(y_true, y_pred, **kwargs):
        return recall_score(y_true, y_pred, pos_label=0, zero_division=1)

    # add metric to PyCaret
    exp.add_metric("specificity", "specificity", specificity, greater_is_better=True)

    lr = exp.create_model("lr")
    assert exp.pull()["specificity"].sum() != 0

    exp.predict_model(lr)
    assert exp.pull()["specificity"].sum() != 0


class TestClassificationExperimentCustomTags:
    def test_classification_setup_fails_with_experiment_custom_tags(
        self, juice_dataframe
    ):
        with pytest.raises(Exception):
            # init setup
            _ = pycaret.classification.setup(
                juice_dataframe,
                target="Purchase",
                remove_multicollinearity=True,
                multicollinearity_threshold=0.95,
                log_experiment=True,
                html=False,
                session_id=123,
                n_jobs=1,
                experiment_name=uuid.uuid4().hex,
                experiment_custom_tags="custom_tag",
            )

    @pytest.mark.parametrize("custom_tag", [1, ("pytest", "True"), True, 1000.0])
    def test_classification_setup_fails_with_experiment_custom_multiples_inputs(
        self, custom_tag
    ):
        with pytest.raises(Exception):
            # init setup
            _ = pycaret.classification.setup(
                pycaret.datasets.get_data("juice"),
                target="Purchase",
                remove_multicollinearity=True,
                multicollinearity_threshold=0.95,
                log_experiment=True,
                html=False,
                session_id=123,
                n_jobs=1,
                experiment_name=uuid.uuid4().hex,
                experiment_custom_tags=custom_tag,
            )

    def test_classification_models_with_experiment_custom_tags(self, juice_dataframe):
        # init setup
        experiment_name = uuid.uuid4().hex
        _ = pycaret.classification.setup(
            juice_dataframe,
            target="Purchase",
            remove_multicollinearity=True,
            multicollinearity_threshold=0.95,
            log_experiment=True,
            html=False,
            session_id=123,
            n_jobs=1,
            experiment_name=experiment_name,
        )

        # compare models
        _ = pycaret.classification.compare_models(
            errors="raise", n_select=100, experiment_custom_tags={"pytest": "testing"}
        )[:3]

        # get experiment data
        tracking_api = MlflowClient()
        experiment = tracking_api.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        # get run's info
        experiment_run = tracking_api.search_runs(experiment_id)[0]
        # get run id
        run_id = experiment_run.info.run_id
        # get run data
        run_data = tracking_api.get_run(run_id)
        # assert that custom tag was inserted
        assert "testing" == run_data.to_dictionary().get("data").get("tags").get(
            "pytest"
        )


if __name__ == "__main__":
    test_classification()
    test_classification_predict_on_unseen()
    TestClassificationExperimentCustomTags()
