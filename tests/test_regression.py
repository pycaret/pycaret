import uuid

import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

import pycaret.datasets
import pycaret.regression


@pytest.fixture(scope="module")
def boston_dataframe():
    return pycaret.datasets.get_data("boston")


@pytest.mark.parametrize("return_train_score", [True, False])
def test_regression(boston_dataframe, return_train_score):
    # loading dataset
    assert isinstance(boston_dataframe, pd.DataFrame)

    # init setup
    pycaret.regression.setup(
        boston_dataframe,
        target="medv",
        remove_multicollinearity=True,
        multicollinearity_threshold=0.95,
        log_experiment=True,
        html=False,
        session_id=123,
        n_jobs=1,
        experiment_name=uuid.uuid4().hex,
    )

    # compare models
    top3 = pycaret.regression.compare_models(
        n_select=100,
        exclude=["catboost"],
        errors="raise",
    )[:3]
    assert isinstance(top3, list)

    # tune model
    tuned_top3 = [
        pycaret.regression.tune_model(
            i, n_iter=3, return_train_score=return_train_score
        )
        for i in top3
    ]
    assert isinstance(tuned_top3, list)

    pycaret.regression.tune_model(
        top3[0], n_iter=3, choose_better=True, return_train_score=return_train_score
    )

    # ensemble model
    bagged_top3 = [
        pycaret.regression.ensemble_model(i, return_train_score=return_train_score)
        for i in tuned_top3
    ]
    assert isinstance(bagged_top3, list)

    # blend models
    pycaret.regression.blend_models(top3, return_train_score=return_train_score)

    # stack models
    pycaret.regression.stack_models(
        estimator_list=top3[1:],
        meta_model=top3[0],
        return_train_score=return_train_score,
    )

    # plot model
    lr = pycaret.regression.create_model("lr", return_train_score=return_train_score)
    pycaret.regression.plot_model(
        lr, save=True
    )  # scale removed because build failed due to large image size

    # select best model
    pycaret.regression.automl(optimize="MAPE", use_holdout=True)
    best = pycaret.regression.automl(optimize="MAPE")

    # hold out predictions
    predict_holdout = pycaret.regression.predict_model(best)
    assert isinstance(predict_holdout, pd.DataFrame)

    # predictions on new dataset
    predict_holdout = pycaret.regression.predict_model(best, data=boston_dataframe)
    assert isinstance(predict_holdout, pd.DataFrame)

    # finalize model
    pycaret.regression.finalize_model(best)

    # save model
    pycaret.regression.save_model(best, "best_model_23122019")

    # load model
    pycaret.regression.load_model("best_model_23122019")

    # returns table of models
    all_models = pycaret.regression.models()
    assert isinstance(all_models, pd.DataFrame)

    # get config
    X_train = pycaret.regression.get_config("X_train")
    X_test = pycaret.regression.get_config("X_test")
    y_train = pycaret.regression.get_config("y_train")
    y_test = pycaret.regression.get_config("y_test")
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

    # set config
    pycaret.regression.set_config("seed", 124)
    seed = pycaret.regression.get_config("seed")
    assert seed == 124

    assert 1 == 1


def test_regression_predict_on_unseen(boston_dataframe):
    exp = pycaret.regression.RegressionExperiment()
    # init setup
    exp.setup(
        boston_dataframe,
        target="medv",
        remove_multicollinearity=True,
        multicollinearity_threshold=0.95,
        log_experiment=True,
        html=False,
        session_id=123,
        n_jobs=1,
        experiment_name=uuid.uuid4().hex,
    )
    model = exp.create_model("dt", cross_validation=False)

    # save model
    exp.save_model(model, "best_model_23122019")

    exp = pycaret.regression.RegressionExperiment()
    # load model
    model = exp.load_model("best_model_23122019")
    exp.predict_model(model, boston_dataframe)


def test_regression_target_transformation(boston_dataframe):
    exp = pycaret.regression.RegressionExperiment()
    # init setup
    exp.setup(
        boston_dataframe,
        target="medv",
        transform_target=True,
        log_experiment=True,
        html=False,
        session_id=123,
        n_jobs=1,
        experiment_name=uuid.uuid4().hex,
    )
    model = exp.create_model("dt", cross_validation=False)
    preds = exp.predict_model(model)
    assert np.isclose(preds["prediction_label"].iloc[0], 49.999989)


class TestRegressionExperimentCustomTags:
    def test_regression_setup_fails_with_experiment_custom_tags(self, boston_dataframe):
        with pytest.raises(Exception):
            # init setup
            _ = pycaret.regression.setup(
                boston_dataframe,
                target="medv",
                log_experiment=True,
                html=False,
                session_id=123,
                n_jobs=1,
                experiment_name=uuid.uuid4().hex,
                experiment_custom_tags="custom_tag",
            )

    @pytest.mark.parametrize("custom_tag", [1, ("pytest", "True"), True, 1000.0])
    def test_regression_setup_fails_with_experiment_custom_multiples_inputs(
        self, custom_tag
    ):
        with pytest.raises(Exception):
            # init setup
            _ = pycaret.regression.setup(
                pycaret.datasets.get_data("boston"),
                target="medv",
                log_experiment=True,
                html=False,
                session_id=123,
                n_jobs=1,
                experiment_name=uuid.uuid4().hex,
                experiment_custom_tags=custom_tag,
            )

    def test_regression_models_with_experiment_custom_tags(self, boston_dataframe):
        # init setup
        experiment_name = uuid.uuid4().hex
        _ = pycaret.regression.setup(
            boston_dataframe,
            target="medv",
            log_experiment=True,
            html=False,
            session_id=123,
            n_jobs=1,
            experiment_name=experiment_name,
        )
        _ = pycaret.regression.compare_models(
            n_select=100, experiment_custom_tags={"pytest": "testing"}
        )[:2]

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
    test_regression()
    test_regression_predict_on_unseen()
    TestRegressionExperimentCustomTags()
