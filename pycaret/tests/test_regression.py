import os, sys
from typing import Type

sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import pytest
import pycaret.regression
import pycaret.datasets
from mlflow.tracking.client import MlflowClient
import uuid


@pytest.fixture(scope="module")
def boston_dataframe():
    return pycaret.datasets.get_data("boston")


@pytest.fixture(scope="module")
def tracking_api():
    client = MlflowClient()
    return client


def test(boston_dataframe):
    # loading dataset
    assert isinstance(boston_dataframe, pd.core.frame.DataFrame)

    # init setup
    reg1 = pycaret.regression.setup(
        boston_dataframe,
        target="medv",
        remove_multicollinearity=True,
        multicollinearity_threshold=0.95,
        silent=True,
        log_experiment=True,
        html=False,
        session_id=123,
        n_jobs=1,
        experiment_name=uuid.uuid4().hex,
    )

    # compare models
    top3 = pycaret.regression.compare_models(n_select=100, exclude=["catboost"])[:3]
    assert isinstance(top3, list)

    # tune model
    tuned_top3 = [pycaret.regression.tune_model(i, n_iter=3) for i in top3]
    assert isinstance(tuned_top3, list)

    pycaret.regression.tune_model(top3[0], n_iter=3, choose_better=True)

    # ensemble model
    bagged_top3 = [pycaret.regression.ensemble_model(i) for i in tuned_top3]
    assert isinstance(bagged_top3, list)

    # blend models
    blender = pycaret.regression.blend_models(top3)

    # stack models
    stacker = pycaret.regression.stack_models(
        estimator_list=top3[1:], meta_model=top3[0]
    )

    # plot model
    lr = pycaret.regression.create_model("lr")
    pycaret.regression.plot_model(
        lr, save=True
    )  # scale removed because build failed due to large image size

    # select best model
    best = pycaret.regression.automl(optimize="MAPE")

    # hold out predictions
    predict_holdout = pycaret.regression.predict_model(best)
    assert isinstance(predict_holdout, pd.core.frame.DataFrame)

    # predictions on new dataset
    predict_holdout = pycaret.regression.predict_model(best, data=boston_dataframe)
    assert isinstance(predict_holdout, pd.core.frame.DataFrame)

    # finalize model
    final_best = pycaret.regression.finalize_model(best)

    # save model
    pycaret.regression.save_model(best, "best_model_23122019")

    # load model
    saved_best = pycaret.regression.load_model("best_model_23122019")

    # returns table of models
    all_models = pycaret.regression.models()
    assert isinstance(all_models, pd.core.frame.DataFrame)

    # get config
    X_train = pycaret.regression.get_config("X_train")
    X_test = pycaret.regression.get_config("X_test")
    y_train = pycaret.regression.get_config("y_train")
    y_test = pycaret.regression.get_config("y_test")
    assert isinstance(X_train, pd.core.frame.DataFrame)
    assert isinstance(X_test, pd.core.frame.DataFrame)
    assert isinstance(y_train, pd.core.series.Series)
    assert isinstance(y_test, pd.core.series.Series)

    # set config
    pycaret.regression.set_config("seed", 124)
    seed = pycaret.regression.get_config("seed")
    assert seed == 124

    assert 1 == 1


class TestRegressionExperimentCustomTags:
    def test_regression_setup_fails_with_experiment_custom_tags(self, boston_dataframe):
        with pytest.raises(TypeError):
            # init setup
            _ = pycaret.regression.setup(
                boston_dataframe,
                target="medv",
                silent=True,
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
        with pytest.raises(TypeError):
            # init setup
            _ = pycaret.regression.setup(
                pycaret.datasets.get_data("boston"),
                target="medv",
                silent=True,
                log_experiment=True,
                html=False,
                session_id=123,
                n_jobs=1,
                experiment_name=uuid.uuid4().hex,
                experiment_custom_tags=custom_tag,
            )

    def test_regression_compare_models_fails_with_experiment_custom_tags(
        self, boston_dataframe
    ):
        with pytest.raises(TypeError):
            # init setup
            _ = pycaret.regression.setup(
                boston_dataframe,
                target="medv",
                silent=True,
                log_experiment=True,
                html=False,
                session_id=123,
                n_jobs=1,
                experiment_name=uuid.uuid4().hex,
            )

            # compare models
            _ = pycaret.regression.compare_models(
                n_select=100, experiment_custom_tags="custom_tag"
            )[:3]

    def test_regression_finalize_models_fails_with_experiment_custom_tags(
        self, boston_dataframe
    ):
        with pytest.raises(TypeError):
            # init setup
            _ = pycaret.regression.setup(
                boston_dataframe,
                target="medv",
                silent=True,
                log_experiment=True,
                html=False,
                session_id=123,
                n_jobs=1,
                experiment_name=uuid.uuid4().hex,
            )

            # compare models
            _ = pycaret.regression.compare_models(
                n_select=100, experiment_custom_tags={"pytest": "testing"}
            )[:2]

            # select best model
            best = pycaret.regression.automl(optimize="MAPE")

            # finalize model
            _ = pycaret.regression.finalize_model(best, experiment_custom_tags="pytest")

    def test_regression_models_with_experiment_custom_tags(
        self, boston_dataframe, tracking_api
    ):
        # init setup
        experiment_name = uuid.uuid4().hex
        _ = pycaret.regression.setup(
            boston_dataframe,
            target="medv",
            silent=True,
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
        experiment = [
            e for e in tracking_api.list_experiments() if e.name == experiment_name
        ][0]
        experiment_id = experiment.experiment_id
        # get run's info
        experiment_run = tracking_api.list_run_infos(experiment_id)[0]
        # get run id
        run_id = experiment_run.run_id
        # get run data
        run_data = tracking_api.get_run(run_id)
        # assert that custom tag was inserted
        assert "testing" == run_data.to_dictionary().get("data").get("tags").get(
            "pytest"
        )


if __name__ == "__main__":
    test()
    TestRegressionExperimentCustomTags()
