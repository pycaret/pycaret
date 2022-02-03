import os, sys

sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import pytest
import pycaret.classification
import pycaret.datasets
from mlflow.tracking.client import MlflowClient
import uuid

@pytest.fixture(scope='module')
def juice_dataframe():
    # loading dataset
    return pycaret.datasets.get_data("juice")

@pytest.fixture(scope='module')
def tracking_api():
    client = MlflowClient()
    return client

def test(juice_dataframe):

    assert isinstance(juice_dataframe, pd.core.frame.DataFrame)

    # init setup
    clf1 = pycaret.classification.setup(
        juice_dataframe,
        target="Purchase",
        remove_multicollinearity=True,
        multicollinearity_threshold=0.95,
        log_experiment=True,
        silent=True,
        html=False,
        session_id=123,
        n_jobs=1,
    )

    # compare models
    top3 = pycaret.classification.compare_models(errors="raise", n_select=100)[:3]
    assert isinstance(top3, list)

    # tune model
    tuned_top3 = [pycaret.classification.tune_model(i, n_iter=3) for i in top3]
    assert isinstance(tuned_top3, list)

    pycaret.classification.tune_model(top3[0], n_iter=3, choose_better=True)

    # ensemble model
    bagged_top3 = [pycaret.classification.ensemble_model(i) for i in tuned_top3]
    assert isinstance(bagged_top3, list)

    # blend models
    blender = pycaret.classification.blend_models(top3)

    # stack models
    stacker = pycaret.classification.stack_models(estimator_list=top3)
    predict_holdout = pycaret.classification.predict_model(stacker)

    # plot model
    lr = pycaret.classification.create_model("lr")
    pycaret.classification.plot_model(lr, save=True, scale=5)

    # select best model
    best = pycaret.classification.automl(optimize="MCC")

    # hold out predictions
    predict_holdout = pycaret.classification.predict_model(best)
    assert isinstance(predict_holdout, pd.core.frame.DataFrame)

    # predictions on new dataset
    predict_holdout = pycaret.classification.predict_model(best, data=juice_dataframe)
    assert isinstance(predict_holdout, pd.core.frame.DataFrame)

    # calibrate model
    calibrated_best = pycaret.classification.calibrate_model(best)

    # finalize model
    final_best = pycaret.classification.finalize_model(best)

    # save model
    pycaret.classification.save_model(best, "best_model_23122019")

    # load model
    saved_best = pycaret.classification.load_model("best_model_23122019")

    # returns table of models
    all_models = pycaret.classification.models()
    assert isinstance(all_models, pd.core.frame.DataFrame)

    # get config
    X_train = pycaret.classification.get_config("X_train")
    X_test = pycaret.classification.get_config("X_test")
    y_train = pycaret.classification.get_config("y_train")
    y_test = pycaret.classification.get_config("y_test")
    assert isinstance(X_train, pd.core.frame.DataFrame)
    assert isinstance(X_test, pd.core.frame.DataFrame)
    assert isinstance(y_train, pd.core.series.Series)
    assert isinstance(y_test, pd.core.series.Series)

    # set config
    pycaret.classification.set_config("seed", 124)
    seed = pycaret.classification.get_config("seed")
    assert seed == 124

    assert 1 == 1


class TestClassificationExperimentCustomTags:
    def test_classification_setup_fails_with_experiment_custom_tags(self, juice_dataframe):
        with pytest.raises(TypeError):
            # init setup
            _ = pycaret.classification.setup(
                juice_dataframe,
                target="Purchase",
                remove_multicollinearity=True,
                multicollinearity_threshold=0.95,
                log_experiment=True,
                silent=True,
                html=False,
                session_id=123,
                n_jobs=1,
                experiment_name=uuid.uuid4().hex,
                experiment_custom_tags='custom_tag'
            )

    @pytest.mark.parametrize('custom_tag', [1, ('pytest', 'True'), True, 1000.0])
    def test_classification_setup_fails_with_experiment_custom_multiples_inputs(self, custom_tag):
        with pytest.raises(TypeError):
            # init setup
            _ = pycaret.classification.setup(
                pycaret.datasets.get_data("juice"),
                target="Purchase",
                remove_multicollinearity=True,
                multicollinearity_threshold=0.95,
                log_experiment=True,
                silent=True,
                html=False,
                session_id=123,
                n_jobs=1,
                experiment_name=uuid.uuid4().hex,
                experiment_custom_tags=custom_tag
            )

    def test_classification_compare_models_fails_with_experiment_custom_tags(self, juice_dataframe):
        with pytest.raises(TypeError):
            # init setup
            _ = pycaret.classification.setup(
                juice_dataframe,
                target="Purchase",
                remove_multicollinearity=True,
                multicollinearity_threshold=0.95,
                log_experiment=True,
                silent=True,
                html=False,
                session_id=123,
                n_jobs=1,
                experiment_name=uuid.uuid4().hex,
                experiment_custom_tags={'pytest' : 'awesome_framework'}
            )

            # compare models
            _ = pycaret.classification.compare_models(errors="raise", n_select=100, experiment_custom_tags='invalid_tag')[:3]

    def test_classification_finalize_models_fails_with_experiment_custom_tags(self, juice_dataframe):
        with pytest.raises(TypeError):
            # init setup
            _ = pycaret.classification.setup(
                juice_dataframe,
                target="Purchase",
                remove_multicollinearity=True,
                multicollinearity_threshold=0.95,
                log_experiment=True,
                silent=True,
                html=False,
                session_id=123,
                n_jobs=1,
                experiment_name=uuid.uuid4().hex,
                experiment_custom_tags={'pytest' : 'awesome_framework'}
            )


            # compare models
            _ = pycaret.classification.compare_models(errors="raise", n_select=100)[:3]

            # select best model
            best = pycaret.classification.automl(optimize="MCC")

            # finalize model
            _ = pycaret.classification.finalize_model(best, experiment_custom_tags='pytest')


    def test_classification_models_with_experiment_custom_tags(self, juice_dataframe, tracking_api):
        # init setup
        experiment_name = uuid.uuid4().hex
        _ = pycaret.classification.setup(
            juice_dataframe,
            target="Purchase",
            remove_multicollinearity=True,
            multicollinearity_threshold=0.95,
            log_experiment=True,
            silent=True,
            html=False,
            session_id=123,
            n_jobs=1,
            experiment_name=experiment_name,
        )

        # compare models
        _ = pycaret.classification.compare_models(errors="raise", n_select=100, experiment_custom_tags={'pytest' : 'testing'})[:3]
        #get experiment data
        experiment = [e for e in tracking_api.list_experiments() if e.name == experiment_name][0]
        experiment_id = experiment.experiment_id
        #get run's info
        experiment_run = tracking_api.list_run_infos(experiment_id)[0]
        #get run id
        run_id = experiment_run.run_id
        #get run data
        run_data = tracking_api.get_run(run_id)
        #assert that custom tag was inserted
        assert 'testing' == run_data.to_dictionary().get('data').get("tags").get("pytest")


if __name__ == "__main__":
    test()
    TestClassificationExperimentCustomTags()
