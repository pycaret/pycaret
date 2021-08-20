import os, sys

sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import pytest
import pycaret.anomaly
import pycaret.datasets
import uuid
from mlflow.tracking.client import MlflowClient

@pytest.fixture(scope='module')
def anomaly_dataframe():
    return pycaret.datasets.get_data("anomaly")

@pytest.fixture(scope='module')
def tracking_api():
    client = MlflowClient()
    return client

def test(anomaly_dataframe):
    # loading dataset
    assert isinstance(anomaly_dataframe, pd.core.frame.DataFrame)

    # init setup
    ano1 = pycaret.anomaly.setup(
        anomaly_dataframe,
        normalize=True,
        log_experiment=True,
        silent=True,
        html=False,
        session_id=123,
        n_jobs=1,
    )

    # create model
    iforest = pycaret.anomaly.create_model("iforest")
    knn = pycaret.anomaly.create_model("knn")

    # assign model
    iforest_results = pycaret.anomaly.assign_model(iforest)
    knn_results = pycaret.anomaly.assign_model(knn)
    assert isinstance(iforest_results, pd.core.frame.DataFrame)
    assert isinstance(knn_results, pd.core.frame.DataFrame)

    # predict model
    iforest_predictions = pycaret.anomaly.predict_model(model=iforest, data=anomaly_dataframe)
    knn_predictions = pycaret.anomaly.predict_model(model=knn, data=anomaly_dataframe)
    assert isinstance(iforest_predictions, pd.core.frame.DataFrame)
    assert isinstance(knn_predictions, pd.core.frame.DataFrame)

    # get config
    X = pycaret.anomaly.get_config("X")
    seed = pycaret.anomaly.get_config("seed")
    assert isinstance(X, pd.core.frame.DataFrame)
    assert isinstance(seed, int)

    # set config
    pycaret.anomaly.set_config("seed", 124)
    seed = pycaret.anomaly.get_config("seed")
    assert seed == 124

    # save model
    pycaret.anomaly.save_model(knn, "knn_model_23122019")

    # load model
    saved_knn = pycaret.anomaly.load_model("knn_model_23122019")

    # returns table of models
    all_models = pycaret.anomaly.models()
    assert isinstance(all_models, pd.core.frame.DataFrame)

    assert 1 == 1

class TestAnomalyExperimentCustomTags:
    def test_anomaly_setup_fails_with_experiment_custom_tags(self, anomaly_dataframe):
        with pytest.raises(TypeError):
            # init setup
            _ = pycaret.anomaly.setup(
                anomaly_dataframe,
                normalize=True,
                log_experiment=True,
                silent=True,
                html=False,
                session_id=123,
                n_jobs=1,
                experiment_name=uuid.uuid4().hex,
                experiment_custom_tags='custom_tag'
            )
    def test_anomaly_create_model_fails_with_experiment_custom_tags(self, anomaly_dataframe):
        with pytest.raises(TypeError):
            # init setup
            _ = pycaret.anomaly.setup(
                anomaly_dataframe,
                normalize=True,
                log_experiment=True,
                silent=True,
                html=False,
                session_id=123,
                n_jobs=1,
                experiment_name=uuid.uuid4().hex,
            )
            _ = pycaret.anomaly.create_model("iforest", experiment_custom_tags=('pytest', 'evaluate'))
    
    @pytest.mark.parametrize('custom_tag', [1, ('pytest', 'True'), True, 1000.0])
    def test_anomaly_setup_fails_with_experiment_custom_multiples_inputs(self, custom_tag):
        with pytest.raises(TypeError):
            # init setup
            _ = pycaret.anomaly.setup(
                pycaret.datasets.get_data("anomaly"),
                normalize=True,
                log_experiment=True,
                silent=True,
                html=False,
                session_id=123,
                n_jobs=1,
                experiment_name=uuid.uuid4().hex,
                experiment_custom_tags=custom_tag
            )
    def test_anomaly_setup_with_experiment_custom_tags(self, anomaly_dataframe, tracking_api):
            experiment_name = uuid.uuid4().hex
            # init setup
            _ = pycaret.anomaly.setup(
                anomaly_dataframe,
                normalize=True,
                log_experiment=True,
                silent=True,
                html=False,
                session_id=123,
                n_jobs=1,
                experiment_name=experiment_name,
                experiment_custom_tags={'pytest' : 'testing'}
            )
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

    def test_anomaly_create_models_with_experiment_custom_tags(self, anomaly_dataframe, tracking_api):
            experiment_name = uuid.uuid4().hex
            # init setup
            _ = pycaret.anomaly.setup(
                anomaly_dataframe,
                normalize=True,
                log_experiment=True,
                silent=True,
                html=False,
                session_id=123,
                n_jobs=1,
                experiment_name=experiment_name,
            )
            _ = pycaret.anomaly.create_model("iforest", experiment_custom_tags={'pytest' : 'testing'})
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
