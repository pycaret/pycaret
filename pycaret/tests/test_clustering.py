import os, sys

sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import pytest
import pycaret.clustering
import pycaret.datasets
import uuid
from mlflow.tracking.client import MlflowClient

@pytest.fixture(scope='module')
def jewellery_dataframe():
    return pycaret.datasets.get_data("jewellery")

@pytest.fixture(scope='module')
def tracking_api():
    client = MlflowClient()
    return client

def test(jewellery_dataframe):
    # loading dataset
    assert isinstance(jewellery_dataframe, pd.core.frame.DataFrame)

    # init setup
    clu1 = pycaret.clustering.setup(
        jewellery_dataframe,
        normalize=True,
        log_experiment=True,
        silent=True,
        html=False,
        session_id=123,
        n_jobs=1,
    )

    # create model
    kmeans = pycaret.clustering.create_model("kmeans")
    kmodes = pycaret.clustering.create_model("kmodes")

    # assign model
    kmeans_results = pycaret.clustering.assign_model(kmeans)
    kmodes_results = pycaret.clustering.assign_model(kmodes)
    assert isinstance(kmeans_results, pd.core.frame.DataFrame)
    assert isinstance(kmodes_results, pd.core.frame.DataFrame)

    # save model
    pycaret.clustering.save_model(kmeans, "kmeans_model_23122019")

    # load model
    saved_kmeans = pycaret.clustering.load_model("kmeans_model_23122019")

    # predict model
    kmeans_predictions = pycaret.clustering.predict_model(model=kmeans, data=jewellery_dataframe)
    assert isinstance(kmeans_predictions, pd.core.frame.DataFrame)

    # returns table of models
    all_models = pycaret.clustering.models()
    assert isinstance(all_models, pd.core.frame.DataFrame)

    # get config
    X = pycaret.clustering.get_config("X")
    seed = pycaret.clustering.get_config("seed")
    assert isinstance(X, pd.core.frame.DataFrame)
    assert isinstance(seed, int)

    # set config
    pycaret.clustering.set_config("seed", 124)
    seed = pycaret.clustering.get_config("seed")
    assert seed == 124

    assert 1 == 1

class TestClusteringExperimentCustomTags:
    def test_clustering_setup_fails_with_experiment_custom_tags(self, jewellery_dataframe):
        with pytest.raises(TypeError):
            # init setup
            _ = pycaret.clustering.setup(
                jewellery_dataframe,
                normalize=True,
                log_experiment=True,
                silent=True,
                html=False,
                session_id=123,
                n_jobs=1,
                experiment_name=uuid.uuid4().hex,
                experiment_custom_tags='custom_tag'
            )
    def test_clustering_create_model_fails_with_experiment_custom_tags(self, jewellery_dataframe):
        with pytest.raises(TypeError):
            # init setup
            _ = pycaret.clustering.setup(
                jewellery_dataframe,
                normalize=True,
                log_experiment=True,
                silent=True,
                html=False,
                session_id=123,
                n_jobs=1,
                experiment_name=uuid.uuid4().hex,
            )
            _ = pycaret.clustering.create_model("kmeans", experiment_custom_tags=('pytest', 'testing'))

    @pytest.mark.parametrize('custom_tag', [1, ('pytest', 'True'), True, 1000.0])
    def test_clustering_setup_fails_with_experiment_custom_multiples_inputs(self, custom_tag):
        with pytest.raises(TypeError):
            # init setup
            _ = pycaret.clustering.setup(
                pycaret.datasets.get_data("jewellery"),
                normalize=True,
                log_experiment=True,
                silent=True,
                html=False,
                session_id=123,
                n_jobs=1,
                experiment_name=uuid.uuid4().hex,
                experiment_custom_tags=custom_tag
            )
    def test_clustering_setup_with_experiment_custom_tags(self, jewellery_dataframe, tracking_api):
            experiment_name = uuid.uuid4().hex
            # init setup
            _ = pycaret.clustering.setup(
                jewellery_dataframe,
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

    def test_clustering_create_models_with_experiment_custom_tags(self, jewellery_dataframe, tracking_api):
            experiment_name = uuid.uuid4().hex
            # init setup
            _ = pycaret.clustering.setup(
                jewellery_dataframe,
                normalize=True,
                log_experiment=True,
                silent=True,
                html=False,
                session_id=123,
                n_jobs=1,
                experiment_name=experiment_name,
            )
            _ = pycaret.clustering.create_model("kmeans", experiment_custom_tags={'pytest' : 'testing'})
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
    TestClusteringExperimentCustomTags()
