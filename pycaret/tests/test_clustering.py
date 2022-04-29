import os, sys

sys.path.insert(0, os.path.abspath(".."))

import uuid
import pytest
import pandas as pd
import pycaret.clustering
import pycaret.datasets
from mlflow.tracking.client import MlflowClient


@pytest.fixture(scope="module")
def data():
    return pycaret.datasets.get_data("jewellery")


def test(data):
    experiment_name = uuid.uuid4().hex
    pycaret.clustering.setup(
        data,
        normalize=True,
        log_experiment=True,
        experiment_name=experiment_name,
        experiment_custom_tags={"tag": 1},
        log_plots=True,
        silent=True,
        html=False,
        session_id=123,
        n_jobs=1,
    )

    # create model
    kmeans = pycaret.clustering.create_model(
        "kmeans", experiment_custom_tags={"tag": 1}
    )
    kmodes = pycaret.clustering.create_model(
        "kmodes", experiment_custom_tags={"tag": 1}
    )

    # assign model
    kmeans_results = pycaret.clustering.assign_model(kmeans)
    kmodes_results = pycaret.clustering.assign_model(kmodes)
    assert isinstance(kmeans_results, pd.DataFrame)
    assert isinstance(kmodes_results, pd.DataFrame)

    # save model
    pycaret.clustering.save_model(kmeans, "kmeans_model_23122019")

    # load model
    pycaret.clustering.load_model("kmeans_model_23122019")

    # predict model
    kmeans_predictions = pycaret.clustering.predict_model(model=kmeans, data=data)
    assert isinstance(kmeans_predictions, pd.DataFrame)

    # returns table of models
    all_models = pycaret.clustering.models()
    assert isinstance(all_models, pd.DataFrame)

    # get config
    X = pycaret.clustering.get_config("X")
    seed = pycaret.clustering.get_config("seed")
    assert isinstance(X, pd.DataFrame)
    assert isinstance(seed, int)

    # set config
    pycaret.clustering.set_config("seed", 124)
    seed = pycaret.clustering.get_config("seed")
    assert seed == 124

    # Assert the custom tags are created
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    for experiment_run in client.list_run_infos(experiment.experiment_id):
        run = client.get_run(experiment_run.run_id)
        assert run.data.tags.get("tag") == "1"
