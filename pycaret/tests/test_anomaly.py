import os, sys

sys.path.insert(0, os.path.abspath(".."))

import uuid
import pytest
import pandas as pd
import pycaret.anomaly
import pycaret.datasets
from mlflow.tracking.client import MlflowClient


@pytest.fixture(scope='module')
def data():
    return pycaret.datasets.get_data("anomaly")


def test(data):
    experiment_name = uuid.uuid4().hex
    pycaret.anomaly.setup(
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
    iforest = pycaret.anomaly.create_model("iforest", experiment_custom_tags={"tag": 1})
    knn = pycaret.anomaly.create_model("knn", experiment_custom_tags={"tag": 1})

    # assign model
    iforest_results = pycaret.anomaly.assign_model(iforest)
    knn_results = pycaret.anomaly.assign_model(knn)
    assert isinstance(iforest_results, pd.DataFrame)
    assert isinstance(knn_results, pd.DataFrame)

    # predict model
    iforest_predictions = pycaret.anomaly.predict_model(model=iforest, data=data)
    knn_predictions = pycaret.anomaly.predict_model(model=knn, data=data)
    assert isinstance(iforest_predictions, pd.DataFrame)
    assert isinstance(knn_predictions, pd.DataFrame)

    # get config
    X = pycaret.anomaly.get_config("X")
    seed = pycaret.anomaly.get_config("seed")
    assert isinstance(X, pd.DataFrame)
    assert isinstance(seed, int)

    # set config
    pycaret.anomaly.set_config("seed", 124)
    seed = pycaret.anomaly.get_config("seed")
    assert seed == 124

    # save model
    pycaret.anomaly.save_model(knn, "knn_model_23122019")

    # load model
    pycaret.anomaly.load_model("knn_model_23122019")

    # returns table of models
    all_models = pycaret.anomaly.models()
    assert isinstance(all_models, pd.DataFrame)

    # Assert the custom tags are created
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    for experiment_run in client.list_run_infos(experiment.experiment_id):
        run = client.get_run(experiment_run.run_id)
        assert run.data.tags.get("tag") == "1"
