import uuid

import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

import pycaret.anomaly
import pycaret.datasets


@pytest.fixture(scope="module")
def data():
    return pycaret.datasets.get_data("anomaly")


def test_anomaly(data):
    experiment_name = uuid.uuid4().hex
    pycaret.anomaly.setup(
        data,
        normalize=True,
        log_experiment=True,
        experiment_name=experiment_name,
        experiment_custom_tags={"tag": 1},
        log_plots=True,
        html=False,
        session_id=123,
        n_jobs=1,
    )

    # create model
    iforest = pycaret.anomaly.create_model("iforest", experiment_custom_tags={"tag": 1})
    knn = pycaret.anomaly.create_model("knn", experiment_custom_tags={"tag": 1})
    # https://github.com/pycaret/pycaret/issues/3606
    cluster = pycaret.anomaly.create_model("cluster", experiment_custom_tags={"tag": 1})

    # Plot model
    pycaret.anomaly.plot_model(iforest)
    pycaret.anomaly.plot_model(knn)

    # assign model
    iforest_results = pycaret.anomaly.assign_model(iforest)
    knn_results = pycaret.anomaly.assign_model(knn)
    cluster_results = pycaret.anomaly.assign_model(cluster)
    assert isinstance(iforest_results, pd.DataFrame)
    assert isinstance(knn_results, pd.DataFrame)
    assert isinstance(cluster_results, pd.DataFrame)

    # predict model
    iforest_predictions = pycaret.anomaly.predict_model(model=iforest, data=data)
    knn_predictions = pycaret.anomaly.predict_model(model=knn, data=data)
    cluster_predictions = pycaret.anomaly.predict_model(model=cluster, data=data)
    assert isinstance(iforest_predictions, pd.DataFrame)
    assert isinstance(knn_predictions, pd.DataFrame)
    assert isinstance(cluster_predictions, pd.DataFrame)

    # get config
    X = pycaret.anomaly.get_config("X")
    seed = pycaret.anomaly.get_config("seed")
    assert isinstance(X, pd.DataFrame)
    assert isinstance(seed, int)

    # set config
    pycaret.anomaly.set_config("seed", 124)
    seed = pycaret.anomaly.get_config("seed")
    assert seed == 124

    # returns table of models
    all_models = pycaret.anomaly.models()
    assert isinstance(all_models, pd.DataFrame)

    # Assert the custom tags are created
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    for experiment_run in client.search_runs(experiment.experiment_id):
        run = client.get_run(experiment_run.info.run_id)
        assert run.data.tags.get("tag") == "1"

    # save model
    pycaret.anomaly.save_model(knn, "knn_model_23122019")

    # reset
    pycaret.anomaly.set_current_experiment(pycaret.anomaly.AnomalyExperiment())

    # load model
    knn = pycaret.anomaly.load_model("knn_model_23122019")

    # predict model
    knn_predictions = pycaret.anomaly.predict_model(model=knn, data=data)
    assert isinstance(knn_predictions, pd.DataFrame)


if __name__ == "__main__":
    test_anomaly()
