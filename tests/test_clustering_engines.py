import daal4py
import pytest
import sklearn

import pycaret.clustering
import pycaret.datasets


def test_engines_setup_global_args():
    """Tests the setting of engines using global arguments in setup."""

    jewellery_dataframe = pycaret.datasets.get_data("jewellery")
    exp = pycaret.clustering.ClusteringExperiment()

    # init setup
    exp.setup(
        jewellery_dataframe,
        normalize=True,
        log_experiment=True,
        experiment_custom_tags={"tag": 1},
        log_plots=True,
        html=False,
        session_id=123,
        n_jobs=1,
        engines={"kmeans": "sklearnex"},
    )

    # Default Model Engine ----
    assert exp.get_engine("kmeans") == "sklearnex"
    model = exp.create_model("kmeans")
    assert isinstance(model, daal4py.sklearn.cluster.KMeans)


def test_engines_global_methods():
    """Tests the setting of engines using methods like set_engine (global changes)."""

    jewellery_dataframe = pycaret.datasets.get_data("jewellery")
    exp = pycaret.clustering.ClusteringExperiment()

    # init setup
    exp.setup(
        jewellery_dataframe,
        normalize=True,
        log_experiment=True,
        experiment_custom_tags={"tag": 1},
        log_plots=True,
        html=False,
        session_id=123,
        n_jobs=1,
        engines={"kmeans": "sklearnex"},
    )

    assert exp.get_engine("kmeans") == "sklearnex"

    # Globally reset engine ----
    exp._set_engine("kmeans", "sklearn")
    assert exp.get_engine("kmeans") == "sklearn"
    model = exp.create_model("kmeans")
    assert isinstance(model, sklearn.cluster.KMeans)


def test_create_model_engines_local_args():
    """Tests the setting of engines for create_model using local args."""

    jewellery_dataframe = pycaret.datasets.get_data("jewellery")
    exp = pycaret.clustering.ClusteringExperiment()

    # init setup
    exp.setup(
        jewellery_dataframe,
        normalize=True,
        log_experiment=True,
        experiment_custom_tags={"tag": 1},
        log_plots=True,
        html=False,
        session_id=123,
        n_jobs=1,
    )

    # Default Model Engine ----
    assert exp.get_engine("kmeans") == "sklearn"
    model = exp.create_model("kmeans")
    assert isinstance(model, sklearn.cluster.KMeans)

    # Override model engine locally ----
    model = exp.create_model("kmeans", engine="sklearnex")
    assert isinstance(model, daal4py.sklearn.cluster.KMeans)
    # Original engine should remain the same
    assert exp.get_engine("kmeans") == "sklearn"


@pytest.mark.parametrize("algo", ("kmeans", "dbscan"))
def test_all_sklearnex_models(algo: str):
    jewellery_dataframe = pycaret.datasets.get_data("jewellery")
    exp = pycaret.clustering.ClusteringExperiment()

    # init setup
    exp.setup(
        jewellery_dataframe,
        normalize=True,
        log_experiment=True,
        experiment_custom_tags={"tag": 1},
        log_plots=True,
        html=False,
        session_id=123,
        n_jobs=1,
    )

    model = exp.create_model(algo)
    parent_library = model.__module__
    assert parent_library.startswith("sklearn")

    model = exp.create_model(algo, engine="sklearnex")
    parent_library = model.__module__
    assert parent_library.startswith("sklearnex") or parent_library.startswith(
        "daal4py"
    )
