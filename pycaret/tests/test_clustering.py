import os, sys

sys.path.insert(0, os.path.abspath(".."))

# compare_models_test
import pytest
import pycaret.clustering
import pycaret.datasets


def test():
    # loading dataset
    data = pycaret.datasets.get_data("jewellery")

    # init setup
    clu1 = pycaret.clustering.setup(data, normalize=True, silent=True, html=False, session_id=123)

    # create model
    kmeans = pycaret.clustering.create_model("kmeans")
    kmodes = pycaret.clustering.create_model("kmodes")

    # assign model
    kmeans_results = pycaret.clustering.assign_model(kmeans)
    kmodes_results = pycaret.clustering.assign_model(kmodes)

    # get config
    X = pycaret.clustering.get_config("X")
    seed = pycaret.clustering.get_config("seed")

    assert 1 == 1


if __name__ == "__main__":
    test()
