import os, sys

sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import pytest
import pycaret.clustering
import pycaret.datasets


def test():
    # loading dataset
    data = pycaret.datasets.get_data("jewellery")
    assert isinstance(data, pd.core.frame.DataFrame)

    # init setup
    clu1 = pycaret.clustering.setup(
        data,
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
    kmeans_predictions = pycaret.clustering.predict_model(model=kmeans, data=data)
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


if __name__ == "__main__":
    test()
