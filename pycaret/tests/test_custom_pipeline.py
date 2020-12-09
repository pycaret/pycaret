import os, sys

sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import pytest
import pycaret.classification
import pycaret.datasets
import sklearn.pipeline
import sklearn.decomposition


def test():
    # loading dataset
    data = pycaret.datasets.get_data("juice")
    assert isinstance(data, pd.core.frame.DataFrame)

    # init setup
    clf1 = pycaret.classification.setup(
        data,
        target="Purchase",
        log_experiment=True,
        silent=True,
        html=False,
        custom_pipeline=[sklearn.decomposition.PCA()],
        session_id=123,
        n_jobs=1,
    )

    clf1 = pycaret.classification.setup(
        data,
        target="Purchase",
        log_experiment=True,
        silent=True,
        html=False,
        custom_pipeline={"CUSTOM_PCA": sklearn.decomposition.PCA()},
        session_id=123,
        n_jobs=1,
    )

    clf1 = pycaret.classification.setup(
        data,
        target="Purchase",
        log_experiment=True,
        silent=True,
        html=False,
        custom_pipeline=sklearn.pipeline.Pipeline(
            [("CUSTOM_PCA", sklearn.decomposition.PCA())]
        ),
        session_id=123,
        n_jobs=1,
    )

    clf1 = pycaret.classification.setup(
        data,
        target="Purchase",
        log_experiment=True,
        silent=True,
        html=False,
        custom_pipeline=[("CUSTOM_PCA", sklearn.decomposition.PCA())],
        session_id=123,
        n_jobs=1,
    )

    model = pycaret.classification.create_model("dt", fold=2)

    # finalize model
    final_best = pycaret.classification.finalize_model(model, model_only=False)
    assert isinstance(final_best, sklearn.pipeline.Pipeline)
    print(final_best)
    assert isinstance(final_best.named_steps["CUSTOM_PCA"], sklearn.decomposition.PCA)

    final_best.predict(data.drop("Purchase", axis=1))

    assert 1 == 1


if __name__ == "__main__":
    test()
