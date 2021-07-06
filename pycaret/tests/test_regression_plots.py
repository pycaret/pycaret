import os, sys

sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import pytest
import pycaret.regression
import pycaret.datasets


def test():
    # loading dataset
    data = pycaret.datasets.get_data("boston")
    assert isinstance(data, pd.core.frame.DataFrame)

    # init setup
    reg1 = pycaret.regression.setup(
        data,
        target="medv",
        log_experiment=True,
        silent=True,
        html=False,
        session_id=123,
        fold=2,
        n_jobs=1,
    )

    model = pycaret.regression.create_model("rf")

    available_plots = [
        "parameter",
        "residuals",
        "error",
        "cooks",
        "rfe",
        "learning",
        "manifold",
        "vc",
        "feature",
        "feature_all",
    ]

    for plot in available_plots:
        pycaret.regression.plot_model(model, plot=plot, use_train_data=False)
        pycaret.regression.plot_model(model, plot=plot, use_train_data=True)

    models = [
        pycaret.regression.create_model("et"),
        pycaret.regression.create_model("xgboost"),
    ]

    # no pfi due to dependency hell
    available_shap = ["summary", "correlation", "reason", "pdp", "msa"]

    for model in models:
        for plot in available_shap:
            pycaret.regression.interpret_model(model, plot=plot)
            pycaret.regression.interpret_model(
                model, plot=plot, X_new_sample=data.iloc[:10]
            )

    assert 1 == 1


if __name__ == "__main__":
    test()
