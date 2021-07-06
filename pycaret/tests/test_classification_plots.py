import os, sys

sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import pytest
import pycaret.classification
import pycaret.datasets


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
        session_id=123,
        fold=2,
        n_jobs=1,
    )

    model = pycaret.classification.create_model("lr")

    available_plots = [
        "parameter",
        "auc",
        "confusion_matrix",
        "threshold",
        "pr",
        "error",
        "class_report",
        "rfe",
        "learning",
        "manifold",
        "calibration",
        "vc",
        "dimension",
        "feature",
        "boundary",
        "lift",
        "gain",
        "ks",
    ]

    for plot in available_plots:
        pycaret.classification.plot_model(model, plot=plot, use_train_data=False)
        pycaret.classification.plot_model(model, plot=plot, use_train_data=True)

    models = [
        pycaret.classification.create_model("et"),
        pycaret.classification.create_model("xgboost"),
    ]

    # no pfi due to dependency hell
    available_shap = ["summary", "correlation", "reason", "pdp", "msa"]

    for model in models:
        for plot in available_shap:
            pycaret.classification.interpret_model(model, plot=plot)
            pycaret.classification.interpret_model(
                model, plot=plot, X_new_sample=data.iloc[:10]
            )

    assert 1 == 1


if __name__ == "__main__":
    test()
