import pandas as pd
import pytest

import pycaret.classification
import pycaret.datasets


@pytest.mark.plotting
def test_plot():
    # loading dataset
    data = pycaret.datasets.get_data("juice")
    assert isinstance(data, pd.DataFrame)

    # init setup
    pycaret.classification.setup(
        data,
        target="Purchase",
        log_experiment=True,
        log_plots=True,
        html=False,
        session_id=123,
        fold=2,
        n_jobs=1,
    )

    model = pycaret.classification.create_model("rf", max_depth=2, n_estimators=5)

    exp = pycaret.classification.ClassificationExperiment()
    available_plots = exp._available_plots

    for plot in available_plots:
        pycaret.classification.plot_model(model, plot=plot)

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
                model, plot=plot, X_new_sample=data.drop("Purchase", axis=1).iloc[:10]
            )

    assert 1 == 1


if __name__ == "__main__":
    test_plot()
