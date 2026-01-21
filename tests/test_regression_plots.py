import matplotlib
import pandas as pd
import pytest
from packaging import version

import pycaret.datasets
import pycaret.regression


@pytest.mark.plotting
def test_plot():
    data = pycaret.datasets.get_data("boston")
    assert isinstance(data, pd.DataFrame)

    pycaret.regression.setup(
        data,
        target="medv",
        log_experiment=True,
        log_plots=True,
        html=False,
        session_id=123,
        fold=2,
        n_jobs=1,
    )

    model = pycaret.regression.create_model("rf", max_depth=2, n_estimators=5)

    exp = pycaret.regression.RegressionExperiment()
    available_plots = exp._available_plots

    skip_plots = set()
    if version.parse(matplotlib.__version__) >= version.parse("3.8.0"):
        skip_plots.add("cooks")

    for plot in available_plots:
        if plot in skip_plots:
            continue
        pycaret.regression.plot_model(model, plot=plot)

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
                model, plot=plot, X_new_sample=data.drop("medv", axis=1).iloc[:10]
            )

    assert 1 == 1


if __name__ == "__main__":
    test_plot()
