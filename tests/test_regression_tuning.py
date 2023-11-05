import sys

import pandas as pd
import pytest

import pycaret.datasets
import pycaret.regression
from pycaret.utils.generic import can_early_stop

if sys.platform == "linux":
    pytest.skip("Skipping test module on Linux", allow_module_level=True)


@pytest.mark.skip(reason="no way of currently testing this")
def test_regression_tuning():
    # loading dataset
    data = pycaret.datasets.get_data("boston")
    assert isinstance(data, pd.DataFrame)

    # init setup
    pycaret.regression.setup(
        data,
        target="medv",
        train_size=0.99,
        fold=2,
        html=False,
        session_id=123,
        n_jobs=1,
    )

    models = pycaret.regression.compare_models(turbo=False, n_select=100)

    models.append(pycaret.regression.stack_models(models[:3]))
    models.append(pycaret.regression.ensemble_model(models[0]))

    for model in models:
        print(f"Testing model {model}")
        if "Dummy" in str(model):
            continue
        pycaret.regression.tune_model(
            model,
            fold=2,
            n_iter=2,
            search_library="scikit-learn",
            search_algorithm="random",
            early_stopping=False,
        )
        pycaret.regression.tune_model(
            model,
            fold=2,
            n_iter=2,
            search_library="scikit-optimize",
            search_algorithm="bayesian",
            early_stopping=False,
        )
        pycaret.regression.tune_model(
            model,
            fold=2,
            n_iter=2,
            search_library="optuna",
            search_algorithm="tpe",
            early_stopping=False,
        )
        # TODO: Enable ray after fix is released
        # pycaret.regression.tune_model(
        #     model,
        #     fold=2,
        #     n_iter=2,
        #     search_library="tune-sklearn",
        #     search_algorithm="random",
        #     early_stopping=False,
        # )
        # pycaret.regression.tune_model(
        #     model,
        #     fold=2,
        #     n_iter=2,
        #     search_library="tune-sklearn",
        #     search_algorithm="optuna",
        #     early_stopping=False,
        # )
        pycaret.regression.tune_model(
            model,
            fold=2,
            n_iter=2,
            search_library="optuna",
            search_algorithm="tpe",
            early_stopping="asha",
        )
        # pycaret.regression.tune_model(
        #     model,
        #     fold=2,
        #     n_iter=2,
        #     search_library="tune-sklearn",
        #     search_algorithm="hyperopt",
        #     early_stopping="asha",
        # )
        # pycaret.regression.tune_model(
        #     model,
        #     fold=2,
        #     n_iter=2,
        #     search_library="tune-sklearn",
        #     search_algorithm="bayesian",
        #     early_stopping="asha",
        # )
        if can_early_stop(model, True, True, True, {}):
            pycaret.regression.tune_model(
                model,
                fold=2,
                n_iter=2,
                search_library="tune-sklearn",
                search_algorithm="bohb",
                early_stopping=True,
            )

    assert 1 == 1


if __name__ == "__main__":
    test_regression_tuning()
