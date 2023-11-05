import os

import pandas as pd
import pytest

import pycaret.classification
import pycaret.datasets
from pycaret.utils.generic import can_early_stop

os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
os.environ["TUNE_MAX_LEN_IDENTIFIER"] = "1"


if "CI" in os.environ:
    pytest.skip("Skipping test module on CI", allow_module_level=True)


@pytest.mark.skip(reason="no way of currently testing this")
def test_classification_tuning():
    # loading dataset
    data = pycaret.datasets.get_data("juice")
    assert isinstance(data, pd.DataFrame)

    # init setup
    pycaret.classification.setup(
        data,
        target="Purchase",
        train_size=0.7,
        fold=2,
        html=False,
        session_id=123,
        n_jobs=1,
    )

    models = pycaret.classification.compare_models(
        turbo=False, n_select=100, verbose=False
    )

    models.append(pycaret.classification.stack_models(models[:3], verbose=False))
    models.append(pycaret.classification.ensemble_model(models[0], verbose=False))

    for model in models:
        print(f"Testing model {model}")
        if "Dummy" in str(model):
            continue
        pycaret.classification.tune_model(
            model,
            fold=2,
            n_iter=2,
            search_library="scikit-learn",
            search_algorithm="random",
            early_stopping=False,
        )
        pycaret.classification.tune_model(
            model,
            fold=2,
            n_iter=2,
            search_library="scikit-optimize",
            search_algorithm="bayesian",
            early_stopping=False,
        )
        pycaret.classification.tune_model(
            model,
            fold=2,
            n_iter=2,
            search_library="optuna",
            search_algorithm="tpe",
            early_stopping=False,
        )
        # TODO: Enable ray after fix is released
        # pycaret.classification.tune_model(
        #     model,
        #     fold=2,
        #     n_iter=2,
        #     search_library="tune-sklearn",
        #     search_algorithm="random",
        #     early_stopping=False,
        # )
        # pycaret.classification.tune_model(
        #     model,
        #     fold=2,
        #     n_iter=2,
        #     search_library="tune-sklearn",
        #     search_algorithm="optuna",
        #     early_stopping=False,
        # )
        pycaret.classification.tune_model(
            model,
            fold=2,
            n_iter=2,
            search_library="optuna",
            search_algorithm="tpe",
            early_stopping="asha",
        )
        # pycaret.classification.tune_model(
        #     model,
        #     fold=2,
        #     n_iter=2,
        #     search_library="tune-sklearn",
        #     search_algorithm="hyperopt",
        #     early_stopping="asha",
        # )
        # pycaret.classification.tune_model(
        #     model,
        #     fold=2,
        #     n_iter=2,
        #     search_library="tune-sklearn",
        #     search_algorithm="bayesian",
        #     early_stopping="asha",
        # )
        if can_early_stop(model, True, True, True, {}):
            pycaret.classification.tune_model(
                model,
                fold=2,
                n_iter=2,
                search_library="tune-sklearn",
                search_algorithm="bohb",
                early_stopping=True,
            )

    assert 1 == 1


if __name__ == "__main__":
    test_classification_tuning()
