import os, sys

sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import pytest
import pycaret.classification
import pycaret.datasets
from pycaret.internal.meta_estimators import CustomProbabilityThresholdClassifier


def test():
    # loading dataset
    data = pycaret.datasets.get_data("juice")
    assert isinstance(data, pd.core.frame.DataFrame)

    # init setup
    clf1 = pycaret.classification.setup(
        data,
        target="Purchase",
        silent=True,
        log_experiment=True,
        html=False,
        session_id=123,
        n_jobs=1,
    )

    probability_threshold = 0.75

    # compare models
    top3 = pycaret.classification.compare_models(
        n_select=100, exclude=["catboost"], probability_threshold=probability_threshold
    )[:3]
    assert isinstance(top3, list)
    assert isinstance(top3[0], CustomProbabilityThresholdClassifier)
    assert top3[0].probability_threshold == probability_threshold

    # tune model
    tuned_top3 = [pycaret.classification.tune_model(i, n_iter=3) for i in top3]
    assert isinstance(tuned_top3, list)
    assert isinstance(tuned_top3[0], CustomProbabilityThresholdClassifier)
    assert tuned_top3[0].probability_threshold == probability_threshold

    # ensemble model
    bagged_top3 = [
        pycaret.classification.ensemble_model(
            i, probability_threshold=probability_threshold
        )
        for i in tuned_top3
    ]
    assert isinstance(bagged_top3, list)
    assert isinstance(bagged_top3[0], CustomProbabilityThresholdClassifier)
    assert bagged_top3[0].probability_threshold == probability_threshold

    # blend models
    blender = pycaret.classification.blend_models(
        top3, probability_threshold=probability_threshold
    )
    assert isinstance(blender, CustomProbabilityThresholdClassifier)
    assert blender.probability_threshold == probability_threshold

    # stack models
    stacker = pycaret.classification.stack_models(
        estimator_list=top3[1:],
        meta_model=top3[0],
        probability_threshold=probability_threshold,
    )
    assert isinstance(stacker, CustomProbabilityThresholdClassifier)
    assert stacker.probability_threshold == probability_threshold

    # calibrate model
    calibrated = pycaret.classification.calibrate_model(estimator=top3[0])
    assert isinstance(calibrated, CustomProbabilityThresholdClassifier)
    assert calibrated.probability_threshold == probability_threshold

    # plot model
    lr = pycaret.classification.create_model(
        "lr", probability_threshold=probability_threshold
    )
    pycaret.classification.plot_model(
        lr, save=True
    )  # scale removed because build failed due to large image size

    # select best model
    best = pycaret.classification.automl()
    assert isinstance(calibrated, CustomProbabilityThresholdClassifier)
    assert calibrated.probability_threshold == probability_threshold

    # hold out predictions
    predict_holdout = pycaret.classification.predict_model(lr)
    predict_holdout_0_5 = pycaret.classification.predict_model(
        lr, probability_threshold=0.5
    )
    predict_holdout_0_75 = pycaret.classification.predict_model(
        lr, probability_threshold=probability_threshold
    )
    assert isinstance(predict_holdout, pd.core.frame.DataFrame)
    assert predict_holdout.equals(predict_holdout_0_75)
    assert not predict_holdout.equals(predict_holdout_0_5)

    # predictions on new dataset
    predict_holdout = pycaret.classification.predict_model(lr, data=data)
    predict_holdout_0_5 = pycaret.classification.predict_model(
        lr, data=data, probability_threshold=0.5
    )
    predict_holdout_0_75 = pycaret.classification.predict_model(
        lr, data=data, probability_threshold=probability_threshold
    )
    assert isinstance(predict_holdout, pd.core.frame.DataFrame)
    assert predict_holdout.equals(predict_holdout_0_75)
    assert not predict_holdout.equals(predict_holdout_0_5)

    # finalize model
    final_best = pycaret.classification.finalize_model(best)
    assert isinstance(final_best, CustomProbabilityThresholdClassifier)
    assert final_best.probability_threshold == probability_threshold

    # save model
    pycaret.classification.save_model(best, "best_model_23122019")

    # load model
    saved_best = pycaret.classification.load_model("best_model_23122019")
    assert isinstance(saved_best._final_estimator, CustomProbabilityThresholdClassifier)
    assert saved_best._final_estimator.probability_threshold == probability_threshold

    assert 1 == 1


if __name__ == "__main__":
    test()
