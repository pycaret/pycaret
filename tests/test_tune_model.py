import pytest

from pycaret.classification import ClassificationExperiment
from pycaret.datasets import get_data
from pycaret.regression import RegressionExperiment


@pytest.mark.parametrize("usecase", ("classification", "regression"))
def test_tunable_voting_estimator(usecase):
    # load dataset
    diabetes = get_data("diabetes")

    # init setup
    if usecase == "classification":
        exp_cls = ClassificationExperiment
    else:
        exp_cls = RegressionExperiment
    exp = exp_cls()
    exp.setup(data=diabetes, target="Class variable", session_id=1, fold=2)

    # train a few models
    lr = exp.create_model("lr")
    dt = exp.create_model("dt")
    knn = exp.create_model("knn")

    # blend models
    blender_weighted = exp.blend_models([lr, dt, knn], weights=[0.5, 0.2, 0.3])
    assert blender_weighted.get_params()["weights"] == [0.5, 0.2, 0.3]

    # tune blender
    tuned_blender = exp.tune_model(blender_weighted, choose_better=False)
    assert (
        tuned_blender.get_params()["weights"]
        != blender_weighted.get_params()["weights"]
    )
    assert tuned_blender.get_params()["weights"] is not None


@pytest.mark.parametrize("usecase", ("classification", "regression"))
def test_tunable_mlp(usecase):
    # load dataset
    diabetes = get_data("diabetes")

    # init setup
    if usecase == "classification":
        exp_cls = ClassificationExperiment
    else:
        exp_cls = RegressionExperiment
    exp = exp_cls()
    exp.setup(data=diabetes, target="Class variable", session_id=1, fold=2)

    mlp = exp.create_model("mlp", hidden_layer_sizes=[15, 15])

    # tune blender
    tuned_mlp = exp.tune_model(mlp, choose_better=False)
    print(tuned_mlp.get_params())
    assert (
        tuned_mlp.get_params()["hidden_layer_sizes"]
        != mlp.get_params()["hidden_layer_sizes"]
    )
    assert tuned_mlp.get_params()["hidden_layer_sizes"] is not None
