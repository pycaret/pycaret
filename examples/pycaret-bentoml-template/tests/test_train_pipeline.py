import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from conf.config_core import config
from src.train_pipeline import run_training
from xgboost.sklearn import XGBRegressor


def test_train_pipeline_func() -> None:
    """
    This function tests if the train pipeline function is working properly,
    producing a trained pipeline.
    """
    trained_model = run_training()

    assert type(trained_model) == type(XGBRegressor())