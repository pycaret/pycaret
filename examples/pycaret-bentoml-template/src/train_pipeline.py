import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.pipeline import Pipeline
from conf.config_core import config
from pycaret.regression import (setup, create_model, 
                                finalize_model, save_model)


def load_data() -> pd.DataFrame:
    """
    This function loads the processed dataset for training the pipeline.
    """
    return pd.read_csv(config.data_config.processed.path)


def save_pipeline(pipeline: Pipeline, filepath: str) -> None:
    """
    This function serializes a trained pipeline in the folder
    meant to store it.
    """
    save_model(pipeline, filepath, verbose=False)


def run_training() -> Pipeline:
    """
    This function trains and serializes the entire pipeline.
    """
    # loading data
    data = load_data()

    # # log transforming target
    data.fare_amount = np.log(data.fare_amount)

    # setting up the pipeline
    pipeline = setup(
        data=data,
        target=config.pipeline_config.target,
        train_size=config.pipeline_config.train_size,
        categorical_features=config.pipeline_config.categorical_features,
        numeric_features=config.pipeline_config.numeric_features,
        normalize=config.pipeline_config.normalize,
        ignore_low_variance=config.pipeline_config.ignore_low_variance,
        use_gpu=config.pipeline_config.use_gpu,
        silent=config.pipeline_config.silent,
        html=config.pipeline_config.html
    )

    # fitting the model
    model = create_model(
        estimator=config.model_config.algorithm,
        objective=config.model_config.objective,
        base_score=config.model_config.base_score,
        booster=config.model_config.booster,
        colsample_bylevel=config.model_config.colsample_bylevel,
        colsample_bynode=config.model_config.colsample_bynode,
        colsample_bytree=config.model_config.colsample_bytree,
        enable_categorical=config.model_config.enable_categorical,
        gamma=config.model_config.gamma,
        gpu_id=config.model_config.gpu_id,
        learning_rate=config.model_config.learning_rate,
        max_delta_step=config.model_config.max_delta_step,
        max_depth=config.model_config.max_depth,
        min_child_weight=config.model_config.min_child_weight,
        n_estimators=config.model_config.n_estimators,
        n_jobs=config.model_config.n_jobs,
        num_parallel_tree=config.model_config.num_parallel_tree,
        predictor=config.model_config.predictor,
        random_state=config.model_config.random_state,
        reg_alpha=config.model_config.reg_alpha,
        reg_lambda=config.model_config.reg_lambda,
        scale_pos_weight=config.model_config.scale_pos_weight,
        subsample=config.model_config.subsample,
        tree_method=config.model_config.tree_method,
        validate_parameters=config.model_config.validate_parameters,
        verbosity=config.model_config.verbosity
    )

    # serializing model
    finalized_pipeline = finalize_model(model)
    save_pipeline(pipeline=finalized_pipeline, filepath=config.pipeline_config.path)

    return finalized_pipeline


if __name__ == "__main__":
    run_training()