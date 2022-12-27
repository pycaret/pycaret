from copy import deepcopy
from typing import Optional, Tuple

import pandas as pd
from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.compose import ForecastingPipeline
from sktime.transformations.compose import TransformerPipeline
from sktime.transformations.series.impute import Imputer


def _add_model_to_pipeline(
    pipeline: ForecastingPipeline, model: BaseForecaster
) -> ForecastingPipeline:
    """Removes the dummy model from the preprocessing pipeline and adds the
    passed model to it instead.

    # TODO: Check if sktime can provide a convenience method for this functionality

    Parameters
    ----------
    pipeline : ForecastingPipeline
        A ForecastingPipeline to be used as the base to replace the final model.
    model : BaseForecaster
        sktime compatible model (without the pipeline). i.e. last step of
        the pipeline TransformedTargetForecaster

    Returns
    -------
    ForecastingPipeline
        The forecasting pipeline with the dummy model replaced with the correct one.
    """
    pipeline_with_model = deepcopy(pipeline)

    # Make sure to update `steps` and `steps_` for both the `ForecastingPipeline`
    # as well as the `TransformedTargetForecaster` else this causes issues when
    # the method is used in `predict_model` where it tries to predict using the
    # "dummy" Forecaster instead of the correct one.
    pipeline_with_model.steps[-1][1].steps.pop()
    pipeline_with_model.steps[-1][1].steps_.pop()
    pipeline_with_model.steps_[-1][1].steps.pop()
    pipeline_with_model.steps_[-1][1].steps_.pop()

    pipeline_with_model.steps[-1][1].steps.extend([("model", model)])
    pipeline_with_model.steps[-1][1].steps_.extend([("model", model)])
    pipeline_with_model.steps_[-1][1].steps.extend([("model", model)])
    pipeline_with_model.steps_[-1][1].steps_.extend([("model", model)])

    # Clone Tags so that the ability to get prediction intervals can be set correctly
    # based on the replacement model and not based on the pipeline model
    # https://github.com/sktime/sktime/blob/v0.14.0/sktime/forecasting/compose/_pipeline.py#L313-L318
    tags_to_clone = [
        "ignores-exogeneous-X",  # does estimator ignore the exogeneous X?
        "capability:pred_int",  # can the estimator produce prediction intervals?
        "requires-fh-in-fit",  # is forecasting horizon already required in fit?
        "enforce_index_type",  # index type that needs to be enforced in X/y
    ]

    # Clone tags for TransformedTargetForecaster
    pipeline_with_model.steps[-1][1].clone_tags(model, tags_to_clone)
    pipeline_with_model.steps_[-1][1].clone_tags(model, tags_to_clone)
    # Clone tags for ForecastingPipeline
    pipeline_with_model.clone_tags(model, tags_to_clone)

    return pipeline_with_model


def _transformations_present_X(pipeline: ForecastingPipeline) -> bool:
    """Returns whether transformations are present for the exogenous variables

    Parameters
    ----------
    pipeline : ForecastingPipeline
        PyCaret's internal Pipeline

    Returns
    -------
    bool
        True if there are exogenous transformations, False otherwise
    """
    return isinstance(pipeline.steps[0][1], TransformerPipeline)


def _transformations_present_y(pipeline: ForecastingPipeline) -> bool:
    """Returns whether transformations are present for the target variables

    Parameters
    ----------
    pipeline : ForecastingPipeline
        PyCaret's internal Pipeline

    Returns
    -------
    bool
        True if there are target transformations, False otherwise
    """
    return isinstance(pipeline.steps[-1][1].steps[0][1], TransformerPipeline)


def _are_pipeline_tansformations_empty(pipeline: ForecastingPipeline) -> bool:
    """Returns whether the pipeline has transformations for either the target
    or exogenous variables or whether there are no transformatons for either.

    Reminder: The pipeline structure is as follows:
    ForecastingPipeline
        - TransformerPipeline(exogenous_steps) [Optional]
        - TransformedTargetForecaster
            - TransformerPipeline(target_steps) [Optional]
            - model

    Parameters
    ----------
    pipeline : ForecastingPipeline
        PyCaret's internal Pipeline

    Returns
    -------
    bool
        True if there are no transformations, False otherwise
    """
    transform_X_present = _transformations_present_X(pipeline)
    transform_y_present = _transformations_present_y(pipeline)
    return (transform_X_present is False) and (transform_y_present is False)


def _get_imputed_data(
    pipeline: ForecastingPipeline, y: pd.Series, X: Optional[pd.DataFrame] = None
) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
    """Passes y and X through the pipeline and returns the imputed data.

    Assumption
    ----------
    If imputer is present in the pipeline, then it must be the first step.
    This check is done at the time the pipeline is created and is not repeated
    here. Also, note that the user may append a custom imputer after the first
    step as well. This is fine as long as the first step is an Imputer.
    This function will only return the output of the Imputer in the first step.

    Parameters
    ----------
    pipeline : ForecastingPipeline
        The pipeline used to get the imputed values
    y : pd.Series
        target data to be used for imputation
    X : pd.DataFrame, optional
        Exogenous variable data to be used for imputation, by default None

    Reminder: The pipeline structure is as follows:
    ForecastingPipeline
        - TransformerPipeline(exogenous_steps) [Optional]
        - TransformedTargetForecaster
            - TransformerPipeline(target_steps) [Optional]
            - model

    Returns
    -------
    Tuple[pd.Series, Optional[pd.DataFrame]]
        Imputed y and X values respectively
    """
    # Since we are fitting (fit_transform) with new data, need to make sure
    # we do not have side effects on pipeline
    pipeline_ = deepcopy(pipeline)

    if X is None:
        # No exogenous variables
        # Note: fit_transform does not work when X is None
        X_imputed = None
    else:
        # Exogenous variables present
        X_imputed = X.copy()

        if _transformations_present_X(pipeline_):
            transformer_pipeline_X = pipeline_.steps_[0][1]
            for _, transformer_X in transformer_pipeline_X.steps_:
                if isinstance(transformer_X, Imputer):
                    X_imputed = transformer_X.fit_transform(X, y)
                    continue

    y_imputed = y.copy()
    if _transformations_present_y(pipeline_):
        transformer_pipeline_y = pipeline_.steps_[-1][1].steps_[0][1]
        for _, transformer_y in transformer_pipeline_y.steps_:
            if isinstance(transformer_y, Imputer):
                y_imputed = transformer_y.fit_transform(y)
                continue

    return y_imputed, X_imputed


def _get_pipeline_estimator_label(
    pipeline: ForecastingPipeline,
) -> str:
    """Returns the name of the Transformed Target Forecaster in the pipeline along
    with the name of the final model step in the pipeline.

    These names can be used to adjust the search space while tuning the pipeline in
    tune_models (defined search space is only for the models, but we pass the entire
    pipeline for tuning. Hence the model search space has to be adjusted appropriately
    by adding the step names in front).

    Parameters
    ----------
    pipeline : ForecastingPipeline
        The pipeline used for modeling

    Returns
    -------
    str
        Name of the TransformedTargetForecaster in the pipeline and the name of
        the final model inside the TransformedTargetForecaster. Returns a single
        string appended with "__".
    """
    name_ttf, transformed_target_forecaster = pipeline.steps_[-1]
    name_model, _ = transformed_target_forecaster.steps_[-1]
    return name_ttf + "__" + name_model


def _pipeline_transform(
    pipeline: ForecastingPipeline, y: pd.Series, X: Optional[pd.DataFrame] = None
) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
    """Transforms y and X based on the transformations present in the pipeline.

    Reminder: The pipeline structure is as follows:
    ForecastingPipeline
        - TransformerPipeline(exogenous_steps) [Optional]
        - TransformedTargetForecaster
            - TransformerPipeline(target_steps) [Optional]
            - model

    Parameters
    ----------
    pipeline : ForecastingPipeline
        The pipeline used for modeling
    y : pd.Series
        Original target values
    X : pd.DataFrame, optional
        Original exogenous values, by default None

    Returns
    -------
    Tuple[pd.Series, Optional[pd.DataFrame]]
        The transformed values for y and X. If no transformations are present in
        the pipeline, then the values are returned as is ("unity" transformation).
    """
    # Get X Transformations ----
    if X is not None:
        _, potential_tx = pipeline.steps_[0]
        if isinstance(potential_tx, TransformerPipeline):
            X = potential_tx.transform(X)

    # Get y Transformations ----
    _, forecaster = pipeline.steps_[-1]
    _, potential_tx = forecaster.steps_[0]
    if isinstance(potential_tx, TransformerPipeline):
        y = potential_tx.transform(y)

    return y, X
