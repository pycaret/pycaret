from copy import deepcopy
from typing import Optional, Tuple

import pandas as pd
from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.compose import ForecastingPipeline, TransformedTargetForecaster
from sktime.transformations.series.impute import Imputer


class PyCaretForecastingPipeline(ForecastingPipeline):
    """Workaround to sktime ForecastingPipeline not having a transform method."""

    def transform(self, y, X=None):
        # If X is not given, just passthrough the data without transformation
        if self._X is not None:
            # transform X
            for _, _, transformer in self._iter_transformers():
                X = transformer.transform(X)

        name, forecaster = self.steps_[-1]
        # If forecaster is not a TransformedTargetForecaster, just passthrough
        if isinstance(forecaster, TransformedTargetForecaster):
            y = forecaster.transform(Z=y, X=X)

        return y, X


def _add_model_to_pipeline(
    pipeline: PyCaretForecastingPipeline, model: BaseForecaster
) -> PyCaretForecastingPipeline:
    """Removes the dummy model from the preprocessing pipeline and adds the
    passed model to it instead.

    # TODO: Check if sktime can provide a convenience method for this functionality

    Parameters
    ----------
    pipeline : PyCaretForecastingPipeline
        A ForecastingPipeline to be used as the base to replace the final model.
    model : BaseForecaster
        sktime compatible model (without the pipeline). i.e. last step of
        the pipeline TransformedTargetForecaster

    Returns
    -------
    PyCaretForecastingPipeline
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
    # https://github.com/alan-turing-institute/sktime/blob/4d874c1c20a94d9006604a3916b6b434750b4735/sktime/forecasting/compose/_pipeline.py#L283
    tags_to_clone = [
        "scitype:y",  # which y are fine? univariate/multivariate/both
        "ignores-exogeneous-X",  # does estimator ignore the exogeneous X?
        "capability:pred_int",  # can the estimator produce prediction intervals?
        "handles-missing-data",  # can estimator handle missing data?
        "requires-fh-in-fit",  # is forecasting horizon already required in fit?
        "X-y-must-have-same-index",  # can estimator handle different X/y index?
        "enforce_index_type",  # index type that needs to be enforced in X/y
    ]

    # Clone tags for TransformedTargetForecaster
    pipeline_with_model.steps[-1][1].clone_tags(model, tags_to_clone)
    pipeline_with_model.steps_[-1][1].clone_tags(model, tags_to_clone)
    # Clone tags for ForecastingPipeline
    pipeline_with_model.clone_tags(model, tags_to_clone)

    return pipeline_with_model


def _are_pipeline_tansformations_empty(pipeline: PyCaretForecastingPipeline) -> bool:
    """Returns whether the pipeline has transformations for either the target
    or exogenous variables or whether there are no transformatons for either.

    Reminder: The pipeline structure is as follows:
        PyCaretForecastingPipeline
            - exogenous_steps
            - TransformedTargetForecaster
                - target_steps
                - model

    Parameters
    ----------
    pipeline : PyCaretForecastingPipeline
        PyCaret Pipeline

    Returns
    -------
    bool
        True if there are no transformations, False otherwise
    """
    num_steps_transform_X = len(pipeline.steps) - 1
    num_steps_transform_y = len(pipeline.steps[-1][1].steps) - 1
    return num_steps_transform_X == 0 and num_steps_transform_y == 0


def _get_imputed_data(
    pipeline: PyCaretForecastingPipeline, y: pd.Series, X: Optional[pd.DataFrame] = None
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
    pipeline : PyCaretForecastingPipeline
        The pipeline used to get the imputed values
    y : pd.Series
        target data to be used for imputation
    X : pd.DataFrame, optional
        Exogenous variable data to be used for imputation, by default None

    Reminder: The pipeline structure is as follows:
        PyCaretForecastingPipeline
            - exogenous_steps
            - TransformedTargetForecaster
                - target_steps
                - model

    Returns
    -------
    Tuple[pd.Series, Optional[pd.DataFrame]]
        Imputed y and X values respectively
    """

    if X is None:
        # No exogenous variables
        # Note: fit_transform does not work when X is None
        X_imputed = None
    else:
        # Exogenous variables present
        X_imputed = X.copy()
        for _, transformer_X in pipeline.steps_:
            if isinstance(transformer_X, Imputer):
                X_imputed = transformer_X.fit_transform(X, y)
                continue

    y_imputed = y.copy()
    for _, transformer_y in pipeline.steps_[-1][1].steps_:
        if isinstance(transformer_y, Imputer):
            y_imputed = transformer_y.fit_transform(y)
            continue

    return y_imputed, X_imputed


def _get_pipeline_estimator_label(
    pipeline: PyCaretForecastingPipeline,
) -> str:
    """Returns the name of the Transformed Target Forecaster in the pipeline along
    with the name of the final model step in the pipeline.

    These names can be used to adjust the search space while tuning the pipeline in
    tune_models (defined search space is only for the models, but we pass the entire
    pipeline for tuning. Hence the model search space has to be adjusted appropriately
    by adding the step names in front).

    Parameters
    ----------
    pipeline : PyCaretForecastingPipeline
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
