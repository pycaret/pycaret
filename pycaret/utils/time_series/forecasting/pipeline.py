from copy import deepcopy
from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.compose import ForecastingPipeline
from sktime.forecasting.compose import TransformedTargetForecaster


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
