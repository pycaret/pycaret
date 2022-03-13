from sktime.forecasting.compose import ForecastingPipeline
from sktime.forecasting.compose import TransformedTargetForecaster


class PyCaretForecastingPipeline(ForecastingPipeline):
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
