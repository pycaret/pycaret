"""Time-series forecasting task — PyCaret 4.0.

PyCaret 4.0 is OOP-only; the 3.x functional API was removed.

The 3.x class name ``TSForecastingExperiment`` was renamed to the cleaner
``TimeSeriesExperiment`` in 4.0 to match the task's module name.

    from pycaret.time_series import TimeSeriesExperiment
    exp = TimeSeriesExperiment(fh=12, session_id=42).fit(y)
    best = exp.compare_models().best
    forecast = exp.predict_model(best).predictions
"""

from pycaret.tasks.time_series import TimeSeriesExperiment

__all__ = ["TimeSeriesExperiment"]
