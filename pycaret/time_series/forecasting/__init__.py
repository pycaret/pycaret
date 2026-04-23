"""Deep import path for the legacy TS forecasting class.

In 4.0, users should import from `pycaret.time_series` (or `pycaret.tasks`).
This module remains only so the legacy `_build_legacy_experiment()` call
inside `TimeSeriesExperiment` can reach the 3.x implementation class.
"""

from pycaret.time_series.forecasting.oop import TSForecastingExperiment

__all__ = ["TSForecastingExperiment"]
