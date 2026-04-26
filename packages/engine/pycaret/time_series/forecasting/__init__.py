"""Phase 6: this submodule was the deep import path for the deleted
legacy ``TSForecastingExperiment``. Kept as an empty namespace for
back-compat — anything that imports from
``pycaret.time_series.forecasting`` should migrate to
``pycaret.time_series`` (or ``pycaret.tasks``):

    from pycaret.time_series import TimeSeriesExperiment

The legacy ``oop.py`` was deleted along with
``pycaret/internal/pycaret_experiment/``.
"""

__all__: list[str] = []
