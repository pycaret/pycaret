import logging
import pandas as pd
from typing import Any
from pycaret.internal.Display import Display
from pycaret.internal.data_profiling.abstract_profiler import AbstractDataProfiler


class PandasProfilingProfiler(AbstractDataProfiler):
    def profile(
        self,
        data: pd.DataFrame,
        verbose: bool,
        logger: logging.Logger,
        **profiler_kwargs,
    ) -> Any:
        profiler_kwargs = profiler_kwargs or {}

        if verbose:
            print("Loading profile... Please Wait!")
        try:
            import pandas_profiling

            return pandas_profiling.ProfileReport(data, **profiler_kwargs)
        except Exception as ex:
            print("Profiler Failed. No output to show, continue with modeling.")
            logger.error(
                f"Data Failed with exception:\n {ex}\n"
                "No output to show, continue with modeling."
            )

    def display(self, data, display: Display) -> None:
        display.display(data)