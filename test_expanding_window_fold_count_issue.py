import numpy as np
from pycaret.time_series import *
from pycaret.datasets import get_data
from sktime.forecasting.model_selection import SlidingWindowSplitter, ExpandingWindowSplitter
from pycaret.regression import RegressionExperiment

y = get_data(114, folder="time_series/seasonal", verbose=False)
cv = SlidingWindowSplitter(fh=np.arange(1, 13), initial_window=24, step_length=4)
exp = TSForecastingExperiment()
exp.setup(y, fh=12, fold_strategy=cv)

# data = get_data("airline")
# # s = setup(data, fh = 3, fold = 5, session_id = 123)

# s = TSForecastingExperiment()
# s.setup(data, fh=3, session_id=123)