# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import time
import numpy as np
import pandas as pd
from typing import Tuple, Union, Any, Optional
from pycaret.datasets import get_data

from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor

from sktime.utils.plotting import plot_series
from sktime.performance_metrics.forecasting import smape_loss, mape_loss
from sktime.performance_metrics.forecasting import sMAPE
from sktime.forecasting.arima import ARIMA
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.model_selection import ExpandingWindowSplitter, SlidingWindowSplitter


from sktime.forecasting.compose import EnsembleForecaster, ReducedForecaster, TransformedTargetForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.transformations.series.detrend import Deseasonalizer, Detrender


from pycaret.time_series import ForecastingGridSearchCV
from pycaret.containers.metrics.time_series import get_all_metric_containers


# # %%  # sklearn cross_validate
# from sklearn import datasets, linear_model
# from sklearn.model_selection import cross_validate
# from sklearn.metrics import make_scorer
# from sklearn.metrics import confusion_matrix
# from sklearn.svm import LinearSVC
# diabetes = datasets.load_diabetes()
# X = diabetes.data[:150]
# y = diabetes.target[:150]
# lasso = linear_model.Lasso()

# cv_results = cross_validate(lasso, X, y, cv=3)
# sorted(cv_results.keys())

# cv_results['test_score']


# # %%  sklearn GridSearchCV
# from sklearn import svm, datasets
# from sklearn.model_selection import GridSearchCV
# iris = datasets.load_iris()
# parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
# svc = svm.SVC()
# clf = GridSearchCV(svc, parameters)
# clf.fit(iris.data, iris.target)
# sorted(clf.cv_results_.keys())


# %%
y = get_data('airline', verbose=False)
X = None


# %%
forecaster = ARIMA()

fh = np.arange(1, 13) # [1,2,3,4,5,6,7,8,9,10,11,12]
initial_window = 108
step_length = len(fh)
window_length = len(fh)
# metrics = {'smape': sMAPE() , 'mape': mape_loss}  # Change this to the Metrics container
metrics = get_all_metric_containers(globals_dict={})
print(metrics)

forecaster_param_grid = {"seasonal_order": [(0,0,0,0), (0,1,0,12)]}

cv = ExpandingWindowSplitter(
    initial_window=initial_window,
    step_length=step_length,
    window_length=window_length,
    fh=fh,
    start_with_window=True
)
fit_kwargs = {'fh': cv.fh}

grid_obj = ForecastingGridSearchCV(
    forecaster=forecaster,
    cv=cv,
    param_grid=forecaster_param_grid,
    scoring=metrics
)
grid_obj.fit(y=y, X=X, **fit_kwargs)

# print(grid_obj.cv_results_)
# print(pd.DataFrame(grid_obj.cv_results_))

print("STANDALONE DONE TEST")


# %% 
from pycaret.internal.PycaretExperiment import TimeSeriesExperiment
fh = np.arange(1,13)
fold = 3

exp = TimeSeriesExperiment()
exp.setup(data=y, fh=fh, fold=fold, fold_strategy='expandingwindow')
arima = exp.create_model("arima")
tuned_arima = exp.tune_model(arima)
print(tuned_arima)

print("TUNE TEST DONE")