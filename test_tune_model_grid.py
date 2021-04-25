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
from sktime.forecasting.model_selection import (
    ExpandingWindowSplitter,
    SlidingWindowSplitter,
)


from sktime.forecasting.compose import (
    EnsembleForecaster,
    ReducedForecaster,
    TransformedTargetForecaster,
)
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
y = get_data("airline", verbose=False)
X = None


# %%
# forecaster = ARIMA()

# fh = np.arange(1, 13)  # [1,2,3,4,5,6,7,8,9,10,11,12]
# initial_window = 108
# step_length = len(fh)
# window_length = len(fh)
# metrics = get_all_metric_containers(globals_dict={})
# metrics_dict = dict([(k, v.scorer) for k, v in metrics.items()])
# print(metrics_dict)


# forecaster_param_grid = {"seasonal_order": [(0, 0, 0, 0), (0, 1, 0, 12)]}

# cv = ExpandingWindowSplitter(
#     initial_window=initial_window,
#     step_length=step_length,
#     window_length=window_length,
#     fh=fh,
#     start_with_window=True,
# )
# fit_kwargs = {"fh": cv.fh}

# grid_obj = ForecastingGridSearchCV(
#     forecaster=forecaster, cv=cv, param_grid=forecaster_param_grid, scoring=metrics_dict
# )
# grid_obj.fit(y=y, X=X, **fit_kwargs)

# # print(grid_obj.cv_results_)
# # print(pd.DataFrame(grid_obj.cv_results_))

# # print("STANDALONE DONE TEST")


# %%
from pycaret.internal.pycaret_experiment import TimeSeriesExperiment

fh = np.arange(1, 13)
fold = 3

exp = TimeSeriesExperiment()
exp.setup(data=y, fh=fh, fold=fold, fold_strategy="expandingwindow", session_id=42)

# # Works
# naive = exp.create_model("naive")
# tuned_naive = exp.tune_model(naive)
# print(naive)
# print(tuned_naive)

# # Works
# arima = exp.create_model("arima")
# tuned_arima = exp.tune_model(arima)
# print(arima)
# print(tuned_arima)

# # Works
# exp_smooth = exp.create_model("exp_smooth")
# tuned_exp_smooth = exp.tune_model(exp_smooth)
# print(exp_smooth)
# print(tuned_exp_smooth)

# # Works
# poly_trend = exp.create_model("poly_trend")
# tuned_poly_trend = exp.tune_model(poly_trend)
# print(poly_trend)
# print(tuned_poly_trend)


# # Works
# theta = exp.create_model("theta")
# tuned_theta = exp.tune_model(theta)
# print(theta)
# print(tuned_theta)

# # from pycaret.containers.models.time_series import RandomForestDTS
# # obj = RandomForestDTS()
# # print(obj.get_params().keys())

# # Works
# rf_cds_dt = exp.create_model("rf_cds_dt")
# tuned_rf_cds_dt = exp.tune_model(rf_cds_dt)
# print(rf_cds_dt)
# print(tuned_rf_cds_dt)

# # Works
# et_cds_dt = exp.create_model("et_cds_dt")
# tuned_et_cds_dt = exp.tune_model(et_cds_dt)
# print(et_cds_dt)
# print(tuned_et_cds_dt)

# # Works
# xgboost_cds_dt = exp.create_model("xgboost_cds_dt")
# tuned_xgboost_cds_dt = exp.tune_model(xgboost_cds_dt)
# print(xgboost_cds_dt)
# print(tuned_xgboost_cds_dt)

lightgbm_cds_dt = exp.create_model("lightgbm_cds_dt")
tuned_lightgbm_cds_dt = exp.tune_model(lightgbm_cds_dt)
print(lightgbm_cds_dt)
print(tuned_lightgbm_cds_dt)

print("TUNE TEST DONE")