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


from pycaret.time_series import ForecastingGridSearchCV, ForecastingRandomizedSearchCV
from pycaret.containers.metrics.time_series import get_all_metric_containers


# %%
y = get_data("airline", verbose=False)
X = None


# %%
# forecaster = ExponentialSmoothing()

# fh = np.arange(1, 13)  # [1,2,3,4,5,6,7,8,9,10,11,12]
# initial_window = 108
# step_length = len(fh)
# window_length = len(fh)
# metrics = get_all_metric_containers(globals_dict={})
# metrics_dict = dict([(k, v.scorer) for k, v in metrics.items()])
# print(metrics_dict)


# forecaster_param_grid = {"seasonal_order": [(0, 0, 0, 0), (0, 1, 0, 12)]}
# tune_distributions = {
#     "trend": ["add", "mul", "additive", "multiplicative", None],
#     "seasonal": ["add", "mul", "additive", "multiplicative", None],
#     "use_boxcox": [True, False],  # 'log', float
# }

# cv = ExpandingWindowSplitter(
#     initial_window=initial_window,
#     step_length=step_length,
#     window_length=window_length,
#     fh=fh,
#     start_with_window=True,
# )
# fit_kwargs = {"fh": cv.fh}

# grid_obj = ForecastingRandomizedSearchCV(
#     forecaster=forecaster,
#     cv=cv,
#     param_distributions=tune_distributions,
#     scoring=metrics_dict,
# )
# grid_obj.fit(y=y, X=X, **fit_kwargs)

# # print(grid_obj.cv_results_)
# # print(pd.DataFrame(grid_obj.cv_results_))

# print("STANDALONE DONE TEST")


# %%
from pycaret.internal.pycaret_experiment import TimeSeriesExperiment

fh = np.arange(1, 13)
fold = 3

exp = TimeSeriesExperiment()
exp.setup(
    data=y, fh=fh, fold=fold, fold_strategy="expandingwindow", session_id=42
)  # ), n_jobs=1) # Set n_jobs to 1 for debug

# # Works
# naive = exp.create_model("naive")
# tuned_naive = exp.tune_model(naive, search_algorithm="random")
# print(naive)
# print(tuned_naive)

# # Works
# poly_trend = exp.create_model("poly_trend")
# tuned_poly_trend = exp.tune_model(poly_trend, search_algorithm="random")
# print(poly_trend)
# print(tuned_poly_trend)

# # Works
# arima = exp.create_model("arima")
# start = time.time()
# tuned_arima = exp.tune_model(arima)  # ), search_algorithm="random") # , n_iter=3)
# end = time.time()
# print(f"Tuning Time for ARIMA: {end-start}")
# print(arima)
# print(tuned_arima)

# # Works
# exp_smooth = exp.create_model("exp_smooth")
# tuned_exp_smooth = exp.tune_model(exp_smooth, search_algorithm="random")
# print(exp_smooth)
# print(tuned_exp_smooth)

# # Works
# theta = exp.create_model("theta")
# tuned_theta = exp.tune_model(theta, search_algorithm="random")
# print(theta)
# print(tuned_theta)

# # Works
# rf_cds_dt = exp.create_model("rf_cds_dt")
# tuned_rf_cds_dt = exp.tune_model(rf_cds_dt, search_algorithm="random")
# print(rf_cds_dt)
# print(tuned_rf_cds_dt)

# # Works
# et_cds_dt = exp.create_model("et_cds_dt")
# tuned_et_cds_dt = exp.tune_model(et_cds_dt, search_algorithm="random")
# print(et_cds_dt)
# print(tuned_et_cds_dt)

# # Works
# xgboost_cds_dt = exp.create_model("xgboost_cds_dt")
# tuned_xgboost_cds_dt = exp.tune_model(xgboost_cds_dt, search_algorithm="random")
# print(xgboost_cds_dt)
# print(tuned_xgboost_cds_dt)

# # Works
# lightgbm_cds_dt = exp.create_model("lightgbm_cds_dt")
# tuned_lightgbm_cds_dt = exp.tune_model(lightgbm_cds_dt, search_algorithm="random")
# print(lightgbm_cds_dt)
# print(tuned_lightgbm_cds_dt)

# # Works
# model = exp.create_model("gbr_cds_dt")
# tuned_model = exp.tune_model(model, search_algorithm="random")
# print(model)
# print(tuned_model)

# # Works
# model = exp.create_model("dt_cds_dt")
# tuned_model = exp.tune_model(model, search_algorithm="random")
# print(model)
# print(tuned_model)

# # Works
# model = exp.create_model("ridge_cds_dt")
# tuned_model = exp.tune_model(model, search_algorithm="random")
# print(model)
# print(tuned_model)

# # Works
# model = exp.create_model("lasso_cds_dt")
# tuned_model = exp.tune_model(model, search_algorithm="random")
# print(model)
# print(tuned_model)

# # Works
# model = exp.create_model("llar_cds_dt")
# tuned_model = exp.tune_model(model, search_algorithm="random")
# print(model)
# print(tuned_model)

# # Works
# model = exp.create_model("br_cds_dt")
# tuned_model = exp.tune_model(model, search_algorithm="random")
# print(model)
# print(tuned_model)

# # Works
# model = exp.create_model("lr_cds_dt")
# tuned_model = exp.tune_model(model, search_algorithm="random")
# print(model)
# print(tuned_model)

# # Works
# model = exp.create_model("huber_cds_dt")
# tuned_model = exp.tune_model(model, search_algorithm="random")
# print(model)
# print(tuned_model)

# # Works
# model = exp.create_model("par_cds_dt")
# tuned_model = exp.tune_model(model, search_algorithm="random")
# print(model)
# print(tuned_model)

# # Works
# model = exp.create_model("omp_cds_dt")
# tuned_model = exp.tune_model(model, search_algorithm="random")
# print(model)
# print(tuned_model)

# # Works
# model = exp.create_model("ada_cds_dt")
# tuned_model = exp.tune_model(model, search_algorithm="random")
# print(model)
# print(tuned_model)

# # Works
# model = exp.create_model("knn_cds_dt")
# tuned_model = exp.tune_model(model, search_algorithm="random")
# print(model)
# print(tuned_model)

# # Works
# model = exp.create_model("en_cds_dt")
# tuned_model = exp.tune_model(model, search_algorithm="random")
# print(model)
# print(tuned_model)

# Works
model = exp.create_model("lar_cds_dt")
tuned_model = exp.tune_model(model, search_algorithm="random")
print(model)
print(tuned_model)

print("TUNE TEST DONE")