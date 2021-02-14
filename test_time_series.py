# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import numpy as np
from sklearn import metrics
import pycaret
from pycaret.datasets import get_data

from sktime.utils.plotting import plot_series


# %%
# all_datasets = get_data('index')

# %%
y = get_data('airline', verbose=False)
plot_series(y)

#%%
fh=np.arange(1,13)

#%%
# # fh set explicitly (not recommended since semi-private method)
# from sktime.forecasting.arima import ARIMA
# forecaster = ARIMA()
# forecaster._set_fh(fh)
# forecaster.fit(y)
# y_pred = forecaster.predict()


#%% fh passed to fit
# from sktime.forecasting.arima import ARIMA
# forecaster = ARIMA()
# # forecaster._set_fh(fh)
# forecaster.fit(y, fh=fh)
# y_pred = forecaster.predict()

# print("DONE TEST")

# %% OLD PyCaret Method
# from pycaret.time_series import *

# print(type(y.index))
# print(y.index.dtype)

# exp = setup(y)
# print(models())
# model1 = create_model("arima", fit_kwargs={'fh': np.arange(1, 13)})
# best_model = compare_models()


#%% NEW PyCaret Method (Class Based)

from pycaret.internal.PycaretExperiment import TimeSeriesExperiment

exp = TimeSeriesExperiment()
exp.setup(data=y, fh=fh, fold=2)
exp.create_model("arima")