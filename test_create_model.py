# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import numpy as np
import pandas as pd
from sklearn import metrics
import pycaret
from pycaret.datasets import get_data

from sktime.utils.plotting import plot_series
from pycaret.time_series import _fit_and_score
from sktime.performance_metrics.forecasting import smape_loss

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)



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


from sktime.forecasting.arima import ARIMA
# forecaster = ARIMA()
# forecaster.fit(y, fh=fh)
# y_pred = forecaster.predict()

# %%
from sktime.forecasting.model_selection import ExpandingWindowSplitter, SlidingWindowSplitter

def perform_cv(cv, y):
    train_initial, test_initial = cv.split_initial(y)
    y_train_initial = y.iloc[train_initial]
    y_test_initial = y.iloc[test_initial]  # Includes all entries after y_train

    rolling_y_train = y_train_initial.copy(deep=True)
    y_test = y_test_initial.iloc[np.arange(len(cv.fh))]  # filter to only what is needed

    print("Fold Initial (0):")
    forecaster = ARIMA()
    forecaster.fit(rolling_y_train, fh=cv.fh)
    y_pred = forecaster.predict()
    if (y_test.index.values != y_pred.index.values).any() or (len(y_test) != len(cv.fh)) or ((len(y_pred) != len(cv.fh))):
        print(f"\t rolling_y_train: {rolling_y_train.index.values}, \n\t test: {y_test.index.values}")
        print(f"\t y_pred: {y_pred.index.values}")
        raise ValueError("y_test indices do not match y_pred_indices or split/prediction length does not match forecast horizon.")
    loss = smape_loss(y_test=y_test, y_pred=y_pred)
    print(f"\t == sMAPE Loss: {loss} ==")

    for i, (train, test) in enumerate(cv.split(y_test_initial)):
        print(f"Fold: {i+1}")
        y_train = y_test_initial.iloc[train]
        y_test = y_test_initial.iloc[test]

        rolling_y_train = pd.concat([rolling_y_train, y_train])
        rolling_y_train = rolling_y_train[~rolling_y_train.index.duplicated(keep='first')]
        if isinstance(cv, SlidingWindowSplitter):
            rolling_y_train = rolling_y_train.iloc[-cv.initial_window:]

        forecaster = ARIMA()
        forecaster.fit(rolling_y_train, fh=cv.fh)
        y_pred = forecaster.predict()

        if (y_test.index.values != y_pred.index.values).any() or (len(y_test) != len(cv.fh)) or ((len(y_pred) != len(cv.fh))):
            print(f"\t rolling_y_train: {rolling_y_train.index.values}, \n\t test: {y_test.index.values}")
            print(f"\t y_pred: {y_pred.index.values}")
            raise ValueError("y_test indices do not match y_pred_indices or split/prediction length does not match forecast horizon.")
        loss = smape_loss(y_test=y_test, y_pred=y_pred)
        print(f"\t == sMAPE Loss: {loss} ==")




y_small = y[:72]

fh = [1,2,3,4,5,6,7,8,9,10,11,12]
initial_window = 36
step_length = len(fh)
window_length = len(fh)

cv = ExpandingWindowSplitter(
        initial_window=initial_window,
        step_length=step_length,
        window_length=window_length,
        fh=fh,
        start_with_window=True
    )
perform_cv(cv, y_small)

cv = SlidingWindowSplitter(
        initial_window=initial_window,
        step_length=step_length,
        window_length=window_length,
        fh=fh,
        start_with_window=True
    )
perform_cv(cv, y_small)



print("DONE TEST")
