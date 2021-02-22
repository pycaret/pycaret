# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import numpy as np
import pandas as pd
from typing import Tuple
from pycaret.datasets import get_data

from sktime.utils.plotting import plot_series
from pycaret.time_series import _fit_and_score
from sktime.performance_metrics.forecasting import smape_loss
from sktime.forecasting.arima import ARIMA
from sktime.forecasting.model_selection import ExpandingWindowSplitter, SlidingWindowSplitter

# %%
y = get_data('airline', verbose=False)

# %%
def get_folds(cv, y) -> Tuple[pd.Series, pd.Series]:
    n_folds = int((len(y) - cv.initial_window) / cv.step_length)
    for i in np.arange(n_folds):
        if i == 0:
            # Initial Split in sktime
            train_initial, test_initial = cv.split_initial(y)
            y_train_initial = y.iloc[train_initial]
            y_test_initial = y.iloc[test_initial]  # Includes all entries after y_train

            rolling_y_train = y_train_initial.copy(deep=True)
            y_test = y_test_initial.iloc[np.arange(len(cv.fh))]  # filter to only what is needed
        else:
            # Subsequent Splits in sktime
            for j, (train, test) in enumerate(cv.split(y_test_initial)):
                if j == i-1:
                    y_train = y_test_initial.iloc[train]
                    y_test = y_test_initial.iloc[test]

                    rolling_y_train = pd.concat([rolling_y_train, y_train])
                    rolling_y_train = rolling_y_train[~rolling_y_train.index.duplicated(keep='first')]

                    if isinstance(cv, SlidingWindowSplitter):
                        rolling_y_train = rolling_y_train.iloc[-cv.initial_window:]
        yield rolling_y_train, y_test

def create_model(forecaster, cv, y):
    for i, (y_train, y_test) in enumerate(get_folds(cv, y)):
        forecaster.fit(y_train, fh=cv.fh)
        y_pred = forecaster.predict()
        if (y_test.index.values != y_pred.index.values).any() or (len(y_test) != len(cv.fh)) or ((len(y_pred) != len(cv.fh))):
            print(f"\t y_train: {y_train.index.values}, \n\t y_test: {y_test.index.values}")
            print(f"\t y_pred: {y_pred.index.values}")
            raise ValueError("y_test indices do not match y_pred_indices or split/prediction length does not match forecast horizon.")
        loss = smape_loss(y_test=y_test, y_pred=y_pred)
        print(f"Fold: {i}: sMAPE Loss: {loss}")

# %%
y_small = y[:72]

fh = np.arange(1, 13) # [1,2,3,4,5,6,7,8,9,10,11,12]
initial_window = 36
step_length = len(fh)
window_length = len(fh)
forecaster = ARIMA()

# %%
print("\n== WITH EXPANDING WINDOW ==")
cv = ExpandingWindowSplitter(
        initial_window=initial_window,
        step_length=step_length,
        window_length=window_length,
        fh=fh,
        start_with_window=True
    )
create_model(forecaster, cv, y_small)

# %%
print("\n== WITH SLIDING WINDOW ==")
cv = SlidingWindowSplitter(
        initial_window=initial_window,
        step_length=step_length,
        window_length=window_length,
        fh=fh,
        start_with_window=True
    )
create_model(forecaster, cv, y_small)

# %%
print("DONE TEST")
