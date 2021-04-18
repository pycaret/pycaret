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

# %%
y = get_data('airline', verbose=False)

# %%

def _get_cv_n_folds(y, cv) -> int:
    n_folds = int((len(y) - cv.initial_window) / cv.step_length)
    return n_folds


def get_folds(cv, y) -> Tuple[pd.Series, pd.Series]:
    n_folds = _get_cv_n_folds(y, cv)
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
        yield rolling_y_train.index, y_test.index


def _create_model_with_cv(forecaster, y, X=None, fit_kwargs={}, round=4, cv=None, metrics=None, refit=None, display=None):
    #  Boiler plate code (similar to other tasks)
    # ...

    model_fit_start = time.time()
    scores = cross_validate_ts(
        forecaster=clone(forecaster),
        y=y,
        X=X,
        cv=cv,
        scoring=metrics,
        fit_params=fit_kwargs,
        n_jobs=1,
        return_train_score=False,
        error_score=0)
    model_fit_end = time.time()
    model_fit_time = np.array(model_fit_end - model_fit_start).round(2)

    score_dict = scores
    # Calculating mean and std
    avgs_dict = {k: [np.mean(v), np.std(v)] for k, v in score_dict.items()}
    # Creating metrics dataframe
    model_results = pd.DataFrame(score_dict)
    model_avgs = pd.DataFrame(avgs_dict, index=["Mean", "SD"],)
    model_results = model_results.append(model_avgs)
    # Round the results
    model_results = model_results.round(round)

    if refit:
        # refitting the model on complete X_train, y_train
        model_fit_start = time.time()
        # Finalizing model
        forecaster.fit(y, X, **fit_kwargs)
        model_fit_end = time.time()
        model_fit_time = np.array(model_fit_end - model_fit_start).round(2)
    else:
        model_fit_time /= _get_cv_n_folds(y, cv)

    return forecaster, model_fit_time, model_results, avgs_dict

def cross_validate_ts(forecaster, y, X, cv, scoring, fit_params, n_jobs, return_train_score, error_score=0):
    scores = {f"test_{scorer_name}": [] for scorer_name, _ in scoring.items()}
    for i, (train_index, test_index) in enumerate(get_folds(cv, y)):
        y_train, y_test = y[train_index], y[test_index]
        X_train = None if X is None else X[train_index]
        X_test = None if X is None else X[test_index]

        forecaster.fit(y_train, X_train, **fit_params)
        y_pred = forecaster.predict(X_test)
        if (y_test.index.values != y_pred.index.values).any() or (len(y_test) != len(cv.fh)) or ((len(y_pred) != len(cv.fh))):
            print(f"\t y_train: {y_train.index.values}, \n\t y_test: {y_test.index.values}")
            print(f"\t y_pred: {y_pred.index.values}")
            raise ValueError("y_test indices do not match y_pred_indices or split/prediction length does not match forecast horizon.")

        for scorer_name, scorer_func in scoring.items():
            metric = scorer_func(y_test=y_test, y_pred=y_pred)
            # print(f"test_{scorer_name}, Fold: {i} Metric: {metric}")
            scores[f"test_{scorer_name}"].append(metric)

    return scores


def create_model(
    forecaster: Union[str, Any],
    fold: Optional[Union[int, Any]] = None,
    round: int = 4,
    cross_validation: bool = True,
    fit_kwargs: Optional[dict] = None,
    verbose: bool = True,
    **kwargs,
):
    # NOTE: Work in Progress
    #  Boiler plate code (similar to other tasks)
    # ...
    if cross_validation:
        _create_model_with_cv(forecaster, y, X=None, fit_kwargs={}, round=4, cv=None, metrics=None, refit=None, display=None)
    else:
        raise NotImplementedError

# %%
def eval_wrapper(forecaster, y):
    print("\n\n")
    print("="*75)
    print(f"Evaluating {forecaster.__class__}")
    print("="*75)

    fh = np.arange(1, 13) # [1,2,3,4,5,6,7,8,9,10,11,12]
    initial_window = 36
    step_length = len(fh)
    window_length = len(fh)
    metrics = {'smape': sMAPE() , 'mape': mape_loss}

    print("\n== WITH EXPANDING WINDOW ==")
    cv = ExpandingWindowSplitter(
        initial_window=initial_window,
        step_length=step_length,
        window_length=window_length,
        fh=fh,
        start_with_window=True
    )
    fit_kwargs = {'fh': cv.fh}
    forecaster, model_fit_time, model_results, avgs_dict = _create_model_with_cv(
        forecaster=forecaster,
        y=y, X=None,
        fit_kwargs=fit_kwargs,
        round=4,
        cv=cv,
        metrics=metrics,
        refit=True,
        display=None
    )
    print(f"Model Fit Time: {model_fit_time}")
    print(f"Model Results:\n {model_results}")
    print(f"Averages Dictionary: {avgs_dict}")

    print("\n== WITH SLIDING WINDOW ==")
    cv = SlidingWindowSplitter(
        initial_window=initial_window,
        step_length=step_length,
        window_length=window_length,
        fh=fh,
        start_with_window=True
    )
    fit_kwargs = {'fh': cv.fh}
    forecaster, model_fit_time, model_results, avgs_dict = _create_model_with_cv(
        forecaster=forecaster,
        y=y, X=None,
        fit_kwargs=fit_kwargs,
        round=4,
        cv=cv,
        metrics=metrics,
        refit=True,
        display=None
    )
    print(f"Model Fit Time: {model_fit_time}")
    print(f"Model Results:\n {model_results}")
    print(f"Averages Dictionary: {avgs_dict}")

# %%

# y_small = y[:72]
y_small = y

# %% Testing Standalone (without PyCaret Integration)
# forecaster = ARIMA()
# eval_wrapper(forecaster, y_small)

# forecaster = ExponentialSmoothing()
# eval_wrapper(forecaster, y_small)

# regressor = RandomForestRegressor()
# forecaster = TransformedTargetForecaster(
#     [
#         ("deseasonalise", Deseasonalizer(model="multiplicative", sp=12)),
#         ("detrend", Detrender(forecaster=PolynomialTrendForecaster(degree=1))),
#         (
#             "forecast",
#             ReducedForecaster(
#                 regressor=regressor, scitype='regressor', window_length=12, strategy="recursive"
#             ),
#         ),
#     ]
# )
# eval_wrapper(forecaster, y_small)


# %% Testing with PyCaret Integration

from pycaret.internal.pycaret_experiment import TimeSeriesExperiment
fh = np.arange(1,13)
fold = 3

exp = TimeSeriesExperiment()
exp.setup(data=y, fh=fh, fold=fold, fold_strategy='expandingwindow')

model = exp.create_model("naive")
pred = model.predict()
print(pred)

model = exp.create_model("arima")
pred = model.predict()
print(pred)

# model = exp.create_model("rf_dts")
# pred = model.predict()
# print(pred)

# model = exp.create_model("auto_ets")
# pred = model.predict()
# print(pred)


print("DONE TEST")