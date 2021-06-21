# Module: internal.ensemble
# Author: Miguel Trejo
# License: MIT

# Provides an Ensemble Forecaster supporting voting, mean and median methods.

# This Ensemble is only to be used internally.

import pandas as pd
import numpy as np
import warnings
from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.base._meta import _HeterogenousEnsembleForecaster
from sktime.forecasting.base._sktime import _OptionalForecastingHorizonMixin

_ENSEMBLE_METHODS = ["voting", "mean", "median"]

class _EnsembleForecasterWithVoting(
    _OptionalForecastingHorizonMixin, _HeterogenousEnsembleForecaster
):
    """
    Ensemble of forecasters.

    Parameters
    ----------
    forecasters : list of (str, estimator) tuples

    method : {'mean', 'median', 'voting'}, default='mean'
        Specifies the ensemble method type to be used.
        It must be one of 'mean', 'median', or 'voting.
        If none is given, 'mean' will be used.

    weights : array-like of shape (n_estimators,), default=None
        A sequence of weights (`float` or `int`) to weight the occurrences of
        predicted values before averaging. This parameter is only valid for
        'voting' method, uses uniform weights for 'voting' method if None.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for fit. None means 1 unless
        in a joblib.parallel_backend context.
        -1 means using all processors.
    """

    _required_parameters = ["forecasters"]
    _not_required_weights = ["mean", "median"]
    _required_weights = ["voting", "mean"]
    _available_methods = ["voting", "mean", "median"]

    def __init__(self, forecasters, method="mean", weights=None, n_jobs=None):
        self.forecasters = forecasters
        self.method = method
        self.weights = weights
        super(_EnsembleForecasterWithVoting, self).__init__(
            forecasters=self.forecasters, n_jobs=n_jobs
        )

    def _check_method(self):
        if self.method == "voting" and self.weights is None:
            warnings.warn(
                "Missing 'weights' argument, setting uniform weights."
            )
            self._set_weights_uniform
        elif self.method in self._not_required_weights and self.weights:
            warnings.warn(
                "Unused 'weights' argument. When method='mean' or method='median', 'weights' argument is not provided. Setting weights to `None`"
            )
            self._set_weights_none
        elif self.method not in self._available_methods:
            raise ValueError(
                f"Method {self.method} is not supported. Avaible methods are {', '.join(self._available_methods)}"
            )

    def _check_weights(self):
        if (self.weights is not None and len(self.weights) != len(self.forecasters)):
            raise ValueError(
                f"Number of forecasters and weights must be equal, got {len(self.weights)} weights and {len(self.estimators)} estimators"
            )

    def fit(self, y, X=None, fh=None):
        """Fit to training data.
        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored
        Returns
        -------
        self : returns an instance of self.
        """
        self._set_y_X(y, X)
        self._set_fh(fh)
        names, forecasters = self._check_forecasters()
        self._fit_forecasters(forecasters, y, X, fh)
        self._is_fitted = True
        return self

    def update(self, y, X=None, update_params=True):
        """Update fitted parameters
        Parameters
        ----------
        y : pd.Series
        X : pd.DataFrame
        update_params : bool, optional (default=True)
        Returns
        -------
        self : an instance of self
        """
        self.check_is_fitted()
        self._update_y_X(y, X)

        for forecaster in self.forecasters_:
            forecaster.update(y, X, update_params=update_params)

        return self

    def _predict(self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        if return_pred_int:
            raise NotImplementedError()

        self._check_method()

        pred_forecasters = pd.concat(self._predict_forecasters(fh, X), axis=1)

        if self.method == "median":
            return pred_forecasters.median(axis=1)
        elif self.method in self._required_weights:
            self._check_weights()
            pred_w = np.average(pred_forecasters, axis=1, weights=self.weights)
            return pd.DataFrame(pred_w, index=pred_forecasters.index)

    @property
    def _set_weights_none(self):
        self.weights = None

    @property
    def _set_weights_uniform(self):
        self.weights = np.ones(len(self.forecasters))
