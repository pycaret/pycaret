import pandas as pd
from sktime.forecasting.base import BaseForecaster

from pycaret.utils.time_series import TSExogenousPresent

# from pycaret.time_series import TSForecastingExperiment


def _disable_pred_int_enforcement(forecaster, enforce_pi: bool) -> bool:
    """Checks to see if prediction interval should be enforced. If it should
    but the forecaster does not support it, the container will be disabled.

    Parameters
    ----------
    forecaster : `sktime` compatible forecaster
        forecaster to check for prediction interval capability.
        Can be a dummy object of the forecasting class
    enforce_pi : bool
        Should prediction interval be enforced?

    Returns
    -------
    bool
        True if user wants to enforce prediction interval and forecaster
        does not supports it. False otherwise.
    """
    if enforce_pi and not forecaster.get_tag("capability:pred_int"):
        return True
    return False


def _disable_exogenous_enforcement(
    forecaster, enforce_exogenous: bool, exp_has_exogenous: TSExogenousPresent
) -> bool:
    """Checks to see if exogenous support should be enforced. If it should
    but the forecaster does not support it, the container will be disabled.
    NOTE: Only enforced if the experiment has exogenous variables. If it does not,
    then no models are disabled.

    Parameters
    ----------
    forecaster : `sktime` compatible forecaster
        forecaster to check for prediction interval capability.
        Can be a dummy object of the forecasting class
    enforce_exogenous : bool
        Should exogenous support be enforced?
    exp_has_exogenous : TSExogenousTypes
        Whether the experiment has exogenous variables or not?

    Returns
    -------
    bool
        True if user wants to enforce exogenous support and forecaster
        does not supports it. False otherwise.
    """

    # Disable models only if the experiment has exogenous variables
    if (
        exp_has_exogenous == TSExogenousPresent.YES
        and enforce_exogenous
        and forecaster.get_tag("ignores-exogeneous-X")
    ):
        return True
    return False


def _check_enforcements(forecaster, experiment) -> bool:
    """Checks whether the model supports certain features such as
    (1) Prediction Interval, and (2) support for exogenous variables. The checks
    depend on what features are requested by the user during the experiment setup.

    Parameters
    ----------
    forecaster : sktime compatible forecaster
        The forecaster which needs to be checked for feature support
    experiment : TSForecastingExperiment
        Used to check what features are requested by the user during setup.

    Returns
    -------
    bool
        True if the model should remain active in the experiment, False otherwise.
    """

    active = True

    # Pred Interval Enforcement ----
    disable_pred_int = _disable_pred_int_enforcement(
        forecaster=forecaster, enforce_pi=experiment.enforce_pi
    )

    # Exogenous variable support Enforcement ----
    disable_exog_enforcement = _disable_exogenous_enforcement(
        forecaster=forecaster,
        enforce_exogenous=experiment.enforce_exogenous,
        exp_has_exogenous=experiment.exogenous_present,
    )

    if disable_pred_int or disable_exog_enforcement:
        active = False

    return active


class DummyForecaster(BaseForecaster):
    """Dummy Forecaster for initial pycaret pipeline"""

    _tags = {
        "scitype:y": "univariate",  # which y are fine? univariate/multivariate/both
        "ignores-exogeneous-X": False,  # does estimator use the exogenous X?
        "handles-missing-data": False,  # can estimator handle missing data?
        "y_inner_mtype": "pd.Series",  # which types do _fit, _predict, assume for y?
        "X_inner_mtype": "pd.DataFrame",  # which types do _fit, _predict, assume for X?
        "requires-fh-in-fit": False,  # is forecasting horizon already required in fit?
        "X-y-must-have-same-index": True,  # can estimator handle different X/y index?
        "enforce-index-type": None,  # index type that needs to be enforced in X/y
        "capability:pred_int": False,
    }

    def _fit(self, y, X=None, fh=None):
        self._fh_len = None
        if fh is not None:
            self._fh_len = len(fh)
        self._is_fitted = True
        return self

    def _predict(self, fh=None, X=None):
        self.check_is_fitted()
        if fh is not None:
            preds = pd.Series([-99_999] * len(fh))
        elif self._fh_len is not None:
            # fh seen during fit
            preds = pd.Series([-99_999] * self._fh_len)
        else:
            raise ValueError(
                f"{type(self).__name__}: `fh` is unknown. Unable to make predictions."
            )

        return preds
