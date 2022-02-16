from pycaret.utils.time_series import TSExogenousTypes


def disable_pred_int_enforcement(forecaster, enforce_pi: bool) -> bool:
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
        supports it. False otherwise.
    """
    if enforce_pi and not forecaster.get_tag("capability:pred_int"):
        return False
    return True


def disable_exogenous_enforcement(
    forecaster, enforce_exogenous: bool, exp_has_exogenous: TSExogenousTypes
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
        supports it. False otherwise.
    """

    # Disable models only if the experiment has exogenous variables
    if (
        exp_has_exogenous == TSExogenousTypes.EXO
        and enforce_exogenous
        and forecaster.get_tag("ignores-exogeneous-X")
    ):
        return False
    return True
