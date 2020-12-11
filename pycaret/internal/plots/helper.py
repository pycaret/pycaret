"""
This module contains methods that can be used in various plot modules and don't really belong to a specific module
"""

import numpy as np


def leverage_statistic(X: np.ndarray) -> np.array:
    """
    Calculates the leverage statistic $h_i$ for all $n$ observations $x_i$ within $X$.
    $h_{i}=\frac{1}{n}+\frac{\left(x_{i}-\bar{x}\right)^{2}}{\sum_{i^{\prime}=1}^{n}\left(x_{i^{\prime}}-\bar{x}\right)^{2}}$
    Observations with high leverage have an unusual value for $x_i$.

    Parameters
    ----------
    X: np.array
        A collection of observations $x_i$, where $dim(X)=2$. The first dimension describes the number of observations $n$
        and the second the number of features.

    Returns
    -------
        np.array: A array containing the leverage of each observation $x_i$, hence it has length $n$.

    """

    if X.ndim > 1:
        x_mean = np.mean(X, axis=0)
        divider = np.sum([np.power(np.subtract(element, x_mean), 2) for element in X], axis=0)
        leverage = np.array(
            [1 / X.shape[0] + np.divide(np.power(np.subtract(element, x_mean), 2), divider) for element in X])
        return np.sum(leverage, axis=1)
    else:
        x_mean = np.mean(X)
        divider = np.sum([np.power(np.subtract(element, x_mean), 2) for element in X])
        leverage = np.array(
            [1 / X.shape[0] + np.divide(np.power(np.subtract(element, x_mean), 2), divider) for element in X])
        return leverage


def calculate_standardized_residual(
        predicted: np.ndarray,
        expected: np.ndarray = None,
        featuresize: int = None
) -> np.ndarray:
    """
    Calculates the standardized residuals $\tilde r_i$ of the predictions to the expectations.
    $\tilde{r}_{i}=\frac{r_{i}}{\widehat{\sigma} \sqrt{1-\left(\frac{1}{n}+\frac{\left(y_{i}-\bar{y}\right)^{2}}{\sum_{i}^{n}\left(y_{i}-\bar{y}\right)^{2}}\right)}}$
    where $r_i= y_i - \hat y_i$ and $\hat \sigma$ denoting the estimated standard deviation of the error terms $\epsilon_i\approx r_i$.
    If the error terms $\epsilon_i$ are distributed according to a normal distribution, then for the distribution of the
    standardized residuals, we have $\tilde r_i \sim \mathcal N(0,1)$.

    Parameters
    ----------
    predicted: np.array
        the predicted values (y_pred)
    expected: np.array
        the expected values for the predictions (y_true)
    featuresize: int
        Optional, number of features per observation, If it is None, a featuresize of 1 is assumed

    Returns
    -------
        np.array: A array containing the standardized residuals of the predictions

    """
    if expected is not None:
        residuals = expected - predicted
    else:
        residuals = predicted
        expected = predicted

    if residuals.sum() == 0:
        return residuals

    n = residuals.shape[0]
    m = featuresize
    if m is None:
        m = 1
    s2_hat = 1 / (n - m) * np.sum(residuals ** 2)
    leverage = 1 / n + (expected - np.mean(expected)) / np.sum((expected - np.mean(expected)) ** 2)
    standardized_residuals = residuals / (np.sqrt(s2_hat) * (1 - leverage))
    return standardized_residuals


def cooks_distance(standardized_residuals: np.ndarray, leverage_statistic: np.ndarray, n_model_params: int = None) -> np.array:
    """
    The Cook’s distance $d_i$ measures to which extent the predicted value $\hat y_i$ changes if the ith observation
    is removed. It can be efficiently calculated using the leverage statistics $h_i$, the standardized
    residuals $\tilde r_i$ and the number of predictor variables $p$.
    $d_{i}=\frac{h_{i}}{1-h_{i}} \frac{\widetilde{r}_{i}^{2}}{p+1}$
    The larger the value of Cook’s distance $d_i$ is, the higher is the influence of the corresponding observation on
    the estimation of the predicted value $\hat y_i$. In practice, we consider a value of Cook’s distance larger than 1
    as dangerously influential.

    Parameters
    ----------
    standardized_residuals: np.array
        the standardized residuals for the predictions of some model w.r.t a specific dataset $X$

    leverage_statistic: np.array
        the leverage statistics for the same dataset $X$ used to calculate the standardized residuals

    n_model_params: int, Optional
        the number of parameters contained in the analysed model, used as a indicator for the number of predictor
        variables $p$. If None, a linear model is assumed which has $p=1$.


    Returns
    -------
        np.array: A array containing the Cook's distance for each observation $x_i$ in the dataset $X$

    """
    p = n_model_params if n_model_params is not None and n_model_params >= 1 else 1
    multiplier = [element / (1-element) for element in leverage_statistic]
    distance = np.multiply(np.power(standardized_residuals, 2) / (p + 1), multiplier)
    return distance

