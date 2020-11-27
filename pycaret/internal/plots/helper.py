"""
This module contains methods that can be used in various plot modules and don't really belong to a specific module
"""

import numpy as np


def leverage_statistic(x: np.ndarray):
    if x.ndim > 1:
        x_mean = np.mean(x, axis=0)
        divider = np.sum([np.power(np.subtract(element, x_mean), 2) for element in x], axis=0)
        leverage = np.array(
            [1 / x.shape[0] + np.divide(np.power(np.subtract(element, x_mean), 2), divider) for element in x])
        return np.sum(leverage, axis=1)
    else:
        x_mean = np.mean(x)
        divider = np.sum([np.power(np.subtract(element, x_mean), 2) for element in x])
        leverage = np.array(
            [1 / x.shape[0] + np.divide(np.power(np.subtract(element, x_mean), 2), divider) for element in x])
        return leverage


def cooks_distance(standardized_residuals: np.ndarray, leverage_statistic: np.ndarray):
    multiplier = [element / (2*(1-element)) for element in leverage_statistic]
    distance = np.multiply(np.power(standardized_residuals, 2), multiplier)
    return distance


def calculate_standardized_residual(
        predicted: np.ndarray,
        expected: np.ndarray = None,
        featuresize: int = None
) -> np.ndarray:
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
