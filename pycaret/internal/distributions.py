# Module: internal.distributions
# Author: Antoni Baum (Yard1) <antoni.baum@protonmail.com>
# License: MIT

from typing import Dict, Hashable, Optional

try:
    from collections.abc import Hashable
except:
    from collections import Hashable

import numpy as np


class Distribution:
    def __init__(self):
        raise NotImplementedError("This is an abstract class.")

    def get_skopt(self):
        raise NotImplementedError("This is an abstract class.")

    def get_optuna(self):
        raise NotImplementedError("This is an abstract class.")

    def get_hyperopt(self, label):
        raise NotImplementedError("This is an abstract class.")

    def get_CS(self, label):
        raise NotImplementedError("This is an abstract class.")


class UniformDistribution(Distribution):
    """
    Uniform float distribution.

    Parameters
    ----------
    lower: float
        Inclusive lower bound of distribution.
    upper: float
        Inclusive upper bound of distribution.
    log: bool, default = False:
        If True, the distribution will be log-uniform.
    """

    def __init__(self, lower: float, upper: float, log: bool = False):
        self.lower = lower
        self.upper = upper
        self.log = log

    def get_skopt(self):
        import skopt.space

        if self.log:
            return skopt.space.Real(self.lower, self.upper, prior="log-uniform")
        else:
            return skopt.space.Real(self.lower, self.upper, prior="uniform")

    def get_optuna(self):
        import optuna

        if self.log:
            return optuna.distributions.LogUniformDistribution(self.lower, self.upper)
        else:
            return optuna.distributions.UniformDistribution(self.lower, self.upper)

    def get_hyperopt(self, label):
        from hyperopt import hp

        if self.log:
            return hp.loguniform(label, np.log(self.lower), np.log(self.upper))
        else:
            return hp.uniform(label, self.lower, self.upper)

    def get_CS(self, label):
        import ConfigSpace.hyperparameters as CSH

        return CSH.UniformFloatHyperparameter(
            name=label, lower=self.lower, upper=self.upper, log=self.log
        )

    def __repr__(self):
        return f"UniformDistribution(lower={self.lower}, upper={self.upper}, log={self.log})"


class IntUniformDistribution(Distribution):
    """
    Uniform integer distribution.

    Parameters
    ----------
    lower: int
        Inclusive lower bound of distribution.
    upper: int
        Inclusive upper bound of distribution.
    log: bool, default = False:
        If True, the distribution will be log-uniform.
    """

    def __init__(self, lower: int, upper: int, log: bool = False):
        self.lower = lower
        self.upper = upper
        self.log = log

    def get_skopt(self):
        import skopt.space

        if self.log:
            return skopt.space.Integer(self.lower, self.upper, prior="log-uniform")
        else:
            return skopt.space.Integer(self.lower, self.upper, prior="uniform")

    def get_optuna(self):
        import optuna

        if self.log:
            return optuna.distributions.IntLogUniformDistribution(
                self.lower, self.upper
            )
        else:
            return optuna.distributions.IntUniformDistribution(self.lower, self.upper)

    def get_hyperopt(self, label):
        from hyperopt import hp
        from hyperopt.pyll import scope

        if self.log:
            return scope.int(
                hp.qloguniform(label, np.log(self.lower), np.log(self.upper), 1)
            )
        else:
            return scope.int(hp.quniform(label, self.lower, self.upper, 1))

    def get_CS(self, label):
        import ConfigSpace.hyperparameters as CSH

        return CSH.UniformIntegerHyperparameter(
            name=label, lower=self.lower, upper=self.upper, log=self.log
        )

    def __repr__(self):
        return f"IntUniformDistribution(lower={self.lower}, upper={self.upper}, log={self.log})"


class DiscreteUniformDistribution(Distribution):
    """
    Discrete (with step) uniform float distribution.

    Parameters
    ----------
    lower: float
        Inclusive lower bound of distribution.
    upper: float
        Inclusive upper bound of distribution.
    q: float = None:
        Step. If None, will be equal to UniformDistribution.

    Warnings
    --------
    - Due to scikit-optimize not supporting discrete distributions,
    `get_skopt()` will return a standard uniform distribution.
    """

    def __init__(self, lower: int, upper: int, q: Optional[float] = None):
        self.lower = lower
        self.upper = upper
        self.q = q

    def get_skopt(self):
        import skopt.space

        # not supported, return standard uniform distribution
        return skopt.space.Real(self.lower, self.upper, prior="uniform")

    def get_optuna(self):
        import optuna

        return optuna.distributions.DiscreteUniformDistribution(
            self.lower, self.upper, self.q
        )

    def get_hyperopt(self, label):
        from hyperopt import hp

        return hp.quniform(label, self.lower, self.upper, self.q)

    def get_CS(self, label):
        import ConfigSpace.hyperparameters as CSH

        return CSH.UniformFloatHyperparameter(
            name=label, lower=self.lower, upper=self.upper, q=self.q
        )

    def __repr__(self):
        return f"DiscreteUniformDistribution(lower={self.lower}, upper={self.upper}, q={self.q})"


class CategoricalDistribution(Distribution):
    """
    Categorical distribution.

    Parameters
    ----------
    values: list or other iterable
        Possible values.

    Warnings
    --------
    - `None` is not supported  as a value for ConfigSpace.
    """

    def __init__(self, values):
        self.values = list(values)

    def get_skopt(self):
        import skopt.space

        return skopt.space.Categorical(
            [x if isinstance(x, Hashable) else None for x in self.values],
            transform="identity",
        )

    def get_optuna(self):
        import optuna

        return optuna.distributions.CategoricalDistribution(self.values)

    def get_hyperopt(self, label):
        from hyperopt import hp

        return hp.choice(label, self.values)

    def get_CS(self, label):
        import ConfigSpace.hyperparameters as CSH

        return CSH.CategoricalHyperparameter(
            name=label, choices=[x for x in self.values if isinstance(x, Hashable)]
        )

    def __repr__(self):
        return f"CategoricalDistribution(values={self.values})"


def get_skopt_distributions(distributions: Dict[str, Distribution]) -> dict:
    return {k: v.get_skopt() for k, v in distributions.items()}


def get_optuna_distributions(distributions: Dict[str, Distribution]) -> dict:
    return {k: v.get_optuna() for k, v in distributions.items()}


def get_hyperopt_distributions(distributions: Dict[str, Distribution]) -> dict:
    return {k: v.get_hyperopt(k) for k, v in distributions.items()}


def get_CS_distributions(distributions: Dict[str, Distribution]) -> dict:
    return {k: v.get_CS(k) for k, v in distributions.items()}


def get_min_max(o):
    if isinstance(o, CategoricalDistribution):
        o = o.values
    elif isinstance(o, Distribution):
        return (o.lower, o.upper)

    o = sorted(o)
    return (o[0], o[-1])
