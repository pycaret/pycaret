# Module: internal.distributions
# Author: Antoni Baum (Yard1) <antoni.baum@protonmail.com>
# License: MIT

from typing import Dict, Optional

from scipy.stats import loguniform, randint, uniform

try:
    from collections.abc import Hashable
except Exception:
    from collections import Hashable

from copy import copy

import numpy as np


class Distribution:
    def __init__(self):
        raise NotImplementedError("This is an abstract class.")

    def get_base(self):
        raise NotImplementedError("This is an abstract class.")

    def get_skopt(self):
        raise NotImplementedError("This is an abstract class.")

    def get_optuna(self):
        raise NotImplementedError("This is an abstract class.")

    def get_hyperopt(self, label):
        raise NotImplementedError("This is an abstract class.")

    def get_CS(self, label):
        raise NotImplementedError("This is an abstract class.")

    def get_tune(self):
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

    def get_base(self):
        """get distributions from base libraries such as scipy, numpy, etc."""
        if self.log:
            return loguniform(self.lower, self.upper)
        return uniform(self.lower, self.upper)  # log = False

    def get_skopt(self):
        import skopt.space

        if self.log:
            return skopt.space.Real(self.lower, self.upper, prior="log-uniform")
        else:
            return skopt.space.Real(self.lower, self.upper, prior="uniform")

    def get_optuna(self):
        import optuna

        optuna_version = int(optuna.__version__[0])

        if self.log:
            return (
                optuna.distributions.FloatDistribution(self.lower, self.upper, log=True)
                if optuna_version >= 3
                else optuna.distributions.LogUniformDistribution(self.lower, self.upper)
            )
        else:
            return (
                optuna.distributions.FloatDistribution(self.lower, self.upper)
                if optuna_version >= 3
                else optuna.distributions.UniformDistribution(self.lower, self.upper)
            )

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

    def get_tune(self):
        from ray import tune

        if self.log:
            return tune.loguniform(lower=self.lower, upper=self.upper)
        else:
            return tune.uniform(lower=self.lower, upper=self.upper)

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

    def get_base(self):
        """get distributions from base libraries such as scipy, numpy, etc."""
        if self.log:
            raise NotImplementedError(
                "integer log sampling for base library has not been implemented yet."
            )
        return randint(self.lower, self.upper)  # log = False

    def get_skopt(self):
        import skopt.space

        if self.log:
            return skopt.space.Integer(self.lower, self.upper, prior="log-uniform")
        else:
            return skopt.space.Integer(self.lower, self.upper, prior="uniform")

    def get_optuna(self):
        import optuna

        optuna_version = int(optuna.__version__[0])

        if self.log:
            return (
                optuna.distributions.IntDistribution(self.lower, self.upper, log=True)
                if optuna_version >= 3
                else optuna.distributions.IntLogUniformDistribution(
                    self.lower, self.upper
                )
            )
        else:
            return (
                optuna.distributions.IntDistribution(self.lower, self.upper)
                if optuna_version >= 3
                else optuna.distributions.IntUniformDistribution(self.lower, self.upper)
            )

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

    def get_tune(self):
        from ray.tune.sample import Integer, LogUniform

        class LogUniformInteger(Integer):
            class _LogUniform(LogUniform):
                def sample(self, domain: "Integer", spec=None, size: int = 1):
                    assert (
                        domain.lower > 0
                    ), "LogUniform needs a lower bound greater than 0"
                    assert (
                        0 < domain.upper < float("inf")
                    ), "LogUniform needs a upper bound greater than 0"
                    logmin = np.log(domain.lower) / np.log(self.base)
                    logmax = np.log(domain.upper) / np.log(self.base)

                    items = self.base ** (np.random.uniform(logmin, logmax, size=size))
                    items = np.round(items).astype(int)
                    return items if len(items) > 1 else domain.cast(items[0])

            def loguniform(self, base: float = 10):
                if not self.lower > 0:
                    raise ValueError(
                        "LogUniform requires a lower bound greater than 0."
                        f"Got: {self.lower}. Did you pass a variable that has "
                        "been log-transformed? If so, pass the non-transformed value "
                        "instead."
                    )
                if not 0 < self.upper < float("inf"):
                    raise ValueError(
                        "LogUniform requires a upper bound greater than 0. "
                        f"Got: {self.lower}. Did you pass a variable that has "
                        "been log-transformed? If so, pass the non-transformed value "
                        "instead."
                    )
                new = copy(self)
                new.set_sampler(self._LogUniform(base))
                return new

        if self.log:
            return LogUniformInteger(self.lower, self.upper).loguniform(10)
        else:
            return Integer(self.lower, self.upper).uniform()

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

    def get_base(self):
        """get distributions from base libraries such as scipy, numpy, etc."""
        raise NotImplementedError(
            "DiscreteUniformDistribution for base library has not been implemented yet."
        )

    def get_skopt(self):
        import skopt.space

        # not supported, return standard uniform distribution
        return skopt.space.Real(self.lower, self.upper, prior="uniform")

    def get_optuna(self):
        import optuna

        optuna_version = int(optuna.__version__[0])

        return (
            optuna.distributions.FloatDistribution(self.lower, self.upper, step=self.q)
            if optuna_version >= 3
            else optuna.distributions.DiscreteUniformDistribution(
                self.lower, self.upper, self.q
            )
        )

    def get_hyperopt(self, label):
        from hyperopt import hp

        return hp.quniform(label, self.lower, self.upper, self.q)

    def get_CS(self, label):
        import ConfigSpace.hyperparameters as CSH

        return CSH.UniformFloatHyperparameter(
            name=label, lower=self.lower, upper=self.upper, q=self.q
        )

    def get_tune(self):
        from ray import tune

        return tune.quniform(lower=self.lower, upper=self.upper, q=self.q)

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

    def get_base(self):
        """get distributions from base libraries such as scipy, numpy, etc."""
        return self.values

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

    def get_tune(self):
        from ray import tune

        return tune.choice(self.values)

    def __repr__(self):
        return f"CategoricalDistribution(values={self.values})"


def get_base_distributions(distributions: Dict[str, Distribution]) -> dict:
    """Returns the distributions from the base libraries.
    Distributions are of types that can be used with scikit-learn `ParamSampler`
    """
    return {
        k: (v.get_base() if isinstance(v, Distribution) else v)
        for k, v in distributions.items()
    }


def get_skopt_distributions(distributions: Dict[str, Distribution]) -> dict:
    return {
        k: (v.get_skopt() if isinstance(v, Distribution) else v)
        for k, v in distributions.items()
    }


def get_optuna_distributions(distributions: Dict[str, Distribution]) -> dict:
    return {
        k: (v.get_optuna() if isinstance(v, Distribution) else v)
        for k, v in distributions.items()
    }


def get_hyperopt_distributions(distributions: Dict[str, Distribution]) -> dict:
    return {
        k: (v.get_hyperopt(k) if isinstance(v, Distribution) else v)
        for k, v in distributions.items()
    }


def get_CS_distributions(distributions: Dict[str, Distribution]) -> dict:
    return {
        k: (v.get_CS(k) if isinstance(v, Distribution) else v)
        for k, v in distributions.items()
    }


def get_tune_distributions(distributions: Dict[str, Distribution]) -> dict:
    return {
        k: (v.get_tune() if isinstance(v, Distribution) else v)
        for k, v in distributions.items()
    }


def get_min_max(o):
    if isinstance(o, CategoricalDistribution):
        o = o.values
    elif isinstance(o, Distribution):
        return (o.lower, o.upper)

    o = sorted(o)
    return (o[0], o[-1])
