# Module: internal.distributions
# Author: Antoni Baum (Yard1) <antoni.baum@protonmail.com>
# License: MIT

from typing import Dict, Optional


class Distrubution:
    def __init__(self):
        raise NotImplementedError("This is an abstract class.")

    def get_optuna(self):
        raise NotImplementedError("This is an abstract class.")

    def get_hyperopt(self, label):
        raise NotImplementedError("This is an abstract class.")

    def get_CS(self, label):
        raise NotImplementedError("This is an abstract class.")


class UniformDistribution(Distrubution):
    def __init__(self, lower: float, upper: float, log: bool = False):
        self.lower = lower
        self.upper = upper
        self.log = log

    def get_optuna(self):
        import optuna

        if self.log:
            return optuna.distributions.LogUniformDistribution(self.lower, self.upper)
        else:
            return optuna.distributions.UniformDistribution(self.lower, self.upper)

    def get_hyperopt(self, label):
        from hyperopt import hp

        if self.log:
            return hp.loguniform(label, self.lower, self.upper)
        else:
            return hp.uniform(label, self.lower, self.upper)

    def get_CS(self, label):
        import ConfigSpace.hyperparameters as CSH

        return CSH.UniformFloatHyperparameter(
            name=label, lower=self.lower, upper=self.upper, log=self.log
        )

    def __str__(self):
        return f"UniformDistribution(lower={self.lower}, upper={self.upper}, log={self.log})"


class IntUniformDistribution(Distrubution):
    def __init__(self, lower: int, upper: int, log: bool = False):
        self.lower = lower
        self.upper = upper
        self.log = log

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
            return scope.int(hp.qloguniform(label, self.lower, self.upper, 1))
        else:
            return scope.int(hp.quniform(label, self.lower, self.upper, 1))

    def get_CS(self, label):
        import ConfigSpace.hyperparameters as CSH

        return CSH.UniformIntegerHyperparameter(
            name=label, lower=self.lower, upper=self.upper, log=self.log
        )

    def __str__(self):
        return f"IntUniformDistribution(lower={self.lower}, upper={self.upper}, log={self.log})"


class DiscreteUniformDistribution(Distrubution):
    def __init__(self, lower: int, upper: int, q: Optional[float] = None):
        self.lower = lower
        self.upper = upper
        self.q = q

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

    def __str__(self):
        return f"DiscreteUniformDistribution(lower={self.lower}, upper={self.upper}, q={self.q})"


class CategoricalDistribution(Distrubution):
    def __init__(self, values):
        self.values = values

    def get_optuna(self):
        import optuna

        return optuna.distributions.CategoricalDistribution(self.values)

    def get_hyperopt(self, label):
        from hyperopt import hp

        return hp.choice(label, self.values)

    def get_CS(self, label):
        import ConfigSpace.hyperparameters as CSH

        return CSH.CategoricalHyperparameter(name=label, choices=self.values)

    def __str__(self):
        return f"CategoricalDistribution(values={self.values})"


def get_optuna_distributions(distributions: Dict[str, Distrubution]) -> dict:
    return {k: v.get_optuna() for k, v in distributions.items()}


def get_hyperopt_distributions(distributions: Dict[str, Distrubution]) -> dict:
    return {k: v.get_hyperopt(k) for k, v in distributions.items()}


def get_CS_distributions(distributions: Dict[str, Distrubution]) -> dict:
    return {k: v.get_CS(k) for k, v in distributions.items()}
