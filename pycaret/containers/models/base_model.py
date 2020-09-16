# Module: containers.models.base_model
# Author: Antoni Baum (Yard1) <antoni.baum@protonmail.com>
# License: MIT

from pycaret.internal.distributions import CategoricalDistribution
import pycaret.internal.utils
from pycaret.containers.base_container import BaseContainer
from typing import Dict, Any


class ModelContainer(BaseContainer):
    """
    Base model container class, for easier definition of containers. Ensures consistent format
    before being turned into a dataframe row.

    Parameters
    ----------
    id : str
        ID used as index.
    name : str
        Full display name.
    class_def : type
        The class used for the model, eg. LogisticRegression.
    is_turbo : bool, default = True
        Should the model be used with 'turbo = True' in compare_models().
    args : dict, default = {}
        The arguments to always pass to constructor when initializing object of class_def class.
    is_special : bool, default = False
        Is the model special (not intended to be used on its own, eg. VotingClassifier).

    Attributes
    ----------
    id : str
        ID used as index.
    name : str
        Full display name.
    class_def : type
        The class used for the model, eg. LogisticRegression.
    is_turbo : bool
        Should the model be used with 'turbo = True' in compare_models().
    args : dict
        The arguments to always pass to constructor when initializing object of class_def class.
    is_special : bool
        Is the model special (not intended to be used on its own, eg. VotingClassifier).

    """

    def __init__(
        self,
        id: str,
        name: str,
        class_def: type,
        args: Dict[str, Any] = None,
        is_special: bool = False,
    ) -> None:
        self.id = id
        self.name = name
        self.class_def = class_def
        self.reference = self.get_class_name()
        if not args:
            args = {}
        self.args = args
        self.is_special = is_special

    def get_dict(self, internal: bool = True) -> Dict[str, Any]:
        """
        Returns a dictionary of the model properties, to
        be turned into a pandas DataFrame row.

        Parameters
        ----------
        internal : bool, default = True
            If True, will return all properties. If False, will only
            return properties intended for the user to see.

        Returns
        -------
        dict of str : Any

        """
        d = [("ID", self.id), ("Name", self.name), ("Reference", self.reference)]

        if internal:
            d += [
                ("Special", self.is_special),
                ("Class", self.class_def),
                ("Args", self.args),
            ]

        return dict(d)


def _leftover_parameters_to_categorical_distributions(
    tune_grid: dict, tune_distributions: dict
) -> None:
    for k, v in tune_grid.items():
        if not k in tune_distributions:
            tune_distributions[k] = CategoricalDistribution(v)