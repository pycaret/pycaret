# Module: containers.models.base_model
# Author: Antoni Baum (Yard1) <antoni.baum@protonmail.com>
# License: MIT

from pycaret.internal.distributions import CategoricalDistribution
from pycaret.containers.base_container import BaseContainer
from typing import Dict, Any, Optional


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
    eq_function : type, default = None
        Function to use to check whether an object (model) can be considered equal to the model
        in the container. If None, will be ``is_instance(x, class_def)`` where x is the object.
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
    eq_function : type
        Function to use to check whether an object (model) can be considered equal to the model
        in the container. Must take the checked object as the sole parameter.
        If None, will be ``is_instance(x, class_def)`` where x is the object.
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
        eq_function: Optional[type] = None,
        args: Dict[str, Any] = None,
        is_special: bool = False,
    ) -> None:
        self.id = id
        self.name = name
        self.class_def = class_def
        self.reference = self.get_class_name()
        if not eq_function:
            eq_function = lambda x: isinstance(x, self.class_def)
        self.eq_function = eq_function
        if not args:
            args = {}
        self.args = args
        self.is_special = is_special

    def is_estimator_equal(self, estimator):
        return self.eq_function(estimator)

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
                ("Equality", self.eq_function),
                ("Args", self.args),
            ]

        return dict(d)


def leftover_parameters_to_categorical_distributions(
    tune_grid: dict, tune_distributions: dict
) -> None:
    for k, v in tune_grid.items():
        if not k in tune_distributions:
            tune_distributions[k] = CategoricalDistribution(v)
