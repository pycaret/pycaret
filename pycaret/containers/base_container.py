# Module: containers.base_container
# Author: Antoni Baum (Yard1) <antoni.baum@protonmail.com>
# License: MIT

import inspect
from typing import Dict, Any
import pycaret.internal.utils


class BaseContainer:
    """
    Base container class, for easier definition of containers. Ensures consistent format
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

    Attributes
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

    """

    def __init__(
        self, id: str, name: str, class_def: type, args: Dict[str, Any] = None,
    ) -> None:
        self.id = id
        self.name = name
        self.class_def = class_def
        self.reference = self.get_class_name()
        if not args:
            args = {}
        self.args = args

    def get_class_name(self):
        return pycaret.internal.utils.get_class_name(self.class_def)

    def get_package_name(self):
        return pycaret.internal.utils.get_package_name(self.class_def)

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
                ("Class", self.class_def),
                ("Args", self.args),
            ]

        return dict(d)


def get_all_containers(
    container_globals: dict,
    globals_dict: dict,
    type_var: type,
    raise_errors: bool = True,
) -> Dict[str, BaseContainer]:
    model_container_classes = [
        obj
        for name, obj in container_globals.items()
        if inspect.isclass(obj) and type_var in obj.__bases__
    ]

    model_containers = []

    for obj in model_container_classes:
        if raise_errors:
            model_containers.append(obj(globals_dict))
        else:
            try:
                model_containers.append(obj(globals_dict))
            except:
                pass

    return {container.id: container for container in model_containers}
