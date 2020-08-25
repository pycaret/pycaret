# Module: models.base_model
# Author: Antoni Baum (Yard1) <antoni.baum@protonmail.com>
# License: MIT

import pycaret.internal.utils


class ModelContainer:
    """
    Base model container class, for easier definition of containers. Ensures consistent format
    before being turned into a dataframe row.
    """

    def __init__(
        self,
        id: str,
        name: str,
        class_def: type,
        args: dict = {},
        is_special: bool = False,
    ) -> None:
        self.id = id
        self.name = name
        self.class_def = class_def
        self.reference = self.get_class_name()
        self.args = args
        self.is_special = is_special

    def get_class_name(self):
        return pycaret.internal.utils.get_class_name(self.class_def)

    def get_package_name(self):
        return pycaret.internal.utils.get_package_name(self.class_def)

    def get_dict(self, internal: bool = True) -> dict:
        """
        Returns a dictionary of the model properties, to
        be turned into a pandas DataFrame row.
        """
        d = [("ID", self.id), ("Name", self.name), ("Reference", self.reference)]

        if internal:
            d += [
                ("Special", self.is_special),
                ("Class", self.class_def),
                ("Args", self.args),
            ]

        return dict(d)
