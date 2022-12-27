# Module: containers.models.base_model
# Author: Antoni Baum (Yard1) <antoni.baum@protonmail.com>
# License: MIT

from typing import Any, Dict, List, Optional

from pycaret.containers.base_container import BaseContainer
from pycaret.internal.distributions import CategoricalDistribution
from pycaret.internal.pycaret_experiment.pycaret_experiment import _PyCaretExperiment
from pycaret.utils.generic import get_allowed_engines, get_logger


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
    args : dict, default = {} (empty dict)
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
        super().__init__(id=id, name=name, class_def=class_def, args=args)
        self.reference = self.get_class_name()
        if not eq_function:
            eq_function = lambda x: isinstance(x, self.class_def)
        self.eq_function = eq_function
        self.is_special = is_special
        self.logger = get_logger()
        self.allowed_engines = None
        self.default_engine = None

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

    def _set_engine(
        self,
        id: str,
        experiment: _PyCaretExperiment,
        severity: str = "error",
    ):
        """Sets the engine to use for a particular model based on what is set in
        the experiment.

        Parameters
        ----------
        id : str
            Identifier for the model for which the engine should be set, e.g.
            "auto_arima"
        experiment : _PyCaretExperiment
            The experiment object to which the model belongs. The engine to use
            for this model is extracted from this experiment object.
        severity : str, optional
            How to handle incorrectly specified engines. Allowed values are "error"
            and "warning". If set to "warning", the existing engine is left
            unchanged if the specified engine is not correct., by default "error".

        Raises
        ------
        ValueError
            (1) If specified engine is not in the allowed list of engines and
                severity is set to "error"
            (2) If the value of "severity" is not one of the allowed values
        """
        engine_to_use = experiment.get_engine(id)
        # If not specified, use the default engine
        if engine_to_use is None:
            engine_to_use = self.default_engine

        if engine_to_use is not None and engine_to_use not in self.allowed_engines:
            msg = (
                f"Engine '{engine_to_use}' for estimator '{id}' is not allowed.\n"
                f"Allowed values are '{self.allowed_engines}'."
            )

            if severity == "error":
                raise ValueError(msg)
            elif severity == "warning":
                self.logger.warning(msg)
                print(msg)
            else:
                raise ValueError(
                    "Error in calling set_engine, severity "
                    f'argument must be "error" or "warning", got "{severity}".'
                )

        self.engine = engine_to_use

    def _set_engine_related_vars(
        self,
        id: str,
        all_allowed_engines: Dict[str, List[str]],
        experiment: _PyCaretExperiment,
    ):
        """Sets the engine related variables, i.e.
        (1) Allowed engine values for the model [List]
        (2) Engine to use for the model based on the settings in the experiment.
        (3) Default engine value to use if nothing is specified in the experiment
            (uses first value from allowed engine values list)

        Parameters
        ----------
        id : str
            Identifier for the model for which the engine should be set
        all_allowed_engines : Dict[str, List[str]]
            All allowed engines for models of this experiment class to which the
            model belongs
        experiment : _PyCaretExperiment
            The experiment object to which the model belongs. The engine to use
            for this model is extracted from this experiment object.
        """
        self.allowed_engines = get_allowed_engines(
            estimator=id, all_allowed_engines=all_allowed_engines
        )
        self.default_engine = self.allowed_engines[0]
        self._set_engine(
            id=id,
            experiment=experiment,
            severity="error",
        )


def leftover_parameters_to_categorical_distributions(
    tune_grid: dict, tune_distributions: dict
) -> None:
    """If a key is present in tune_grid but not in tune_distribution,
    then this function will add those values as a CategoricalDistribution
    to tune_distribution (inplace operation).

    This is mainly a helper function that prevents the need to redefine some
    hyperparameters in tune_distribution again if they have already been
    defined in tune_grid.

    Parameters
    ----------
    tune_grid : dict
        Dictionary of parameters and values (fixed)
    tune_distributions : dict
        Dictionary of PyCaret Distributions
    """
    for k, v in tune_grid.items():
        if k not in tune_distributions:
            tune_distributions[k] = CategoricalDistribution(v)
