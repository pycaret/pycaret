# Module: containers.metrics.base_metric
# Author: Antoni Baum (Yard1) <antoni.baum@protonmail.com>
# License: MIT

from typing import Dict, Any, Union
from pycaret.containers.base_container import BaseContainer
from sklearn.metrics._scorer import _BaseScorer
from sklearn.metrics import make_scorer


class MetricContainer(BaseContainer):
    """
    Base metric container class, for easier definition of containers. Ensures consistent format
    before being turned into a dataframe row.

    Parameters
    ----------
    id : str
        ID used as index.
    name : str
        Full name.
    score_func_type : type
        The callable used for the score function, eg. sklearn.metrics.accuracy_score.
    scorer : str or callable, default = None
        The scorer passed to models. Can be a string representing a built-in sklearn scorer,
        a sklearn Scorer object, or None, in which case a Scorer object will be created from
        score_func_type and args.
    args : dict, default = {}
        The arguments to always pass to constructor when initializing score_func_type of class_def class.
    display_name : str, default = None
        Display name (shorter than name). If None or empty, will use name.
    is_custom : bool, default = False
        Is the metric custom. Should be False for all metrics defined in PyCaret.

    Attributes
    ----------
    id : str
        ID used as index.
    name : str
        Full name.
    score_func_type : type
        The callable used for the score function, eg. sklearn.metrics.accuracy_score.
    scorer : str or callable
        The scorer passed to models. Can be a string representing a built-in sklearn scorer,
        a sklearn Scorer object, or None, in which case a Scorer object will be created from
        score_func_type and args.
    args : dict
        The arguments to always pass to constructor when initializing score_func_type of class_def class.
    display_name : str
        Display name (shorter than name). If None or empty, will use name.
    is_custom : bool
        Is the metric custom. Should be False for all metrics defined in PyCaret.

    """

    def __init__(
        self,
        id: str,
        name: str,
        score_func_type: type,
        scorer: Union[str, _BaseScorer] = None,
        args: Dict[str, Any] = {},
        display_name: str = None,
        is_custom: bool = False,
    ) -> None:
        self.id = id
        self.name = name
        self.score_func_type = score_func_type
        self.scorer = scorer if scorer else make_scorer(score_func_type, **args)
        self.display_name = display_name if display_name else name
        self.args = args
        self.is_custom = is_custom

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
        d = {
            "ID": self.id,
            "Name": self.name,
            "Display Name": self.display_name,
            "Score Function": self.score_func_type,
            "Scorer": self.scorer,
            "Args": self.args,
            "Custom": self.is_custom,
        }

        return d
