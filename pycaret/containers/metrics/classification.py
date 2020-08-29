# Module: containers.metrics.classification
# Author: Antoni Baum (Yard1) <antoni.baum@protonmail.com>
# License: MIT

# The purpose of this module is to serve as a central repository of classification metrics. The `classification` module will
# call `get_all_metrics_containers()`, which will return instances of all classes in this module that have `ClassificationMetricContainer`
# as a base (but not `ClassificationMetricContainer` itself). In order to add a new model, you only need to create a new class that has
# `ClassificationMetricContainer` as a base, set all of the required parameters in the `__init__` and then call `super().__init__`
# to complete the process. Refer to the existing classes for examples.

from typing import Optional, Union, Dict, Any
from pycaret.containers.metrics.base_metric import MetricContainer
from sklearn.metrics._scorer import _BaseScorer
import pycaret.containers.base_container
import numpy as np
from sklearn import metrics


class ClassificationMetricContainer(MetricContainer):
    """
    Base classification metric container class, for easier definition of containers. Ensures consistent format
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
    target : str, default = 'pred'
        The target of the score function.
        - 'pred' for the prediction table
        - 'pred_proba' for pred_proba
        - 'threshold' for decision_function or predict_proba
    args : dict, default = {}
        The arguments to always pass to constructor when initializing score_func_type of class_def class.
    display_name : str, default = None
        Display name (shorter than name). If None or empty, will use name.
    is_multiclass : bool,  default = True
        Can the metric be used for multiclass problems.
    is_custom : bool, default = False
        Is the metric custom. Should be False for all metrics defined in PyCaret.

    Attributes
    ----------
    id : str
        ID used as index.
    name : str
        Full name.
    score_func_type : type
        The callable used for the score function, eg. metrics.accuracy_score.
    scorer : str or callable
        The scorer passed to models. Can be a string representing a built-in sklearn scorer,
        a sklearn Scorer object, or None, in which case a Scorer object will be created from
        score_func_type and args.
    target : str
        The target of the score function.
        - 'pred' for the prediction table
        - 'pred_proba' for pred_proba
        - 'threshold' for decision_function or predict_proba
    args : dict
        The arguments to always pass to constructor when initializing score_func_type of class_def class.
    is_multiclass : bool
        Can the metric be used for multiclass problems.
    is_custom : bool
        Is the metric custom. Should be False for all metrics defined in PyCaret.

    """

    def __init__(
        self,
        id: str,
        name: str,
        score_func_type: type,
        scorer: Optional[Union[str, _BaseScorer]] = None,
        target: str = "pred",
        args: Dict[str, Any] = {},
        display_name: Optional[str] = None,
        is_multiclass: bool = True,
        is_custom: bool = False,
    ) -> None:

        if not isinstance(args, dict):
            raise TypeError("args needs to be a dictionary.")

        allowed_targets = ["pred", "pred_proba", "threshold"]
        if not target in allowed_targets:
            raise ValueError(f"Target must be one of {', '.join(allowed_targets)}.")

        self.id = id
        self.name = name
        self.score_func_type = score_func_type
        self.target = target
        self.scorer = (
            scorer
            if scorer
            else metrics.make_scorer(
                score_func_type,
                needs_proba=target == "pred_proba",
                needs_threshold="threshold",
                **args,
            )
        )
        self.display_name = display_name if display_name else name
        self.args = args
        self.is_multiclass = is_multiclass
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
            "Target": self.target,
            "Args": self.args,
            "Multiclass": self.is_multiclass,
            "Custom": self.is_custom,
        }

        return d


class AccuracyMetricContainer(ClassificationMetricContainer):
    def __init__(self, globals_dict: dict) -> None:
        super().__init__(
            id="acc",
            name="Accuracy",
            score_func_type=metrics.accuracy_score,
            scorer="accuracy",
        )


class ROCAUCMetricContainer(ClassificationMetricContainer):
    def __init__(self, globals_dict: dict) -> None:
        super().__init__(
            id="auc",
            name="AUC",
            score_func_type=metrics.roc_auc_score,
            scorer="roc_auc",
            target="pred_proba",
            is_multiclass=False,
        )


class RecallMetricContainer(ClassificationMetricContainer):
    def __init__(self, globals_dict: dict) -> None:
        super().__init__(
            id="recall",
            name="Recall",
            score_func_type=metrics.recall_score,
            scorer=metrics.make_scorer(metrics.recall_score, average="macro")
            if globals_dict["y"].value_counts().count() > 2
            else "recall",
            args={"average": "macro"}
            if globals_dict["y"].value_counts().count() > 2
            else {},
        )


class PrecisionMetricContainer(ClassificationMetricContainer):
    def __init__(self, globals_dict: dict) -> None:
        super().__init__(
            id="precision",
            name="Precision",
            display_name="Prec.",
            score_func_type=metrics.precision_score,
            scorer=metrics.make_scorer(metrics.precision_score, average="weighted")
            if globals_dict["y"].value_counts().count() > 2
            else "precision",
            args={"average": "weighted"}
            if globals_dict["y"].value_counts().count() > 2
            else {},
        )


class F1MetricContainer(ClassificationMetricContainer):
    def __init__(self, globals_dict: dict) -> None:
        super().__init__(
            id="f1",
            name="F1",
            score_func_type=metrics.f1_score,
            scorer=metrics.make_scorer(metrics.f1_score, average="weighted")
            if globals_dict["y"].value_counts().count() > 2
            else "f1",
            args={"average": "weighted"}
            if globals_dict["y"].value_counts().count() > 2
            else {},
        )


class KappaMetricContainer(ClassificationMetricContainer):
    def __init__(self, globals_dict: dict) -> None:
        super().__init__(
            id="kappa",
            name="Kappa",
            score_func_type=metrics.cohen_kappa_score,
            scorer=metrics.make_scorer(metrics.cohen_kappa_score),
        )


class MCCMetricContainer(ClassificationMetricContainer):
    def __init__(self, globals_dict: dict) -> None:
        super().__init__(
            id="mcc",
            name="MCC",
            score_func_type=metrics.matthews_corrcoef,
            scorer=metrics.make_scorer(metrics.matthews_corrcoef),
        )


class TTMetricContainer(ClassificationMetricContainer):
    def __init__(self, globals_dict: dict) -> None:
        # special case
        self.id = "tt"
        self.name = "TT"
        self.display_name = "TT (Sec)"
        self.score_func_type = None
        self.scorer = None
        self.args = None
        self.target = None
        self.is_multiclass = True
        self.is_custom = False


def get_all_metric_containers(
    globals_dict: dict,
) -> Dict[str, ClassificationMetricContainer]:
    return pycaret.containers.base_container.get_all_containers(
        globals(), globals_dict, ClassificationMetricContainer
    )

