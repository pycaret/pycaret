from pycaret.internal.pycaret_experiment.utils import MLUsecase
from pycaret.internal.pycaret_experiment.unsupervised_experiment import (
    _UnsupervisedExperiment,
)
import pycaret.internal.patches.sklearn
import pycaret.internal.patches.yellowbrick
from pycaret.internal.logging import get_logger
from pycaret.internal.distributions import *
from pycaret.internal.validation import *
import pycaret.containers.metrics.clustering
import pycaret.containers.models.clustering
import pycaret.internal.preprocess
import pycaret.internal.persistence
import pandas as pd  # type ignore
import numpy as np  # type: ignore
from typing import List, Tuple, Any, Union, Optional, Dict
import warnings
import plotly.express as px  # type: ignore
import plotly.graph_objects as go  # type: ignore


warnings.filterwarnings("ignore")
LOGGER = get_logger()


class ClusteringExperiment(_UnsupervisedExperiment):
    def __init__(self) -> None:
        super().__init__()
        self._ml_usecase = MLUsecase.CLUSTERING
        self.exp_name_log = "cluster-default-name"
        self._available_plots = {
            "cluster": "t-SNE (3d) Dimension Plot",
            "tsne": "Cluster t-SNE (3d)",
            "elbow": "Elbow Plot",
            "silhouette": "Silhouette Plot",
            "distance": "Distance Plot",
            "distribution": "Distribution Plot",
        }
        return

    def _get_models(self, raise_errors: bool = True) -> Tuple[dict, dict]:
        all_models = {
            k: v
            for k, v in pycaret.containers.models.clustering.get_all_model_containers(
                self, raise_errors=raise_errors
            ).items()
            if not v.is_special
        }
        all_models_internal = pycaret.containers.models.clustering.get_all_model_containers(
            self, raise_errors=raise_errors
        )
        return all_models, all_models_internal

    def _get_metrics(self, raise_errors: bool = True) -> dict:
        return pycaret.containers.metrics.clustering.get_all_metric_containers(
            self.variables, raise_errors=raise_errors
        )

    def _get_default_plots_to_log(self) -> List[str]:
        return ["cluster", "distribution", "elbow"]

    def get_metrics(
        self,
        reset: bool = False,
        include_custom: bool = True,
        raise_errors: bool = True,
    ) -> pd.DataFrame:
        """
        Returns table of metrics available.

        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> jewellery = get_data('jewellery')
        >>> from pycaret.clustering import *
        >>> exp_name = setup(data = jewellery)
        >>> all_metrics = get_metrics()

        This will return pandas dataframe with all available
        metrics and their metadata.

        Parameters
        ----------
        reset: bool, default = False
            If True, will reset all changes made using add_metric() and get_metric().
        include_custom: bool, default = True
            Whether to include user added (custom) metrics or not.
        raise_errors: bool, default = True
            If False, will suppress all exceptions, ignoring models
            that couldn't be created.

        Returns
        -------
        pandas.DataFrame

        """

        if reset and not self._setup_ran:
            raise ValueError("setup() needs to be ran first.")

        np.random.seed(self.seed)

        if reset:
            self._all_metrics = self._get_metrics(raise_errors=raise_errors)

        metric_containers = self._all_metrics
        rows = [v.get_dict() for k, v in metric_containers.items()]

        df = pd.DataFrame(rows)
        df.set_index("ID", inplace=True, drop=True)

        if not include_custom:
            df = df[df["Custom"] == False]

        return df

    def add_metric(
        self,
        id: str,
        name: str,
        score_func: type,
        target: str = "pred",
        greater_is_better: bool = True,
        needs_ground_truth: bool = False,
        **kwargs,
    ) -> pd.Series:
        """
        Adds a custom metric to be used in all functions.

        Parameters
        ----------
        id: str
            Unique id for the metric.

        name: str
            Display name of the metric.

        score_func: type
            Score function (or loss function) with signature score_func(y, y_pred, **kwargs).

        target: str, default = 'pred'
            The target of the score function.
            - 'pred' for the prediction table
            - 'pred_proba' for pred_proba
            - 'threshold' for decision_function or predict_proba

        greater_is_better: bool, default = True
            Whether score_func is a score function (default), meaning high is good,
            or a loss function, meaning low is good. In the latter case, the
            scorer object will sign-flip the outcome of the score_func.

        needs_ground_truth: bool, default = False
            Whether the metric needs ground truth to be calculated.

        **kwargs:
            Arguments to be passed to score function.

        Returns
        -------
        pandas.Series
            The created row as Series.

        """

        if not self._setup_ran:
            raise ValueError("setup() needs to be ran first.")

        if id in self._all_metrics:
            raise ValueError("id already present in metrics dataframe.")

        new_metric = pycaret.containers.metrics.clustering.ClusterMetricContainer(
            id=id,
            name=name,
            score_func=score_func,
            args=kwargs,
            display_name=name,
            greater_is_better=greater_is_better,
            needs_ground_truth=needs_ground_truth,
            is_custom=True,
        )

        self._all_metrics[id] = new_metric

        new_metric = new_metric.get_dict()

        new_metric = pd.Series(new_metric, name=id.replace(" ", "_")).drop("ID")

        return new_metric

    def remove_metric(self, name_or_id: str):
        """
        Removes a metric used in all functions.

        Parameters
        ----------
        name_or_id: str
            Display name or ID of the metric.

        """
        if not self._setup_ran:
            raise ValueError("setup() needs to be ran first.")

        try:
            self._all_metrics.pop(name_or_id)
            return
        except:
            pass

        try:
            k_to_remove = next(
                k for k, v in self._all_metrics.items() if v.name == name_or_id
            )
            self._all_metrics.pop(k_to_remove)
            return
        except:
            pass

        raise ValueError(
            f"No metric 'Display Name' or 'ID' (index) {name_or_id} present in the metrics repository."
        )

