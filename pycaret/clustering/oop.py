from typing import Any, List, Optional, Tuple, Union

import numpy as np  # type: ignore
import pandas as pd  # type ignore

import pycaret.internal.patches.sklearn
import pycaret.internal.patches.yellowbrick
import pycaret.internal.persistence
import pycaret.internal.preprocess
from pycaret.containers.metrics import get_all_clust_metric_containers
from pycaret.containers.models import get_all_clust_model_containers
from pycaret.internal.logging import get_logger
from pycaret.internal.pycaret_experiment.unsupervised_experiment import (
    _UnsupervisedExperiment,
)
from pycaret.utils.generic import MLUsecase

LOGGER = get_logger()


class ClusteringExperiment(_UnsupervisedExperiment):
    def __init__(self) -> None:
        super().__init__()
        self._ml_usecase = MLUsecase.CLUSTERING
        self.exp_name_log = "cluster-default-name"
        self._available_plots = {
            "pipeline": "Pipeline Plot",
            "cluster": "t-SNE (3d) Dimension Plot",
            "tsne": "Cluster t-SNE (3d)",
            "elbow": "Elbow Plot",
            "silhouette": "Silhouette Plot",
            "distance": "Distance Plot",
            "distribution": "Distribution Plot",
        }

    def _get_models(self, raise_errors: bool = True) -> Tuple[dict, dict]:
        all_models = {
            k: v
            for k, v in get_all_clust_model_containers(
                self, raise_errors=raise_errors
            ).items()
            if not v.is_special
        }
        all_models_internal = get_all_clust_model_containers(
            self, raise_errors=raise_errors
        )
        return all_models, all_models_internal

    def _get_metrics(self, raise_errors: bool = True) -> dict:
        return get_all_clust_metric_containers(
            self.variables, raise_errors=raise_errors
        )

    def _get_default_plots_to_log(self) -> List[str]:
        return ["cluster", "distribution", "elbow"]

    def predict_model(
        self, estimator, data: pd.DataFrame, ml_usecase: Optional[MLUsecase] = None
    ) -> pd.DataFrame:
        """
        This function generates cluster labels using a trained model.

        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> jewellery = get_data('jewellery')
        >>> from pycaret.clustering import *
        >>> exp_name = setup(data = jewellery)
        >>> kmeans = create_model('kmeans')
        >>> kmeans_predictions = predict_model(model = kmeans, data = unseen_data)


        model: scikit-learn compatible object
            Trained Model Object.


        data : pandas.DataFrame
            Shape (n_samples, n_features) where n_samples is the number of samples and
            n_features is the number of features.


        Returns:
            pandas.DataFrame


        Warnings
        --------
        - Models that do not support 'predict' method cannot be used in the ``predict_model``.

        - The behavior of the predict_model is changed in version 2.1 without backward compatibility.
        As such, the pipelines trained using the version (<= 2.0), may not work for inference
        with version >= 2.1. You can either retrain your models with a newer version or downgrade
        the version for inference.


        """

        return super().predict_model(estimator, data, ml_usecase)

    def plot_model(
        self,
        estimator,
        plot: str = "auc",
        scale: float = 1,
        save: Union[str, bool] = False,
        fold: Optional[Union[int, Any]] = None,
        fit_kwargs: Optional[dict] = None,
        plot_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        feature_name: Optional[str] = None,
        label: bool = False,
        use_train_data: bool = False,
        verbose: bool = True,
        display_format: Optional[str] = None,
    ) -> Optional[str]:
        """
        This function analyzes the performance of a trained model.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> jewellery = get_data('jewellery')
        >>> from pycaret.clustering import *
        >>> exp_name = setup(data = jewellery)
        >>> kmeans = create_model('kmeans')
        >>> plot_model(kmeans, plot = 'cluster')


        model: scikit-learn compatible object
            Trained Model Object


        plot: str, default = 'cluster'
            List of available plots (ID - Name):

            * 'cluster' - Cluster PCA Plot (2d)
            * 'tsne' - Cluster t-SNE (3d)
            * 'elbow' - Elbow Plot
            * 'silhouette' - Silhouette Plot
            * 'distance' - Distance Plot
            * 'distribution' - Distribution Plot


        feature: str, default = None
            Feature to be evaluated when plot = 'distribution'. When ``plot`` type is
            'cluster' or 'tsne' feature column is used as a hoverover tooltip and/or
            label when the ``label`` param is set to True. When the ``plot`` type is
            'cluster' or 'tsne' and feature is None, first column of the dataset is
            used.


        label: bool, default = False
            Name of column to be used as data labels. Ignored when ``plot`` is not
            'cluster' or 'tsne'.


        scale: float, default = 1
            The resolution scale of the figure.


        save: bool, default = False
            When set to True, plot is saved in the current working directory.


        display_format: str, default = None
            To display plots in Streamlit (https://www.streamlit.io/), set this to 'streamlit'.
            Currently, not all plots are supported.


        Returns:
            Path to saved file, if any.

        """
        return super().plot_model(
            estimator,
            plot,
            scale,
            save,
            fold,
            fit_kwargs,
            plot_kwargs,
            groups,
            feature_name,
            label,
            use_train_data,
            verbose,
            display_format,
        )

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


        reset: bool, default = False
            If True, will reset all changes made using add_metric() and get_metric().


        include_custom: bool, default = True
            Whether to include user added (custom) metrics or not.


        raise_errors: bool, default = True
            If False, will suppress all exceptions, ignoring models
            that couldn't be created.


        Returns:
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
            df = df[df["Custom"] is False]

        return df

    def add_metric(
        self,
        id: str,
        name: str,
        score_func: type,
        greater_is_better: bool = True,
        needs_ground_truth: bool = False,
        **kwargs,
    ) -> pd.Series:
        """
        Adds a custom metric to be used in all functions.


        id: str
            Unique id for the metric.


        name: str
            Display name of the metric.


        score_func: type
            Score function (or loss function) with signature ``score_func(y, y_pred, **kwargs)``.


        greater_is_better: bool, default = True
            Whether score_func is a score function (default), meaning high is good,
            or a loss function, meaning low is good. In the latter case, the
            scorer object will sign-flip the outcome of the score_func.


        multiclass: bool, default = True
            Whether the metric supports multiclass problems.


        **kwargs:
            Arguments to be passed to score function.

        Returns:
            pandas.Series

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
        Removes a metric used for evaluation.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> jewellery = get_data('jewellery')
        >>> from pycaret.clustering import *
        >>> exp_name = setup(data = jewellery)
        >>> remove_metric('cs')


        name_or_id: str
            Display name or ID of the metric.


        Returns:
            None

        """

        if not self._setup_ran:
            raise ValueError("setup() needs to be ran first.")

        try:
            self._all_metrics.pop(name_or_id)
            return
        except Exception:
            pass

        try:
            k_to_remove = next(
                k for k, v in self._all_metrics.items() if v.name == name_or_id
            )
            self._all_metrics.pop(k_to_remove)
            return
        except Exception:
            pass

        raise ValueError(
            f"No metric 'Display Name' or 'ID' (index) {name_or_id} present in the metrics repository."
        )
