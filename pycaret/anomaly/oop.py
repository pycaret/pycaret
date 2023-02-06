from typing import Any, List, Optional, Tuple, Union

import pandas as pd

from pycaret.containers.metrics import get_all_anomaly_metric_containers
from pycaret.containers.models import get_all_anomaly_model_containers
from pycaret.internal.logging import get_logger
from pycaret.internal.pycaret_experiment.unsupervised_experiment import (
    _UnsupervisedExperiment,
)
from pycaret.utils.generic import MLUsecase

LOGGER = get_logger()


class AnomalyExperiment(_UnsupervisedExperiment):
    def __init__(self):
        super().__init__()
        self._ml_usecase = MLUsecase.ANOMALY
        self.exp_name_log = "anomaly-default-name"
        self._available_plots = {
            "pipeline": "Pipeline Plot",
            "tsne": "t-SNE (3d) Dimension Plot",
            "umap": "UMAP Dimensionality Plot",
        }

    def _get_models(self, raise_errors: bool = True) -> Tuple[dict, dict]:
        all_models = {
            k: v
            for k, v in get_all_anomaly_model_containers(
                self, raise_errors=raise_errors
            ).items()
            if not v.is_special
        }
        all_models_internal = get_all_anomaly_model_containers(
            self, raise_errors=raise_errors
        )
        return all_models, all_models_internal

    def _get_metrics(self, raise_errors: bool = True) -> dict:
        return get_all_anomaly_metric_containers(
            self.variables, raise_errors=raise_errors
        )

    def _get_default_plots_to_log(self) -> List[str]:
        return ["tsne"]

    def predict_model(
        self, estimator, data: pd.DataFrame, ml_usecase: Optional[MLUsecase] = None
    ) -> pd.DataFrame:
        """
        This function generates anomaly labels on using a trained model.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> anomaly = get_data('anomaly')
        >>> from pycaret.anomaly import *
        >>> exp_name = setup(data = anomaly)
        >>> knn = create_model('knn')
        >>> knn_predictions = predict_model(model = knn, data = unseen_data)


        model: scikit-learn compatible object
            Trained Model Object.


        data : pandas.DataFrame
            Shape (n_samples, n_features) where n_samples is the number of samples and
            n_features is the number of features.


        Returns:
            pandas.DataFrame


        Warnings
        --------
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
        >>> anomaly = get_data('anomaly')
        >>> from pycaret.anomaly import *
        >>> exp_name = setup(data = anomaly)
        >>> knn = create_model('knn')
        >>> plot_model(knn, plot = 'tsne')


        model: scikit-learn compatible object
            Trained Model Object


        plot: str, default = 'tsne'
            List of available plots (ID - Name):

            * 'tsne' - t-SNE (3d) Dimension Plot
            * 'umap' - UMAP Dimensionality Plot


        feature: str, default = None
            Feature to be used as a hoverover tooltip and/or label when the ``label``
            param is set to True. When feature is None, first column of the dataset
            is used.


        label: bool, default = False
            Name of column to be used as data labels.


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
