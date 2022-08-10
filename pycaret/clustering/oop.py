from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np  # type: ignore
import pandas as pd  # type ignore

import pycaret.containers.metrics.clustering
import pycaret.containers.models.clustering
import pycaret.internal.patches.sklearn
import pycaret.internal.patches.yellowbrick
import pycaret.internal.persistence
import pycaret.internal.preprocess
from pycaret.containers.models.clustering import (
    ALL_ALLOWED_ENGINES,
    get_container_default_engines,
)
from pycaret.internal.logging import get_logger
from pycaret.internal.pycaret_experiment.unsupervised_experiment import (
    _UnsupervisedExperiment,
)
from pycaret.internal.pycaret_experiment.utils import MLUsecase

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
            for k, v in pycaret.containers.models.clustering.get_all_model_containers(
                self, raise_errors=raise_errors
            ).items()
            if not v.is_special
        }
        all_models_internal = (
            pycaret.containers.models.clustering.get_all_model_containers(
                self, raise_errors=raise_errors
            )
        )
        return all_models, all_models_internal

    def _get_metrics(self, raise_errors: bool = True) -> dict:
        return pycaret.containers.metrics.clustering.get_all_metric_containers(
            self.variables, raise_errors=raise_errors
        )

    def _get_default_plots_to_log(self) -> List[str]:
        return ["cluster", "distribution", "elbow"]

    def create_model(
        self,
        estimator: Union[str, Any],
        fold: Optional[Union[int, Any]] = None,
        round: int = 4,
        cross_validation: bool = True,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        experiment_custom_tags: Optional[Dict[str, Any]] = None,
        engine: Optional[str] = None,
        verbose: bool = True,
        return_train_score: bool = False,
        **kwargs,
    ) -> Any:

        """
        This function trains and evaluates the performance of a given estimator
        using cross validation. The output of this function is a score grid with
        CV scores by fold. Metrics evaluated during CV can be accessed using the
        ``get_metrics`` function. Custom metrics can be added or removed using
        ``add_metric`` and ``remove_metric`` function. All the available models
        can be accessed using the ``models`` function.

        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase')
        >>> lr = create_model('lr')


        estimator: str or scikit-learn compatible object
            ID of an estimator available in model library or pass an untrained
            model object consistent with scikit-learn API. Estimators available
            in the model library (ID - Name):

            * 'lr' - Logistic Regression
            * 'knn' - K Neighbors Classifier
            * 'nb' - Naive Bayes
            * 'dt' - Decision Tree Classifier
            * 'svm' - SVM - Linear Kernel
            * 'rbfsvm' - SVM - Radial Kernel
            * 'gpc' - Gaussian Process Classifier
            * 'mlp' - MLP Classifier
            * 'ridge' - Ridge Classifier
            * 'rf' - Random Forest Classifier
            * 'qda' - Quadratic Discriminant Analysis
            * 'ada' - Ada Boost Classifier
            * 'gbc' - Gradient Boosting Classifier
            * 'lda' - Linear Discriminant Analysis
            * 'et' - Extra Trees Classifier
            * 'xgboost' - Extreme Gradient Boosting
            * 'lightgbm' - Light Gradient Boosting Machine
            * 'catboost' - CatBoost Classifier


        fold: int or scikit-learn compatible CV generator, default = None
            Controls cross-validation. If None, the CV generator in the ``fold_strategy``
            parameter of the ``setup`` function is used. When an integer is passed,
            it is interpreted as the 'n_splits' parameter of the CV generator in the
            ``setup`` function.


        round: int, default = 4
            Number of decimal places the metrics in the score grid will be rounded to.


        cross_validation: bool, default = True
            When set to False, metrics are evaluated on holdout set. ``fold`` param
            is ignored when cross_validation is set to False.


        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.


        groups: str or array-like, with shape (n_samples,), default = None
            Optional group labels when GroupKFold is used for the cross validation.
            It takes an array with shape (n_samples, ) where n_samples is the number
            of rows in training dataset. When string is passed, it is interpreted as
            the column name in the dataset containing group labels.


        experiment_custom_tags: dict, default = None
            Dictionary of tag_name: String -> value: (String, but will be string-ified
            if not) passed to the mlflow.set_tags to add new custom tags for the experiment.


        engine: Optional[str] = None
            The execution engine to use for the model, e.g. for K-Means Clustering ("kmeans"), users can
            switch between "sklearn" and "sklearnex" by specifying
            `engine="sklearnex"`.


        verbose: bool, default = True
            Score grid is not printed when verbose is set to False.


        return_train_score: bool, default = False
            If False, returns the CV Validation scores only.
            If True, returns the CV training scores along with the CV validation scores.
            This is useful when the user wants to do bias-variance tradeoff. A high CV
            training score with a low corresponding CV validation score indicates overfitting.


        **kwargs:
            Additional keyword arguments to pass to the estimator.


        Returns:
            Trained Model


        Warnings
        --------
        - AUC for estimators that does not support 'predict_proba' is shown as 0.0000.

        - Models are not logged on the ``MLFlow`` server when ``cross_validation`` param
        is set to False.

        """

        if engine is not None:
            # Save current engines, then set to user specified options
            initial_default_model_engines = self.exp_model_engines.copy()
            self._set_engine(estimator=estimator, engine=engine, severity="error")

        try:
            return_values = super().create_model(
                estimator=estimator,
                fold=fold,
                round=round,
                cross_validation=cross_validation,
                fit_kwargs=fit_kwargs,
                groups=groups,
                verbose=verbose,
                experiment_custom_tags=experiment_custom_tags,
                return_train_score=return_train_score,
                **kwargs,
            )
        finally:
            if engine is not None:
                # Reset the models back to the default engines
                self._set_exp_model_engines(
                    container_default_engines=get_container_default_engines(),
                    engines=initial_default_model_engines,
                )

        return return_values

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


        id: str
            Unique id for the metric.


        name: str
            Display name of the metric.


        score_func: type
            Score function (or loss function) with signature ``score_func(y, y_pred, **kwargs)``.


        target: str, default = 'pred'
            The target of the score function.

            - 'pred' for the prediction table
            - 'pred_proba' for pred_proba
            - 'threshold' for decision_function or predict_proba


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
