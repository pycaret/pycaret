import datetime
import gc
import logging
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np  # type: ignore
import pandas as pd
from joblib.memory import Memory
from sklearn.base import clone  # type: ignore

from pycaret.containers.metrics.clustering import get_all_metric_containers
from pycaret.containers.models.clustering import (
    ALL_ALLOWED_ENGINES,
    get_container_default_engines,
)
from pycaret.internal.display import CommonDisplay
from pycaret.internal.logging import get_logger, redirect_output
from pycaret.internal.pipeline import Pipeline as InternalPipeline
from pycaret.internal.pipeline import estimator_pipeline, get_pipeline_fit_kwargs
from pycaret.internal.pycaret_experiment.unsupervised_experiment import _UnsupervisedExperiment
from pycaret.internal.validation import is_sklearn_pipeline
from pycaret.loggers.base_logger import BaseLogger
from pycaret.utils.constants import DATAFRAME_LIKE, SEQUENCE_LIKE
from pycaret.utils.generic import MLUsecase, highlight_setup, calculate_ts_unsupervised_metrics

LOGGER = get_logger()


class _TSUnsupervisedExperiment(_UnsupervisedExperiment):
    def __init__(self) -> None:
        super().__init__()
        self._variable_keys = self._variable_keys.union({"X"})
        return

    def _calculate_metrics(self, X, labels, ground_truth=None, ml_usecase=None) -> dict:
        """
        Calculate all metrics in _all_metrics.
        """
        from pycaret.utils.generic import calculate_unsupervised_metrics

        if ml_usecase is None:
            ml_usecase = self._ml_usecase

        try:
            return calculate_ts_unsupervised_metrics(
                metrics=self._all_metrics, X=X, labels=labels, ground_truth=ground_truth
            )
        except Exception:
            if ml_usecase == MLUsecase.TIME_SERIES_CLUSTERING:
                metrics = get_all_metric_containers(self.variables, True)
            return calculate_ts_unsupervised_metrics(
                metrics=metrics,  # type: ignore
                X=X,
                labels=labels,
                ground_truth=ground_truth,
            )

    def _is_ts_unsupervised(self) -> bool:
        return True

    def _set_up_logging(
        self,
        runtime,
        log_data,
        log_profile,
        experiment_custom_tags=None,
    ):
        if self.logging_param:
            self.logging_param.log_experiment(
                self,
                log_profile,
                log_data,
                experiment_custom_tags,
                runtime,
            )

    ''' Revamped setup method for time series clustering , for now just calls super class method'''
    def setup(
        self,
        data: Optional[DATAFRAME_LIKE] = None,
        data_func: Optional[Callable[[], Union[pd.Series, pd.DataFrame]]] = None,
        index: Union[bool, int, str, SEQUENCE_LIKE] = True,
        ordinal_features: Optional[Dict[str, list]] = None,
        numeric_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        date_features: Optional[List[str]] = None,
        text_features: Optional[List[str]] = None,
        ignore_features: Optional[List[str]] = None,
        keep_features: Optional[List[str]] = None,
        preprocess: bool = True,
        create_date_columns: List[str] = ["day", "month", "year"],
        imputation_type: Optional[str] = "simple",
        numeric_imputation: str = "mean",
        categorical_imputation: str = "mode",
        text_features_method: str = "tf-idf",
        max_encoding_ohe: int = -1,
        encoding_method: Optional[Any] = None,
        rare_to_value: Optional[float] = None,
        rare_value: str = "rare",
        polynomial_features: bool = False,
        polynomial_degree: int = 2,
        low_variance_threshold: Optional[float] = None,
        group_features: Optional[dict] = None,
        drop_groups: bool = False,
        remove_multicollinearity: bool = False,
        multicollinearity_threshold: float = 0.9,
        bin_numeric_features: Optional[List[str]] = None,
        remove_outliers: bool = False,
        outliers_method: str = "iforest",
        outliers_threshold: float = 0.05,
        transformation: bool = False,
        transformation_method: str = "yeo-johnson",
        normalize: bool = False,
        normalize_method: str = "zscore",
        pca: bool = False,
        pca_method: str = "linear",
        pca_components: Optional[Union[int, float, str]] = None,
        custom_pipeline: Optional[Any] = None,
        custom_pipeline_position: int = -1,
        n_jobs: Optional[int] = -1,
        use_gpu: bool = False,
        html: bool = True,
        session_id: Optional[int] = None,
        system_log: Union[bool, str, logging.Logger] = True,
        log_experiment: Union[
            bool, str, BaseLogger, List[Union[str, BaseLogger]]
        ] = False,
        experiment_name: Optional[str] = None,
        experiment_custom_tags: Optional[Dict[str, Any]] = None,
        log_plots: Union[bool, list] = False,
        log_profile: bool = False,
        log_data: bool = False,
        verbose: bool = True,
        memory: Union[bool, str, Memory] = True,
        profile: bool = False,
        profile_kwargs: Optional[Dict[str, Any]] = None,
        engines: Optional[Dict[str, str]] = None,
    ):
        """

        This function initializes the training environment and creates the transformation
        pipeline. Setup function must be called before executing any other function. It
        takes one mandatory parameter: ``data``. All the other parameters are optional.  The
        example below shows the sktime timeseries dataset called arrow_head


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> arrow_head = get_data('arrow_head')
        >>> from pycaret.clustering import *
        >>> exp_name = setup(data = arrow_head)


        data: dataframe-like
            Data set with shape (n_samples, n_features), where n_samples is the
            number of samples and n_features is the number of features. If data
            is not a pandas dataframe, it's converted to one using default column
            names.


        data_func: Callable[[], DATAFRAME_LIKE] = None
            The function that generate ``data`` (the dataframe-like input). This
            is useful when the dataset is large, and you need parallel operations
            such as ``compare_models``. It can avoid broadcasting large dataset
            from driver to workers. Notice one and only one of ``data`` and
            ``data_func`` must be set.

        index: bool, int, str or sequence, default = True
            Handle indices in the `data` dataframe.
                - If False: Reset to RangeIndex.
                - If True: Keep the provided index.
                - If int: Position of the column to use as index.
                - If str: Name of the column to use as index.
                - If sequence: Array with shape=(n_samples,) to use as index.


        ordinal_features: dict, default = None
            Categorical features to be encoded ordinally. For example, a categorical
            feature with 'low', 'medium', 'high' values where low < medium < high can
            be passed as ordinal_features = {'column_name' : ['low', 'medium', 'high']}.


        numeric_features: list of str, default = None
            If the inferred data types are not correct, the numeric_features param can
            be used to define the data types. It takes a list of strings with column
            names that are numeric.


        categorical_features: list of str, default = None
            If the inferred data types are not correct, the categorical_features param
            can be used to define the data types. It takes a list of strings with column
            names that are categorical.


        date_features: list of str, default = None
            If the inferred data types are not correct, the date_features param can be
            used to overwrite the data types. It takes a list of strings with column
            names that are DateTime.


        text_features: list of str, default = None
            Column names that contain a text corpus. If None, no text features are
            selected.


        ignore_features: list of str, default = None
            ignore_features param can be used to ignore features during preprocessing
            and model training. It takes a list of strings with column names that are
            to be ignored.


        keep_features: list of str, default = None
            keep_features param can be used to always keep specific features during
            preprocessing, i.e. these features are never dropped by any kind of
            feature selection. It takes a list of strings with column names that are
            to be kept.


        preprocess: bool, default = True
            When set to False, no transformations are applied except for train_test_split
            and custom transformations passed in ``custom_pipeline`` param. Data must be
            ready for modeling (no missing values, no dates, categorical data encoding),
            when preprocess is set to False.


        create_date_columns: list of str, default = ["day", "month", "year"]
            Columns to create from the date features. Note that created features
            with zero variance (e.g. the feature hour in a column that only contains
            dates) are ignored. Allowed values are datetime attributes from
            `pandas.Series.dt`. The datetime format of the feature is inferred
            automatically from the first non NaN value.


        imputation_type: str or None, default = 'simple'
            The type of imputation to use. Unsupervised learning only supports
            'imputation_type=simple'. If None, no imputation of missing values
            is performed.


        numeric_imputation: str, default = 'mean'
            Missing values in numeric features are imputed with 'mean' value of the feature
            in the training dataset. The other available option is 'median' or 'zero'.


        categorical_imputation: str, default = 'constant'
            Missing values in categorical features are imputed with a constant 'not_available'
            value. The other available option is 'mode'.


        text_features_method: str, default = "tf-idf"
            Method with which to embed the text features in the dataset. Choose
            between "bow" (Bag of Words - CountVectorizer) or "tf-idf" (TfidfVectorizer).
            Be aware that the sparse matrix output of the transformer is converted
            internally to its full array. This can cause memory issues for large
            text embeddings.


        max_encoding_ohe: int, default = -1
            Categorical columns with `max_encoding_ohe` or less unique values are
            encoded using OneHotEncoding. If more, the `encoding_method` estimator
            is used. Note that columns with exactly two classes are always encoded
            ordinally. Set to below 0 to always use OneHotEncoding.


        encoding_method: category-encoders estimator, default = None
            A `category-encoders` estimator to encode the categorical columns
            with more than `max_encoding_ohe` unique values. If None,
            `category_encoders.basen.BaseN` is used.


        rare_to_value: float or None, default=None
            Minimum fraction of category occurrences in a categorical column.
            If a category is less frequent than `rare_to_value * len(X)`, it is
            replaced with the string in `rare_value`. Use this parameter to group
            rare categories before encoding the column. If None, ignores this step.


        rare_value: str, default="rare"
            Value with which to replace rare categories. Ignored when
            ``rare_to_value`` is None.


        polynomial_features: bool, default = False
            When set to True, new features are derived using existing numeric features.


        polynomial_degree: int, default = 2
            Degree of polynomial features. For example, if an input sample is two dimensional
            and of the form [a, b], the polynomial features with degree = 2 are:
            [1, a, b, a^2, ab, b^2]. Ignored when ``polynomial_features`` is not True.


        low_variance_threshold: float or None, default = None
            Remove features with a training-set variance lower than the provided
            threshold. If 0, keep all features with non-zero variance, i.e. remove
            the features that have the same value in all samples. If None, skip
            this transformation step.


        group_features: dict or None, default = None
            When the dataset contains features with related characteristics,
            add new fetaures with the following statistical properties of that
            group: min, max, mean, std, median and mode. The parameter takes a
            dict with the group name as key and a list of feature names
            belonging to that group as value.


        drop_groups: bool, default=False
            Whether to drop the original features in the group. Ignored when
            ``group_features`` is None.


        remove_multicollinearity: bool, default = False
            When set to True, features with the inter-correlations higher than
            the defined threshold are removed. For each group, it removes all
            except the first feature.


        multicollinearity_threshold: float, default = 0.9
            Minimum absolute Pearson correlation to identify correlated
            features. The default value removes equal columns. Ignored when
            ``remove_multicollinearity`` is not True.


        bin_numeric_features: list of str, default = None
            To convert numeric features into categorical, bin_numeric_features parameter can
            be used. It takes a list of strings with column names to be discretized. It does
            so by using 'sturges' rule to determine the number of clusters and then apply
            KMeans algorithm. Original values of the feature are then replaced by the
            cluster label.


        remove_outliers: bool, default = False
            When set to True, outliers from the training data are removed using an
            Isolation Forest.


        outliers_method: str, default = "iforest"
            Method with which to remove outliers. Possible values are:
                - 'iforest': Uses sklearn's IsolationForest.
                - 'ee': Uses sklearn's EllipticEnvelope.
                - 'lof': Uses sklearn's LocalOutlierFactor.


        outliers_threshold: float, default = 0.05
            The percentage outliers to be removed from the dataset. Ignored
            when ``remove_outliers=False``.


        transformation: bool, default = False
            When set to True, it applies the power transform to make data more Gaussian-like.
            Type of transformation is defined by the ``transformation_method`` parameter.


        transformation_method: str, default = 'yeo-johnson'
            Defines the method for transformation. By default, the transformation method is
            set to 'yeo-johnson'. The other available option for transformation is 'quantile'.
            Ignored when ``transformation`` is not True.


        normalize: bool, default = False
            When set to True, it transforms the features by scaling them to a given
            range. Type of scaling is defined by the ``normalize_method`` parameter.


        normalize_method: str, default = 'zscore'
            Defines the method for scaling. By default, normalize method is set to 'zscore'
            The standard zscore is calculated as z = (x - u) / s. Ignored when ``normalize``
            is not True. The other options are:

            - minmax: scales and translates each feature individually such that it is in
            the range of 0 - 1.
            - maxabs: scales and translates each feature individually such that the
            maximal absolute value of each feature will be 1.0. It does not
            shift/center the data, and thus does not destroy any sparsity.
            - robust: scales and translates each feature according to the Interquartile
            range. When the dataset contains outliers, robust scaler often gives
            better results.


        pca: bool, default = False
            When set to True, dimensionality reduction is applied to project the data into
            a lower dimensional space using the method defined in ``pca_method`` parameter.


        pca_method: str, default = 'linear'
            Method with which to apply PCA. Possible values are:
                - 'linear': Uses Singular Value  Decomposition.
                - 'kernel': Dimensionality reduction through the use of RBF kernel.
                - 'incremental': Similar to 'linear', but more efficient for large datasets.


        pca_components: int, float, str or None, default = None
            Number of components to keep. This parameter is ignored when `pca=False`.
                - If None: All components are kept.
                - If int: Absolute number of components.
                - If float: Such an amount that the variance that needs to be explained
                            is greater than the percentage specified by `n_components`.
                            Value should lie between 0 and 1 (ony for pca_method='linear').
                - If "mle": Minka’s MLE is used to guess the dimension (ony for pca_method='linear').


        custom_pipeline: list of (str, transformer), dict or Pipeline, default = None
            Addidiotnal custom transformers. If passed, they are applied to the
            pipeline last, after all the build-in transformers.


        custom_pipeline_position: int, default = -1
            Position of the custom pipeline in the overal preprocessing pipeline.
            The default value adds the custom pipeline last.


        n_jobs: int, default = -1
            The number of jobs to run in parallel (for functions that supports parallel
            processing) -1 means using all processors. To run all functions on single
            processor set n_jobs to None.


        use_gpu: bool or str, default = False
            When set to True, it will use GPU for training with algorithms that support it,
            and fall back to CPU if they are unavailable. When set to 'force', it will only
            use GPU-enabled algorithms and raise exceptions when they are unavailable. When
            False, all algorithms are trained using CPU only.

            GPU enabled algorithms:

            - None at this moment.


        html: bool, default = True
            When set to False, prevents runtime display of monitor. This must be set to False
            when the environment does not support IPython. For example, command line terminal,
            Databricks Notebook, Spyder and other similar IDEs.


        session_id: int, default = None
            Controls the randomness of experiment. It is equivalent to 'random_state' in
            scikit-learn. When None, a pseudo random number is generated. This can be used
            for later reproducibility of the entire experiment.


        system_log: bool or str or logging.Logger, default = True
            Whether to save the system logging file (as logs.log). If the input
            is a string, use that as the path to the logging file. If the input
            already is a logger object, use that one instead.


        log_experiment: bool, default = False
            A (list of) PyCaret ``BaseLogger`` or str (one of 'mlflow', 'wandb', 'comet_ml')
            corresponding to a logger to determine which experiment loggers to use.
            Setting to True will use just MLFlow.
            If ``wandb`` (Weights & Biases) or ``comet_ml``is installed, will also log there.


        experiment_name: str, default = None
            Name of the experiment for logging. Ignored when ``log_experiment`` is False.


        experiment_custom_tags: dict, default = None
            Dictionary of tag_name: String -> value: (String, but will be string-ified
            if not) passed to the mlflow.set_tags to add new custom tags for the experiment.


        log_plots: bool or list, default = False
            When set to True, certain plots are logged automatically in the ``MLFlow`` server.
            To change the type of plots to be logged, pass a list containing plot IDs. Refer
            to documentation of ``plot_model``. Ignored when ``log_experiment`` is False.


        log_profile: bool, default = False
            When set to True, data profile is logged on the ``MLflow`` server as a html file.
            Ignored when ``log_experiment`` is False.


        log_data: bool, default = False
            When set to True, dataset is logged on the ``MLflow`` server as a csv file.
            Ignored when ``log_experiment`` is False.


        verbose: bool, default = True
            When set to False, Information grid is not printed.


        memory: str, bool or Memory, default=True
            Used to cache the fitted transformers of the pipeline.
                If False: No caching is performed.
                If True: A default temp directory is used.
                If str: Path to the caching directory.


        profile: bool, default = False
            When set to True, an interactive EDA report is displayed.


        profile_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the ProfileReport method used
            to create the EDA report. Ignored if ``profile`` is False.


        Returns:
            Global variables that can be changed using the ``set_config`` function.

        """

        return super().setup(
            data,
            data_func,
            index,
            ordinal_features,
            numeric_features,
            categorical_features,
            date_features,
            text_features,
            ignore_features,
            keep_features,
            preprocess,
            create_date_columns,
            imputation_type,
            numeric_imputation,
            categorical_imputation,
            text_features_method,
            max_encoding_ohe,
            encoding_method,
            rare_to_value,
            rare_value,
            polynomial_features,
            polynomial_degree,
            low_variance_threshold,
            group_features,
            drop_groups,
            remove_multicollinearity,
            multicollinearity_threshold,
            bin_numeric_features,
            remove_outliers,
            outliers_method,
            outliers_threshold,
            transformation,
            transformation_method,
            normalize,
            normalize_method,
            pca,
            pca_method,
            pca_components,
            custom_pipeline,
            custom_pipeline_position,
            n_jobs,
            use_gpu,
            html,
            session_id,
            system_log,
            log_experiment,
            experiment_name,
            experiment_custom_tags,
            log_plots,
            log_profile,
            log_data,
            verbose,
            memory,
            profile,
            profile_kwargs,
            engines
        )

    def assign_model(
        self,
        model,
        transformation: bool = False,
        score: bool = True,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        This function assigns cluster labels to the dataset for a given model.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> jewellery = get_data('arrow_head')
        >>> from pycaret.clustering import *
        >>> exp_name = setup(data = jewellery)
        >>> timeseries_kmeans = create_model('timeseries_kmeans')
        >>> kmeans_df = assign_model(timeseries_kmeans)



        model: scikit-learn compatible object
            Trained model object


        transformation: bool, default = False
            Whether to apply cluster labels on the transformed dataset.


        verbose: bool, default = True
            Status update is not printed when verbose is set to False.


        Returns:
            pandas.DataFrame

        """

        function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

        self.logger.info("Initializing assign_model()")
        self.logger.info(f"assign_model({function_params_str})")

        self.logger.info("Checking exceptions")

        # checking transformation parameter
        if type(transformation) is not bool:
            raise TypeError(
                "Transformation parameter can only take argument as True or False."
            )

        # checking verbose parameter
        if type(verbose) is not bool:
            raise TypeError(
                "Verbose parameter can only take argument as True or False."
            )

        """
        error handling ends here
        """

        if is_sklearn_pipeline(model):
            model = model.steps[-1][1]

        self.logger.info("Determining Trained Model")

        name = self._get_model_name(model)

        self.logger.info(f"Trained Model : {name}")

        self.logger.info("Copying data")
        # copy data_
        if transformation:
            data = self.X_transformed.copy()
            self.logger.info(
                "Transformation parameter set to True. Assigned time series clusters are attached on transformed dataset."
            )
        else:
            data = self.X.copy()

        # calculation labels and attaching to dataframe

        if self._ml_usecase == MLUsecase.TIME_SERIES_CLUSTERING:
            labels = [f"Cluster {i}" for i in model.labels_]
            data["Cluster"] = labels
        else:
            data["Anomaly"] = model.labels_
            if score:
                data["Anomaly_Score"] = model.decision_scores_

        self.logger.info(data.shape)
        self.logger.info(
            "assign_model() successfully completed......................................"
        )

        return data

    def predict_model(
        self,
        estimator,
        data: pd.DataFrame,
        ml_usecase: Optional[MLUsecase] = None,
    ) -> pd.DataFrame:
        function_params_str = ", ".join(
            [f"{k}={v}" for k, v in locals().items() if k != "data"]
        )

        self.logger.info("Initializing predict_model()")
        self.logger.info(f"predict_model({function_params_str})")

        if ml_usecase is None:
            ml_usecase = self._ml_usecase

        if data is None:
            # Can be any Pipeline (pycaret, sklearn, imblearn, etc...)
            if estimator.__class__.__name__ == "Pipeline":
                data = self.X
            else:
                data = self.X_transformed
        else:
            if estimator.__class__.__name__ == "Pipeline":
                data = self._prepare_dataset(data)
            else:
                data = self.pipeline.transform(data)

        # Select features to use
        if hasattr(estimator, "feature_names_in_"):
            data = data[list(estimator.feature_names_in_)]

        # exception checking for predict method
        if hasattr(estimator, "predict"):
            pass
        else:
            raise TypeError("Model doesn't have the predict method.")

        output = data.copy()
        pred = estimator.predict(data)
        if ml_usecase == MLUsecase.CLUSTERING:
            output["Cluster"] = [f"Cluster {i}" for i in pred]
        else:
            output["Anomaly"] = pred
            output["Anomaly_Score"] = estimator.decision_function(data)

        return output

    def _create_model(
        self,
        estimator,
        num_clusters: int = 4,
        fraction: float = 0.05,
        ground_truth: Optional[str] = None,
        round: int = 4,
        fit_kwargs: Optional[dict] = None,
        experiment_custom_tags: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
        system: bool = True,
        add_to_model_list: bool = True,
        raise_num_clusters: bool = False,
        X_data: Optional[pd.DataFrame] = None,  # added in pycaret==2.2.0
        display: Optional[CommonDisplay] = None,  # added in pycaret==2.2.0
        **kwargs,
    ) -> Any:
        """
        Internal version of ``create_model`` with private arguments.
        """
        function_params_str = ", ".join(
            [f"{k}={v}" for k, v in locals().items() if k not in ("X_data")]
        )

        self.logger.info("Initializing create_model()")
        self.logger.info(f"create_model({function_params_str})")

        self.logger.info("Checking exceptions")

        # run_time
        runtime_start = time.time()

        available_estimators = set(self._all_models_internal.keys())

        if not fit_kwargs:
            fit_kwargs = {}

        # only raise exception of estimator is of type string.
        if isinstance(estimator, str):
            if estimator not in available_estimators:
                raise ValueError(
                    f"Estimator {estimator} not available. Please see docstring for list of available estimators."
                )
        elif not hasattr(estimator, "fit"):
            raise ValueError(
                f"Estimator {estimator} does not have the required fit() method."
            )

        # checking round parameter
        if type(round) is not int:
            raise TypeError("Round parameter only accepts integer value.")

        # checking verbose parameter
        if type(verbose) is not bool:
            raise TypeError(
                "Verbose parameter can only take argument as True or False."
            )

        # checking system parameter
        if type(system) is not bool:
            raise TypeError("System parameter can only take argument as True or False.")

        # checking fraction type:
        if fraction <= 0 or fraction >= 1:
            raise TypeError(
                "Fraction parameter can only take value as float between 0 to 1."
            )

        # checking num_clusters type:
        if num_clusters <= 1:
            raise TypeError(
                "num_clusters parameter can only take value integer value greater than 1."
            )

        # check ground truth exist in data_
        if ground_truth is not None:
            if ground_truth not in self.dataset.columns:
                raise ValueError(
                    f"ground_truth {ground_truth} doesn't exist in the dataset."
                )

        """

        ERROR HANDLING ENDS HERE

        """

        if not display:
            progress_args = {"max": 3}
            timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
            monitor_rows = [
                ["Initiated", ". . . . . . . . . . . . . . . . . .", timestampStr],
                [
                    "Status",
                    ". . . . . . . . . . . . . . . . . .",
                    "Loading Dependencies",
                ],
                [
                    "Estimator",
                    ". . . . . . . . . . . . . . . . . .",
                    "Compiling Library",
                ],
            ]
            display = CommonDisplay(
                verbose=verbose,
                html_param=self.html_param,
                progress_args=progress_args,
                monitor_rows=monitor_rows,
            )

        np.random.seed(self.seed)

        # Storing X_train and y_train in data_X and data_y parameter
        data_X = self.X if X_data is None else X_data
        transformed_data = (
            self.X_transformed if X_data is None else self.pipeline.transform(X_data)
        )

        """
        MONITOR UPDATE STARTS
        """
        display.update_monitor(1, "Selecting Estimator")
        """
        MONITOR UPDATE ENDS
        """

        self.logger.info("Importing untrained model")

        is_cblof = False

        if isinstance(estimator, str) and estimator in available_estimators:
            is_cblof = estimator == "cluster"
            model_definition = self._all_models_internal[estimator]
            model_args = model_definition.args
            model_args = {**model_args, **kwargs}
            model = model_definition.class_def(**model_args)
            full_name = model_definition.name
        else:
            self.logger.info("Declaring custom model")

            model = clone(estimator)
            model.set_params(**kwargs)

            full_name = self._get_model_name(model)

        display.update_monitor(2, full_name)

        if self._ml_usecase == MLUsecase.CLUSTERING:
            if raise_num_clusters:
                model.set_params(n_clusters=num_clusters)
            else:
                try:
                    model.set_params(n_clusters=num_clusters)
                except Exception:
                    pass
        else:
            model.set_params(contamination=fraction)

        # workaround for an issue with set_params in cuML
        try:
            model = clone(model)
        except Exception:
            self.logger.warning(
                f"create_model() for {model} raised an exception when cloning:"
            )
            self.logger.warning(traceback.format_exc())

        self.logger.info(f"{full_name} Imported successfully")

        display.move_progress()

        """
        MONITOR UPDATE STARTS
        """

        if self._ml_usecase == MLUsecase.TIME_SERIES_CLUSTERING:
            display.update_monitor(1, f"Fitting {num_clusters} TimeSeries Clusters")
        else:
            display.update_monitor(1, f"Fitting {fraction} Fraction")

        """
        MONITOR UPDATE ENDS
        """

        with estimator_pipeline(self.pipeline, model) as pipeline_with_model:
            fit_kwargs = get_pipeline_fit_kwargs(pipeline_with_model, fit_kwargs)

            self.logger.info("Fitting Model")
            model_fit_start = time.time()
            with redirect_output(self.logger):
                if is_cblof and "n_clusters" not in kwargs:
                    try:
                        pipeline_with_model.fit(data_X, **fit_kwargs)
                    except Exception:
                        try:
                            pipeline_with_model.set_params(
                                actual_estimator__n_clusters=12
                            )
                            model_fit_start = time.time()
                            pipeline_with_model.fit(data_X, **fit_kwargs)
                        except Exception as e:
                            raise RuntimeError(
                                "Could not form valid cluster separation. Try a different dataset or model."
                            ) from e
                else:
                    pipeline_with_model.fit(data_X, **fit_kwargs)
            model_fit_end = time.time()

            model_fit_time = np.array(model_fit_end - model_fit_start).round(2)

        display.move_progress()

        if ground_truth is not None:
            self.logger.info(f"ground_truth parameter set to {ground_truth}")

            gt = np.array(self.dataset[ground_truth])
        else:
            gt = None

        if self._ml_usecase == MLUsecase.TIME_SERIES_CLUSTERING:
            with redirect_output(self.logger):
                metrics = self._calculate_metrics(
                    transformed_data, model.labels_, ground_truth=gt
                )
        else:
            metrics = {}

        self.logger.info(str(model))
        self.logger.info(
            "create_models() successfully completed......................................"
        )

        runtime = time.time() - runtime_start

        # mlflow logging
        if self.logging_param and system:
            metrics_log = {k: v for k, v in metrics.items()}

            self._log_model(
                model=model,
                model_results=None,
                score_dict=metrics_log,
                source="create_model",
                runtime=runtime,
                model_fit_time=model_fit_time,
                pipeline=self.pipeline,
                log_plots=self.log_plots_param,
                experiment_custom_tags=experiment_custom_tags,
                display=display,
            )

        display.move_progress()

        self.logger.info("Uploading results into container")

        if metrics:
            model_results = pd.DataFrame(metrics, index=[0])
            model_results = model_results.round(round)
            self._display_container.append(model_results)
        else:
            model_results = None

        if add_to_model_list:
            # storing results in _master_model_container
            self.logger.info("Uploading model into container now")
            self._master_model_container.append(
                {"model": model, "scores": model_results, "cv": None}
            )

        if model_results is not None and system:
            display.display(model_results.style.format(precision=round))
        else:
            display.close()

        self.logger.info(
            f"_master_model_container: {len(self._master_model_container)}"
        )
        self.logger.info(f"_display_container: {len(self._display_container)}")

        self.logger.info(str(model))
        self.logger.info(
            "create_model() successfully completed......................................"
        )
        gc.collect()

        if not system:
            return (model, model_fit_time)

        return model

    def create_model(
        self,
        estimator,
        num_clusters: int = 4,
        fraction: float = 0.05,
        ground_truth: Optional[str] = None,
        round: int = 4,
        fit_kwargs: Optional[dict] = None,
        experiment_custom_tags: Optional[Dict[str, Any]] = None,
        engine: Optional[str] = None,
        verbose: bool = True,
        **kwargs,
    ) -> Any:
        """
        This function trains and evaluates the performance of a given model.
        Metrics evaluated can be accessed using the ``get_metrics`` function.
        Custom metrics can be added or removed using the ``add_metric`` and
        ``remove_metric`` function. All the available models can be accessed
        using the ``models`` function.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> arrow_head = get_data('arrow_head')
        >>> from pycaret.clustering import *
        >>> exp_name = setup(data = arrow_head)
        >>> timeseries_kmeans = create_model('timeseries_kmeans')


        model: str or scikit-learn compatible object
            ID of an model available in the model library or pass an untrained
            model object consistent with scikit-learn API. Models available
            in the model library (ID - Name):

            * 'kmeans' - K-Means Clustering
            * 'ap' - Affinity Propagation
            * 'meanshift' - Mean shift Clustering
            * 'sc' - Spectral Clustering
            * 'hclust' - Agglomerative Clustering
            * 'dbscan' - Density-Based Spatial Clustering
            * 'optics' - OPTICS Clustering
            * 'birch' - Birch Clustering
            * 'kmodes' - K-Modes Clustering


        num_clusters: int, default = 4
            The number of clusters to form.


        ground_truth: str, default = None
            ground_truth to be provided to evaluate metrics that require true labels.
            When None, such metrics are returned as 0.0. All metrics evaluated can
            be accessed using ``get_metrics`` function.


        round: int, default = 4
            Number of decimal places the metrics in the score grid will be rounded to.


        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.


        verbose: bool, default = True
            Status update is not printed when verbose is set to False.


        engine: Optional[str] = None
            The execution engine to use for the model, e.g. for K-Means Clustering ("kmeans"), users can
            switch between "sklearn" and "sklearnex" by specifying
            `engine="sklearnex"`.


        experiment_custom_tags: dict, default = None
            Dictionary of tag_name: String -> value: (String, but will be string-ified
            if not) passed to the mlflow.set_tags to add new custom tags for the experiment.


        **kwargs:
            Additional keyword arguments to pass to the estimator.


        Returns:
            Trained Model


        Warnings
        --------
        - ``num_clusters`` param not required for Affinity Propagation ('ap'),
        Mean shift ('meanshift'), Density-Based Spatial Clustering ('dbscan')
        and OPTICS Clustering ('optics').

        - When fit doesn't converge in Affinity Propagation ('ap') model, all
        datapoints are labelled as -1.

        - Noisy samples are given the label -1, when using Density-Based Spatial
        ('dbscan') or OPTICS Clustering ('optics').

        - OPTICS ('optics') clustering may take longer training times on large
        datasets.


        """

        # TODO improve error message
        assert not any(
            x
            in (
                "system",
                "add_to_model_list",
                "raise_num_clusters",
                "X_data",
                "metrics",
            )
            for x in kwargs
        )
        if engine is not None:
            # Save current engines, then set to user specified options
            initial_default_model_engines = self.exp_model_engines.copy()
            self._set_engine(estimator=estimator, engine=engine, severity="error")

        try:
            return_values = self._create_model(
                estimator=estimator,
                num_clusters=num_clusters,
                fraction=fraction,
                ground_truth=ground_truth,
                round=round,
                fit_kwargs=fit_kwargs,
                experiment_custom_tags=experiment_custom_tags,
                verbose=verbose,
                **kwargs,
            )
        finally:
            if engine is not None:
                # Reset the models back to the default engines
                self._set_exp_model_engines(
                    container_default_engines=get_container_default_engines(),
                    engine=initial_default_model_engines,
                )

        return return_values

    def evaluate_model(
        self,
        estimator,
        fold: Optional[Union[int, Any]] = None,
        fit_kwargs: Optional[dict] = None,
        plot_kwargs: Optional[dict] = None,
        feature_name: Optional[str] = None,
        groups: Optional[Union[str, Any]] = None,
    ):
        """
        This function displays a user interface for analyzing performance of a trained
        model. It calls the ``plot_model`` function internally.

        Example
        --------
        >>> from pycaret.datasets import get_data
        >>> arrow_head = get_data('arrow_head')
        >>> from pycaret.clustering import *
        >>> exp_name = setup(data = arrow_head)
        >>> timeseries_kmeans = create_model('timeseries_kmeans')
        >>> evaluate_model(timeseries_kmeans)


        model: scikit-learn compatible object
            Trained model object


        feature: str, default = None
            Feature to be evaluated when plot = 'distribution'. When ``plot`` type is
            'cluster' or 'tsne' feature column is used as a hoverover tooltip and/or
            label when the ``label`` param is set to True. When the ``plot`` type is
            'cluster' or 'tsne' and feature is None, first column of the dataset is
            used.


        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.


        Returns:
            None


        Warnings
        --------
        -   This function only works in IPython enabled Notebook.

        """

        return super().evaluate_model(
            estimator,
            fold,
            fit_kwargs,
            plot_kwargs,
            feature_name,
            groups,
        )
