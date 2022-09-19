import datetime
import gc
import logging
import time
import traceback
from typing import Any, Dict, List, Optional, Union

import numpy as np  # type: ignore
import pandas as pd
import plotly.graph_objects as go  # type: ignore
from joblib.memory import Memory
from sklearn.base import clone  # type: ignore
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from pycaret.containers.metrics import (
    get_all_class_metric_containers,
    get_all_clust_metric_containers,
    get_all_reg_metric_containers,
)
from pycaret.containers.models import (
    get_all_class_model_containers,
    get_all_reg_model_containers,
)
from pycaret.internal.display import CommonDisplay
from pycaret.internal.logging import get_logger, redirect_output
from pycaret.internal.pipeline import Pipeline as InternalPipeline
from pycaret.internal.pipeline import estimator_pipeline, get_pipeline_fit_kwargs
from pycaret.internal.preprocess.preprocessor import Preprocessor
from pycaret.internal.pycaret_experiment.tabular_experiment import _TabularExperiment
from pycaret.internal.validation import is_sklearn_pipeline
from pycaret.loggers.base_logger import BaseLogger
from pycaret.utils.constants import DATAFRAME_LIKE, SEQUENCE_LIKE
from pycaret.utils.generic import MLUsecase, highlight_setup, infer_ml_usecase

LOGGER = get_logger()


class _UnsupervisedExperiment(_TabularExperiment, Preprocessor):
    def __init__(self) -> None:
        super().__init__()
        self.variable_keys = self.variable_keys.union({"X"})
        return

    def _calculate_metrics(self, X, labels, ground_truth=None, ml_usecase=None) -> dict:
        """
        Calculate all metrics in _all_metrics.
        """
        from pycaret.utils.generic import calculate_unsupervised_metrics

        if ml_usecase is None:
            ml_usecase = self._ml_usecase

        try:
            return calculate_unsupervised_metrics(
                metrics=self._all_metrics, X=X, labels=labels, ground_truth=ground_truth
            )
        except Exception:
            if ml_usecase == MLUsecase.CLUSTERING:
                metrics = get_all_clust_metric_containers(self.variables, True)
            return calculate_unsupervised_metrics(
                metrics=metrics,  # type: ignore
                X=X,
                labels=labels,
                ground_truth=ground_truth,
            )

    def _is_unsupervised(self) -> bool:
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

    def setup(
        self,
        data: DATAFRAME_LIKE,
        index: Union[bool, int, str, SEQUENCE_LIKE] = False,
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
        categorical_imputation: str = "constant",
        text_features_method: str = "tf-idf",
        max_encoding_ohe: int = -1,
        encoding_method: Optional[Any] = None,
        rare_to_value: Optional[float] = None,
        rare_value: str = "rare",
        polynomial_features: bool = False,
        polynomial_degree: int = 2,
        low_variance_threshold: Optional[float] = 0,
        group_features: Optional[list] = None,
        group_names: Optional[Union[str, list]] = None,
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
    ):
        """

        This function initializes the training environment and creates the transformation
        pipeline. Setup function must be called before executing any other function. It
        takes one mandatory parameter: ``data``. All the other parameters are optional.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> jewellery = get_data('jewellery')
        >>> from pycaret.clustering import *
        >>> exp_name = setup(data = jewellery)


        data: dataframe-like
            Data set with shape (n_samples, n_features), where n_samples is the
            number of samples and n_features is the number of features. If data
            is not a pandas dataframe, it's converted to one using default column
            names.


        index: bool, int, str or sequence, default = False
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
            `category_encoders.leave_one_out.LeaveOneOutEncoder` is used.


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


        low_variance_threshold: float or None, default = 0
            Remove features with a training-set variance lower than the provided
            threshold. The default is to keep all features with non-zero variance,
            i.e. remove the features that have the same value in all samples. If
            None, skip this transformation step.


        group_features: list, list of lists or None, default = None
            When the dataset contains features with related characteristics,
            replace those fetaures with the following statistical properties
            of that group: min, max, mean, std, median and mode. The parameter
            takes a list of feature names or a list of lists of feature names
            to specify multiple groups.


        group_names: str, list, or None, default = None
            Group names to be used when naming the new features. The length
            should match with the number of groups specified in ``group_features``.
            If None, new features are named using the default form, e.g. group_1,
            group_2, etc... Ignored when ``group_features`` is None.


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
            A (list of) PyCaret ``BaseLogger`` or str (one of 'mlflow', 'wandb')
            corresponding to a logger to determine which experiment loggers to use.
            Setting to True will use just MLFlow.
            If ``wandb`` (Weights & Biases) is installed, will also log there.


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

        # Setup initialization ===================================== >>

        runtime_start = time.time()

        self._initialize_setup(
            n_jobs=n_jobs,
            use_gpu=use_gpu,
            html=html,
            session_id=session_id,
            system_log=system_log,
            log_experiment=log_experiment,
            experiment_name=experiment_name,
            memory=memory,
            verbose=verbose,
        )

        # Prepare experiment specific params ======================= >>

        self.log_plots_param = log_plots
        if self.log_plots_param is True:
            self.log_plots_param = self._get_default_plots_to_log()
        elif isinstance(self.log_plots_param, list):
            for i in self.log_plots_param:
                if i not in self._available_plots:
                    raise ValueError(
                        f"Invalid value for log_plots '{i}'. Possible values "
                        f"are: {', '.join(self._available_plots.keys())}."
                    )

        # Set up data ============================================== >>

        self.index = index
        self.data = self._set_index(self._prepare_dataset(data))

        # Train and Test indices
        self.idx = [self.data.index, None]

        self._prepare_column_types(
            ordinal_features=ordinal_features,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            date_features=date_features,
            text_features=text_features,
            ignore_features=ignore_features,
            keep_features=keep_features,
        )

        # Preprocessing ============================================ >>

        # Initialize empty pipeline
        self.pipeline = InternalPipeline(
            steps=[("placeholder", None)],
            memory=self.memory,
        )

        if preprocess:
            self.logger.info("Preparing preprocessing pipeline...")

            # Convert date feature to numerical values
            if self._fxs["Date"]:
                self._date_feature_engineering(create_date_columns)

            # Impute missing values
            if imputation_type == "simple":
                self._simple_imputation(numeric_imputation, categorical_imputation)
            elif imputation_type is not None:
                raise ValueError(
                    "Invalid value for the imputation_type parameter, got "
                    f"{imputation_type}. Possible values are: simple."
                )

            # Convert text features to meaningful vectors
            if self._fxs["Text"]:
                self._text_embedding(text_features_method)

            # Encode non-numerical features
            if self._fxs["Ordinal"] or self._fxs["Categorical"]:
                self._encoding(
                    max_encoding_ohe=max_encoding_ohe,
                    encoding_method=encoding_method,
                    rare_to_value=rare_to_value,
                    rare_value=rare_value,
                )

            # Create polynomial features from the existing ones
            if polynomial_features:
                self._polynomial_features(polynomial_degree)

            # Drop features with too low variance
            if low_variance_threshold is not None:
                self._low_variance(low_variance_threshold)

            # Get statistical properties of a group of features
            if group_features:
                self._group_features(group_features, group_names)

            # Drop features that are collinear with other features
            if remove_multicollinearity:
                self._remove_multicollinearity(multicollinearity_threshold)

            # Bin numerical features to 5 clusters
            if bin_numeric_features:
                self._bin_numerical_features(bin_numeric_features)

            # Remove outliers from the dataset
            if remove_outliers:
                self._remove_outliers(outliers_method, outliers_threshold)

            # Power transform the data to be more Gaussian-like
            if transformation:
                self._transformation(transformation_method)

            # Scale the features
            if normalize:
                self._normalization(normalize_method)

            # Apply Principal Component Analysis
            if pca:
                self._pca(pca_method, pca_components)

        # Add custom transformers to the pipeline
        if custom_pipeline:
            self._add_custom_pipeline(custom_pipeline, custom_pipeline_position)

        # Remove placeholder step
        if ("placeholder", None) in self.pipeline.steps and len(self.pipeline) > 1:
            self.pipeline.steps.remove(("placeholder", None))

        self.pipeline.fit(self.X)

        self.logger.info("Finished creating preprocessing pipeline.")
        self.logger.info(f"Pipeline: {self.pipeline}")

        # Final display ============================================ >>

        self.logger.info("Creating final display dataframe.")

        container = []
        container.append(["Session id", self.seed])
        container.append(["Original data shape", self.dataset.shape])
        container.append(["Transformed data shape", self.dataset_transformed.shape])
        for fx, cols in self._fxs.items():
            if len(cols) > 0:
                container.append([f"{fx} features", len(cols)])
        if self.data.isna().sum().sum():
            n_nans = 100 * self.data.isna().any(axis=1).sum() / len(self.data)
            container.append(["Rows with missing values", f"{round(n_nans, 1)}%"])
        if preprocess:
            container.append(["Preprocess", preprocess])
            container.append(["Imputation type", imputation_type])
            if imputation_type == "simple":
                container.append(["Numeric imputation", numeric_imputation])
                container.append(["Categorical imputation", categorical_imputation])
            if self._fxs["Text"]:
                container.append(
                    ["Text features embedding method", text_features_method]
                )
            if self._fxs["Categorical"]:
                container.append(["Maximum one-hot encoding", max_encoding_ohe])
                container.append(["Encoding method", encoding_method])
            if polynomial_features:
                container.append(["Polynomial features", polynomial_features])
                container.append(["Polynomial degree", polynomial_degree])
            if low_variance_threshold is not None:
                container.append(["Low variance threshold", low_variance_threshold])
            if remove_multicollinearity:
                container.append(["Remove multicollinearity", remove_multicollinearity])
                container.append(
                    ["Multicollinearity threshold", multicollinearity_threshold]
                )
            if remove_outliers:
                container.append(["Remove outliers", remove_outliers])
                container.append(["Outliers threshold", outliers_threshold])
            if transformation:
                container.append(["Transformation", transformation])
                container.append(["Transformation method", transformation_method])
            if normalize:
                container.append(["Normalize", normalize])
                container.append(["Normalize method", normalize_method])
            if pca:
                container.append(["PCA", pca])
                container.append(["PCA method", pca_method])
                container.append(["PCA components", pca_components])
            if custom_pipeline:
                container.append(["Custom pipeline", "Yes"])
            container.append(["CPU Jobs", self.n_jobs_param])
            container.append(["Use GPU", self.gpu_param])
            container.append(["Log Experiment", self.logging_param])
            container.append(["Experiment Name", self.exp_name_log])
            container.append(["USI", self.USI])

        self.display_container = [
            pd.DataFrame(container, columns=["Description", "Value"])
        ]
        self.logger.info(f"Setup display_container: {self.display_container[0]}")
        display = CommonDisplay(
            verbose=self.verbose,
            html_param=self.html_param,
        )
        if self.verbose:
            pd.set_option("display.max_rows", 100)
            display.display(self.display_container[0].style.apply(highlight_setup))
            pd.reset_option("display.max_rows")  # Reset option

        # Wrap-up ================================================== >>

        # Create a profile report
        self._profile(profile, profile_kwargs)

        # Define models and metrics
        self._all_models, self._all_models_internal = self._get_models()
        self._all_metrics = self._get_metrics()

        runtime = np.array(time.time() - runtime_start).round(2)
        self._set_up_logging(
            runtime,
            log_data,
            log_profile,
            experiment_custom_tags=experiment_custom_tags,
        )

        self._setup_ran = True
        self.logger.info(f"setup() successfully completed in {runtime}s...............")

        return self

    def tune_model(
        self,
        model,
        supervised_target: str,
        supervised_type: Optional[str] = None,
        supervised_estimator: Union[str, Any] = "lr",
        optimize: Optional[str] = None,
        custom_grid: Optional[List[int]] = None,
        fold: Optional[Union[int, Any]] = None,
        groups: Optional[Union[str, Any]] = None,
        ground_truth: Optional[str] = None,
        method: str = "drop",
        fit_kwargs: Optional[dict] = None,
        round: int = 4,
        verbose: bool = True,
        **kwargs,
    ):
        """
        This function tunes the ``num_clusters`` parameter of a given model.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> from pycaret.clustering import *
        >>> exp_name = setup(data = juice)
        >>> tuned_kmeans = tune_model(model = 'kmeans', supervised_target = 'Purchase')


        model: str
            ID of an model available in the model library. Models that can be
            tuned in this function (ID - Model):

            * 'kmeans' - K-Means Clustering
            * 'sc' - Spectral Clustering
            * 'hclust' - Agglomerative Clustering
            * 'birch' - Birch Clustering
            * 'kmodes' - K-Modes Clustering


        supervised_target: str
            Name of the target column containing labels.


        supervised_type: str, default = None
            Type of task. 'classification' or 'regression'. Automatically inferred
            when None.


        supervised_estimator: str, default = None
            Classification (ID - Name):
                * 'lr' - Logistic Regression (Default)
                * 'knn' - K Nearest Neighbour
                * 'nb' - Naive Bayes
                * 'dt' - Decision Tree Classifier
                * 'svm' - SVM - Linear Kernel
                * 'rbfsvm' - SVM - Radial Kernel
                * 'gpc' - Gaussian Process Classifier
                * 'mlp' - Multi Level Perceptron
                * 'ridge' - Ridge Classifier
                * 'rf' - Random Forest Classifier
                * 'qda' - Quadratic Discriminant Analysis
                * 'ada' - Ada Boost Classifier
                * 'gbc' - Gradient Boosting Classifier
                * 'lda' - Linear Discriminant Analysis
                * 'et' - Extra Trees Classifier
                * 'xgboost' - Extreme Gradient Boosting
                * 'lightgbm' - Light Gradient Boosting
                * 'catboost' - CatBoost Classifier

            Regression (ID - Name):
                * 'lr' - Linear Regression (Default)
                * 'lasso' - Lasso Regression
                * 'ridge' - Ridge Regression
                * 'en' - Elastic Net
                * 'lar' - Least Angle Regression
                * 'llar' - Lasso Least Angle Regression
                * 'omp' - Orthogonal Matching Pursuit
                * 'br' - Bayesian Ridge
                * 'ard' - Automatic Relevance Determ.
                * 'par' - Passive Aggressive Regressor
                * 'ransac' - Random Sample Consensus
                * 'tr' - TheilSen Regressor
                * 'huber' - Huber Regressor
                * 'kr' - Kernel Ridge
                * 'svm' - Support Vector Machine
                * 'knn' - K Neighbors Regressor
                * 'dt' - Decision Tree
                * 'rf' - Random Forest
                * 'et' - Extra Trees Regressor
                * 'ada' - AdaBoost Regressor
                * 'gbr' - Gradient Boosting
                * 'mlp' - Multi Level Perceptron
                * 'xgboost' - Extreme Gradient Boosting
                * 'lightgbm' - Light Gradient Boosting
                * 'catboost' - CatBoost Regressor


        optimize: str, default = None
            For Classification tasks:
                Accuracy, AUC, Recall, Precision, F1, Kappa (default = 'Accuracy')

            For Regression tasks:
                MAE, MSE, RMSE, R2, RMSLE, MAPE (default = 'R2')


        custom_grid: list, default = None
            By default, a pre-defined number of clusters is iterated over to
            optimize the supervised objective. To overwrite default iteration,
            pass a list of num_clusters to iterate over in custom_grid param.


        fold: int, default = 10
            Number of folds to be used in Kfold CV. Must be at least 2.


        verbose: bool, default = True
            Status update is not printed when verbose is set to False.


        Returns:
            Trained Model with optimized ``num_clusters`` parameter.


        Warnings
        --------
        - Affinity Propagation, Mean shift, Density-Based Spatial Clustering
        and OPTICS Clustering cannot be used in this function since they donot
        support the ``num_clusters`` param.


        """

        function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

        self.logger.info("Initializing tune_model()")
        self.logger.info(f"tune_model({function_params_str})")

        self.logger.info("Checking exceptions")

        # run_time
        runtime_start = time.time()

        if not fit_kwargs:
            fit_kwargs = {}

        if supervised_target not in self.dataset.columns:
            raise ValueError(
                f"{supervised_target} is not present as a column in the dataset."
            )

        np.random.seed(self.seed)

        cols_to_drop = [x for x in self.X.columns if x.startswith(supervised_target)]
        data_X = self.X.drop(cols_to_drop, axis=1)
        data_y = self.dataset[[supervised_target]]
        if data_y.dtypes[0] not in [int, float, bool]:
            data_y[supervised_target] = LabelEncoder().fit_transform(
                data_y[supervised_target]
            )
        data_y = data_y[supervised_target]

        temp_globals = self.variables
        temp_globals["y_train"] = data_y

        if supervised_type is None:
            supervised_type, _ = infer_ml_usecase(data_y)
            self.logger.info(f"supervised_type inferred as {supervised_type}")

        if supervised_type == "classification":
            metrics = get_all_class_metric_containers(self, raise_errors=True)
            available_estimators = get_all_class_model_containers(
                self, raise_errors=True
            )
            ml_usecase = MLUsecase.CLASSIFICATION
        elif supervised_type == "regression":
            metrics = get_all_reg_metric_containers(self, raise_errors=True)
            available_estimators = get_all_reg_model_containers(self, raise_errors=True)
            ml_usecase = MLUsecase.REGRESSION
        else:
            raise ValueError(
                "supervised_type parameter must be either 'classification' or 'regression'."
            )

        fold = self._get_cv_splitter(fold, ml_usecase)

        if isinstance(supervised_estimator, str):
            if supervised_estimator in available_estimators:
                estimator_definition = available_estimators[supervised_estimator]
                estimator_args = estimator_definition.args
                estimator_args = {**estimator_args}
                supervised_estimator = estimator_definition.class_def(**estimator_args)
            else:
                raise ValueError(
                    f"Unknown supervised_estimator {supervised_estimator}."
                )
        else:
            self.logger.info("Declaring custom model")

            supervised_estimator = clone(supervised_estimator)

        supervised_estimator_name = self._get_model_name(
            supervised_estimator, models=available_estimators
        )

        if optimize is None:
            optimize = "Accuracy" if supervised_type == "classification" else "R2"
        optimize = self._get_metric_by_name_or_id(optimize, metrics=metrics)
        if optimize is None:
            raise ValueError(
                "Optimize method not supported. See docstring for list of available parameters."
            )

        if custom_grid is not None and not isinstance(custom_grid, list):
            raise ValueError("custom_grid parameter must be a list.")

        # checking round parameter
        if type(round) is not int:
            raise TypeError("Round parameter only accepts integer value.")

        # checking verbose parameter
        if type(verbose) is not bool:
            raise TypeError(
                "Verbose parameter can only take argument as True or False."
            )

        if custom_grid is None:
            if self._ml_usecase == MLUsecase.CLUSTERING:
                param_grid = [2, 4, 5, 6, 8, 10, 14, 18, 25, 30, 40]
            else:
                param_grid = [
                    0.01,
                    0.02,
                    0.03,
                    0.04,
                    0.05,
                    0.06,
                    0.07,
                    0.08,
                    0.09,
                    0.10,
                ]
        else:
            param_grid = custom_grid
            try:
                param_grid.remove(0)
            except ValueError:
                pass
        param_grid.sort()

        progress_args = {"max": len(param_grid) * 3 + (len(param_grid) + 1) * 4}
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

        unsupervised_models = {}
        unsupervised_models_results = {}
        unsupervised_grids = {0: data_X}

        self.logger.info("Fitting unsupervised models")

        for k in param_grid:
            if self._ml_usecase == MLUsecase.CLUSTERING:
                try:
                    new_model, _ = self._create_model(
                        model,
                        num_clusters=k,
                        X_data=data_X,
                        display=display,
                        system=False,
                        ground_truth=ground_truth,
                        round=round,
                        fit_kwargs=fit_kwargs,
                        raise_num_clusters=True,
                        **kwargs,
                    )
                except ValueError:
                    raise ValueError(
                        f"Model {model} cannot be used in this function as its number of clusters cannot be set (n_clusters parameter required)."
                    )
            else:
                new_model, _ = self._create_model(
                    model,
                    fraction=k,
                    X_data=data_X,
                    display=display,
                    system=False,
                    ground_truth=ground_truth,
                    round=round,
                    fit_kwargs=fit_kwargs,
                    **kwargs,
                )
            unsupervised_models_results[k] = self.pull(pop=True)
            unsupervised_models[k] = new_model
            unsupervised_grids[k] = (
                self.assign_model(new_model, verbose=False, transformation=True)
                .reset_index(drop=True)
                .drop(cols_to_drop, axis=1)
            )
            if self._ml_usecase == MLUsecase.CLUSTERING:
                unsupervised_grids[k] = pd.get_dummies(
                    unsupervised_grids[k],
                    columns=["Cluster"],
                )
            elif method == "drop":
                unsupervised_grids[k] = unsupervised_grids[k][
                    unsupervised_grids[k]["Anomaly"] == 0
                ].drop(["Anomaly", "Anomaly_Score"], axis=1)

        results = {}

        self.logger.info("Fitting supervised estimator")

        for k, v in unsupervised_grids.items():
            self._create_model(
                supervised_estimator,
                fold=fold,
                display=display,
                system=False,
                X_train_data=v,
                y_train_data=data_y[data_y.index.isin(v.index)],
                metrics=metrics,
                groups=groups,
                round=round,
                refit=False,
            )
            results[k] = self.pull(pop=True).loc["Mean"]
            display.move_progress()

        self.logger.info("Compiling results")

        results = pd.DataFrame(results).T

        greater_is_worse_columns = {
            v.display_name for k, v in metrics.items() if not v.greater_is_better
        }

        best_model_idx = (
            results.drop(0)
            .sort_values(
                by=optimize.display_name, ascending=optimize in greater_is_worse_columns
            )
            .index[0]
        )

        def highlight_max(s):
            to_highlight = s == s.max()
            return ["background-color: yellow" if v else "" for v in to_highlight]

        def highlight_min(s):
            to_highlight = s == s.min()
            return ["background-color: yellow" if v else "" for v in to_highlight]

        # end runtime
        runtime_end = time.time()
        runtime = np.array(runtime_end - runtime_start).round(2)

        if self._ml_usecase == MLUsecase.CLUSTERING:
            best_model, best_model_fit_time = self._create_model(
                unsupervised_models[best_model_idx],
                num_clusters=best_model_idx,
                system=False,
                round=round,
                ground_truth=ground_truth,
                fit_kwargs=fit_kwargs,
                display=display,
                **kwargs,
            )
        else:
            best_model, best_model_fit_time = self._create_model(
                unsupervised_models[best_model_idx],
                fraction=best_model_idx,
                system=False,
                round=round,
                fit_kwargs=fit_kwargs,
                display=display,
                **kwargs,
            )
        best_model_results = self.pull(pop=True)

        if self.logging_param:

            metrics_log = {k: v[0] for k, v in best_model_results.items()}

            self._log_model(
                model=model,
                model_results=None,
                score_dict=metrics_log,
                source="tune_model",
                runtime=runtime,
                model_fit_time=best_model_fit_time,
                pipeline=self.pipeline,
                log_plots=self.log_plots_param,
                display=display,
            )

        self.display_container.append(results)

        results = results.style.apply(
            highlight_max,
            subset=[x for x in results.columns if x not in greater_is_worse_columns],
        ).apply(
            highlight_min,
            subset=[x for x in results.columns if x in greater_is_worse_columns],
        )
        display.display(results.format(precision=round))

        if self.html_param and verbose:
            self.logger.info("Rendering Visual")
            plot_df = results.data.drop(
                [x for x in results.columns if x != optimize.display_name], axis=1
            )

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=plot_df.index,
                    y=plot_df[optimize.display_name],
                    mode="lines+markers",
                    name=optimize.display_name,
                )
            )
            msg = (
                "Number of Clusters"
                if self._ml_usecase == MLUsecase.CLUSTERING
                else "Anomaly Fraction"
            )
            title = f"{supervised_estimator_name} Metrics and {msg} by {self._get_model_name(best_model)}"
            fig.update_layout(
                plot_bgcolor="rgb(245,245,245)",
                title={
                    "text": title,
                    "y": 0.95,
                    "x": 0.45,
                    "xanchor": "center",
                    "yanchor": "top",
                },
                xaxis_title=msg,
                yaxis_title=optimize.display_name,
            )
            fig.show()
            self.logger.info("Visual Rendered Successfully")

        self.logger.info(f"master_model_container: {len(self.master_model_container)}")
        self.logger.info(f"display_container: {len(self.display_container)}")

        self.logger.info(str(best_model))
        self.logger.info(
            "tune_model() successfully completed......................................"
        )

        gc.collect()

        return best_model

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
        >>> jewellery = get_data('jewellery')
        >>> from pycaret.clustering import *
        >>> exp_name = setup(data = jewellery)
        >>> kmeans = create_model('kmeans')
        >>> kmeans_df = assign_model(kmeans)



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
                "Transformation parameter set to True. Assigned clusters are attached on transformed dataset."
            )
        else:
            data = self.X.copy()

        # calculation labels and attaching to dataframe

        if self._ml_usecase == MLUsecase.CLUSTERING:
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

        if isinstance(estimator, Pipeline):
            if not hasattr(estimator, "feature_names_in_"):
                raise ValueError(
                    "If estimator is a Pipeline, it must implement `feature_names_in_`."
                )
            pipeline = estimator
            # Temporarily remove final estimator so it's not used for transform
            final_step = pipeline.steps[-1]
            estimator = final_step[-1]
            pipeline.steps = pipeline.steps[:-1]
        elif not self._setup_ran:
            raise ValueError(
                "If estimator is not a Pipeline, you must run setup() first."
            )
        else:
            pipeline = self.pipeline
            final_step = None

        if ml_usecase is None:
            ml_usecase = self._ml_usecase

        X_columns = pipeline.feature_names_in_
        if data is None:
            data_transformed = self.X_transformed
        else:
            data = self._prepare_dataset(data)[X_columns]
            data_transformed = pipeline.transform(data)
            if final_step:
                pipeline.steps.append(final_step)

        # exception checking for predict param
        if hasattr(estimator, "predict"):
            pass
        else:
            raise TypeError("Model doesn't support predict parameter.")

        pred = estimator.predict(data_transformed)
        if ml_usecase == MLUsecase.CLUSTERING:
            data_transformed["Cluster"] = [f"Cluster {i}" for i in pred]
        else:
            pred_score = estimator.decision_function(data_transformed)
            data_transformed["Anomaly"] = pred
            data_transformed["Anomaly_Score"] = pred_score

        return data_transformed

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

        if self._ml_usecase == MLUsecase.CLUSTERING:
            display.update_monitor(1, f"Fitting {num_clusters} Clusters")
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
                        except Exception:
                            raise RuntimeError(
                                "Could not form valid cluster separation. Try a different dataset or model."
                            )
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

        if self._ml_usecase == MLUsecase.CLUSTERING:
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
            self.display_container.append(model_results)
        else:
            model_results = None

        if add_to_model_list:
            # storing results in master_model_container
            self.logger.info("Uploading model into container now")
            self.master_model_container.append(
                {"model": model, "scores": model_results, "cv": None}
            )

        if model_results is not None and system:
            display.display(model_results.style.format(precision=round))
        else:
            display.close()

        self.logger.info(f"master_model_container: {len(self.master_model_container)}")
        self.logger.info(f"display_container: {len(self.display_container)}")

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
        >>> jewellery = get_data('jewellery')
        >>> from pycaret.clustering import *
        >>> exp_name = setup(data = jewellery)
        >>> kmeans = create_model('kmeans')


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

        return self._create_model(
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

    def evaluate_model(
        self,
        estimator,
        fold: Optional[Union[int, Any]] = None,
        fit_kwargs: Optional[dict] = None,
        plot_kwargs: Optional[dict] = None,
        feature_name: Optional[str] = None,
        groups: Optional[Union[str, Any]] = None,
        use_train_data: bool = False,
    ):
        """
        This function displays a user interface for analyzing performance of a trained
        model. It calls the ``plot_model`` function internally.

        Example
        --------
        >>> from pycaret.datasets import get_data
        >>> jewellery = get_data('jewellery')
        >>> from pycaret.clustering import *
        >>> exp_name = setup(data = jewellery)
        >>> kmeans = create_model('kmeans')
        >>> evaluate_model(kmeans)


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
            use_train_data,
        )
