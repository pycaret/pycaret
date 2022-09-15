import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np  # type: ignore
import pandas as pd
from joblib.memory import Memory

from pycaret.containers.metrics import get_all_reg_metric_containers
from pycaret.containers.models import get_all_reg_model_containers
from pycaret.containers.models.regression import (
    ALL_ALLOWED_ENGINES,
    get_container_default_engines,
)
from pycaret.internal.display import CommonDisplay
from pycaret.internal.logging import get_logger
from pycaret.internal.parallel.parallel_backend import ParallelBackend

# Own module
from pycaret.internal.pipeline import Pipeline as InternalPipeline
from pycaret.internal.preprocess.preprocessor import Preprocessor
from pycaret.internal.pycaret_experiment.supervised_experiment import (
    _SupervisedExperiment,
)
from pycaret.loggers.base_logger import BaseLogger
from pycaret.utils.constants import DATAFRAME_LIKE, SEQUENCE_LIKE, TARGET_LIKE
from pycaret.utils.generic import MLUsecase, highlight_setup

LOGGER = get_logger()


class RegressionExperiment(_SupervisedExperiment, Preprocessor):
    def __init__(self) -> None:
        super().__init__()
        self._ml_usecase = MLUsecase.REGRESSION
        self.exp_name_log = "reg-default-name"
        self.variable_keys = self.variable_keys.union(
            {
                "transform_target_param",
                "transform_target_method_param",
            }
        )
        self._available_plots = {
            "pipeline": "Pipeline Plot",
            "parameter": "Hyperparameters",
            "residuals": "Residuals",
            "error": "Prediction Error",
            "cooks": "Cooks Distance",
            "rfe": "Feature Selection",
            "learning": "Learning Curve",
            "manifold": "Manifold Learning",
            "vc": "Validation Curve",
            "feature": "Feature Importance",
            "feature_all": "Feature Importance (All)",
            "tree": "Decision Tree",
            "residuals_interactive": "Interactive Residuals",
        }

    def _get_models(self, raise_errors: bool = True) -> Tuple[dict, dict]:
        all_models = {
            k: v
            for k, v in get_all_reg_model_containers(
                self, raise_errors=raise_errors
            ).items()
            if not v.is_special
        }
        all_models_internal = get_all_reg_model_containers(
            self, raise_errors=raise_errors
        )
        return all_models, all_models_internal

    def _get_metrics(self, raise_errors: bool = True) -> dict:
        return get_all_reg_metric_containers(self.variables, raise_errors=raise_errors)

    def _get_default_plots_to_log(self) -> List[str]:
        return ["residuals", "error", "feature"]

    def setup(
        self,
        data: Optional[DATAFRAME_LIKE] = None,
        data_func: Optional[Callable[[], DATAFRAME_LIKE]] = None,
        target: TARGET_LIKE = -1,
        index: Union[bool, int, str, SEQUENCE_LIKE] = False,
        train_size: float = 0.7,
        test_data: Optional[DATAFRAME_LIKE] = None,
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
        iterative_imputation_iters: int = 5,
        numeric_iterative_imputer: Union[str, Any] = "lightgbm",
        categorical_iterative_imputer: Union[str, Any] = "lightgbm",
        text_features_method: str = "tf-idf",
        max_encoding_ohe: int = 5,
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
        feature_selection: bool = False,
        feature_selection_method: str = "classic",
        feature_selection_estimator: Union[str, Any] = "lightgbm",
        n_features_to_select: int = 10,
        transform_target: bool = False,
        transform_target_method: str = "yeo-johnson",
        custom_pipeline: Optional[Any] = None,
        custom_pipeline_position: int = -1,
        data_split_shuffle: bool = True,
        data_split_stratify: Union[bool, List[str]] = False,
        fold_strategy: Union[str, Any] = "kfold",
        fold: int = 10,
        fold_shuffle: bool = False,
        fold_groups: Optional[Union[str, pd.DataFrame]] = None,
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
        engine: Optional[Dict[str, str]] = None,
        verbose: bool = True,
        memory: Union[bool, str, Memory] = True,
        profile: bool = False,
        profile_kwargs: Optional[Dict[str, Any]] = None,
    ):

        """
        This function initializes the training environment and creates the transformation
        pipeline. Setup function must be called before executing any other function. It takes
        two mandatory parameters: ``data`` and ``target``. All the other parameters are
        optional.

        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase')


        data: dataframe-like = None
            Data set with shape (n_samples, n_features), where n_samples is the
            number of samples and n_features is the number of features. If data
            is not a pandas dataframe, it's converted to one using default column
            names.


        data_func: Callable[[], DATAFRAME_LIKE] = None
            The function that generate ``data`` (the dataframe-like input). This
            is useful when the dataset is large, and you need parallel operations
            such as ``compare_models``. It can avoid boradcasting large dataset
            from driver to workers. Notice one and only one of ``data`` and
            ``data_func`` must be set.


        target: int, str or sequence, default = -1
            If int or str, respectivcely index or name of the target column in data.
            The default value selects the last column in the dataset. If sequence,
            it should have shape (n_samples,). The target can be either binary or
            multiclass.


        index: bool, int, str or sequence, default = False
            Handle indices in the `data` dataframe.
                - If False: Reset to RangeIndex.
                - If True: Keep the provided index.
                - If int: Position of the column to use as index.
                - If str: Name of the column to use as index.
                - If sequence: Array with shape=(n_samples,) to use as index.


        train_size: float, default = 0.7
            Proportion of the dataset to be used for training and validation. Should be
            between 0.0 and 1.0.


        test_data: dataframe-like or None, default = None
            If not None, test_data is used as a hold-out set and `train_size` parameter
            is ignored. The columns of data and test_data must match.


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
            The type of imputation to use. Can be either 'simple' or 'iterative'.
            If None, no imputation of missing values is performed.


        numeric_imputation: int, float or str, default = 'mean'
            Imputing strategy for numerical columns. Ignored when ``imputation_type=
            iterative``. Choose from:
                - "drop": Drop rows containing missing values.
                - "mean": Impute with mean of column.
                - "median": Impute with median of column.
                - "mode": Impute with most frequent value.
                - "knn": Impute using a K-Nearest Neighbors approach.
                - int or float: Impute with provided numerical value.


        categorical_imputation: str, default = 'mode'
            Imputing strategy for categorical columns. Ignored when ``imputation_type=
            iterative``. Choose from:
                - "drop": Drop rows containing missing values.
                - "mode": Impute with most frequent value.
                - str: Impute with provided string.


        iterative_imputation_iters: int, default = 5
            Number of iterations. Ignored when ``imputation_type=simple``.


        numeric_iterative_imputer: str or sklearn estimator, default = 'lightgbm'
            Regressor for iterative imputation of missing values in numeric features.
            If None, it uses LGBClassifier. Ignored when ``imputation_type=simple``.


        categorical_iterative_imputer: str or sklearn estimator, default = 'lightgbm'
            Regressor for iterative imputation of missing values in categorical features.
            If None, it uses LGBClassifier. Ignored when ``imputation_type=simple``.


        text_features_method: str, default = "tf-idf"
            Method with which to embed the text features in the dataset. Choose
            between "bow" (Bag of Words - CountVectorizer) or "tf-idf" (TfidfVectorizer).
            Be aware that the sparse matrix output of the transformer is converted
            internally to its full array. This can cause memory issues for large
            text embeddings.


        max_encoding_ohe: int, default = 5
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
            except the feature with the highest correlation to `y`.


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
            Method with which to remove outliers. Ignored when `remove_outliers=False`.
            Possible values are:
                - 'iforest': Uses sklearn's IsolationForest.
                - 'ee': Uses sklearn's EllipticEnvelope.
                - 'lof': Uses sklearn's LocalOutlierFactor.


        outliers_threshold: float, default = 0.05
            The percentage of outliers to be removed from the dataset. Ignored
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


        feature_selection: bool, default = False
            When set to True, a subset of features is selected based on a feature
            importance score determined by ``feature_selection_estimator``.


        feature_selection_method: str, default = 'classic'
            Algorithm for feature selection. Choose from:
                - 'univariate': Uses sklearn's SelectKBest.
                - 'classic': Uses sklearn's SelectFromModel.
                - 'sequential': Uses sklearn's SequentialFeatureSelector.


        feature_selection_estimator: str or sklearn estimator, default = 'lightgbm'
            Classifier used to determine the feature importances. The
            estimator should have a `feature_importances_` or `coef_`
            attribute after fitting. If None, it uses LGBRegressor. This
            parameter is ignored when `feature_selection_method=univariate`.


        n_features_to_select: int, default = 10
            The number of features to select. Note that this parameter doesn't
            take features in ``ignore_features`` or ``keep_features`` into account
            when counting.


        transform_target: bool, default = False
            When set to True, target variable is transformed using the method defined in
            ``transform_target_method`` param. Target transformation is applied separately
            from feature transformations.


        transform_target_method: str, default = 'yeo-johnson'
            Defines the method for transformation. By default, the transformation method is
            set to 'yeo-johnson'. The other available option for transformation is 'quantile'.
            Ignored when ``transform_target`` is not True.

        custom_pipeline: list of (str, transformer), dict or Pipeline, default = None
            Addidiotnal custom transformers. If passed, they are applied to the
            pipeline last, after all the build-in transformers.


        custom_pipeline_position: int, default = -1
            Position of the custom pipeline in the overal preprocessing pipeline.
            The default value adds the custom pipeline last.


        data_split_shuffle: bool, default = True
            When set to False, prevents shuffling of rows during 'train_test_split'.


        data_split_stratify: bool or list, default = False
            Controls stratification during 'train_test_split'. When set to True, will
            stratify by target column. To stratify on any other columns, pass a list of
            column names. Ignored when ``data_split_shuffle`` is False.


        fold_strategy: str or sklearn CV generator object, default = 'kfold'
            Choice of cross validation strategy. Possible values are:

            * 'kfold'
            * 'groupkfold'
            * 'timeseries'
            * a custom CV generator object compatible with scikit-learn.

            For ``groupkfold``, column name must be passed in ``fold_groups`` parameter.
            Example: ``setup(fold_strategy="groupkfold", fold_groups="COLUMN_NAME")``

        fold: int, default = 10
            Number of folds to be used in cross validation. Must be at least 2. This is
            a global setting that can be over-written at function level by using ``fold``
            parameter. Ignored when ``fold_strategy`` is a custom object.


        fold_shuffle: bool, default = False
            Controls the shuffle parameter of CV. Only applicable when ``fold_strategy``
            is 'kfold' or 'stratifiedkfold'. Ignored when ``fold_strategy`` is a custom
            object.


        fold_groups: str or array-like, with shape (n_samples,), default = None
            Optional group labels when 'GroupKFold' is used for the cross validation.
            It takes an array with shape (n_samples, ) where n_samples is the number
            of rows in the training dataset. When string is passed, it is interpreted
            as the column name in the dataset containing group labels.


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

            - Extreme Gradient Boosting, requires no further installation

            - CatBoost Classifier, requires no further installation
            (GPU is only enabled when data > 50,000 rows)

            - Light Gradient Boosting Machine, requires GPU installation
            https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html

            - Linear Regression, Lasso Regression, Ridge Regression, K Neighbors Regressor,
            Random Forest, Support Vector Regression, Elastic Net requires cuML >= 0.15
            https://github.com/rapidsai/cuml


        html: bool, default = True
            When set to False, prevents runtime display of monitor. This must be set to False
            when the environment does not support IPython. For example, command line terminal,
            Databricks Notebook, Spyder and other similar IDEs.


        session_id: int, default = None
            Controls the randomness of experiment. It is equivalent to 'random_state' in
            scikit-learn. When None, a pseudo random number is generated. This can be used
            for later reproducibility of the entire experiment.


        log_experiment: bool, default = False
            A (list of) PyCaret ``BaseLogger`` or str (one of 'mlflow', 'wandb')
            corresponding to a logger to determine which experiment loggers to use.
            Setting to True will use just MLFlow.
            If ``wandb`` (Weights & Biases) is installed, will also log there.


        system_log: bool or str or logging.Logger, default = True
            Whether to save the system logging file (as logs.log). If the input
            is a string, use that as the path to the logging file. If the input
            already is a logger object, use that one instead.


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


        engine: Optional[Dict[str, str]] = None
            The execution engines to use for the models in the form of a dict
            of `model_id: engine` - e.g. for Linear Regression ("lr", users can
            switch between "sklearn" and "sklearnex" by specifying
            `engine={"lr": "sklearnex"}`


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

        self._register_setup_params(dict(locals()))

        if (data is None and data_func is None) or (
            data is not None and data_func is not None
        ):
            raise ValueError("One and only one of data and data_func must be set")

        # No extra code above this line
        # Setup initialization ===================================== >>

        runtime_start = time.time()

        if data_func is not None:
            data = data_func()

        self.all_allowed_engines = ALL_ALLOWED_ENGINES

        # Define parameter attrs
        self.fold_shuffle_param = fold_shuffle
        self.fold_groups_param = fold_groups

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

        # Check transform_target_method
        allowed_transform_target_method = ["quantile", "yeo-johnson"]
        if transform_target_method not in allowed_transform_target_method:
            raise ValueError(
                "Invalid value for the transform_target_method parameter. "
                f"Choose from: {', '.join(allowed_transform_target_method)}."
            )
        self.transform_target_param = transform_target
        self.transform_target_method = transform_target_method

        # Set up data ============================================== >>

        self.data = self._prepare_dataset(data, target)
        self.target_param = self.data.columns[-1]
        self.index = index

        self._prepare_train_test(
            train_size=train_size,
            test_data=test_data,
            data_split_stratify=data_split_stratify,
            data_split_shuffle=data_split_shuffle,
        )
        self._prepare_folds(
            fold_strategy=fold_strategy,
            fold=fold,
            fold_shuffle=fold_shuffle,
            fold_groups=fold_groups,
        )
        self._prepare_column_types(
            ordinal_features=ordinal_features,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            date_features=date_features,
            text_features=text_features,
            ignore_features=ignore_features,
            keep_features=keep_features,
        )

        self._set_exp_model_engines(
            container_default_engines=get_container_default_engines(),
            engine=engine,
        )

        # Preprocessing ============================================ >>

        # Initialize empty pipeline
        self.pipeline = InternalPipeline(
            steps=[("placeholder", None)],
            memory=self.memory,
        )

        if preprocess:
            self.logger.info("Preparing preprocessing pipeline...")

            # Encode the target column
            if self.y.dtype.kind not in "ifu":
                self._encode_target_column()

            # Power transform the target to be more Gaussian-like
            if transform_target:
                self._target_transformation(transform_target_method)

            # Convert date feature to numerical values
            if self._fxs["Date"]:
                self._date_feature_engineering(create_date_columns)

            # Impute missing values
            if imputation_type == "simple":
                self._simple_imputation(numeric_imputation, categorical_imputation)
            elif imputation_type == "iterative":
                self._iterative_imputation(
                    iterative_imputation_iters=iterative_imputation_iters,
                    numeric_iterative_imputer=numeric_iterative_imputer,
                    categorical_iterative_imputer=categorical_iterative_imputer,
                )
            elif imputation_type is not None:
                raise ValueError(
                    "Invalid value for the imputation_type parameter, got "
                    f"{imputation_type}. Possible values are: simple, iterative."
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

            # Select relevant features
            if feature_selection:
                self._feature_selection(
                    feature_selection_method=feature_selection_method,
                    feature_selection_estimator=feature_selection_estimator,
                    n_features_to_select=n_features_to_select,
                )

        # Add custom transformers to the pipeline
        if custom_pipeline:
            self._add_custom_pipeline(custom_pipeline, custom_pipeline_position)

            # Remove placeholder step
        if ("placeholder", None) in self.pipeline.steps and len(self.pipeline) > 1:
            self.pipeline.steps.remove(("placeholder", None))

        self.pipeline.fit(self.X_train, self.y_train)

        self.logger.info("Finished creating preprocessing pipeline.")
        self.logger.info(f"Pipeline: {self.pipeline}")

        # Final display ============================================ >>

        self.logger.info("Creating final display dataframe.")

        container = []
        container.append(["Session id", self.seed])
        container.append(["Target", self.target_param])
        container.append(["Target type", "Regression"])
        container.append(["Data shape", self.dataset_transformed.shape])
        container.append(["Train data shape", self.train_transformed.shape])
        container.append(["Test data shape", self.test_transformed.shape])
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
            elif imputation_type == "iterative":
                if isinstance(numeric_iterative_imputer, str):
                    num_imputer = numeric_iterative_imputer
                else:
                    num_imputer = numeric_iterative_imputer.__class__.__name__

                if isinstance(categorical_iterative_imputer, str):
                    cat_imputer = categorical_iterative_imputer
                else:
                    cat_imputer = categorical_iterative_imputer.__class__.__name__

                container.append(
                    ["Iterative imputation iterations", iterative_imputation_iters]
                )
                container.append(["Numeric iterative imputer", num_imputer])
                container.append(["Categorical iterative imputer", cat_imputer])
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
            if feature_selection:
                container.append(["Feature selection", feature_selection])
                container.append(["Feature selection method", feature_selection_method])
                container.append(
                    ["Feature selection estimator", feature_selection_estimator]
                )
                container.append(["Number of features selected", n_features_to_select])
            if transform_target:
                container.append(["Transform target", transform_target])
                container.append(["Transform target method", transform_target_method])
            if custom_pipeline:
                container.append(["Custom pipeline", "Yes"])
            container.append(["Fold Generator", self.fold_generator.__class__.__name__])
            container.append(["Fold Number", fold])
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

    def compare_models(
        self,
        include: Optional[List[Union[str, Any]]] = None,
        exclude: Optional[List[str]] = None,
        fold: Optional[Union[int, Any]] = None,
        round: int = 4,
        cross_validation: bool = True,
        sort: str = "R2",
        n_select: int = 1,
        budget_time: Optional[float] = None,
        turbo: bool = True,
        errors: str = "ignore",
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        experiment_custom_tags: Optional[Dict[str, Any]] = None,
        engine: Optional[Dict[str, str]] = None,
        verbose: bool = True,
        parallel: Optional[ParallelBackend] = None,
    ):

        """
        This function trains and evaluates performance of all estimators available in the
        model library using cross validation. The output of this function is a score grid
        with average cross validated scores. Metrics evaluated during CV can be accessed
        using the ``get_metrics`` function. Custom metrics can be added or removed using
        ``add_metric`` and ``remove_metric`` function.


        Example
        --------
        >>> from pycaret.datasets import get_data
        >>> boston = get_data('boston')
        >>> from pycaret.regression import *
        >>> exp_name = setup(data = boston,  target = 'medv')
        >>> best_model = compare_models()


        include: list of str or scikit-learn compatible object, default = None
            To train and evaluate select models, list containing model ID or scikit-learn
            compatible object can be passed in include param. To see a list of all models
            available in the model library use the ``models`` function.


        exclude: list of str, default = None
            To omit certain models from training and evaluation, pass a list containing
            model id in the exclude parameter. To see a list of all models available
            in the model library use the ``models`` function.


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


        sort: str, default = 'R2'
            The sort order of the score grid. It also accepts custom metrics that are
            added through the ``add_metric`` function.


        n_select: int, default = 1
            Number of top_n models to return. For example, to select top 3 models use
            n_select = 3.


        budget_time: int or float, default = None
            If not None, will terminate execution of the function after budget_time
            minutes have passed and return results up to that point.


        turbo: bool, default = True
            When set to True, it excludes estimators with longer training times. To
            see which algorithms are excluded use the ``models`` function.


        errors: str, default = 'ignore'
            When set to 'ignore', will skip the model with exceptions and continue.
            If 'raise', will break the function when exceptions are raised.


        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.


        groups: str or array-like, with shape (n_samples,), default = None
            Optional group labels when 'GroupKFold' is used for the cross validation.
            It takes an array with shape (n_samples, ) where n_samples is the number
            of rows in the training dataset. When string is passed, it is interpreted
            as the column name in the dataset containing group labels.


        engine: Optional[Dict[str, str]] = None
            The execution engines to use for the models in the form of a dict
            of `model_id: engine` - e.g. for Linear Regression ("lr", users can
            switch between "sklearn" and "sklearnex" by specifying
            `engine={"lr": "sklearnex"}`


        verbose: bool, default = True
            Score grid is not printed when verbose is set to False.


        parallel: pycaret.internal.parallel.parallel_backend.ParallelBackend, default = None
            A ParallelBackend instance. For example if you have a SparkSession ``session``,
            you can use ``FugueBackend(session)`` to make this function running using
            Spark. For more details, see
            :class:`~pycaret.parallel.fugue_backend.FugueBackend`


        Returns:
            Trained model or list of trained models, depending on the ``n_select`` param.


        Warnings
        --------
        - Changing turbo parameter to False may result in very high training times with
        datasets exceeding 10,000 rows.

        - No models are logged in ``MLFlow`` when ``cross_validation`` parameter is False.

        """

        caller_params = dict(locals())

        if engine is not None:
            # Save current engines, then set to user specified options
            initial_model_engines = self.exp_model_engines.copy()
            for estimator, eng in engine.items():
                self._set_engine(estimator=estimator, engine=eng, severity="error")

        try:
            return_values = super().compare_models(
                include=include,
                exclude=exclude,
                fold=fold,
                round=round,
                cross_validation=cross_validation,
                sort=sort,
                n_select=n_select,
                budget_time=budget_time,
                turbo=turbo,
                errors=errors,
                fit_kwargs=fit_kwargs,
                groups=groups,
                experiment_custom_tags=experiment_custom_tags,
                verbose=verbose,
                parallel=parallel,
                caller_params=caller_params,
            )

        finally:
            if engine is not None:
                # Reset the models back to the default engines
                self._set_exp_model_engines(
                    container_default_engines=get_container_default_engines(),
                    engine=initial_model_engines,
                )

        return return_values

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
    ):

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
        >>> boston = get_data('boston')
        >>> from pycaret.regression import *
        >>> exp_name = setup(data = boston,  target = 'medv')
        >>> lr = create_model('lr')


        estimator: str or scikit-learn compatible object
            ID of an estimator available in model library or pass an untrained
            model object consistent with scikit-learn API. Estimators available
            in the model library (ID - Name):

            * 'lr' - Linear Regression
            * 'lasso' - Lasso Regression
            * 'ridge' - Ridge Regression
            * 'en' - Elastic Net
            * 'lar' - Least Angle Regression
            * 'llar' - Lasso Least Angle Regression
            * 'omp' - Orthogonal Matching Pursuit
            * 'br' - Bayesian Ridge
            * 'ard' - Automatic Relevance Determination
            * 'par' - Passive Aggressive Regressor
            * 'ransac' - Random Sample Consensus
            * 'tr' - TheilSen Regressor
            * 'huber' - Huber Regressor
            * 'kr' - Kernel Ridge
            * 'svm' - Support Vector Regression
            * 'knn' - K Neighbors Regressor
            * 'dt' - Decision Tree Regressor
            * 'rf' - Random Forest Regressor
            * 'et' - Extra Trees Regressor
            * 'ada' - AdaBoost Regressor
            * 'gbr' - Gradient Boosting Regressor
            * 'mlp' - MLP Regressor
            * 'xgboost' - Extreme Gradient Boosting
            * 'lightgbm' - Light Gradient Boosting Machine
            * 'catboost' - CatBoost Regressor


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
            The execution engine to use for the model, e.g. for Linear Regression ("lr"), users can
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
                experiment_custom_tags=experiment_custom_tags,
                verbose=verbose,
                return_train_score=return_train_score,
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

    def tune_model(
        self,
        estimator,
        fold: Optional[Union[int, Any]] = None,
        round: int = 4,
        n_iter: int = 10,
        custom_grid: Optional[Union[Dict[str, list], Any]] = None,
        optimize: str = "R2",
        custom_scorer=None,
        search_library: str = "scikit-learn",
        search_algorithm: Optional[str] = None,
        early_stopping: Any = False,
        early_stopping_max_iters: int = 10,
        choose_better: bool = True,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        return_tuner: bool = False,
        verbose: bool = True,
        tuner_verbose: Union[int, bool] = True,
        return_train_score: bool = False,
        **kwargs,
    ):

        """
        This function tunes the hyperparameters of a given estimator. The output of
        this function is a score grid with CV scores by fold of the best selected
        model based on ``optimize`` parameter. Metrics evaluated during CV can be
        accessed using the ``get_metrics`` function. Custom metrics can be added
        or removed using ``add_metric`` and ``remove_metric`` function.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> boston = get_data('boston')
        >>> from pycaret.regression import *
        >>> exp_name = setup(data = boston,  target = 'medv')
        >>> lr = create_model('lr')
        >>> tuned_lr = tune_model(lr)


        estimator: scikit-learn compatible object
            Trained model object


        fold: int or scikit-learn compatible CV generator, default = None
            Controls cross-validation. If None, the CV generator in the ``fold_strategy``
            parameter of the ``setup`` function is used. When an integer is passed,
            it is interpreted as the 'n_splits' parameter of the CV generator in the
            ``setup`` function.


        round: int, default = 4
            Number of decimal places the metrics in the score grid will be rounded to.


        n_iter: int, default = 10
            Number of iterations in the grid search. Increasing 'n_iter' may improve
            model performance but also increases the training time.


        custom_grid: dictionary, default = None
            To define custom search space for hyperparameters, pass a dictionary with
            parameter name and values to be iterated. Custom grids must be in a format
            supported by the defined ``search_library``.


        optimize: str, default = 'R2'
            Metric name to be evaluated for hyperparameter tuning. It also accepts custom
            metrics that are added through the ``add_metric`` function.


        custom_scorer: object, default = None
            custom scoring strategy can be passed to tune hyperparameters of the model.
            It must be created using ``sklearn.make_scorer``. It is equivalent of adding
            custom metric using the ``add_metric`` function and passing the name of the
            custom metric in the ``optimize`` parameter.
            Will be deprecated in future.


        search_library: str, default = 'scikit-learn'
            The search library used for tuning hyperparameters. Possible values:

            - 'scikit-learn' - default, requires no further installation
                https://github.com/scikit-learn/scikit-learn

            - 'scikit-optimize' - ``pip install scikit-optimize``
                https://scikit-optimize.github.io/stable/

            - 'tune-sklearn' - ``pip install tune-sklearn ray[tune]``
                https://github.com/ray-project/tune-sklearn

            - 'optuna' - ``pip install optuna``
                https://optuna.org/


        search_algorithm: str, default = None
            The search algorithm depends on the ``search_library`` parameter.
            Some search algorithms require additional libraries to be installed.
            If None, will use search library-specific default algorithm.

            - 'scikit-learn' possible values:
                - 'random' : random grid search (default)
                - 'grid' : grid search

            - 'scikit-optimize' possible values:
                - 'bayesian' : Bayesian search (default)

            - 'tune-sklearn' possible values:
                - 'random' : random grid search (default)
                - 'grid' : grid search
                - 'bayesian' : ``pip install scikit-optimize``
                - 'hyperopt' : ``pip install hyperopt``
                - 'optuna' : ``pip install optuna``
                - 'bohb' : ``pip install hpbandster ConfigSpace``

            - 'optuna' possible values:
                - 'random' : randomized search
                - 'tpe' : Tree-structured Parzen Estimator search (default)


        early_stopping: bool or str or object, default = False
            Use early stopping to stop fitting to a hyperparameter configuration
            if it performs poorly. Ignored when ``search_library`` is scikit-learn,
            or if the estimator does not have 'partial_fit' attribute. If False or
            None, early stopping will not be used. Can be either an object accepted
            by the search library or one of the following:

            - 'asha' for Asynchronous Successive Halving Algorithm
            - 'hyperband' for Hyperband
            - 'median' for Median Stopping Rule
            - If False or None, early stopping will not be used.


        early_stopping_max_iters: int, default = 10
            Maximum number of epochs to run for each sampled configuration.
            Ignored if ``early_stopping`` is False or None.


        choose_better: bool, default = True
            When set to True, the returned object is always better performing. The
            metric used for comparison is defined by the ``optimize`` parameter.


        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the tuner.


        groups: str or array-like, with shape (n_samples,), default = None
            Optional group labels when GroupKFold is used for the cross validation.
            It takes an array with shape (n_samples, ) where n_samples is the number
            of rows in training dataset. When string is passed, it is interpreted as
            the column name in the dataset containing group labels.


        return_tuner: bool, default = False
            When set to True, will return a tuple of (model, tuner_object).


        verbose: bool, default = True
            Score grid is not printed when verbose is set to False.


        tuner_verbose: bool or in, default = True
            If True or above 0, will print messages from the tuner. Higher values
            print more messages. Ignored when ``verbose`` param is False.


        return_train_score: bool, default = False
            If False, returns the CV Validation scores only.
            If True, returns the CV training scores along with the CV validation scores.
            This is useful when the user wants to do bias-variance tradeoff. A high CV
            training score with a low corresponding CV validation score indicates overfitting.


        **kwargs:
            Additional keyword arguments to pass to the optimizer.


        Returns:
            Trained Model and Optional Tuner Object when ``return_tuner`` is True.


        Warnings
        --------
        - Using 'grid' as ``search_algorithm`` may result in very long computation.
        Only recommended with smaller search spaces that can be defined in the
        ``custom_grid`` parameter.

        - ``search_library`` 'tune-sklearn' does not support GPU models.

        """

        return super().tune_model(
            estimator=estimator,
            fold=fold,
            round=round,
            n_iter=n_iter,
            custom_grid=custom_grid,
            optimize=optimize,
            custom_scorer=custom_scorer,
            search_library=search_library,
            search_algorithm=search_algorithm,
            early_stopping=early_stopping,
            early_stopping_max_iters=early_stopping_max_iters,
            choose_better=choose_better,
            fit_kwargs=fit_kwargs,
            groups=groups,
            return_tuner=return_tuner,
            verbose=verbose,
            tuner_verbose=tuner_verbose,
            return_train_score=return_train_score,
            **kwargs,
        )

    def ensemble_model(
        self,
        estimator,
        method: str = "Bagging",
        fold: Optional[Union[int, Any]] = None,
        n_estimators: int = 10,
        round: int = 4,
        choose_better: bool = False,
        optimize: str = "R2",
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        verbose: bool = True,
        return_train_score: bool = False,
    ) -> Any:

        """
        This function ensembles a given estimator. The output of this function is
        a score grid with CV scores by fold. Metrics evaluated during CV can be
        accessed using the ``get_metrics`` function. Custom metrics can be added
        or removed using ``add_metric`` and ``remove_metric`` function.


        Example
        --------
        >>> from pycaret.datasets import get_data
        >>> boston = get_data('boston')
        >>> from pycaret.regression import *
        >>> exp_name = setup(data = boston,  target = 'medv')
        >>> dt = create_model('dt')
        >>> bagged_dt = ensemble_model(dt, method = 'Bagging')


        estimator: scikit-learn compatible object
            Trained model object


        method: str, default = 'Bagging'
            Method for ensembling base estimator. It can be 'Bagging' or 'Boosting'.


        fold: int or scikit-learn compatible CV generator, default = None
            Controls cross-validation. If None, the CV generator in the ``fold_strategy``
            parameter of the ``setup`` function is used. When an integer is passed,
            it is interpreted as the 'n_splits' parameter of the CV generator in the
            ``setup`` function.


        n_estimators: int, default = 10
            The number of base estimators in the ensemble. In case of perfect fit, the
            learning procedure is stopped early.


        round: int, default = 4
            Number of decimal places the metrics in the score grid will be rounded to.


        choose_better: bool, default = False
            When set to True, the returned object is always better performing. The
            metric used for comparison is defined by the ``optimize`` parameter.


        optimize: str, default = 'R2'
            Metric to compare for model selection when ``choose_better`` is True.


        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.


        groups: str or array-like, with shape (n_samples,), default = None
            Optional group labels when GroupKFold is used for the cross validation.
            It takes an array with shape (n_samples, ) where n_samples is the number
            of rows in training dataset. When string is passed, it is interpreted as
            the column name in the dataset containing group labels.


        return_train_score: bool, default = False
        If False, returns the CV Validation scores only.
        If True, returns the CV training scores along with the CV validation scores.
        This is useful when the user wants to do bias-variance tradeoff. A high CV
        training score with a low corresponding CV validation score indicates overfitting.


        verbose: bool, default = True
            Score grid is not printed when verbose is set to False.


        Returns:
            Trained Model

        """

        return super().ensemble_model(
            estimator=estimator,
            method=method,
            fold=fold,
            n_estimators=n_estimators,
            round=round,
            choose_better=choose_better,
            optimize=optimize,
            fit_kwargs=fit_kwargs,
            groups=groups,
            verbose=verbose,
            return_train_score=return_train_score,
        )

    def blend_models(
        self,
        estimator_list: list,
        fold: Optional[Union[int, Any]] = None,
        round: int = 4,
        choose_better: bool = False,
        optimize: str = "R2",
        weights: Optional[List[float]] = None,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        verbose: bool = True,
        return_train_score: bool = False,
    ):

        """
        This function trains a Voting Regressor for select models passed in the
        ``estimator_list`` param. The output of this function is a score grid with
        CV scores by fold. Metrics evaluated during CV can be accessed using the
        ``get_metrics`` function. Custom metrics can be added or removed using
        ``add_metric`` and ``remove_metric`` function.


        Example
        --------
        >>> from pycaret.datasets import get_data
        >>> boston = get_data('boston')
        >>> from pycaret.regression import *
        >>> exp_name = setup(data = boston,  target = 'medv')
        >>> top3 = compare_models(n_select = 3)
        >>> blender = blend_models(top3)


        estimator_list: list of scikit-learn compatible objects
            List of trained model objects


        fold: int or scikit-learn compatible CV generator, default = None
            Controls cross-validation. If None, the CV generator in the ``fold_strategy``
            parameter of the ``setup`` function is used. When an integer is passed,
            it is interpreted as the 'n_splits' parameter of the CV generator in the
            ``setup`` function.


        round: int, default = 4
            Number of decimal places the metrics in the score grid will be rounded to.


        choose_better: bool, default = False
            When set to True, the returned object is always better performing. The
            metric used for comparison is defined by the ``optimize`` parameter.


        optimize: str, default = 'R2'
            Metric to compare for model selection when ``choose_better`` is True.


        weights: list, default = None
            Sequence of weights (float or int) to weight the occurrences of predicted class
            labels (hard voting) or class probabilities before averaging (soft voting). Uses
            uniform weights when None.


        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.


        groups: str or array-like, with shape (n_samples,), default = None
            Optional group labels when GroupKFold is used for the cross validation.
            It takes an array with shape (n_samples, ) where n_samples is the number
            of rows in training dataset. When string is passed, it is interpreted as
            the column name in the dataset containing group labels.


        verbose: bool, default = True
            Score grid is not printed when verbose is set to False.


        return_train_score: bool, default = False
            If False, returns the CV Validation scores only.
            If True, returns the CV training scores along with the CV validation scores.
            This is useful when the user wants to do bias-variance tradeoff. A high CV
            training score with a low corresponding CV validation score indicates overfitting.


        Returns:
            Trained Model


        """

        return super().blend_models(
            estimator_list=estimator_list,
            fold=fold,
            round=round,
            choose_better=choose_better,
            optimize=optimize,
            method="auto",
            weights=weights,
            fit_kwargs=fit_kwargs,
            groups=groups,
            verbose=verbose,
            return_train_score=return_train_score,
        )

    def stack_models(
        self,
        estimator_list: list,
        meta_model=None,
        meta_model_fold: Optional[Union[int, Any]] = 5,
        fold: Optional[Union[int, Any]] = None,
        round: int = 4,
        restack: bool = False,
        choose_better: bool = False,
        optimize: str = "R2",
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        verbose: bool = True,
        return_train_score: bool = False,
    ):

        """
        This function trains a meta model over select estimators passed in
        the ``estimator_list`` parameter. The output of this function is a
        score grid with CV scores by fold. Metrics evaluated during CV can
        be accessed using the ``get_metrics`` function. Custom metrics
        can be added or removed using ``add_metric`` and ``remove_metric``
        function.


        Example
        --------
        >>> from pycaret.datasets import get_data
        >>> boston = get_data('boston')
        >>> from pycaret.regression import *
        >>> exp_name = setup(data = boston,  target = 'medv')
        >>> top3 = compare_models(n_select = 3)
        >>> stacker = stack_models(top3)


        estimator_list: list of scikit-learn compatible objects
            List of trained model objects


        meta_model: scikit-learn compatible object, default = None
            When None, Linear Regression is trained as a meta model.


        meta_model_fold: integer or scikit-learn compatible CV generator, default = 5
            Controls internal cross-validation. Can be an integer or a scikit-learn
            CV generator. If set to an integer, will use (Stratifed)KFold CV with
            that many folds. See scikit-learn documentation on Stacking for
            more details.


        fold: int or scikit-learn compatible CV generator, default = None
            Controls cross-validation. If None, the CV generator in the ``fold_strategy``
            parameter of the ``setup`` function is used. When an integer is passed,
            it is interpreted as the 'n_splits' parameter of the CV generator in the
            ``setup`` function.


        round: int, default = 4
            Number of decimal places the metrics in the score grid will be rounded to.


        restack: bool, default = False
            When set to False, only the predictions of estimators will be used as
            training data for the ``meta_model``.


        choose_better: bool, default = False
            When set to True, the returned object is always better performing. The
            metric used for comparison is defined by the ``optimize`` parameter.


        optimize: str, default = 'R2'
            Metric to compare for model selection when ``choose_better`` is True.


        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.


        groups: str or array-like, with shape (n_samples,), default = None
            Optional group labels when GroupKFold is used for the cross validation.
            It takes an array with shape (n_samples, ) where n_samples is the number
            of rows in training dataset. When string is passed, it is interpreted as
            the column name in the dataset containing group labels.


        verbose: bool, default = True
            Score grid is not printed when verbose is set to False.


        return_train_score: bool, default = False
            If False, returns the CV Validation scores only.
            If True, returns the CV training scores along with the CV validation scores.
            This is useful when the user wants to do bias-variance tradeoff. A high CV
            training score with a low corresponding CV validation score indicates overfitting.


        Returns:
            Trained Model

        """

        return super().stack_models(
            estimator_list=estimator_list,
            meta_model=meta_model,
            meta_model_fold=meta_model_fold,
            fold=fold,
            round=round,
            method="auto",
            restack=restack,
            choose_better=choose_better,
            optimize=optimize,
            fit_kwargs=fit_kwargs,
            groups=groups,
            verbose=verbose,
            return_train_score=return_train_score,
        )

    def plot_model(
        self,
        estimator,
        plot: str = "residuals",
        scale: float = 1,
        save: bool = False,
        fold: Optional[Union[int, Any]] = None,
        fit_kwargs: Optional[dict] = None,
        plot_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        use_train_data: bool = False,
        verbose: bool = True,
        display_format: Optional[str] = None,
    ) -> Optional[str]:

        """
        This function analyzes the performance of a trained model on holdout set.
        It may require re-training the model in certain cases.


        Example
        --------
        >>> from pycaret.datasets import get_data
        >>> boston = get_data('boston')
        >>> from pycaret.regression import *
        >>> exp_name = setup(data = boston,  target = 'medv')
        >>> lr = create_model('lr')
        >>> plot_model(lr, plot = 'residual')


        estimator: scikit-learn compatible object
            Trained model object


        plot: str, default = 'residual'
            List of available plots (ID - Name):

            * 'pipeline' - Schematic drawing of the preprocessing pipeline
            * 'residuals_interactive' - Interactive Residual plots
            * 'residuals' - Residuals Plot
            * 'error' - Prediction Error Plot
            * 'cooks' - Cooks Distance Plot
            * 'rfe' - Recursive Feat. Selection
            * 'learning' - Learning Curve
            * 'vc' - Validation Curve
            * 'manifold' - Manifold Learning
            * 'feature' - Feature Importance
            * 'feature_all' - Feature Importance (All)
            * 'parameter' - Model Hyperparameter
            * 'tree' - Decision Tree


        scale: float, default = 1
            The resolution scale of the figure.


        save: bool, default = False
            When set to True, plot is saved in the current working directory.


        fold: int or scikit-learn compatible CV generator, default = None
            Controls cross-validation. If None, the CV generator in the ``fold_strategy``
            parameter of the ``setup`` function is used. When an integer is passed,
            it is interpreted as the 'n_splits' parameter of the CV generator in the
            ``setup`` function.


        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.


        plot_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the visualizer class.
                - pipeline: fontsize -> int


        plot_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the visualizer class.
                - pipeline: fontsize -> int


        groups: str or array-like, with shape (n_samples,), default = None
            Optional group labels when GroupKFold is used for the cross validation.
            It takes an array with shape (n_samples, ) where n_samples is the number
            of rows in training dataset. When string is passed, it is interpreted as
            the column name in the dataset containing group labels.


        use_train_data: bool, default = False
            When set to true, train data will be used for plots, instead
            of test data.


        verbose: bool, default = True
            When set to False, progress bar is not displayed.


        display_format: str, default = None
            To display plots in Streamlit (https://www.streamlit.io/), set this to 'streamlit'.
            Currently, not all plots are supported.


        Returns:
            Path to saved file, if any.

        """

        return super().plot_model(
            estimator=estimator,
            plot=plot,
            scale=scale,
            save=save,
            fold=fold,
            fit_kwargs=fit_kwargs,
            plot_kwargs=plot_kwargs,
            groups=groups,
            verbose=verbose,
            use_train_data=use_train_data,
            display_format=display_format,
        )

    def evaluate_model(
        self,
        estimator,
        fold: Optional[Union[int, Any]] = None,
        fit_kwargs: Optional[dict] = None,
        plot_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        use_train_data: bool = False,
    ):

        """
        This function displays a user interface for analyzing performance of a trained
        model. It calls the ``plot_model`` function internally.

        Example
        --------
        >>> from pycaret.datasets import get_data
        >>> boston = get_data('boston')
        >>> from pycaret.regression import *
        >>> exp_name = setup(data = boston,  target = 'medv')
        >>> lr = create_model('lr')
        >>> evaluate_model(lr)


        estimator: scikit-learn compatible object
            Trained model object


        fold: int or scikit-learn compatible CV generator, default = None
            Controls cross-validation. If None, the CV generator in the ``fold_strategy``
            parameter of the ``setup`` function is used. When an integer is passed,
            it is interpreted as the 'n_splits' parameter of the CV generator in the
            ``setup`` function.


        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.


        plot_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the visualizer class.


        groups: str or array-like, with shape (n_samples,), default = None
            Optional group labels when GroupKFold is used for the cross validation.
            It takes an array with shape (n_samples, ) where n_samples is the number
            of rows in training dataset. When string is passed, it is interpreted as
            the column name in the dataset containing group labels.


        use_train_data: bool, default = False
            When set to true, train data will be used for plots, instead
            of test data.


        Returns:
            None


        Warnings
        --------
        -   This function only works in IPython enabled Notebook.

        """

        return super().evaluate_model(
            estimator=estimator,
            fold=fold,
            fit_kwargs=fit_kwargs,
            plot_kwargs=plot_kwargs,
            groups=groups,
            use_train_data=use_train_data,
        )

    def interpret_model(
        self,
        estimator,
        plot: str = "summary",
        feature: Optional[str] = None,
        observation: Optional[int] = None,
        use_train_data: bool = False,
        X_new_sample: Optional[pd.DataFrame] = None,
        y_new_sample: Optional[pd.DataFrame] = None,  # add for pfi explainer
        save: Union[str, bool] = False,
        **kwargs,
    ):

        """
        This function takes a trained model object and returns an interpretation plot
        based on the test / hold-out set.

        This function is implemented based on the SHAP (SHapley Additive exPlanations),
        which is a unified approach to explain the output of any machine learning model.
        SHAP connects game theory with local explanations.

        For more information: https://shap.readthedocs.io/en/latest/

        For more information on Partial Dependence Plot: https://github.com/SauceCat/PDPbox


        Example
        --------
        >>> from pycaret.datasets import get_data
        >>> boston = get_data('boston')
        >>> from pycaret.regression import *
        >>> exp = setup(data = boston,  target = 'medv')
        >>> xgboost = create_model('xgboost')
        >>> interpret_model(xgboost)


        estimator: scikit-learn compatible object
            Trained model object


        plot : str, default = 'summary'
            Abbreviation of type of plot. The current list of plots supported
            are (Plot - Name):

            * 'summary' - Summary Plot using SHAP
            * 'correlation' - Dependence Plot using SHAP
            * 'reason' - Force Plot using SHAP
            * 'pdp' - Partial Dependence Plot
            * 'msa' - Morris Sensitivity Analysis
            * 'pfi' - Permutation Feature Importance


        feature: str, default = None
            This parameter is only needed when plot = 'correlation' or 'pdp'.
            By default feature is set to None which means the first column of the
            dataset will be used as a variable. A feature parameter must be passed
            to change this.


        observation: integer, default = None
            This parameter only comes into effect when plot is set to 'reason'. If no
            observation number is provided, it will return an analysis of all observations
            with the option to select the feature on x and y axes through drop down
            interactivity. For analysis at the sample level, an observation parameter must
            be passed with the index value of the observation in test / hold-out set.


        use_train_data: bool, default = False
            When set to true, train data will be used for plots, instead
            of test data.


        X_new_sample: pd.DataFrame, default = None
            Row from an out-of-sample dataframe (neither train nor test data) to be plotted.
            The sample must have the same columns as the raw input train data, and it is transformed
            by the preprocessing pipeline automatically before plotting.


        y_new_sample: pd.DataFrame, default = None
            Row from an out-of-sample dataframe (neither train nor test data) to be plotted.
            The sample must have the same columns as the raw input label data, and it is transformed
            by the preprocessing pipeline automatically before plotting.


        save: string or bool, default = False
            When set to True, Plot is saved as a 'png' file in current working directory.
            When a path destination is given, Plot is saved as a 'png' file the given path to the directory of choice.


        **kwargs:
            Additional keyword arguments to pass to the plot.


        Returns:
            None

        """

        return super().interpret_model(
            estimator=estimator,
            plot=plot,
            feature=feature,
            observation=observation,
            use_train_data=use_train_data,
            X_new_sample=X_new_sample,
            y_new_sample=y_new_sample,
            save=save,
            **kwargs,
        )

    def predict_model(
        self,
        estimator,
        data: Optional[pd.DataFrame] = None,
        drift_report: bool = False,
        round: int = 4,
        verbose: bool = True,
    ) -> pd.DataFrame:

        """
        This function predicts ``Label`` using a trained model. When ``data`` is
        None, it predicts label on the holdout set.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> boston = get_data('boston')
        >>> from pycaret.regression import *
        >>> exp_name = setup(data = boston,  target = 'medv')
        >>> lr = create_model('lr')
        >>> pred_holdout = predict_model(lr)
        >>> pred_unseen = predict_model(lr, data = unseen_dataframe)


        estimator: scikit-learn compatible object
            Trained model object


        data : pandas.DataFrame
            Shape (n_samples, n_features). All features used during training
            must be available in the unseen dataset.


        drift_report: bool, default = False
            When set to True, interactive drift report is generated on test set
            with the evidently library.


        round: int, default = 4
            Number of decimal places to round predictions to.


        verbose: bool, default = True
            When set to False, holdout score grid is not printed.


        Returns:
            pandas.DataFrame


        Warnings
        --------
        - The behavior of the ``predict_model`` is changed in version 2.1 without backward
        compatibility. As such, the pipelines trained using the version (<= 2.0), may not
        work for inference with version >= 2.1. You can either retrain your models with a
        newer version or downgrade the version for inference.


        """

        return super().predict_model(
            estimator=estimator,
            data=data,
            probability_threshold=None,
            encoded_labels=False,
            drift_report=drift_report,
            round=round,
            verbose=verbose,
        )

    def finalize_model(
        self,
        estimator,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        model_only: bool = False,
        experiment_custom_tags: Optional[Dict[str, Any]] = None,
    ) -> Any:

        """
        This function trains a given estimator on the entire dataset including the
        holdout set.


        Example
        --------
        >>> from pycaret.datasets import get_data
        >>> boston = get_data('boston')
        >>> from pycaret.regression import *
        >>> exp_name = setup(data = boston,  target = 'medv')
        >>> lr = create_model('lr')
        >>> final_lr = finalize_model(lr)


        estimator: scikit-learn compatible object
            Trained model object


        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.


        groups: str or array-like, with shape (n_samples,), default = None
            Optional group labels when GroupKFold is used for the cross validation.
            It takes an array with shape (n_samples, ) where n_samples is the number
            of rows in training dataset. When string is passed, it is interpreted as
            the column name in the dataset containing group labels.


        model_only : bool, default = False
            Whether to return the complete fitted pipeline or only the fitted model.


        experiment_custom_tags: dict, default = None
            Dictionary of tag_name: String -> value: (String, but will be string-ified if
            not) passed to the mlflow.set_tags to add new custom tags for the experiment.

        Returns:
            Trained pipeline or model object fitted on complete dataset.

        """

        return super().finalize_model(
            estimator=estimator,
            fit_kwargs=fit_kwargs,
            groups=groups,
            model_only=model_only,
            experiment_custom_tags=experiment_custom_tags,
        )

    def deploy_model(
        self,
        model,
        model_name: str,
        authentication: dict,
        platform: str = "aws",
    ):

        """
        This function deploys the transformation pipeline and trained model on cloud.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> boston = get_data('boston')
        >>> from pycaret.regression import *
        >>> exp_name = setup(data = boston,  target = 'medv')
        >>> lr = create_model('lr')
        >>> # sets appropriate credentials for the platform as environment variables
        >>> import os
        >>> os.environ["AWS_ACCESS_KEY_ID"] = str("foo")
        >>> os.environ["AWS_SECRET_ACCESS_KEY"] = str("bar")
        >>> deploy_model(model = lr, model_name = 'lr-for-deployment', platform = 'aws', authentication = {'bucket' : 'S3-bucket-name'})


        Amazon Web Service (AWS) users:
            To deploy a model on AWS S3 ('aws'), the credentials have to be passed. The easiest way is to use environment
            variables in your local environment. Following information from the IAM portal of amazon console account
            are required:

            - AWS Access Key ID
            - AWS Secret Key Access

            More info: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html#environment-variables


        Google Cloud Platform (GCP) users:
            To deploy a model on Google Cloud Platform ('gcp'), project must be created
            using command line or GCP console. Once project is created, you must create
            a service account and download the service account key as a JSON file to set
            environment variables in your local environment.

            More info: https://cloud.google.com/docs/authentication/production


        Microsoft Azure (Azure) users:
            To deploy a model on Microsoft Azure ('azure'), environment variables for connection
            string must be set in your local environment. Go to settings of storage account on
            Azure portal to access the connection string required.

            - AZURE_STORAGE_CONNECTION_STRING (required as environment variable)

            More info: https://docs.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python?toc=%2Fpython%2Fazure%2FTOC.json


        model: scikit-learn compatible object
            Trained model object


        model_name: str
            Name of model.


        authentication: dict
            Dictionary of applicable authentication tokens.

            When platform = 'aws':
            {'bucket' : 'S3-bucket-name', 'path': (optional) folder name under the bucket}

            When platform = 'gcp':
            {'project': 'gcp-project-name', 'bucket' : 'gcp-bucket-name'}

            When platform = 'azure':
            {'container': 'azure-container-name'}


        platform: str, default = 'aws'
            Name of the platform. Currently supported platforms: 'aws', 'gcp' and 'azure'.


        Returns:
            None

        """

        return super().deploy_model(
            model=model,
            model_name=model_name,
            authentication=authentication,
            platform=platform,
        )

    def save_model(
        self,
        model,
        model_name: str,
        model_only: bool = False,
        verbose: bool = True,
        **kwargs,
    ):

        """
        This function saves the transformation pipeline and trained model object
        into the current working directory as a pickle file for later use.

        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> boston = get_data('boston')
        >>> from pycaret.regression import *
        >>> exp_name = setup(data = boston,  target = 'medv')
        >>> lr = create_model('lr')
        >>> save_model(lr, 'saved_lr_model')


        model: scikit-learn compatible object
            Trained model object


        model_name: str
            Name of the model.


        model_only: bool, default = False
            When set to True, only trained model object is saved instead of the
            entire pipeline.


        **kwargs:
            Additional keyword arguments to pass to joblib.dump().


        verbose: bool, default = True
            Success message is not printed when verbose is set to False.


        Returns:
            Tuple of the model object and the filename.

        """

        return super().save_model(
            model=model,
            model_name=model_name,
            model_only=model_only,
            verbose=verbose,
            **kwargs,
        )

    def load_model(
        self,
        model_name: str,
        platform: Optional[str] = None,
        authentication: Optional[Dict[str, str]] = None,
        verbose: bool = True,
    ):

        """
        This function loads a previously saved pipeline.

        Example
        -------
        >>> from pycaret.regression import load_model
        >>> saved_lr = load_model('saved_lr_model')


        model_name: str
            Name of the model.


        platform: str, default = None
            Name of the cloud platform. Currently supported platforms:
            'aws', 'gcp' and 'azure'.


        authentication: dict, default = None
            dictionary of applicable authentication tokens.

            when platform = 'aws':
            {'bucket' : 'Name of Bucket on S3', 'path': (optional) folder name under the bucket}

            when platform = 'gcp':
            {'project': 'gcp-project-name', 'bucket' : 'gcp-bucket-name'}

            when platform = 'azure':
            {'container': 'azure-container-name'}


        verbose: bool, default = True
            Success message is not printed when verbose is set to False.


        Returns:
            Trained Model

        """

        return super().load_model(
            model_name=model_name,
            platform=platform,
            authentication=authentication,
            verbose=verbose,
        )

    def automl(
        self,
        optimize: str = "Accuracy",
        use_holdout: bool = False,
        turbo: bool = True,
        return_train_score: bool = False,
    ) -> Any:

        """
        This function returns the best model out of all trained models in
        current session based on the ``optimize`` parameter. Metrics
        evaluated can be accessed using the ``get_metrics`` function.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> boston = get_data('boston')
        >>> from pycaret.regression import *
        >>> exp_name = setup(data = boston,  target = 'medv')
        >>> top3 = compare_models(n_select = 3)
        >>> tuned_top3 = [tune_model(i) for i in top3]
        >>> blender = blend_models(tuned_top3)
        >>> stacker = stack_models(tuned_top3)
        >>> best_mae_model = automl(optimize = 'MAE')


        optimize: str, default = 'R2'
            Metric to use for model selection. It also accepts custom metrics
            added using the ``add_metric`` function.


        use_holdout: bool, default = False
            When set to True, metrics are evaluated on holdout set instead of CV.


        turbo: bool, default = True
            When set to True and use_holdout is False, only models created with default fold
            parameter will be considered. If set to False, models created with a non-default
            fold parameter will be scored again using default fold settings, so that they can be
            compared.


        return_train_score: bool, default = False
            If False, returns the CV Validation scores only.
            If True, returns the CV training scores along with the CV validation scores.
            This is useful when the user wants to do bias-variance tradeoff. A high CV
            training score with a low corresponding CV validation score indicates overfitting.


        Returns:
            Trained Model


        """

        return super().automl(
            optimize=optimize,
            use_holdout=use_holdout,
            turbo=turbo,
            return_train_score=return_train_score,
        )

    def models(
        self,
        type: Optional[str] = None,
        internal: bool = False,
        raise_errors: bool = True,
    ) -> pd.DataFrame:

        """
        Returns table of models available in the model library.

        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> boston = get_data('boston')
        >>> from pycaret.regression import *
        >>> exp_name = setup(data = boston,  target = 'medv')
        >>> all_models = models()


        type: str, default = None
            - linear : filters and only return linear models
            - tree : filters and only return tree based models
            - ensemble : filters and only return ensemble models


        internal: bool, default = False
            When True, will return extra columns and rows used internally.


        raise_errors: bool, default = True
            When False, will suppress all exceptions, ignoring models
            that couldn't be created.


        Returns:
            pandas.DataFrame

        """

        return super().models(type=type, internal=internal, raise_errors=raise_errors)

    def get_metrics(
        self,
        reset: bool = False,
        include_custom: bool = True,
        raise_errors: bool = True,
    ) -> pd.DataFrame:

        """
        Returns table of available metrics used in the experiment.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> boston = get_data('boston')
        >>> from pycaret.regression import *
        >>> exp_name = setup(data = boston,  target = 'medv')
        >>> all_metrics = get_metrics()


        reset: bool, default = False
            When True, will reset all changes made using the ``add_metric``
            and ``remove_metric`` function.


        include_custom: bool, default = True
            Whether to include user added (custom) metrics or not.


        raise_errors: bool, default = True
            If False, will suppress all exceptions, ignoring models that
            couldn't be created.


        Returns:
            pandas.DataFrame

        """

        return super().get_metrics(
            reset=reset,
            include_custom=include_custom,
            raise_errors=raise_errors,
        )

    def add_metric(
        self,
        id: str,
        name: str,
        score_func: type,
        greater_is_better: bool = True,
        **kwargs,
    ) -> pd.Series:

        """
        Adds a custom metric to be used in the experiment.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> boston = get_data('boston')
        >>> from pycaret.regression import *
        >>> exp_name = setup(data = boston,  target = 'medv')
        >>> from sklearn.metrics import explained_variance_score
        >>> add_metric('evs', 'EVS', explained_variance_score)


        id: str
            Unique id for the metric.


        name: str
            Display name of the metric.


        score_func: type
            Score function (or loss function) with signature ``score_func(y, y_pred, **kwargs)``.


        greater_is_better: bool, default = True
            Whether ``score_func`` is higher the better or not.


        **kwargs:
            Arguments to be passed to score function.


        Returns:
            pandas.Series

        """

        return super().add_metric(
            id=id,
            name=name,
            score_func=score_func,
            target="pred",
            greater_is_better=greater_is_better,
            **kwargs,
        )

    def remove_metric(self, name_or_id: str):

        """
        Removes a metric from experiment.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> boston = get_data('boston')
        >>> from pycaret.regression import *
        >>> exp_name = setup(data = boston,  target = 'mredv')
        >>> remove_metric('MAPE')


        name_or_id: str
            Display name or ID of the metric.


        Returns:
            None

        """

        return super().remove_metric(name_or_id=name_or_id)

    def get_logs(
        self, experiment_name: Optional[str] = None, save: bool = False
    ) -> pd.DataFrame:

        """
        Returns a table of experiment logs. Only works when ``log_experiment``
        is True when initializing the ``setup`` function.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> boston = get_data('boston')
        >>> from pycaret.regression import *
        >>> exp_name = setup(data = boston,  target = 'medv', log_experiment = True)
        >>> best = compare_models()
        >>> exp_logs = get_logs()


        experiment_name: str, default = None
            When None current active run is used.


        save: bool, default = False
            When set to True, csv file is saved in current working directory.


        Returns:
            pandas.DataFrame

        """

        return super().get_logs(experiment_name=experiment_name, save=save)

    def dashboard(
        self,
        estimator,
        display_format: str = "dash",
        dashboard_kwargs: Optional[Dict[str, Any]] = None,
        run_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        This function generates the interactive dashboard for a trained model. The
        dashboard is implemented using ExplainerDashboard (explainerdashboard.readthedocs.io)


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase')
        >>> lr = create_model('lr')
        >>> dashboard(lr)


        estimator: scikit-learn compatible object
            Trained model object


        display_format: str, default = 'dash'
            Render mode for the dashboard. The default is set to ``dash`` which will
            render a dashboard in browser. There are four possible options:

            - 'dash' - displays the dashboard in browser
            - 'inline' - displays the dashboard in the jupyter notebook cell.
            - 'jupyterlab' - displays the dashboard in jupyterlab pane.
            - 'external' - displays the dashboard in a separate tab. (use in Colab)


        dashboard_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the ``ExplainerDashboard`` class.


        run_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the ``run`` method of ``ExplainerDashboard``.


        **kwargs:
            Additional keyword arguments to pass to the ``ClassifierExplainer`` or
            ``RegressionExplainer`` class.


        Returns:
            ExplainerDashboard
        """

        # soft dependencies check
        super().dashboard(
            estimator, display_format, dashboard_kwargs, run_kwargs, **kwargs
        )

        dashboard_kwargs = dashboard_kwargs or {}
        run_kwargs = run_kwargs or {}

        from explainerdashboard import ExplainerDashboard, RegressionExplainer

        # Replaceing chars which dash doesnt accept for column name `.` , `{`, `}`
        X_test_df = self.X_test_transformed.copy()
        X_test_df.columns = [
            col.replace(".", "__").replace("{", "__").replace("}", "__")
            for col in X_test_df.columns
        ]
        explainer = RegressionExplainer(
            estimator, X_test_df, self.y_test_transformed, **kwargs
        )
        return ExplainerDashboard(
            explainer, mode=display_format, **dashboard_kwargs
        ).run(**run_kwargs)
