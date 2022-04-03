import gc
import time
import logging
import datetime
import warnings
import traceback
import numpy as np  # type: ignore
from typing import List, Tuple, Union, Any
from joblib.memory import Memory
import plotly.express as px  # type: ignore
import plotly.graph_objects as go  # type: ignore

# Own modules
from pycaret.internal.pipeline import Pipeline as InternalPipeline
from pycaret.internal.pycaret_experiment.utils import highlight_setup, MLUsecase
from pycaret.internal.pycaret_experiment.supervised_experiment import (
    _SupervisedExperiment,
)
from pycaret.internal.preprocess.preprocessor import Preprocessor
from pycaret.internal.meta_estimators import (
    CustomProbabilityThresholdClassifier,
    get_estimator_from_meta_estimator,
)
from pycaret.internal.utils import color_df, get_label_encoder
import pycaret.internal.patches.sklearn
import pycaret.internal.patches.yellowbrick
from pycaret.internal.distributions import *
from pycaret.internal.validation import *
import pycaret.containers.metrics.classification
import pycaret.containers.models.classification
import pycaret.internal.preprocess
import pycaret.internal.persistence
from pycaret.internal.Display import Display


warnings.filterwarnings("ignore")
LOGGER = get_logger()


class ClassificationExperiment(_SupervisedExperiment, Preprocessor):
    def __init__(self) -> None:
        super().__init__()
        self._ml_usecase = MLUsecase.CLASSIFICATION
        self.exp_name_log = "clf-default-name"
        self.variable_keys = self.variable_keys.union(
            {"fix_imbalance_param", "fix_imbalance_method_param"}
        )
        self._available_plots = {
            "parameter": "Hyperparameters",
            "auc": "AUC",
            "confusion_matrix": "Confusion Matrix",
            "threshold": "Threshold",
            "pr": "Precision Recall",
            "error": "Prediction Error",
            "class_report": "Class Report",
            "rfe": "Feature Selection",
            "learning": "Learning Curve",
            "manifold": "Manifold Learning",
            "calibration": "Calibration Curve",
            "vc": "Validation Curve",
            "dimension": "Dimensions",
            "feature": "Feature Importance",
            "feature_all": "Feature Importance (All)",
            "boundary": "Decision Boundary",
            "lift": "Lift Chart",
            "gain": "Gain Chart",
            "tree": "Decision Tree",
            "ks": "KS Statistic Plot",
        }

    def _get_models(self, raise_errors: bool = True) -> Tuple[dict, dict]:
        all_models = {
            k: v
            for k, v in pycaret.containers.models.classification.get_all_model_containers(
                self, raise_errors=raise_errors
            ).items()
            if not v.is_special
        }
        all_models_internal = (
            pycaret.containers.models.classification.get_all_model_containers(
                self, raise_errors=raise_errors
            )
        )
        return all_models, all_models_internal

    def _get_metrics(self, raise_errors: bool = True) -> dict:
        return pycaret.containers.metrics.classification.get_all_metric_containers(
            self.variables, raise_errors=raise_errors
        )

    def _is_multiclass(self) -> bool:
        """
        Method to check if the problem is multiclass.
        """
        try:
            return self.y.value_counts().count() > 2
        except Exception:
            return False

    def setup(
        self,
        data: pd.DataFrame,
        target: Union[int, str] = -1,
        train_size: float = 0.7,
        test_data: Optional[pd.DataFrame] = None,
        ordinal_features: Optional[Dict[str, list]] = None,
        numeric_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        date_features: Optional[List[str]] = None,
        text_features: Optional[List[str]] = None,
        ignore_features: Optional[List[str]] = None,
        keep_features: Optional[List[str]] = None,
        preprocess: bool = True,
        imputation_type: Optional[str] = "simple",
        numeric_imputation: str = "mean",
        categorical_imputation: str = "constant",
        iterative_imputation_iters: int = 5,
        numeric_iterative_imputer: Union[str, Any] = "lightgbm",
        categorical_iterative_imputer: Union[str, Any] = "lightgbm",
        text_features_method: str = "tf-idf",
        max_encoding_ohe: int = 5,
        encoding_method: Optional[Any] = None,
        polynomial_features: bool = False,
        polynomial_degree: int = 2,
        low_variance_threshold: float = 0,
        remove_multicollinearity: bool = False,
        multicollinearity_threshold: float = 0.9,
        bin_numeric_features: Optional[List[str]] = None,
        remove_outliers: bool = False,
        outliers_method: str = "iforest",
        outliers_threshold: float = 0.05,
        fix_imbalance: bool = False,
        fix_imbalance_method: Optional[Any] = None,
        transformation: bool = False,
        transformation_method: str = "yeo-johnson",
        normalize: bool = False,
        normalize_method: str = "zscore",
        pca: bool = False,
        pca_method: str = "linear",
        pca_components: Union[int, float] = 1.0,
        feature_selection: bool = False,
        feature_selection_method: str = "classic",
        feature_selection_estimator: Union[str, Any] = "lightgbm",
        n_features_to_select: int = 10,
        custom_pipeline: Any = None,
        data_split_shuffle: bool = True,
        data_split_stratify: Union[bool, List[str]] = False,
        fold_strategy: Union[str, Any] = "stratifiedkfold",
        fold: int = 10,
        fold_shuffle: bool = False,
        fold_groups: Optional[Union[str, pd.DataFrame]] = None,
        n_jobs: Optional[int] = -1,
        use_gpu: bool = False,
        html: bool = True,
        session_id: Optional[int] = None,
        system_log: Union[bool, logging.Logger] = True,
        log_experiment: bool = False,
        experiment_name: Optional[str] = None,
        experiment_custom_tags: Optional[Dict[str, Any]] = None,
        log_plots: Union[bool, list] = False,
        log_profile: bool = False,
        log_data: bool = False,
        silent: bool = False,
        verbose: bool = True,
        memory: Union[bool, str, Memory] = True,
        profile: bool = False,
        profile_kwargs: Dict[str, Any] = None,
    ):
        # Setup initialization ===================================== >>

        runtime_start = time.time()

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
            self.log_plots_param = ["auc", "confusion_matrix", "feature"]
        elif isinstance(self.log_plots_param, list):
            for i in self.log_plots_param:
                if i not in self._available_plots:
                    raise ValueError(
                        f"Invalid value for log_plots '{i}'. Possible values "
                        f"are: {', '.join(self._available_plots.keys())}."
                    )

        # Set up data ============================================== >>

        self._prepare_dataset(data)
        self._prepare_target(target)

        self._prepare_column_types(
            ordinal_features=ordinal_features,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            date_features=date_features,
            text_features=text_features,
            ignore_features=ignore_features,
            keep_features=keep_features,
        )

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

            # Convert date feature to numerical values
            if self._fxs["Date"]:
                self._date_feature_engineering()

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
                self._encoding(max_encoding_ohe, encoding_method)

            # Create polynomial features from the existing ones
            if polynomial_features:
                self._polynomial_features(polynomial_degree)

            # Drop features with too low variance
            if low_variance_threshold:
                self._low_variance(low_variance_threshold)

            # Drop features that are collinear with other features
            if remove_multicollinearity:
                self._remove_multicollinearity(multicollinearity_threshold)

            # Bin numerical features to 5 clusters
            if bin_numeric_features:
                self._bin_numerical_features(bin_numeric_features)

            # Remove outliers from the dataset
            if remove_outliers:
                self._remove_outliers(outliers_method, outliers_threshold)

            # Balance the classes in the target column
            if fix_imbalance:
                self._balance(fix_imbalance_method)

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
            self._add_custom_pipeline(custom_pipeline)

        # Remove placeholder step
        if len(self.pipeline) > 1:
            self.pipeline.steps.pop(0)

        self.pipeline.fit(self.X_train, self.y_train)

        self.logger.info(f"Finished creating preprocessing pipeline.")
        self.logger.info(f"Pipeline: {self.pipeline}")

        # Final display ============================================ >>

        self.logger.info("Creating final display dataframe.")

        container = []
        container.append(["Session id", self.seed])
        container.append(["Target", self.target_param])
        container.append(["Target type", "classification"])
        le = get_label_encoder(self.pipeline)
        if le:
            mapping = {str(v): i for i, v in enumerate(le.classes_)}
            container.append(
                ["Target mapping", ", ".join([f"{k}: {v}" for k, v in mapping.items()])]
            )
        container.append(["Data shape", self.dataset.shape])
        container.append(["Train data shape", self.train.shape])
        container.append(["Test data shape", self.test.shape])
        for fx, cols in self._fxs.items():
            if len(cols) > 0:
                container.append([f"{fx} features", len(cols)])
        if self.data.isna().sum().sum():
            container.append(["Missing Values", self.data.isna().sum().sum()])
        if preprocess:
            container.append(["Preprocess", preprocess])
            container.append(["Imputation type", imputation_type])
            if imputation_type == "simple":
                container.append(["Numeric imputation", numeric_imputation])
                container.append(["Categorical imputation", categorical_imputation])
            else:
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
            if low_variance_threshold:
                container.append(["Low variance threshold", low_variance_threshold])
            if remove_multicollinearity:
                container.append(["Remove multicollinearity", remove_multicollinearity])
                container.append(
                    ["Multicollinearity threshold", multicollinearity_threshold]
                )
            if remove_outliers:
                container.append(["Remove outliers", remove_outliers])
                container.append(["Outliers threshold", outliers_threshold])
            if fix_imbalance:
                container.append(["Fix imbalance", fix_imbalance])
                container.append(["Fix imbalance method", fix_imbalance_method])
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
            if custom_pipeline:
                container.append(["Custom pipeline", "Yes"])
            container.append(["Fold Generator", self.fold_generator.__class__.__name__])
            container.append(["Fold Number", fold])
            container.append(["CPU Jobs", self.n_jobs_param])
            container.append(["Log Experiment", self.logging_param])
            container.append(["Experiment Name", self.exp_name_log])
            container.append(["USI", self.USI])

        self.display_container = [
            pd.DataFrame(container, columns=["Description", "Value"])
        ]
        self.logger.info(f"Setup display_container: {self.display_container[0]}")
        display = Display(
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
        self._set_up_mlflow(runtime, log_data, log_profile, experiment_custom_tags=experiment_custom_tags)

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
        sort: str = "Accuracy",
        n_select: int = 1,
        budget_time: Optional[float] = None,
        turbo: bool = True,
        errors: str = "ignore",
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        experiment_custom_tags: Optional[Dict[str, Any]] = None,
        probability_threshold: Optional[float] = None,
        verbose: bool = True,
    ) -> Union[Any, List[Any]]:

        """
        This function trains and evaluates performance of all estimators available in the
        model library using cross validation. The output of this function is a score grid
        with average cross validated scores. Metrics evaluated during CV can be accessed
        using the ``get_metrics`` function. Custom metrics can be added or removed using
        ``add_metric`` and ``remove_metric`` function.

        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase')
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


        sort: str, default = 'Accuracy'
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


        probability_threshold: float, default = None
            Threshold for converting predicted probability to class label.
            It defaults to 0.5 for all classifiers unless explicitly defined
            in this parameter. Only applicable for binary classification.


        verbose: bool, default = True
            Score grid is not printed when verbose is set to False.


        Returns:
            Trained model or list of trained models, depending on the ``n_select`` param.

        Warnings
        --------
        - Changing turbo parameter to False may result in very high training times with
        datasets exceeding 10,000 rows.

        - AUC for estimators that does not support 'predict_proba' is shown as 0.0000.

        - No models are logged in ``MLFlow`` when ``cross_validation`` parameter is False.
        """

        return super().compare_models(
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
            probability_threshold=probability_threshold,
        )

    def create_model(
        self,
        estimator: Union[str, Any],
        fold: Optional[Union[int, Any]] = None,
        round: int = 4,
        cross_validation: bool = True,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        experiment_custom_tags: Optional[Dict[str, Any]] = None,
        probability_threshold: Optional[float] = None,
        verbose: bool = True,
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


        probability_threshold: float, default = None
            Threshold for converting predicted probability to class label.
            It defaults to 0.5 for all classifiers unless explicitly defined
            in this parameter. Only applicable for binary classification.


        verbose: bool, default = True
            Score grid is not printed when verbose is set to False.


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

        return super().create_model(
            estimator=estimator,
            fold=fold,
            round=round,
            cross_validation=cross_validation,
            fit_kwargs=fit_kwargs,
            groups=groups,
            verbose=verbose,
            experiment_custom_tags=experiment_custom_tags,
            probability_threshold=probability_threshold,
            **kwargs,
        )

    def tune_model(
        self,
        estimator,
        fold: Optional[Union[int, Any]] = None,
        round: int = 4,
        n_iter: int = 10,
        custom_grid: Optional[Union[Dict[str, list], Any]] = None,
        optimize: str = "Accuracy",
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
        **kwargs,
    ) -> Any:

        """
        This function tunes the hyperparameters of a given estimator. The output of
        this function is a score grid with CV scores by fold of the best selected
        model based on ``optimize`` parameter. Metrics evaluated during CV can be
        accessed using the ``get_metrics`` function. Custom metrics can be added
        or removed using ``add_metric`` and ``remove_metric`` function.

        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase')
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


        optimize: str, default = 'Accuracy'
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
            print more messages. Ignored when ``verbose`` parameter is False.


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
        optimize: str = "Accuracy",
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        probability_threshold: Optional[float] = None,
        verbose: bool = True,
    ) -> Any:

        """
        This function ensembles a given estimator. The output of this function is
        a score grid with CV scores by fold. Metrics evaluated during CV can be
        accessed using the ``get_metrics`` function. Custom metrics can be added
        or removed using ``add_metric`` and ``remove_metric`` function.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase')
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


        optimize: str, default = 'Accuracy'
            Metric to compare for model selection when ``choose_better`` is True.


        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.


        groups: str or array-like, with shape (n_samples,), default = None
            Optional group labels when GroupKFold is used for the cross validation.
            It takes an array with shape (n_samples, ) where n_samples is the number
            of rows in training dataset. When string is passed, it is interpreted as
            the column name in the dataset containing group labels.


        probability_threshold: float, default = None
            Threshold for converting predicted probability to class label.
            It defaults to 0.5 for all classifiers unless explicitly defined
            in this parameter. Only applicable for binary classification.


        verbose: bool, default = True
            Score grid is not printed when verbose is set to False.


        Returns:
            Trained Model


        Warnings
        --------
        - Method 'Boosting' is not supported for estimators that do not have 'class_weights'
        or 'predict_proba' attributes.

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
            probability_threshold=probability_threshold,
            verbose=verbose,
        )

    def blend_models(
        self,
        estimator_list: list,
        fold: Optional[Union[int, Any]] = None,
        round: int = 4,
        choose_better: bool = False,
        optimize: str = "Accuracy",
        method: str = "auto",
        weights: Optional[List[float]] = None,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        probability_threshold: Optional[float] = None,
        verbose: bool = True,
    ) -> Any:

        """
        This function trains a Soft Voting / Majority Rule classifier for select
        models passed in the ``estimator_list`` param. The output of this function
        is a score grid with CV scores by fold. Metrics evaluated during CV can be
        accessed using the ``get_metrics`` function. Custom metrics can be added
        or removed using ``add_metric`` and ``remove_metric`` function.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase')
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


        optimize: str, default = 'Accuracy'
            Metric to compare for model selection when ``choose_better`` is True.


        method: str, default = 'auto'
            'hard' uses predicted class labels for majority rule voting. 'soft', predicts
            the class label based on the argmax of the sums of the predicted probabilities,
            which is recommended for an ensemble of well-calibrated classifiers. Default
            value, 'auto', will try to use 'soft' and fall back to 'hard' if the former is
            not supported.


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


        probability_threshold: float, default = None
            Threshold for converting predicted probability to class label.
            It defaults to 0.5 for all classifiers unless explicitly defined
            in this parameter. Only applicable for binary classification.


        verbose: bool, default = True
            Score grid is not printed when verbose is set to False.


        Returns:
            Trained Model

        """

        return super().blend_models(
            estimator_list=estimator_list,
            fold=fold,
            round=round,
            choose_better=choose_better,
            optimize=optimize,
            method=method,
            weights=weights,
            fit_kwargs=fit_kwargs,
            groups=groups,
            verbose=verbose,
            probability_threshold=probability_threshold,
        )

    def stack_models(
        self,
        estimator_list: list,
        meta_model=None,
        meta_model_fold: Optional[Union[int, Any]] = 5,
        fold: Optional[Union[int, Any]] = None,
        round: int = 4,
        method: str = "auto",
        restack: bool = False,
        choose_better: bool = False,
        optimize: str = "Accuracy",
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        probability_threshold: Optional[float] = None,
        verbose: bool = True,
    ) -> Any:

        """
        This function trains a meta model over select estimators passed in
        the ``estimator_list`` parameter. The output of this function is a
        score grid with CV scores by fold. Metrics evaluated during CV can
        be accessed using the ``get_metrics`` function. Custom metrics
        can be added or removed using ``add_metric`` and ``remove_metric``
        function.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase')
        >>> top3 = compare_models(n_select = 3)
        >>> stacker = stack_models(top3)


        estimator_list: list of scikit-learn compatible objects
            List of trained model objects


        meta_model: scikit-learn compatible object, default = None
            When None, Logistic Regression is trained as a meta model.


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


        method: str, default = 'auto'
            When set to 'auto', it will invoke, for each estimator, 'predict_proba',
            'decision_function' or 'predict' in that order. Other, manually pass one
            of the value from 'predict_proba', 'decision_function' or 'predict'.


        restack: bool, default = False
            When set to False, only the predictions of estimators will be used as
            training data for the ``meta_model``.


        choose_better: bool, default = False
            When set to True, the returned object is always better performing. The
            metric used for comparison is defined by the ``optimize`` parameter.


        optimize: str, default = 'Accuracy'
            Metric to compare for model selection when ``choose_better`` is True.


        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.


        groups: str or array-like, with shape (n_samples,), default = None
            Optional group labels when GroupKFold is used for the cross validation.
            It takes an array with shape (n_samples, ) where n_samples is the number
            of rows in training dataset. When string is passed, it is interpreted as
            the column name in the dataset containing group labels.


        probability_threshold: float, default = None
            Threshold for converting predicted probability to class label.
            It defaults to 0.5 for all classifiers unless explicitly defined
            in this parameter. Only applicable for binary classification.


        verbose: bool, default = True
            Score grid is not printed when verbose is set to False.


        Returns:
            Trained Model


        Warnings
        --------
        - When ``method`` is not set to 'auto', it will check if the defined method
        is available for all estimators passed in ``estimator_list``. If the method is
        not implemented by any estimator, it will raise an error.

        """

        return super().stack_models(
            estimator_list=estimator_list,
            meta_model=meta_model,
            meta_model_fold=meta_model_fold,
            fold=fold,
            round=round,
            method=method,
            restack=restack,
            choose_better=choose_better,
            optimize=optimize,
            fit_kwargs=fit_kwargs,
            groups=groups,
            verbose=verbose,
            probability_threshold=probability_threshold,
        )

    def plot_model(
        self,
        estimator,
        plot: str = "auc",
        scale: float = 1,
        save: bool = False,
        fold: Optional[Union[int, Any]] = None,
        fit_kwargs: Optional[dict] = None,
        plot_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        use_train_data: bool = False,
        verbose: bool = True,
        display_format: Optional[str] = None,
    ) -> str:

        """
        This function analyzes the performance of a trained model on holdout set.
        It may require re-training the model in certain cases.

        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase')
        >>> lr = create_model('lr')
        >>> plot_model(lr, plot = 'auc')


        estimator: scikit-learn compatible object
            Trained model object


        plot: str, default = 'auc'
            List of available plots (ID - Name):

            * 'residuals_interactive' - Interactive Residual plots
            * 'auc' - Area Under the Curve
            * 'threshold' - Discrimination Threshold
            * 'pr' - Precision Recall Curve
            * 'confusion_matrix' - Confusion Matrix
            * 'error' - Class Prediction Error
            * 'class_report' - Classification Report
            * 'boundary' - Decision Boundary
            * 'rfe' - Recursive Feature Selection
            * 'learning' - Learning Curve
            * 'manifold' - Manifold Learning
            * 'calibration' - Calibration Curve
            * 'vc' - Validation Curve
            * 'dimension' - Dimension Learning
            * 'feature' - Feature Importance
            * 'feature_all' - Feature Importance (All)
            * 'parameter' - Model Hyperparameter
            * 'lift' - Lift Curve
            * 'gain' - Gain Chart
            * 'tree' - Decision Tree
            * 'ks' - KS Statistic Plot


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
            None


        Warnings
        --------
        -   Estimators that does not support 'predict_proba' attribute cannot be used for
            'AUC' and 'calibration' plots.

        -   When the target is multiclass, 'calibration', 'threshold', 'manifold' and 'rfe'
            plots are not available.

        -   When the 'max_features' parameter of a trained model object is not equal to
            the number of samples in training set, the 'rfe' plot is not available.

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
            system=True,
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
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase')
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
        based on the test / hold-out set. It only supports tree based algorithms.

        This function is implemented based on the SHAP (SHapley Additive exPlanations),
        which is a unified approach to explain the output of any machine learning model.
        SHAP connects game theory with local explanations.

        For more information : https://shap.readthedocs.io/en/latest/

        For Partial Dependence Plot : https://github.com/SauceCat/PDPbox


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase')
        >>> xgboost = create_model('xgboost')
        >>> interpret_model(xgboost)


        estimator : object, default = none
            A trained model object to be passed as an estimator. Only tree-based
            models are accepted when plot type is 'summary', 'correlation', or
            'reason'. 'pdp' plot is model agnostic.


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

    def calibrate_model(
        self,
        estimator,
        method: str = "sigmoid",
        calibrate_fold: Optional[Union[int, Any]] = 5,
        fold: Optional[Union[int, Any]] = None,
        round: int = 4,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        verbose: bool = True,
        display: Optional[Display] = None,  # added in pycaret==2.2.0
    ) -> Any:

        """
        This function takes the input of trained estimator and performs probability
        calibration with sigmoid or isotonic regression. The output prints a score
        grid that shows Accuracy, AUC, Recall, Precision, F1, Kappa and MCC by fold
        (default = 10 Fold). The ouput of the original estimator and the calibrated
        estimator (created using this function) might not differ much. In order
        to see the calibration differences, use 'calibration' plot in plot_model to
        see the difference before and after.

        This function returns a trained model object.

        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> experiment_name = setup(data = juice,  target = 'Purchase')
        >>> dt_boosted = create_model('dt', ensemble = True, method = 'Boosting')
        >>> calibrated_dt = calibrate_model(dt_boosted)

        This will return Calibrated Boosted Decision Tree Model.

        Parameters
        ----------
        estimator : object

        method : str, default = 'sigmoid'
            The method to use for calibration. Can be 'sigmoid' which corresponds to Platt's
            method or 'isotonic' which is a non-parametric approach. It is not advised to use
            isotonic calibration with too few calibration samples

        calibrate_fold: integer or scikit-learn compatible CV generator, default = 5
            Controls internal cross-validation. Can be an integer or a scikit-learn
            CV generator. If set to an integer, will use (Stratifed)KFold CV with
            that many folds. See scikit-learn documentation on Stacking for
            more details.

        fold: integer or scikit-learn compatible CV generator, default = None
            Controls cross-validation. If None, will use the CV generator defined in setup().
            If integer, will use KFold CV with that many folds.
            When cross_validation is False, this parameter is ignored.

        round: integer, default = 4
            Number of decimal places the metrics in the score grid will be rounded to.

        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.

        groups: str or array-like, with shape (n_samples,), default = None
            Optional Group labels for the samples used while splitting the dataset into train/test set.
            If string is passed, will use the data column with that name as the groups.
            Only used if a group based cross-validation generator is used (eg. GroupKFold).
            If None, will use the value set in fold_groups parameter in setup().

        verbose: bool, default = True
            Score grid is not printed when verbose is set to False.

        Returns
        -------
        score_grid
            A table containing the scores of the model across the kfolds.
            Scoring metrics used are Accuracy, AUC, Recall, Precision, F1,
            Kappa and MCC. Mean and standard deviation of the scores across
            the folds are also returned.

        model
            trained and calibrated model object.

        Warnings
        --------
        - Avoid isotonic calibration with too few calibration samples (<1000) since it
        tends to overfit.

        - calibration plot not available for multiclass problems.


        """

        function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

        self.logger.info("Initializing calibrate_model()")
        self.logger.info(f"calibrate_model({function_params_str})")

        self.logger.info("Checking exceptions")

        # run_time
        runtime_start = time.time()

        if not fit_kwargs:
            fit_kwargs = {}

        # checking fold parameter
        if fold is not None and not (
            type(fold) is int or is_sklearn_cv_generator(fold)
        ):
            raise TypeError(
                "fold parameter must be either None, an integer or a scikit-learn compatible CV generator object."
            )

        # checking round parameter
        if type(round) is not int:
            raise TypeError("Round parameter only accepts integer value.")

        # checking verbose parameter
        if type(verbose) is not bool:
            raise TypeError(
                "Verbose parameter can only take argument as True or False."
            )

        """

        ERROR HANDLING ENDS HERE

        """

        fold = self._get_cv_splitter(fold)

        groups = self._get_groups(groups)

        self.logger.info("Preloading libraries")

        # pre-load libraries

        self.logger.info("Preparing display monitor")

        if not display:
            progress_args = {"max": 2 + 4}
            master_display_columns = [
                v.display_name for k, v in self._all_metrics.items()
            ]
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
            display = Display(
                verbose=verbose,
                html_param=self.html_param,
                progress_args=progress_args,
                master_display_columns=master_display_columns,
                monitor_rows=monitor_rows,
            )

            display.display_progress()
            display.display_monitor()
            display.display_master_display()

        np.random.seed(self.seed)

        probability_threshold = None
        if isinstance(estimator, CustomProbabilityThresholdClassifier):
            probability_threshold = estimator.probability_threshold
            estimator = get_estimator_from_meta_estimator(estimator)

        self.logger.info("Getting model name")

        full_name = self._get_model_name(estimator)

        self.logger.info(f"Base model : {full_name}")

        display.update_monitor(2, full_name)
        display.display_monitor()

        """
        MONITOR UPDATE STARTS
        """

        display.update_monitor(1, "Selecting Estimator")
        display.display_monitor()

        """
        MONITOR UPDATE ENDS
        """

        # calibrating estimator

        self.logger.info("Importing untrained CalibratedClassifierCV")

        calibrated_model_definition = self._all_models_internal["CalibratedCV"]
        model = calibrated_model_definition.class_def(
            base_estimator=estimator,
            method=method,
            cv=calibrate_fold,
            **calibrated_model_definition.args,
        )

        display.move_progress()

        self.logger.info(
            "SubProcess create_model() called =================================="
        )
        model, model_fit_time = self.create_model(
            estimator=model,
            system=False,
            display=display,
            fold=fold,
            round=round,
            fit_kwargs=fit_kwargs,
            groups=groups,
            probability_threshold=probability_threshold,
        )
        model_results = self.pull()
        self.logger.info(
            "SubProcess create_model() end =================================="
        )

        model_results = model_results.round(round)

        display.move_progress()

        # end runtime
        runtime_end = time.time()
        runtime = np.array(runtime_end - runtime_start).round(2)

        # mlflow logging
        if self.logging_param:

            avgs_dict_log = {k: v for k, v in model_results.loc["Mean"].items()}

            try:
                self._mlflow_log_model(
                    model=model,
                    model_results=model_results,
                    score_dict=avgs_dict_log,
                    source="calibrate_models",
                    runtime=runtime,
                    model_fit_time=model_fit_time,
                    pipeline=self.pipeline,
                    log_plots=self.log_plots_param,
                    display=display,
                )
            except:
                self.logger.error(
                    f"_mlflow_log_model() for {model} raised an exception:"
                )
                self.logger.error(traceback.format_exc())

        model_results = color_df(model_results, "yellow", ["Mean"], axis=1)
        model_results = model_results.set_precision(round)
        display.display(model_results, clear=True)

        self.logger.info(f"master_model_container: {len(self.master_model_container)}")
        self.logger.info(f"display_container: {len(self.display_container)}")

        self.logger.info(str(model))
        self.logger.info(
            "calibrate_model() succesfully completed......................................"
        )

        gc.collect()
        return model

    def optimize_threshold(
        self,
        estimator,
        optimize: str = "Accuracy",
        grid_interval: float = 0.1,
        return_data: bool = False,
        plot_kwargs: Optional[dict] = None,
    ):

        """
        This function optimizes probability threshold for a trained classifier. It
        iterates over performance metrics at different ``probability_threshold`` with
        a step size defined in ``grid_interval`` parameter. This function will display
        a plot of the performance metrics at each probability threshold and returns the
        best model based on the metric defined under ``optimize`` parameter.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> experiment_name = setup(data = juice,  target = 'Purchase')
        >>> lr = create_model('lr')
        >>> best_lr_threshold = optimize_threshold(lr)


        Parameters
        ----------
        estimator : object
            A trained model object should be passed as an estimator.


        optimize : str, default = 'Accuracy'
            Metric to be used for selecting best model.


        grid_interval : float, default = 0.0001
            Grid interval for threshold grid search. Default 10 iterations.


        return_data :  bool, default = False
            When set to True, data used for visualization is also returned.


        plot_kwargs :  dict, default = {} (empty dict)
            Dictionary of arguments passed to the visualizer class.


        Returns
        -------
        Trained Model

        Warnings
        --------
        - This function is not supported for multiclass problems.


        """

        function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

        self.logger.info("Initializing optimize_threshold()")
        self.logger.info(f"optimize_threshold({function_params_str})")

        self.logger.info("Importing libraries")

        # import libraries

        np.random.seed(self.seed)

        """
        ERROR HANDLING STARTS HERE
        """

        self.logger.info("Checking exceptions")

        # exception 1 for multi-class
        if self._is_multiclass():
            raise TypeError(
                "optimize_threshold() cannot be used when target is multi-class."
            )

        # check predict_proba value
        if type(estimator) is not list:
            if not hasattr(estimator, "predict_proba"):
                raise TypeError(
                    "Estimator doesn't support predict_proba function and cannot be used in optimize_threshold()."
                )

        allowed_types = [int, float]

        if type(grid_interval) not in allowed_types or grid_interval > 1.0:
            raise TypeError("grid_interval should be float and less than 1.0.")

        if isinstance(optimize, str):
            # checking optimize parameter
            optimize = self._get_metric_by_name_or_id(optimize)
            if optimize is None:
                raise ValueError(
                    "Optimize method not supported. See docstring for list of available parameters."
                )
            optimize = optimize.display_name

        """
        ERROR HANDLING ENDS HERE
        """

        self.logger.info("defining variables")
        # get estimator name
        model_name = self._get_model_name(estimator)

        # defines grid
        grid = np.arange(0, 1.0001, grid_interval)

        # defines empty list
        models_by_threshold = []
        results_df = []

        self.logger.info("starting optimization loop")
        # loop starts here
        for i in grid:
            model = self.create_model(estimator, verbose=False, system=False, probability_threshold=i)
            try:
                models_by_threshold.append(model[0])
            except:
                models_by_threshold.append(model)
            model_results = self.pull(pop=True).loc[['Mean']]
            model_results['probability_threshold'] = i
            results_df.append(model_results)

        self.logger.info("optimization loop finished successfully")

        results_concat = pd.concat(results_df, axis=0)
        results_concat_melted = results_concat.melt(id_vars = ['probability_threshold'], value_vars=list(results_concat.columns[:-1]))
        optimized_metric_index = np.array(results_concat_melted[results_concat_melted['variable'] == optimize]['value']).argmax()
        best_model_by_metric = models_by_threshold[optimized_metric_index]

        self.logger.info("plotting optimization threshold using plotly")

        # plotting threshold
        import plotly.express as px
        title = f"{model_name} Probability Threshold Optimization (default = 0.5)"
        plot_kwargs = plot_kwargs or {}
        fig = px.line(results_concat_melted, x="probability_threshold", y="value", title = title,\
                        color='variable', **plot_kwargs)
        self.logger.info("Figure ready for render")
        fig.show()

        self.logger.info("returning model with best metric")
        if return_data:
            self.logger.info("also returning data as return_data = True")
            self.logger.info("optimize_threshold() succesfully completed......................................")
            return (results_concat_melted, best_model_by_metric)
        else:
            self.logger.info("optimize_threshold() succesfully completed......................................")
            return best_model_by_metric

    def predict_model(
        self,
        estimator,
        data: Optional[pd.DataFrame] = None,
        probability_threshold: Optional[float] = None,
        encoded_labels: bool = False,
        raw_score: bool = False,
        drift_report: bool = False,
        round: int = 4,
        verbose: bool = True,
    ) -> pd.DataFrame:

        """
        This function predicts ``Label`` and ``Score`` (probability of predicted
        class) using a trained model. When ``data`` is None, it predicts label and
        score on the holdout set.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase')
        >>> lr = create_model('lr')
        >>> pred_holdout = predict_model(lr)
        >>> pred_unseen = predict_model(lr, data = unseen_dataframe)


        estimator: scikit-learn compatible object
            Trained model object


        data: pandas.DataFrame
            Shape (n_samples, n_features). All features used during training
            must be available in the unseen dataset.


        probability_threshold: float, default = None
            Threshold for converting predicted probability to class label.
            It defaults to 0.5 for all classifiers unless explicitly defined
            in this parameter.


        encoded_labels: bool, default = False
            When set to True, will return labels encoded as an integer.


        raw_score: bool, default = False
            When set to True, scores for all labels will be returned.


        round: int, default = 4
            Number of decimal places the metrics in the score grid will be rounded to.


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
            probability_threshold=probability_threshold,
            encoded_labels=encoded_labels,
            raw_score=raw_score,
            drift_report=drift_report,
            round=round,
            verbose=verbose,
        )

    def finalize_model(
        self,
        estimator,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        model_only: bool = True,
        experiment_custom_tags: Optional[Dict[str, Any]] = None,
    ) -> Any:

        """
        This function trains a given estimator on the entire dataset including the
        holdout set.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase')
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


        model_only: bool, default = True
            When set to False, only model object is re-trained and all the
            transformations in Pipeline are ignored.


        Returns:
            Trained Model

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
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase')
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
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase')
        >>> lr = create_model('lr')
        >>> save_model(lr, 'saved_lr_model')


        model: scikit-learn compatible object
            Trained model object


        model_name: str
            Name of the model.


        model_only: bool, default = False
            When set to True, only trained model object is saved instead of the
            entire pipeline.


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
        model_name,
        platform: Optional[str] = None,
        authentication: Optional[Dict[str, str]] = None,
        verbose: bool = True,
    ):

        """
        This function loads a previously saved pipeline.


        Example
        -------
        >>> from pycaret.classification import load_model
        >>> saved_lr = load_model('saved_lr_model')


        model_name: str
            Name of the model.


        platform: str, default = None
            Name of the cloud platform. Currently supported platforms:
            'aws', 'gcp' and 'azure'.


        authentication: dict, default = None
            dictionary of applicable authentication tokens.

            when platform = 'aws':
            {'bucket' : 'S3-bucket-name'}

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
        self, optimize: str = "Accuracy", use_holdout: bool = False, turbo: bool = True
    ) -> Any:

        """
        This function returns the best model out of all trained models in
        current session based on the ``optimize`` parameter. Metrics
        evaluated can be accessed using the ``get_metrics`` function.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase')
        >>> top3 = compare_models(n_select = 3)
        >>> tuned_top3 = [tune_model(i) for i in top3]
        >>> blender = blend_models(tuned_top3)
        >>> stacker = stack_models(tuned_top3)
        >>> best_auc_model = automl(optimize = 'AUC')


        optimize: str, default = 'Accuracy'
            Metric to use for model selection. It also accepts custom metrics
            added using the ``add_metric`` function.


        use_holdout: bool, default = False
            When set to True, metrics are evaluated on holdout set instead of CV.


        turbo: bool, default = True
            When set to True and use_holdout is False, only models created with default fold
            parameter will be considered. If set to False, models created with a non-default
            fold parameter will be scored again using default fold settings, so that they can be
            compared.


        Returns:
            Trained Model

        """
        return super().automl(optimize=optimize, use_holdout=use_holdout, turbo=turbo)

    def pull(self, pop: bool = False) -> pd.DataFrame:

        """
        Returns last printed score grid. Use ``pull`` function after
        any training function to store the score grid in pandas.DataFrame.


        pop: bool, default = False
            If True, will pop (remove) the returned dataframe from the
            display container.


        Returns:
            pandas.DataFrame

        """
        return super().pull(pop=pop)

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
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase')
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
        Returns table of available metrics used for CV.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase')
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
        target: str = "pred",
        greater_is_better: bool = True,
        multiclass: bool = True,
        **kwargs,
    ) -> pd.Series:

        """
        Adds a custom metric to be used for CV.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase')
        >>> from sklearn.metrics import log_loss
        >>> add_metric('logloss', 'Log Loss', log_loss, greater_is_better = False)


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
            Whether ``score_func`` is higher the better or not.


        multiclass: bool, default = True
            Whether the metric supports multiclass target.


        **kwargs:
            Arguments to be passed to score function.


        Returns:
            pandas.Series

        """

        return super().add_metric(
            id=id,
            name=name,
            score_func=score_func,
            target=target,
            greater_is_better=greater_is_better,
            multiclass=multiclass,
            **kwargs,
        )

    def remove_metric(self, name_or_id: str):

        """
        Removes a metric from CV.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase')
        >>> remove_metric('MCC')


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
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase', log_experiment = True)
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
