import os
import gc
import logging
import warnings
import traceback
import random
import secrets
import time
import traceback
import warnings
from typing import List, Tuple, Any, Union, Optional, Dict
from unittest.mock import patch
from packaging import version

import numpy as np  # type: ignore
import plotly.express as px  # type: ignore
import plotly.graph_objects as go  # type: ignore
import pycaret.internal.patches.sklearn
import pycaret.internal.patches.yellowbrick
import pycaret.internal.persistence
import pycaret.internal.preprocess
import scikitplot as skplt  # type: ignore
import sklearn
from boruta import BorutaPy
from imblearn.over_sampling import SMOTE
from pandas.io.formats.style import Styler
from pycaret.internal.logging import create_logger
from pycaret.internal.preprocess import (
    TransfomerWrapper,
    ExtractDateTimeFeatures,
    EmbedTextFeatures,
    RemoveMulticollinearity,
    RemoveOutliers,
    FixImbalancer,
)
from pycaret.internal.meta_estimators import get_estimator_from_meta_estimator
from pycaret.internal.pipeline import Pipeline as InternalPipeline
from pycaret.internal.pipeline import (
    estimator_pipeline,
    get_pipeline_estimator_label,
    get_pipeline_fit_kwargs,
)
from pycaret.internal.plots.helper import MatplotlibDefaultDPI
from pycaret.internal.plots.yellowbrick import show_yellowbrick_plot
from pycaret.internal.pycaret_experiment.pycaret_experiment import _PyCaretExperiment
from pycaret.containers.models.classification import get_all_model_containers as get_classifiers
from pycaret.containers.models.regression import get_all_model_containers as get_regressors
from pycaret.internal.pycaret_experiment.utils import MLUsecase
from pycaret.internal.utils import (
    check_features_exist,
    get_columns_to_stratify_by,
    get_model_name,
    mlflow_remove_bad_chars,
    normalize_custom_transformers,
    df_shrink_dtypes,
)
from pycaret.internal.validation import *

from category_encoders.one_hot import OneHotEncoder
from category_encoders.ordinal import OrdinalEncoder
from category_encoders.leave_one_out import LeaveOneOutEncoder
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.experimental import enable_iterative_imputer
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectFromModel,
    SequentialFeatureSelector,
)
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.model_selection import BaseCrossValidator  # type: ignore
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
)
from sklearn.preprocessing import (
    LabelEncoder,
    MaxAbsScaler,
    MinMaxScaler,
    PolynomialFeatures,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
    KBinsDiscretizer,
)


warnings.filterwarnings("ignore")
LOGGER = get_logger()


class _TabularExperiment(_PyCaretExperiment):
    def __init__(self) -> None:
        super().__init__()
        self.variable_keys = self.variable_keys.union(
            {
                "_ml_usecase",
                "_available_plots",
                "variable_keys",
                "USI",
                "html_param",
                "seed",
                "_internal_pipeline",
                "experiment__",
                "n_jobs_param",
                "_gpu_n_jobs_param",
                "master_model_container",
                "display_container",
                "exp_name_log",
                "exp_id",
                "logging_param",
                "log_plots_param",
                "data",
                "gpu_param",
                "_all_models",
                "_all_models_internal",
                "_all_metrics",
                "_internal_pipeline",
                "imputation_regressor",
                "imputation_classifier",
                "iterative_imputation_iters_param",
            }
        )
        return

    def _get_setup_display(self, **kwargs) -> Styler:
        return pd.DataFrame().style

    def _get_default_plots_to_log(self) -> List[str]:
        return []

    def _get_groups(
        self,
        groups,
        data: Optional[pd.DataFrame] = None,
        fold_groups=None,
        ml_usecase: Optional[MLUsecase] = None,
    ):
        import pycaret.internal.utils

        data = data if data is not None else self.X_train
        fold_groups = fold_groups if fold_groups is not None else self.fold_groups_param
        return pycaret.internal.utils.get_groups(groups, data, fold_groups)

    def _get_cv_splitter(
        self, fold, ml_usecase: Optional[MLUsecase] = None
    ) -> BaseCrossValidator:
        """Returns the cross validator object used to perform cross validation"""
        if not ml_usecase:
            ml_usecase = self._ml_usecase

        import pycaret.internal.utils

        return pycaret.internal.utils.get_cv_splitter(
            fold,
            default=self.fold_generator,
            seed=self.seed,
            shuffle=self.fold_shuffle_param,
            int_default="stratifiedkfold"
            if ml_usecase == MLUsecase.CLASSIFICATION
            else "kfold",
        )

    def _is_unsupervised(self) -> bool:
        return False

    def _get_model_id(self, e, models=None) -> str:
        """
        Get model id.
        """
        if models is None:
            models = self._all_models_internal

        return pycaret.internal.utils.get_model_id(e, models)

    def _get_metric_by_name_or_id(self, name_or_id: str, metrics: Optional[Any] = None):
        """
        Gets a metric from get_metrics() by name or index.
        """
        if metrics is None:
            metrics = self._all_metrics
        metric = None
        try:
            metric = metrics[name_or_id]
            return metric
        except Exception:
            pass

        try:
            metric = next(
                v for k, v in metrics.items() if name_or_id in (v.display_name, v.name)
            )
            return metric
        except Exception:
            pass

        return metric

    def _get_model_name(self, e, deep: bool = True, models=None) -> str:
        """
        Get model name.
        """
        if models is None:
            models = self._all_models_internal

        return get_model_name(e, models, deep=deep)

    def _mlflow_log_model(
        self,
        model,
        model_results,
        score_dict: dict,
        source: str,
        runtime: float,
        model_fit_time: float,
        _internal_pipeline,
        log_holdout: bool = True,
        log_plots: bool = False,
        tune_cv_results=None,
        URI=None,
        display: Optional[Display] = None,
    ):
        self.logger.info("Creating MLFlow logs")

        # Creating Logs message monitor
        if display:
            display.update_monitor(1, "Creating Logs")
            display.display_monitor()

        # import mlflow
        import mlflow
        import mlflow.sklearn

        mlflow.set_experiment(self.exp_name_log)

        full_name = self._get_model_name(model)
        self.logger.info(f"Model: {full_name}")

        with mlflow.start_run(run_name=full_name, nested=True) as run:

            # Get active run to log as tag
            RunID = mlflow.active_run().info.run_id

            # Log model parameters
            pipeline_estimator_name = get_pipeline_estimator_label(model)
            if pipeline_estimator_name:
                params = model.named_steps[pipeline_estimator_name]
            else:
                params = model

            # get regressor from meta estimator
            params = get_estimator_from_meta_estimator(params)

            try:
                try:
                    params = params.get_all_params()
                except Exception:
                    params = params.get_params()
            except Exception:
                self.logger.warning("Couldn't get params for model. Exception:")
                self.logger.warning(traceback.format_exc())
                params = {}

            for i in list(params):
                v = params.get(i)
                if len(str(v)) > 250:
                    params.pop(i)

            params = {mlflow_remove_bad_chars(k): v for k, v in params.items()}
            self.logger.info(f"logged params: {params}")
            mlflow.log_params(params)

            # Log metrics
            def try_make_float(val):
                try:
                    return np.float64(val)
                except Exception:
                    return np.nan

            score_dict = {k: try_make_float(v) for k, v in score_dict.items()}
            self.logger.info(f"logged metrics: {score_dict}")
            mlflow.log_metrics(score_dict)

            # set tag of compare_models
            mlflow.set_tag("Source", source)

            if not URI:
                import secrets

                URI = secrets.token_hex(nbytes=4)
            mlflow.set_tag("URI", URI)
            mlflow.set_tag("USI", self.USI)
            mlflow.set_tag("Run Time", runtime)
            mlflow.set_tag("Run ID", RunID)

            # Log training time in seconds
            mlflow.log_metric("TT", model_fit_time)

            # Log the CV results as model_results.html artifact
            if not self._is_unsupervised():
                try:
                    model_results.data.to_html(
                        "Results.html", col_space=65, justify="left"
                    )
                except Exception:
                    model_results.to_html("Results.html", col_space=65, justify="left")
                mlflow.log_artifact("Results.html")
                os.remove("Results.html")

                if log_holdout:
                    # Generate hold-out predictions and save as html
                    try:
                        holdout = self.predict_model(model, verbose=False)  # type: ignore
                        holdout_score = self.pull(pop=True)
                        del holdout
                        holdout_score.to_html(
                            "Holdout.html", col_space=65, justify="left"
                        )
                        mlflow.log_artifact("Holdout.html")
                        os.remove("Holdout.html")
                    except Exception:
                        self.logger.warning(
                            "Couldn't create holdout prediction for model, exception below:"
                        )
                        self.logger.warning(traceback.format_exc())

            # Log AUC and Confusion Matrix plot

            if log_plots:

                self.logger.info(
                    "SubProcess plot_model() called =================================="
                )

                def _log_plot(plot):
                    try:
                        plot_name = self.plot_model(
                            model, plot=plot, verbose=False, save=True, system=False
                        )
                        mlflow.log_artifact(plot_name)
                        os.remove(plot_name)
                    except Exception as e:
                        self.logger.warning(e)

                for plot in log_plots:
                    _log_plot(plot)

                self.logger.info(
                    "SubProcess plot_model() end =================================="
                )

            # Log hyperparameter tuning grid
            if tune_cv_results:
                d1 = tune_cv_results.get("params")
                dd = pd.DataFrame.from_dict(d1)
                dd["Score"] = tune_cv_results.get("mean_test_score")
                dd.to_html("Iterations.html", col_space=75, justify="left")
                mlflow.log_artifact("Iterations.html")
                os.remove("Iterations.html")

            # get default conda env
            from mlflow.sklearn import get_default_conda_env

            default_conda_env = get_default_conda_env()
            default_conda_env["name"] = f"{self.exp_name_log}-env"
            default_conda_env.get("dependencies").pop(-3)
            dependencies = default_conda_env.get("dependencies")[-1]
            from pycaret.utils import __version__

            dep = f"pycaret=={__version__}"
            dependencies["pip"] = [dep]

            # define model signature
            from mlflow.models.signature import infer_signature

            try:
                signature = infer_signature(
                    self.data.drop([self.target_param], axis=1)
                )
            except Exception:
                self.logger.warning("Couldn't infer MLFlow signature.")
                signature = None
            if not self._is_unsupervised():
                input_example = (
                    self.data.drop([self.target_param], axis=1)
                    .iloc[0]
                    .to_dict()
                )
            else:
                input_example = self.data.iloc[0].to_dict()

            # log model as sklearn flavor
            _internal_pipeline_temp = deepcopy(_internal_pipeline)
            _internal_pipeline_temp.steps.append(["trained_model", model])
            mlflow.sklearn.log_model(
                _internal_pipeline_temp,
                "model",
                conda_env=default_conda_env,
                signature=signature,
                input_example=input_example,
            )
            del _internal_pipeline_temp
        gc.collect()

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
        imputation_type: str = "simple",
        numeric_imputation: str = "mean",
        categorical_imputation: str = "constant",
        iterative_imputation_iters: int = 5,
        numeric_iterative_imputer: Union[str, Any] = "lightgbm",
        categorical_iterative_imputer: Union[str, Any] = "lightgbm",
        text_features_method: str = "tf-idf",
        max_encoding_ohe: int = 5,
        encoding_method: Optional[Any] = None,
        transformation: bool = False,
        transformation_method: str = "yeo-johnson",
        normalize: bool = False,
        normalize_method: str = "zscore",
        low_variance_threshold: float = 0,
        remove_multicollinearity: bool = False,
        multicollinearity_threshold: float = 0.9,
        bin_numeric_features: Optional[List[str]] = None,
        remove_outliers: bool = False,
        outliers_method: str = "iforest",
        outliers_threshold: float = 0.05,
        polynomial_features: bool = False,
        polynomial_degree: int = 2,
        fix_imbalance: bool = False,
        fix_imbalance_method: Optional[Any] = None,
        pca: bool = False,
        pca_method: str = "linear",
        pca_components: Union[int, float] = 1.0,
        feature_selection: bool = False,
        feature_selection_method: str = "classic",
        feature_selection_estimator: Union[str, Any] = "lightgbm",
        n_features_to_select: int = 10,
        transform_target=False,
        transform_target_method="box-cox",
        custom_pipeline: Any = None,
        data_split_shuffle: bool = True,
        data_split_stratify: Union[bool, List[str]] = False,  # added in pycaret==2.2
        fold_strategy: Union[str, Any] = "kfold",  # added in pycaret==2.2
        fold: int = 10,  # added in pycaret==2.2
        fh: Union[List[int], int, np.array] = 1,
        fold_shuffle: bool = False,
        fold_groups: Optional[Union[str, pd.DataFrame]] = None,
        n_jobs: Optional[int] = -1,
        use_gpu: bool = False,  # added in pycaret==2.1
        html: bool = True,
        session_id: Optional[int] = None,
        system_log: Union[bool, logging.Logger] = True,
        log_experiment: bool = False,
        experiment_name: Optional[str] = None,
        log_plots: Union[bool, list] = False,
        log_profile: bool = False,
        log_data: bool = False,
        silent: bool = False,
        seasonal_period: Optional[int] = None,
        verbose: bool = True,
        profile: bool = False,
        profile_kwargs: Dict[str, Any] = None,
        display: Optional[Display] = None,
    ):
        """
        This function initializes the environment in pycaret and creates the
        transformation pipeline to prepare the data for modeling and deployment.
        setup() must called before executing any other function in pycaret. It
        takes only two mandatory parameters: data and name of the target column.

        """
        from pycaret.utils import __version__

        # Settings ================================================= >>

        pd.set_option("display.max_columns", 100)
        pd.set_option("display.max_rows", 100)

        # Attribute definition ===================================== >>

        # Parameter attrs
        self.data = data
        self.target_param = target
        self.transform_target_param = transform_target
        self.transform_target_method_param = transform_target_method
        self.n_jobs_param = n_jobs
        self.gpu_param = use_gpu
        self.fold_param = fold
        self.fold_groups_param = None
        self.html_param = html
        self.logging_param = log_experiment
        self.log_plots_param = log_plots or False

        # Global attrs
        self.seed = random.randint(150, 9000) if session_id is None else session_id
        np.random.seed(self.seed)

        # Initialization =========================================== >>

        runtime_start = time.time()

        # Get local parameters to write to logger
        function_params_str = ", ".join(
            [f"{k}={v}" for k, v in locals().items() if k != "self"]
        )

        if experiment_name:
            if not isinstance(experiment_name, str):
                raise TypeError(
                    "The experiment_name parameter must be a non-empty str if not None."
                )
            self.exp_name_log = experiment_name

        self.logger = create_logger(system_log)
        self.logger.info(f"PyCaret {type(self).__name__}")
        self.logger.info(f"Logging name: {self.exp_name_log}")
        self.logger.info(f"ML Usecase: {self._ml_usecase}")
        self.logger.info(f"version {__version__}")
        self.logger.info("Initializing setup()")
        self.logger.info(f"setup(data=data, {function_params_str})")

        self.USI = secrets.token_hex(nbytes=2)
        self.logger.info(f"self.USI: {self.USI}")

        self.logger.info(f"self.variable_keys: {self.variable_keys}")

        self._check_enviroment()

        # Checking parameters ====================================== >>

        self.logger.info("Checking parameters...")

        if not isinstance(data, pd.DataFrame):
            raise TypeError("The provided data must be of type pandas.DataFrame.")
        elif data.empty:
            raise ValueError("The provided data cannot be an empty dataframe.")

        if train_size <= 0 or train_size > 1:
            raise ValueError("train_size parameter has to be positive and not above 1.")

        # Checking target parameter
        if not self._is_unsupervised():
            if isinstance(target, str):
                if target not in data.columns:
                    raise ValueError(
                        "Invalid value for the target parameter. "
                        f"Column {target} not found in the data."
                    )
                self.target_param = target
            else:
                self.target_param = data.columns[target]

        # checking session_id
        if session_id is not None:
            if type(session_id) is not int:
                raise TypeError("session_id parameter must be an integer.")

        if profile_kwargs is None:
            profile_kwargs = {}
        elif not isinstance(profile_kwargs, dict):
            raise TypeError("profile_kwargs can only be a dict.")

        # stratify
        if data_split_stratify:
            if (
                type(data_split_stratify) is not list
                and type(data_split_stratify) is not bool
            ):
                raise TypeError(
                    "data_split_stratify parameter only accepts a bool or a list of strings."
                )

            if not data_split_shuffle:
                raise TypeError(
                    "data_split_stratify parameter requires data_split_shuffle to be set to True."
                )

        possible_fold_strategy = [
            "kfold",
            "stratifiedkfold",
            "groupkfold",
            "timeseries",
        ]
        possible_time_series_fold_strategies = ["expanding", "sliding", "rolling"]

        if self._ml_usecase != MLUsecase.TIME_SERIES:
            if not (
                fold_strategy in possible_fold_strategy
                or is_sklearn_cv_generator(fold_strategy)
            ):
                raise TypeError(
                    f"fold_strategy parameter must be either a scikit-learn compatible CV generator object or one of {', '.join(possible_fold_strategy)}."
                )
        elif self._ml_usecase == MLUsecase.TIME_SERIES:
            if not (
                fold_strategy in possible_time_series_fold_strategies
                or is_sklearn_cv_generator(fold_strategy)
            ):
                raise TypeError(
                    f"fold_strategy parameter must be either a sktime compatible CV generator object or one of '{', '.join(possible_time_series_fold_strategies)}'."
                )

        if fold_strategy == "groupkfold" and (
            fold_groups is None or len(fold_groups) == 0
        ):
            raise ValueError(
                "'groupkfold' fold_strategy requires 'fold_groups' to be a non-empty array-like object."
            )

        if isinstance(fold_groups, str):
            if fold_groups not in self.X.columns:
                raise ValueError(
                    f"Column {fold_groups} used for fold_groups is not present in the dataset."
                )

        # checking fold parameter
        if not isinstance(fold, int):
            raise TypeError("fold parameter only accepts integer value.")

        # checking fh parameter
        if not isinstance(fh, (int, list, np.ndarray)):
            raise TypeError(
                f"fh parameter accepts integer. list or np.array value. Provided values is {type(fh)}"
            )

        # fold_shuffle
        if type(fold_shuffle) is not bool:
            raise TypeError("fold_shuffle parameter only accepts True or False.")

        # log_plots
        if isinstance(log_plots, list):
            for i in log_plots:
                if i not in self._available_plots:
                    raise ValueError(
                        f"Incorrect value for log_plots '{i}'. Possible values are: {', '.join(self._available_plots.keys())}."
                    )
        elif type(log_plots) is not bool:
            raise TypeError("log_plots parameter must be a bool or a list.")

        # log_data
        if type(log_data) is not bool:
            raise TypeError("log_data parameter only accepts True or False.")

        # check transform_target
        if type(transform_target) is not bool:
            raise TypeError("transform_target parameter only accepts True or False.")

        # transform_target_method
        allowed_transform_target_method = ["box-cox", "yeo-johnson"]
        if transform_target_method not in allowed_transform_target_method:
            raise ValueError(
                f"transform_target_method parameter only accepts {', '.join(allowed_transform_target_method)}."
            )

        # Data preparation ========================================= >>

        # Standardize dataframe types to save memory
        self.data = df_shrink_dtypes(self.data)

        # Features to be ignored (are not read by self.dataset, self.X, etc...)
        self._ign_cols = ignore_features if ignore_features else []

        # Ordinal features
        if ordinal_features:
            check_features_exist(ordinal_features.keys(), self.X)
        else:
            ordinal_features = {}

        # Numerical features
        if numeric_features:
            check_features_exist(numeric_features, self.X)
        else:
            numeric_features = list(self.X.select_dtypes(include="number").columns)

        # Date features
        if date_features:
            check_features_exist(date_features, self.X)
        else:
            date_features = list(self.X.select_dtypes(include="datetime").columns)

        # Text features
        if text_features:
            check_features_exist(text_features, self.X)
        else:
            text_features = []

        # Categorical features
        if categorical_features:
            check_features_exist(categorical_features, self.X)
        else:
            # Default should exclude datetime and text columns
            categorical_features = [
                col for col in self.X.select_dtypes(include=["object", "category"]).columns
                if col not in date_features + text_features
            ]

        # Features to keep during all preprocessing
        keep_features = keep_features if keep_features else []

        if test_data is None:
            if self._ml_usecase == MLUsecase.TIME_SERIES:
                # TODO: Fix time series for new data properties!
                from sktime.forecasting.model_selection import (
                    temporal_train_test_split,
                )  # sktime is an optional dependency

                X_train, X_test, y_train, y_test = temporal_train_test_split(
                    X=self.X,
                    y=self.y,
                    fh=fh,  # if fh is provided it splits by it
                )

                y_train, y_test = pd.DataFrame(y_train), pd.DataFrame(y_test)

                self.data = pd.concat([X_train, X_test]).reset_index(drop=True)
                self.idx = [len(X_train), len(X_test)]

            else:
                train, test = train_test_split(
                    self.data,
                    test_size=1 - train_size,
                    stratify=get_columns_to_stratify_by(
                        self.X, self.y, data_split_stratify
                    ),
                    random_state=self.seed,
                    shuffle=data_split_shuffle,
                )
                self.data = pd.concat([train, test]).reset_index(drop=True)
                self.idx = (len(train), len(test))

        else:  # test_data is provided
            self.data = pd.concat([data, test_data]).reset_index(drop=True)
            self.idx = [len(data), len(test_data)]

        # Preprocessing ============================================ >>

        self.logger.info("Preparing preprocessing pipeline...")

        # Initialize empty pipeline
        self._internal_pipeline = InternalPipeline(steps=[("placeholder", None)])

        if preprocess:

            # Encode target variable =============================== >>

            if self.y.dtype.kind not in "ifu":
                self._internal_pipeline.steps.append(("label_encoder", LabelEncoder()))

            # Date feature engineering ============================= >>

            # TODO: Could be improved allowing the user to choose which features to add
            if date_features:
                self.logger.info("Extracting features from datetime columns")
                date_estimator = TransfomerWrapper(
                    transformer=ExtractDateTimeFeatures(),
                    include=date_features,
                )

                self._internal_pipeline.steps.append(
                    ("date_feature_extractor", date_estimator),
                )

            # Imputation =========================================== >>

            if self.data.isna().any().any():
                # Checking parameters
                num_dict = {"zero": "constant", "mean": "mean", "median": "median"}
                if numeric_imputation not in num_dict:
                    raise ValueError(
                        "Invalid value for the numeric_imputation parameter, "
                        f"got {numeric_imputation}. Possible values are "
                        f"{' '.join(num_dict)}."
                    )

                cat_dict = {"constant": "constant", "mode": "most_frequent"}
                if categorical_imputation not in cat_dict:
                    raise ValueError(
                        "Invalid value for the categorical_imputation "
                        f"parameter, got {categorical_imputation}. Possible "
                        f"values are {' '.join(cat_dict)}."
                    )

                if imputation_type == "simple":
                    self.logger.info("Setting up simple imputation")

                    num_estimator = TransfomerWrapper(
                        transformer=SimpleImputer(
                            strategy=num_dict[numeric_imputation],
                            fill_value=0,
                        ),
                        include=numeric_features,
                    )
                    cat_estimator = TransfomerWrapper(
                        transformer=SimpleImputer(
                            strategy=cat_dict[categorical_imputation],
                            fill_value="not_available",
                        ),
                        include=categorical_features,
                    )

                elif imputation_type == "iterative":
                    self.logger.info("Setting up iterative imputation")

                    # TODO: Fix iterative imputer for categorical columns

                    # Dict of all regressor models available
                    regressors = {k: v for k, v in get_regressors(self).items() if not v.is_special}

                    if isinstance(numeric_iterative_imputer, str):
                        if numeric_iterative_imputer not in regressors:
                            raise ValueError(
                                "Invalid value for the numeric_iterative_imputer "
                                f"parameter, got {numeric_iterative_imputer}. "
                                f"Allowed estimators are: {', '.join(regressors)}."
                            )
                        numeric_iterative_imputer = regressors[numeric_iterative_imputer].class_def()
                    elif not hasattr(numeric_iterative_imputer, "predict"):
                        raise ValueError(
                            "Invalid value for the numeric_iterative_imputer "
                            "parameter. The provided estimator does not adhere "
                            "to sklearn's API."
                        )

                    if isinstance(categorical_iterative_imputer, str):
                        if categorical_iterative_imputer not in regressors:
                            raise ValueError(
                                "Invalid value for the categorical_iterative_imputer "
                                "parameter, got {categorical_iterative_imputer}. "
                                f"Allowed estimators are: {', '.join(regressors)}."
                            )
                        categorical_iterative_imputer = regressors[categorical_iterative_imputer].class_def()
                    elif not hasattr(categorical_iterative_imputer, "predict"):
                        raise ValueError(
                            "Invalid value for the categorical_iterative_imputer "
                            "parameter. The provided estimator does not adhere "
                            "to sklearn's API."
                        )

                    num_estimator = TransfomerWrapper(
                        transformer=IterativeImputer(
                            estimator=numeric_iterative_imputer,
                            max_iter=iterative_imputation_iters,
                            random_state=self.seed,
                        ),
                        include=numeric_features,
                    )
                    cat_estimator = TransfomerWrapper(
                        transformer=IterativeImputer(
                            estimator=categorical_iterative_imputer,
                            max_iter=iterative_imputation_iters,
                            initial_strategy="most_frequent",
                            random_state=self.seed,
                        ),
                        include=categorical_features,
                    )
                else:
                    raise ValueError(
                        "Invalid value for the imputation_type parameter, got "
                        f"{imputation_type}. Possible values are: simple, iterative."
                    )

                self._internal_pipeline.steps.extend(
                    [
                        ("numerical_imputer", num_estimator),
                        ("categorical_imputer", cat_estimator),
                    ],
                )

            # Text embedding ======================================= >>

            if text_features:
                self.logger.info("Setting text embedding...")
                if text_features_method.lower() in ("bow", "tfidf", "tf-idf"):
                    embed_estimator = TransfomerWrapper(
                        transformer=EmbedTextFeatures(method=text_features_method),
                        include=text_features,
                    )
                else:
                    raise ValueError(
                        "Invalid value for the text_features_method "
                        "parameter. Choose between bow (Bag of Words) "
                        f"or tf-idf, got {text_features_method}."
                    )

                self._internal_pipeline.steps.append(
                    ("text_embedding", embed_estimator)
                )

            # Encoding ============================================= >>

            self.logger.info("Setting up encoding")

            # Select columns for different encoding types
            one_hot_cols, rest_cols = [], []
            for col in categorical_features:
                n_unique = self.X[col].nunique()
                if n_unique == 2:
                    ordinal_features[col] = list(self.X[col].dropna().unique())
                elif n_unique <= max_encoding_ohe:
                    one_hot_cols.append(col)
                else:
                    rest_cols.append(col)

            if ordinal_features:
                self.logger.info("Setting up encoding of ordinal features")

                # Check provided features and levels are correct
                mapping = {}
                for key, value in ordinal_features.items():
                    if self.X[key].nunique() != len(value):
                        raise ValueError(
                            "The levels passed to the ordinal_features parameter "
                            "doesn't match with the levels in the dataset."
                        )
                    for elem in value:
                        if elem not in self.X[key].unique():
                            raise ValueError(
                                f"Feature {key} doesn't contain the {elem} element."
                            )
                    mapping[key] = {v: i for i, v in enumerate(value)}

                    # Encoder always needs mapping of NaN value
                    if np.NaN not in mapping[key]:
                        mapping[key][np.NaN] = -1

                ord_estimator = TransfomerWrapper(
                    transformer=OrdinalEncoder(
                        mapping=[{"col": k, "mapping": val} for k, val in mapping.items()],
                        handle_missing="return_nan",
                        handle_unknown="value",
                    ),
                    include=list(ordinal_features.keys()),
                )

                self._internal_pipeline.steps.append(
                    ("ordinal_encoding", ord_estimator)
                )

            if categorical_features:
                self.logger.info("Setting up encoding of categorical features")

                if len(one_hot_cols) > 0:
                    onehot_estimator = TransfomerWrapper(
                        transformer=OneHotEncoder(
                            use_cat_names=True,
                            handle_missing="return_nan",
                            handle_unknown="value",
                        ),
                        include=one_hot_cols,
                    )

                    self._internal_pipeline.steps.append(
                        ("onehot_encoding", onehot_estimator)
                    )

                # Encode the rest of the categorical columns
                if len(rest_cols) > 0:
                    if not encoding_method:
                        encoding_method = LeaveOneOutEncoder(
                            handle_missing="return_nan",
                            handle_unknown="value",
                            random_state=self.seed,
                        )

                    rest_estimator = TransfomerWrapper(
                        transformer=encoding_method,
                        include=rest_cols,
                    )

                    self._internal_pipeline.steps.append(
                        ("rest_encoding", rest_estimator)
                    )

            # Transformation ======================================= >>

            if transformation:
                self.logger.info("Setting up column transformation")
                if transformation_method == "yeo-johnson":
                    transformation_estimator = PowerTransformer(
                        method="yeo-johnson", standardize=False, copy=True
                    )
                elif transformation_method == "quantile":
                    transformation_estimator = QuantileTransformer(
                        random_state=self.seed,
                        output_distribution="normal",
                    )
                else:
                    raise ValueError(
                        "Invalid value for the transformation_method parameter. "
                        "The value should be either yeo-johnson or quantile, "
                        f"got {transformation_method}."
                    )

                self._internal_pipeline.steps.append(
                    ("transformation", TransfomerWrapper(transformation_estimator))
                )

            # Normalization ======================================== >>

            if normalize:
                self.logger.info("Setting up feature normalization")
                norm_dict = {
                    "zscore": StandardScaler(),
                    "minmax": MinMaxScaler(),
                    "maxabs": MaxAbsScaler(),
                    "robust": RobustScaler(),
                }
                if normalize_method in norm_dict:
                    normalize_estimator = TransfomerWrapper(norm_dict[normalize_method])
                else:
                    raise ValueError(
                        "Invalid value for the normalize_method parameter, got "
                        f"{normalize_method}. Possible values are: {' '.join(norm_dict)}."
                    )

                self._internal_pipeline.steps.append(("normalize", normalize_estimator))

            # Low variance ========================================= >>

            if low_variance_threshold:
                self.logger.info("Setting up variance threshold")
                if low_variance_threshold < 0:
                    raise ValueError(
                        "Invalid value for the ignore_low_variance parameter. "
                        f"The value should be >0, got {low_variance_threshold}."
                    )
                else:
                    variance_estimator = TransfomerWrapper(
                        transformer=VarianceThreshold(low_variance_threshold),
                        exclude=keep_features,
                    )

                self._internal_pipeline.steps.append(
                    ("low_variance", variance_estimator)
                )

            # Remove multicollinearity ============================= >>

            if remove_multicollinearity:
                self.logger.info("Setting up removing multicollinearity")
                if 0 > multicollinearity_threshold or multicollinearity_threshold > 1:
                    raise ValueError(
                        "Invalid value for the multicollinearity_threshold "
                        "parameter. Value should lie between 0 and 1, got "
                        f"{multicollinearity_threshold}."
                    )

                multicollinearity = TransfomerWrapper(
                    transformer=RemoveMulticollinearity(multicollinearity_threshold),
                    exclude=keep_features,
                )

                self._internal_pipeline.steps.append(
                    ("remove_multicollinearity", multicollinearity)
                )

            # Bin numerical features =============================== >>

            if bin_numeric_features:
                self.logger.info("Setting up binning of numerical features")
                check_features_exist(bin_numeric_features, self.X)

                binning_estimator = TransfomerWrapper(
                    transformer=KBinsDiscretizer(encode="ordinal", strategy="kmeans"),
                    include=bin_numeric_features,
                )

                self._internal_pipeline.steps.append(
                    ("bin_numeric_features", binning_estimator)
                )

            # Remove outliers ====================================== >>

            if remove_outliers:
                self.logger.info("Setting up removing outliers")
                if outliers_method.lower() not in ("iforest", "ee", "lof"):
                    raise ValueError(
                        "Invalid value for the outliers_method parameter, "
                        f"got {outliers_method}. Possible values are: "
                        "'iforest', 'ee' or 'lof'."
                    )

                outliers = TransfomerWrapper(
                    RemoveOutliers(
                        method=outliers_method,
                        threshold=outliers_threshold,
                    ),
                )

                self._internal_pipeline.steps.append(
                    ("remove_outliers", outliers)
                )

            # Polynomial features ================================== >>

            if polynomial_features:
                self.logger.info("Setting up polynomial features")
                polynomial = TransfomerWrapper(
                    transformer=PolynomialFeatures(
                        degree=polynomial_degree,
                        interaction_only=False,
                        include_bias=False,
                        order="C",
                    ),
                )

                self._internal_pipeline.steps.append(
                    ("polynomial_features", polynomial)
                )

            # Balance the dataset ================================== >>

            if fix_imbalance:
                self.logger.info("Setting up imbalanced handling")
                if fix_imbalance_method is None:
                    balance_estimator = FixImbalancer(SMOTE())
                elif not hasattr(fix_imbalance_method, "fit_resample"):
                    raise ValueError(
                        "Invalid value for the fix_imbalance_method parameter. "
                        "The provided value must be a imblearn estimator, got "
                        f"{fix_imbalance_method.__class__.__name_}."
                    )
                else:
                    balance_estimator = FixImbalancer(fix_imbalance_method)

                balance_estimator = TransfomerWrapper(balance_estimator)
                self._internal_pipeline.steps.append(("balance", balance_estimator))

            # PCA ================================================== >>

            if pca:
                self.logger.info("Setting up PCA")
                if pca_components <= 0:
                    raise ValueError(
                        "Invalid value for the pca_components parameter. "
                        f"The value should be >0, got {pca_components}."
                    )
                elif pca_components <= 1:
                    pca_components = int(pca_components * self.X.shape[1])
                elif pca_components <= self.X.shape[1]:
                    pca_components = int(pca_components)
                else:
                    raise ValueError(
                        "Invalid value for the pca_components parameter. "
                        "The value should be smaller than the number of "
                        f"features, got {pca_components}."
                    )

                pca_dict = {
                    "linear": PCA(n_components=pca_components),
                    "kernel": KernelPCA(n_components=pca_components, kernel="rbf"),
                    "incremental": IncrementalPCA(n_components=pca_components),
                }
                if pca_method in pca_dict:
                    pca_estimator = TransfomerWrapper(
                        transformer=pca_dict[pca_method],
                        exclude=keep_features,
                    )
                else:
                    raise ValueError(
                        "Invalid value for the pca_method parameter, got "
                        f"{pca_method}. Possible values are: {' '.join(pca_dict)}."
                    )

                self._internal_pipeline.steps.append(("pca", pca_estimator))

            # Feature selection ==================================== >>

            if feature_selection:
                self.logger.info("Setting up feature selection...")

                if self._ml_usecase == MLUsecase.CLASSIFICATION:
                    func = get_classifiers
                else:
                    func = get_regressors

                models = {k: v for k, v in func(self).items() if not v.is_special}
                if isinstance(feature_selection_estimator, str):
                    if feature_selection_estimator not in models:
                        raise ValueError(
                            "Invalid value for the feature_selection_estimator "
                            f"parameter, got {feature_selection_estimator}. Allowed "
                            f"estimators are: {', '.join(models)}."
                        )
                    fs_estimator = models[feature_selection_estimator].class_def()
                elif not hasattr(feature_selection_estimator, "predict"):
                    raise ValueError(
                        "Invalid value for the feature_selection_estimator parameter. "
                        "The provided estimator does not adhere to sklearn's API."
                    )

                if feature_selection_method.lower() == "classic":
                    feature_selector = TransfomerWrapper(
                        transformer=SelectFromModel(
                            estimator=fs_estimator,
                            threshold=-np.inf,
                            max_features=n_features_to_select,
                        ),
                        exclude=keep_features,
                    )
                elif feature_selection_method.lower() == "sequential":
                    feature_selector = TransfomerWrapper(
                        transformer=SequentialFeatureSelector(
                            estimator=fs_estimator,
                            n_features_to_select=n_features_to_select,
                            n_jobs=self.n_jobs_param,
                        ),
                        exclude=keep_features,
                    )
                elif feature_selection_method.lower() == "boruta":
                    # TODO: Fix
                    feature_selector = TransfomerWrapper(
                        transformer=BorutaPy(
                            estimator=fs_estimator,
                            n_estimators="auto",
                        ),
                        exclude=keep_features,
                    )
                else:
                    raise ValueError(
                        "Invalid value for the feature_selection_method parameter, "
                        f"got {feature_selection_method}. Possible values are: "
                        "'classic' or 'boruta'."
                    )

                self._internal_pipeline.steps.append(
                    ("feature_selection", feature_selector)
                )

        # Custom transformers ====================================== >>

        if custom_pipeline:
            self.logger.info("Setting up custom pipeline")
            for name, estimator in normalize_custom_transformers(custom_pipeline):
                self._internal_pipeline.steps.append(
                    (name, TransfomerWrapper(estimator))
                )

        # Remove placeholder step
        if len(self._internal_pipeline) > 1:
            self._internal_pipeline.steps.pop(0)

        self.logger.info(f"Finished creating preprocessing pipeline.")
        self.logger.info(f"Pipeline: {self._internal_pipeline}")

        # Set up GPU usage ========================================= >>

        if self.gpu_param != "force" and type(self.gpu_param) is not bool:
            raise TypeError(
                f"Invalid value for the use_gpu parameter, got {self.gpu_param}. "
                "Possible values are: 'force', True or False."
            )

        cuml_version = None
        if self.gpu_param:
            self.logger.info("Set up GPU usage.")

            try:
                from cuml import __version__

                cuml_version = __version__
                self.logger.info(f"cuml=={cuml_version}")

                cuml_version = cuml_version.split(".")
                cuml_version = (int(cuml_version[0]), int(cuml_version[1]))
            except Exception:
                self.logger.warning("cuML not found")

            if cuml_version is None or not version.parse(cuml_version) >= version.parse(
                "0.15"
            ):
                message = f"cuML is outdated or not found. Required version is >=0.15, got {__version__}"
                if use_gpu == "force":
                    raise ImportError(message)
                else:
                    self.logger.warning(message)

        # Set up folding strategy ================================== >>

        self.logger.info("Set up folding strategy.")

        if fold_groups is not None:
            if isinstance(fold_groups, str):
                self.fold_groups_param = self.X[fold_groups]
            else:
                self.fold_groups_param = fold_groups
            if pd.isnull(self.fold_groups_param).any():
                raise ValueError(f"fold_groups cannot contain NaNs.")
        self.fold_shuffle_param = fold_shuffle

        if not self._is_unsupervised():
            fold_random_state = self.seed if self.fold_shuffle_param else None

            if self._ml_usecase != MLUsecase.TIME_SERIES:
                if fold_strategy == "kfold":
                    self.fold_generator = KFold(
                        self.fold_param,
                        random_state=fold_random_state,
                        shuffle=self.fold_shuffle_param,
                    )
                elif fold_strategy == "stratifiedkfold":
                    self.fold_generator = StratifiedKFold(
                        self.fold_param,
                        random_state=fold_random_state,
                        shuffle=self.fold_shuffle_param,
                    )
                elif fold_strategy == "groupkfold":
                    self.fold_generator = GroupKFold(self.fold_param)
                elif fold_strategy == "timeseries":
                    self.fold_generator = TimeSeriesSplit(self.fold_param)
                else:
                    self.fold_generator = fold_strategy

            elif self._ml_usecase == MLUsecase.TIME_SERIES:
                # Set splitter
                self.fold_generator = None
                if fold_strategy in possible_time_series_fold_strategies:
                    self.fold_strategy = fold_strategy  # save for use in methods later
                    self.fold_generator = self.get_fold_generator(fold=self.fold_param)
                else:
                    self.fold_generator = fold_strategy

                    # Number of folds
                    self.fold_param = fold_strategy.get_n_splits(y=self.y_train)

        if self._ml_usecase == MLUsecase.TIME_SERIES:
            from pycaret.internal.tests.time_series import (
                recommend_uppercase_d,
                recommend_lowercase_d,
            )

            self.white_noise = None
            wn_results = self.check_stats(test="white_noise")
            wn_values = wn_results.query("Property == 'White Noise'")["Value"]

            # There can be multiple lags values tested.
            # Checking the percent of lag values that indicate white noise
            percent_white_noise = sum(wn_values) / len(wn_values)
            if percent_white_noise == 0:
                self.white_noise = "No"
            elif percent_white_noise == 1.00:
                self.white_noise = "Yes"
            else:
                self.white_noise = "Maybe"

            self.lowercase_d = recommend_lowercase_d(data=self.y)
            # TODO: Should sp this overrise the self.seasonal_period since sp
            # will be used for all models and the same checks will need to be
            # done there as well
            sp = self.seasonal_period if self.seasonality_present else 1
            self.uppercase_d = (
                recommend_uppercase_d(data=self.y, sp=sp) if sp > 1 else 0
            )

        # Final display ============================================ >>

        self.logger.info("Creating final display dataframe.")

        if isinstance(numeric_iterative_imputer, str):
            num_imputer = numeric_iterative_imputer
        else:
            num_imputer = numeric_iterative_imputer.__class__.__name__

        if isinstance(categorical_iterative_imputer, str):
            cat_imputer = categorical_iterative_imputer
        else:
            cat_imputer = categorical_iterative_imputer.__class__.__name__

        display_container = self._get_setup_display(
            target_type="Multiclass" if self._is_multiclass() else "Binary",
            train_size=train_size,
            ordinal_features=len(ordinal_features) if ordinal_features else 0,
            numerical_features=len(numeric_features),
            categorical_features=len(categorical_features),
            date_features=len(date_features),
            text_features=len(text_features),
            ignore_features=len(self._ign_cols),
            keep_features=len(keep_features),
            missing_values=self.data.isna().sum().sum(),
            preprocess=preprocess,
            imputation_type=imputation_type,
            numeric_imputation=numeric_imputation,
            categorical_imputation=categorical_imputation,
            iterative_imputation_iters=iterative_imputation_iters,
            numeric_iterative_imputer=num_imputer,
            categorical_iterative_imputer=cat_imputer,
            text_features_method=text_features_method,
            max_encoding_ohe=max_encoding_ohe,
            encoding_method=encoding_method,
            transformation=transformation,
            transformation_method=transformation_method,
            normalize=normalize,
            normalize_method=normalize_method,
            low_variance_threshold=low_variance_threshold,
            remove_multicollinearity=remove_multicollinearity,
            multicollinearity_threshold=multicollinearity_threshold,
            remove_outliers=remove_outliers,
            outliers_threshold=outliers_threshold,
            polynomial_features=polynomial_features,
            polynomial_degree=polynomial_degree,
            fix_imbalance=fix_imbalance,
            fix_imbalance_method=fix_imbalance_method,
            pca=pca,
            pca_method=pca_method,
            pca_components=pca_components,
            feature_selection=feature_selection,
            feature_selection_method=feature_selection_method,
            feature_selection_estimator=feature_selection_estimator,
            n_features_to_select=n_features_to_select,
            custom_pipeline=custom_pipeline,
        )

        # Other attrs
        self.experiment__ = []
        self.master_model_container = []
        self._all_models, self._all_models_internal = self._get_models()
        self._all_metrics = self._get_metrics()

        if verbose:
            print(display_container)

        # Pandas profiling ========================================= >>

        if profile:
            if verbose:
                print("Setup Successfully Completed! Loading profile... Please Wait!")
            try:
                import pandas_profiling

                pf = pandas_profiling.ProfileReport(self.data, **profile_kwargs)
                display.display(pf, clear=True)
            except Exception as ex:
                print("Profiler Failed. No output to show, continue with modeling.")
                self.logger.error(
                    f"Data Failed with exception:\n {ex}\n"
                    "No output to show, continue with modeling."
                )

        # MLflow and wrap-up ======================================= >>

        self._set_up_mlflow(
            display_container,
            np.array(time.time() - runtime_start).round(2),
            log_profile,
            profile_kwargs,
            log_data,
            display,
        )

        self.logger.info(f"self.display_container: {len(self.display_container)}")
        self.logger.info(f"Pipeline: {str(self._internal_pipeline)}")
        self.logger.info("setup() successfully completed............................")

        # Reset pandas option
        pd.reset_option("display.max_rows")
        pd.reset_option("display.max_columns")

        self._setup_ran = True

        ## Disabling of certain metrics.
        ## NOTE: This must be run after _setup_ran has been set, else metrics can
        # not be retrieved.
        if (
            self._ml_usecase == MLUsecase.TIME_SERIES
            and len(self.fh) == 1
            and "r2" in self._get_metrics()
        ):
            # disable R2 metric if it exists in the metrics since R2 needs
            # at least 2 values
            self.remove_metric("R2")

        self.logger.info(
            f"self.master_model_container: {len(self.master_model_container)}"
        )
        self.logger.info(f"self.display_container: {len(self.display_container)}")

        self.logger.info(str(self._internal_pipeline))
        self.logger.info(
            "setup() successfully completed......................................"
        )

        gc.collect()

        return self

    def plot_model(
        self,
        estimator,
        plot: str = "auc",
        scale: float = 1,  # added in pycaret==2.1.0
        save: Union[str, bool] = False,
        fold: Optional[Union[int, Any]] = None,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        feature_name: Optional[str] = None,
        label: bool = False,
        use_train_data: bool = False,
        verbose: bool = True,
        system: bool = True,
        display: Optional[Display] = None,  # added in pycaret==2.2.0
        display_format: Optional[str] = None,
    ) -> str:

        """
        This function takes a trained model object and returns a plot based on the
        test / hold-out set. The process may require the model to be re-trained in
        certain cases. See list of plots supported below.

        Model must be created using create_model() or tune_model().

        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> experiment_name = setup(data = juice,  target = 'Purchase')
        >>> lr = create_model('lr')
        >>> plot_model(lr)

        This will return an AUC plot of a trained Logistic Regression model.

        Parameters
        ----------
        estimator : object, default = none
            A trained model object should be passed as an estimator.

        plot : str, default = auc
            Enter abbreviation of type of plot. The current list of plots supported are (Plot - Name):

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

        scale: float, default = 1
            The resolution scale of the figure.

        save: string or bool, default = False
            When set to True, Plot is saved as a 'png' file in current working directory.
            When a path destination is given, Plot is saved as a 'png' file the given path to the directory of choice.

        fold: integer or scikit-learn compatible CV generator, default = None
            Controls cross-validation used in certain plots. If None, will use the CV generator
            defined in setup(). If integer, will use KFold CV with that many folds.
            When cross_validation is False, this parameter is ignored.

        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.

        groups: str or array-like, with shape (n_samples,), default = None
            Optional Group labels for the samples used while splitting the dataset into train/test set.
            If string is passed, will use the data column with that name as the groups.
            Only used if a group based cross-validation generator is used (eg. GroupKFold).
            If None, will use the value set in fold_groups parameter in setup().

        verbose: bool, default = True
            Progress bar not shown when verbose set to False.

        system: bool, default = True
            Must remain True all times. Only to be changed by internal functions.

        display_format: str, default = None
            To display plots in Streamlit (https://www.streamlit.io/), set this to 'streamlit'.
            Currently, not all plots are supported.

        Returns
        -------
        Visual_Plot
            Prints the visual plot.
        str:
            If save parameter is True, will return the name of the saved file.

        Warnings
        --------
        -  'svm' and 'ridge' doesn't support the predict_proba method. As such, AUC and
            calibration plots are not available for these estimators.

        -   When the 'max_features' parameter of a trained model object is not equal to
            the number of samples in training set, the 'rfe' plot is not available.

        -   'calibration', 'threshold', 'manifold' and 'rfe' plots are not available for
            multiclass problems.


        """

        function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

        self.logger.info("Initializing plot_model()")
        self.logger.info(f"plot_model({function_params_str})")

        self.logger.info("Checking exceptions")

        if not fit_kwargs:
            fit_kwargs = {}

        if not hasattr(estimator, "fit"):
            raise ValueError(
                f"Estimator {estimator} does not have the required fit() method."
            )

        if plot not in self._available_plots:
            raise ValueError(
                "Plot Not Available. Please see docstring for list of available Plots."
            )

        # checking display_format parameter
        plot_formats = [None, "streamlit"]

        if display_format not in plot_formats:
            raise ValueError("display_format can only be None or 'streamlit'.")

        if display_format == "streamlit":
            try:
                import streamlit as st
            except ImportError:
                raise ImportError(
                    "It appears that streamlit is not installed. Do: pip install hpbandster ConfigSpace"
                )

        # multiclass plot exceptions:
        multiclass_not_available = ["calibration", "threshold", "manifold", "rfe"]
        if self._is_multiclass():
            if plot in multiclass_not_available:
                raise ValueError(
                    "Plot Not Available for multiclass problems. Please see docstring for list of available Plots."
                )

        # exception for CatBoost
        # if "CatBoostClassifier" in str(type(estimator)):
        #    raise ValueError(
        #    "CatBoost estimator is not compatible with plot_model function, try using Catboost with interpret_model instead."
        # )

        # checking for auc plot
        if not hasattr(estimator, "predict_proba") and plot == "auc":
            raise TypeError(
                "AUC plot not available for estimators with no predict_proba attribute."
            )

        # checking for auc plot
        if not hasattr(estimator, "predict_proba") and plot == "auc":
            raise TypeError(
                "AUC plot not available for estimators with no predict_proba attribute."
            )

        # checking for calibration plot
        if not hasattr(estimator, "predict_proba") and plot == "calibration":
            raise TypeError(
                "Calibration plot not available for estimators with no predict_proba attribute."
            )

        def is_tree(e):
            from sklearn.ensemble._forest import BaseForest
            from sklearn.tree import BaseDecisionTree

            if "final_estimator" in e.get_params():
                e = e.final_estimator
            if "base_estimator" in e.get_params():
                e = e.base_estimator
            if isinstance(e, BaseForest) or isinstance(e, BaseDecisionTree):
                return True

        # checking for calibration plot
        if plot == "tree" and not is_tree(estimator):
            raise TypeError(
                "Decision Tree plot is only available for scikit-learn Decision Trees and Forests, Ensemble models using those or Stacked models using those as meta (final) estimators."
            )

        # checking for feature plot
        if not (
            hasattr(estimator, "coef_") or hasattr(estimator, "feature_importances_")
        ) and (plot == "feature" or plot == "feature_all" or plot == "rfe"):
            raise TypeError(
                "Feature Importance and RFE plots not available for estimators that doesnt support coef_ or feature_importances_ attribute."
            )

        # checking fold parameter
        if fold is not None and not (
            type(fold) is int or is_sklearn_cv_generator(fold)
        ):
            raise TypeError(
                "fold parameter must be either None, an integer or a scikit-learn compatible CV generator object."
            )

        if type(label) is not bool:
            raise TypeError("Label parameter only accepts True or False.")

        if type(use_train_data) is not bool:
            raise TypeError("use_train_data parameter only accepts True or False.")

        if feature_name is not None and type(feature_name) is not str:
            raise TypeError(
                "feature parameter must be string containing column name of dataset."
            )

        """

        ERROR HANDLING ENDS HERE

        """

        cv = self._get_cv_splitter(fold)

        groups = self._get_groups(groups)

        if not display:
            progress_args = {"max": 5}
            display = Display(
                verbose=verbose, html_param=self.html_param, progress_args=progress_args
            )
            display.display_progress()

        self.logger.info("Preloading libraries")
        # pre-load libraries
        import matplotlib.pyplot as plt

        np.random.seed(self.seed)

        display.move_progress()

        # defining estimator as model locally
        # deepcopy instead of clone so we have a fitted estimator
        if isinstance(estimator, InternalPipeline):
            estimator = estimator.steps[-1][1]
        estimator = deepcopy(estimator)
        model = estimator

        display.move_progress()

        # plots used for logging (controlled through plots_log_param)
        # AUC, #Confusion Matrix and #Feature Importance

        self.logger.info("Copying training dataset")

        # Storing X_train and y_train in data_X and data_y parameter
        data_X = self.X_train.copy()
        if not self._is_unsupervised():
            data_y = self.y_train.copy()

        # reset index
        data_X.reset_index(drop=True, inplace=True)
        if not self._is_unsupervised():
            data_y.reset_index(drop=True, inplace=True)  # type: ignore

            self.logger.info("Copying test dataset")

            # Storing X_train and y_train in data_X and data_y parameter
            test_X = self.X_train.copy() if use_train_data else self.X_test.copy()
            test_y = self.y_train.copy() if use_train_data else self.y_test.copy()

            # reset index
            test_X.reset_index(drop=True, inplace=True)
            test_y.reset_index(drop=True, inplace=True)

        self.logger.info(f"Plot type: {plot}")
        plot_name = self._available_plots[plot]
        display.move_progress()

        # yellowbrick workaround start
        import yellowbrick.utils.helpers
        import yellowbrick.utils.types

        # yellowbrick workaround end

        model_name = self._get_model_name(model)
        plot_filename = f"{plot_name}.png"
        with patch(
            "yellowbrick.utils.types.is_estimator",
            pycaret.internal.patches.yellowbrick.is_estimator,
        ):
            with patch(
                "yellowbrick.utils.helpers.is_estimator",
                pycaret.internal.patches.yellowbrick.is_estimator,
            ):
                with estimator_pipeline(
                    self._internal_pipeline, model
                ) as pipeline_with_model:
                    fit_kwargs = get_pipeline_fit_kwargs(
                        pipeline_with_model, fit_kwargs
                    )

                    _base_dpi = 100

                    def residuals_interactive():
                        from pycaret.internal.plots.residual_plots import (
                            InteractiveResidualsPlot,
                        )

                        resplots = InteractiveResidualsPlot(
                            x=data_X,
                            y=data_y,
                            x_test=test_X,
                            y_test=test_y,
                            model=pipeline_with_model,
                            display=display,
                        )

                        display.clear_output()
                        if system:
                            resplots.show()

                        plot_filename = f"{plot_name}.html"

                        if save:
                            if not isinstance(save, bool):
                                plot_filename = os.path.join(save, plot_filename)
                            else:
                                plot_filename = plot
                            self.logger.info(f"Saving '{plot_filename}'")
                            resplots.write_html(plot_filename)

                        self.logger.info("Visual Rendered Successfully")
                        return plot_filename

                    def cluster():
                        self.logger.info(
                            "SubProcess assign_model() called =================================="
                        )
                        b = self.assign_model(  # type: ignore
                            pipeline_with_model, verbose=False, transformation=True
                        ).reset_index(drop=True)
                        self.logger.info(
                            "SubProcess assign_model() end =================================="
                        )
                        cluster = b["Cluster"].values
                        b.drop("Cluster", axis=1, inplace=True)
                        b = pd.get_dummies(b)  # casting categorical variable

                        from sklearn.decomposition import PCA

                        pca = PCA(n_components=2, random_state=self.seed)
                        self.logger.info("Fitting PCA()")
                        pca_ = pca.fit_transform(b)
                        pca_ = pd.DataFrame(pca_)
                        pca_ = pca_.rename(columns={0: "PCA1", 1: "PCA2"})
                        pca_["Cluster"] = cluster

                        if feature_name is not None:
                            pca_["Feature"] = self.data[feature_name]
                        else:
                            pca_["Feature"] = self.data[
                                self.data.columns[0]
                            ]

                        if label:
                            pca_["Label"] = pca_["Feature"]

                        """
                        sorting
                        """

                        self.logger.info("Sorting dataframe")

                        print(pca_["Cluster"])

                        clus_num = [int(i.split()[1]) for i in pca_["Cluster"]]

                        pca_["cnum"] = clus_num
                        pca_.sort_values(by="cnum", inplace=True)

                        """
                        sorting ends
                        """

                        display.clear_output()

                        self.logger.info("Rendering Visual")

                        if label:
                            fig = px.scatter(
                                pca_,
                                x="PCA1",
                                y="PCA2",
                                text="Label",
                                color="Cluster",
                                opacity=0.5,
                            )
                        else:
                            fig = px.scatter(
                                pca_,
                                x="PCA1",
                                y="PCA2",
                                hover_data=["Feature"],
                                color="Cluster",
                                opacity=0.5,
                            )

                        fig.update_traces(textposition="top center")
                        fig.update_layout(plot_bgcolor="rgb(240,240,240)")

                        fig.update_layout(
                            height=600 * scale, title_text="2D Cluster PCA Plot"
                        )

                        plot_filename = f"{plot_name}.html"

                        if save:
                            if not isinstance(save, bool):
                                plot_filename = os.path.join(save, plot_filename)
                            else:
                                plot_filename = plot
                            self.logger.info(f"Saving '{plot_filename}'")
                            fig.write_html(plot_filename)

                        elif system:
                            if display_format == "streamlit":
                                st.write(fig)
                            else:
                                fig.show()

                        self.logger.info("Visual Rendered Successfully")
                        return plot_filename

                    def umap():
                        self.logger.info(
                            "SubProcess assign_model() called =================================="
                        )
                        b = self.assign_model(  # type: ignore
                            model, verbose=False, transformation=True, score=False
                        ).reset_index(drop=True)
                        self.logger.info(
                            "SubProcess assign_model() end =================================="
                        )

                        label = pd.DataFrame(b["Anomaly"])
                        b.dropna(axis=0, inplace=True)  # droping rows with NA's
                        b.drop(["Anomaly"], axis=1, inplace=True)

                        import umap

                        reducer = umap.UMAP()
                        self.logger.info("Fitting UMAP()")
                        embedding = reducer.fit_transform(b)
                        X = pd.DataFrame(embedding)

                        import plotly.express as px

                        df = X
                        df["Anomaly"] = label

                        if feature_name is not None:
                            df["Feature"] = self.data[feature_name]
                        else:
                            df["Feature"] = self.data[
                                self.data.columns[0]
                            ]

                        display.clear_output()

                        self.logger.info("Rendering Visual")

                        fig = px.scatter(
                            df,
                            x=0,
                            y=1,
                            color="Anomaly",
                            title="uMAP Plot for Outliers",
                            hover_data=["Feature"],
                            opacity=0.7,
                            width=900 * scale,
                            height=800 * scale,
                        )
                        plot_filename = f"{plot_name}.html"

                        if save:
                            if not isinstance(save, bool):
                                plot_filename = os.path.join(save, plot_filename)
                            else:
                                plot_filename = plot
                            self.logger.info(f"Saving '{plot_filename}'")
                            fig.write_html(f"{plot_filename}")

                        elif system:
                            if display_format == "streamlit":
                                st.write(fig)
                            else:
                                fig.show()

                        self.logger.info("Visual Rendered Successfully")
                        return plot_filename

                    def tsne():
                        if self._ml_usecase == MLUsecase.CLUSTERING:
                            return _tsne_clustering()
                        else:
                            return _tsne_anomaly()

                    def _tsne_anomaly():
                        self.logger.info(
                            "SubProcess assign_model() called =================================="
                        )
                        b = self.assign_model(  # type: ignore
                            model, verbose=False, transformation=True, score=False
                        ).reset_index(drop=True)
                        self.logger.info(
                            "SubProcess assign_model() end =================================="
                        )
                        cluster = b["Anomaly"].values
                        b.dropna(axis=0, inplace=True)  # droping rows with NA's
                        b.drop("Anomaly", axis=1, inplace=True)

                        self.logger.info(
                            "Getting dummies to cast categorical variables"
                        )

                        from sklearn.manifold import TSNE

                        self.logger.info("Fitting TSNE()")
                        X_embedded = TSNE(n_components=3).fit_transform(b)

                        X = pd.DataFrame(X_embedded)
                        X["Anomaly"] = cluster
                        if feature_name is not None:
                            X["Feature"] = self.data[feature_name]
                        else:
                            X["Feature"] = self.data[
                                self.data.columns[0]
                            ]

                        df = X

                        display.clear_output()

                        self.logger.info("Rendering Visual")

                        if label:
                            fig = px.scatter_3d(
                                df,
                                x=0,
                                y=1,
                                z=2,
                                text="Feature",
                                color="Anomaly",
                                title="3d TSNE Plot for Outliers",
                                opacity=0.7,
                                width=900 * scale,
                                height=800 * scale,
                            )
                        else:
                            fig = px.scatter_3d(
                                df,
                                x=0,
                                y=1,
                                z=2,
                                hover_data=["Feature"],
                                color="Anomaly",
                                title="3d TSNE Plot for Outliers",
                                opacity=0.7,
                                width=900 * scale,
                                height=800 * scale,
                            )

                        plot_filename = f"{plot_name}.html"

                        if save:
                            if not isinstance(save, bool):
                                plot_filename = os.path.join(save, plot_filename)
                            else:
                                plot_filename = plot
                            self.logger.info(f"Saving '{plot_filename}'")
                            fig.write_html(f"{plot_filename}")

                        elif system:
                            if display_format == "streamlit":
                                st.write(fig)
                            else:
                                fig.show()

                        self.logger.info("Visual Rendered Successfully")
                        return plot_filename

                    def _tsne_clustering():
                        self.logger.info(
                            "SubProcess assign_model() called =================================="
                        )
                        b = self.assign_model(  # type: ignore
                            pipeline_with_model,
                            verbose=False,
                            score=False,
                            transformation=True,
                        ).reset_index(drop=True)
                        self.logger.info(
                            "SubProcess assign_model() end =================================="
                        )

                        cluster = b["Cluster"].values
                        b.drop("Cluster", axis=1, inplace=True)

                        from sklearn.manifold import TSNE

                        self.logger.info("Fitting TSNE()")
                        X_embedded = TSNE(
                            n_components=3, random_state=self.seed
                        ).fit_transform(b)
                        X_embedded = pd.DataFrame(X_embedded)
                        X_embedded["Cluster"] = cluster

                        if feature_name is not None:
                            X_embedded["Feature"] = self.data[
                                feature_name
                            ]
                        else:
                            X_embedded["Feature"] = self.data[
                                data_X.columns[0]
                            ]

                        if label:
                            X_embedded["Label"] = X_embedded["Feature"]

                        """
                        sorting
                        """
                        self.logger.info("Sorting dataframe")

                        clus_num = [int(i.split()[1]) for i in X_embedded["Cluster"]]

                        X_embedded["cnum"] = clus_num
                        X_embedded.sort_values(by="cnum", inplace=True)

                        """
                        sorting ends
                        """

                        df = X_embedded

                        display.clear_output()

                        self.logger.info("Rendering Visual")

                        if label:

                            fig = px.scatter_3d(
                                df,
                                x=0,
                                y=1,
                                z=2,
                                color="Cluster",
                                title="3d TSNE Plot for Clusters",
                                text="Label",
                                opacity=0.7,
                                width=900 * scale,
                                height=800 * scale,
                            )

                        else:
                            fig = px.scatter_3d(
                                df,
                                x=0,
                                y=1,
                                z=2,
                                color="Cluster",
                                title="3d TSNE Plot for Clusters",
                                hover_data=["Feature"],
                                opacity=0.7,
                                width=900 * scale,
                                height=800 * scale,
                            )

                        plot_filename = f"{plot_name}.html"

                        if save:
                            if not isinstance(save, bool):
                                plot_filename = os.path.join(save, plot_filename)
                            else:
                                plot_filename = plot
                            self.logger.info(f"Saving '{plot_filename}'")
                            fig.write_html(f"{plot_filename}")

                        elif system:
                            if display_format == "streamlit":
                                st.write(fig)
                            else:
                                fig.show()

                        self.logger.info("Visual Rendered Successfully")
                        return plot_filename

                    def distribution():
                        self.logger.info(
                            "SubProcess assign_model() called =================================="
                        )
                        d = self.assign_model(  # type: ignore
                            pipeline_with_model, verbose=False
                        ).reset_index(drop=True)
                        self.logger.info(
                            "SubProcess assign_model() end =================================="
                        )

                        """
                        sorting
                        """
                        self.logger.info("Sorting dataframe")

                        clus_num = []
                        for i in d.Cluster:
                            a = int(i.split()[1])
                            clus_num.append(a)

                        d["cnum"] = clus_num
                        d.sort_values(by="cnum", inplace=True)
                        d.reset_index(inplace=True, drop=True)

                        clus_label = []
                        for i in d.cnum:
                            a = "Cluster " + str(i)
                            clus_label.append(a)

                        d.drop(["Cluster", "cnum"], inplace=True, axis=1)
                        d["Cluster"] = clus_label

                        """
                        sorting ends
                        """

                        if feature_name is None:
                            x_col = "Cluster"
                        else:
                            x_col = feature_name

                        display.clear_output()

                        self.logger.info("Rendering Visual")

                        fig = px.histogram(
                            d,
                            x=x_col,
                            color="Cluster",
                            marginal="box",
                            opacity=0.7,
                            hover_data=d.columns,
                        )

                        fig.update_layout(height=600 * scale,)

                        plot_filename = f"{plot_name}.html"

                        if save:
                            if not isinstance(save, bool):
                                plot_filename = os.path.join(save, plot_filename)
                            else:
                                plot_filename = plot
                            self.logger.info(f"Saving '{plot_filename}'")
                            fig.write_html(f"{plot_filename}")

                        elif system:
                            if display_format == "streamlit":
                                st.write(fig)
                            else:
                                fig.show()

                        self.logger.info("Visual Rendered Successfully")
                        return plot_filename

                    def elbow():
                        try:
                            from yellowbrick.cluster import KElbowVisualizer

                            visualizer = KElbowVisualizer(
                                pipeline_with_model, timings=False
                            )
                            show_yellowbrick_plot(
                                visualizer=visualizer,
                                X_train=data_X,
                                y_train=None,
                                X_test=None,
                                y_test=None,
                                name=plot_name,
                                handle_test="",
                                scale=scale,
                                save=save,
                                fit_kwargs=fit_kwargs,
                                groups=groups,
                                display=display,
                                display_format=display_format,
                            )

                        except:
                            self.logger.error("Elbow plot failed. Exception:")
                            self.logger.error(traceback.format_exc())
                            raise TypeError("Plot Type not supported for this model.")

                    def silhouette():
                        from yellowbrick.cluster import SilhouetteVisualizer

                        try:
                            visualizer = SilhouetteVisualizer(
                                pipeline_with_model, colors="yellowbrick"
                            )
                            show_yellowbrick_plot(
                                visualizer=visualizer,
                                X_train=data_X,
                                y_train=None,
                                X_test=None,
                                y_test=None,
                                name=plot_name,
                                handle_test="",
                                scale=scale,
                                save=save,
                                fit_kwargs=fit_kwargs,
                                groups=groups,
                                display=display,
                                display_format=display_format,
                            )
                        except:
                            self.logger.error("Silhouette plot failed. Exception:")
                            self.logger.error(traceback.format_exc())
                            raise TypeError("Plot Type not supported for this model.")

                    def distance():
                        from yellowbrick.cluster import InterclusterDistance

                        try:
                            visualizer = InterclusterDistance(pipeline_with_model)
                            show_yellowbrick_plot(
                                visualizer=visualizer,
                                X_train=data_X,
                                y_train=None,
                                X_test=None,
                                y_test=None,
                                name=plot_name,
                                handle_test="",
                                scale=scale,
                                save=save,
                                fit_kwargs=fit_kwargs,
                                groups=groups,
                                display=display,
                                display_format=display_format,
                            )
                        except:
                            self.logger.error("Distance plot failed. Exception:")
                            self.logger.error(traceback.format_exc())
                            raise TypeError("Plot Type not supported for this model.")

                    def residuals():

                        from yellowbrick.regressor import ResidualsPlot

                        visualizer = ResidualsPlot(pipeline_with_model)
                        show_yellowbrick_plot(
                            visualizer=visualizer,
                            X_train=data_X,
                            y_train=data_y,
                            X_test=test_X,
                            y_test=test_y,
                            name=plot_name,
                            scale=scale,
                            save=save,
                            fit_kwargs=fit_kwargs,
                            groups=groups,
                            display=display,
                            display_format=display_format,
                        )

                    def auc():

                        from yellowbrick.classifier import ROCAUC

                        visualizer = ROCAUC(pipeline_with_model)
                        show_yellowbrick_plot(
                            visualizer=visualizer,
                            X_train=data_X,
                            y_train=data_y,
                            X_test=test_X,
                            y_test=test_y,
                            name=plot_name,
                            scale=scale,
                            save=save,
                            fit_kwargs=fit_kwargs,
                            groups=groups,
                            display=display,
                            display_format=display_format,
                        )

                    def threshold():

                        from yellowbrick.classifier import DiscriminationThreshold

                        visualizer = DiscriminationThreshold(
                            pipeline_with_model, random_state=self.seed
                        )
                        show_yellowbrick_plot(
                            visualizer=visualizer,
                            X_train=data_X,
                            y_train=data_y,
                            X_test=test_X,
                            y_test=test_y,
                            name=plot_name,
                            scale=scale,
                            save=save,
                            fit_kwargs=fit_kwargs,
                            groups=groups,
                            display=display,
                            display_format=display_format,
                        )

                    def pr():

                        from yellowbrick.classifier import PrecisionRecallCurve

                        visualizer = PrecisionRecallCurve(
                            pipeline_with_model, random_state=self.seed
                        )
                        show_yellowbrick_plot(
                            visualizer=visualizer,
                            X_train=data_X,
                            y_train=data_y,
                            X_test=test_X,
                            y_test=test_y,
                            name=plot_name,
                            scale=scale,
                            save=save,
                            fit_kwargs=fit_kwargs,
                            groups=groups,
                            display=display,
                            display_format=display_format,
                        )

                    def confusion_matrix():

                        from yellowbrick.classifier import ConfusionMatrix

                        visualizer = ConfusionMatrix(
                            pipeline_with_model,
                            random_state=self.seed,
                            fontsize=15,
                            cmap="Greens",
                        )
                        show_yellowbrick_plot(
                            visualizer=visualizer,
                            X_train=data_X,
                            y_train=data_y,
                            X_test=test_X,
                            y_test=test_y,
                            name=plot_name,
                            scale=scale,
                            save=save,
                            fit_kwargs=fit_kwargs,
                            groups=groups,
                            display=display,
                            display_format=display_format,
                        )

                    def error():

                        if self._ml_usecase == MLUsecase.CLASSIFICATION:
                            from yellowbrick.classifier import ClassPredictionError

                            visualizer = ClassPredictionError(
                                pipeline_with_model, random_state=self.seed
                            )

                        elif self._ml_usecase == MLUsecase.REGRESSION:
                            from yellowbrick.regressor import PredictionError

                            visualizer = PredictionError(
                                pipeline_with_model, random_state=self.seed
                            )

                        show_yellowbrick_plot(
                            visualizer=visualizer,  # type: ignore
                            X_train=data_X,
                            y_train=data_y,
                            X_test=test_X,
                            y_test=test_y,
                            name=plot_name,
                            scale=scale,
                            save=save,
                            fit_kwargs=fit_kwargs,
                            groups=groups,
                            display=display,
                            display_format=display_format,
                        )

                    def cooks():

                        from yellowbrick.regressor import CooksDistance

                        visualizer = CooksDistance()
                        show_yellowbrick_plot(
                            visualizer=visualizer,
                            X_train=self.X,
                            y_train=self.y,
                            X_test=test_X,
                            y_test=test_y,
                            name=plot_name,
                            scale=scale,
                            save=save,
                            fit_kwargs=fit_kwargs,
                            handle_test="",
                            groups=groups,
                            display=display,
                            display_format=display_format,
                        )

                    def class_report():

                        from yellowbrick.classifier import ClassificationReport

                        visualizer = ClassificationReport(
                            pipeline_with_model, random_state=self.seed, support=True
                        )
                        show_yellowbrick_plot(
                            visualizer=visualizer,
                            X_train=data_X,
                            y_train=data_y,
                            X_test=test_X,
                            y_test=test_y,
                            name=plot_name,
                            scale=scale,
                            save=save,
                            fit_kwargs=fit_kwargs,
                            groups=groups,
                            display=display,
                            display_format=display_format,
                        )

                    def boundary():

                        from sklearn.decomposition import PCA
                        from sklearn.preprocessing import StandardScaler
                        from yellowbrick.contrib.classifier import DecisionViz

                        data_X_transformed = data_X.select_dtypes(include="float32")
                        test_X_transformed = test_X.select_dtypes(include="float32")
                        self.logger.info("Fitting StandardScaler()")
                        data_X_transformed = StandardScaler().fit_transform(
                            data_X_transformed
                        )
                        test_X_transformed = StandardScaler().fit_transform(
                            test_X_transformed
                        )
                        pca = PCA(n_components=2, random_state=self.seed)
                        self.logger.info("Fitting PCA()")
                        data_X_transformed = pca.fit_transform(data_X_transformed)
                        test_X_transformed = pca.fit_transform(test_X_transformed)

                        data_y_transformed = np.array(data_y)
                        test_y_transformed = np.array(test_y)

                        viz_ = DecisionViz(pipeline_with_model)
                        show_yellowbrick_plot(
                            visualizer=viz_,
                            X_train=data_X_transformed,
                            y_train=data_y_transformed,
                            X_test=test_X_transformed,
                            y_test=test_y_transformed,
                            name=plot_name,
                            scale=scale,
                            handle_test="draw",
                            save=save,
                            fit_kwargs=fit_kwargs,
                            groups=groups,
                            display=display,
                            features=["Feature One", "Feature Two"],
                            classes=["A", "B"],
                            display_format=display_format,
                        )

                    def rfe():

                        from yellowbrick.model_selection import RFECV

                        visualizer = RFECV(pipeline_with_model, cv=cv)
                        show_yellowbrick_plot(
                            visualizer=visualizer,
                            X_train=data_X,
                            y_train=data_y,
                            X_test=test_X,
                            y_test=test_y,
                            handle_test="",
                            name=plot_name,
                            scale=scale,
                            save=save,
                            fit_kwargs=fit_kwargs,
                            groups=groups,
                            display=display,
                            display_format=display_format,
                        )

                    def learning():

                        from yellowbrick.model_selection import LearningCurve

                        sizes = np.linspace(0.3, 1.0, 10)
                        visualizer = LearningCurve(
                            pipeline_with_model,
                            cv=cv,
                            train_sizes=sizes,
                            n_jobs=self._gpu_n_jobs_param,
                            random_state=self.seed,
                        )
                        show_yellowbrick_plot(
                            visualizer=visualizer,
                            X_train=data_X,
                            y_train=data_y,
                            X_test=test_X,
                            y_test=test_y,
                            handle_test="",
                            name=plot_name,
                            scale=scale,
                            save=save,
                            fit_kwargs=fit_kwargs,
                            groups=groups,
                            display=display,
                            display_format=display_format,
                        )

                    def lift():

                        display.move_progress()
                        self.logger.info(
                            "Generating predictions / predict_proba on X_test"
                        )
                        with fit_if_not_fitted(
                            pipeline_with_model,
                            data_X,
                            data_y,
                            groups=groups,
                            **fit_kwargs,
                        ) as fitted_pipeline_with_model:
                            y_test__ = test_y
                            predict_proba__ = fitted_pipeline_with_model.predict_proba(
                                test_X
                            )
                        display.move_progress()
                        display.move_progress()
                        display.clear_output()
                        with MatplotlibDefaultDPI(
                            base_dpi=_base_dpi, scale_to_set=scale
                        ):
                            fig = skplt.metrics.plot_lift_curve(
                                y_test__, predict_proba__, figsize=(10, 6)
                            )
                            if save:
                                plot_filename = f"{plot_name}.png"
                                if not isinstance(save, bool):
                                    plot_filename = os.path.join(save, plot_filename)
                                self.logger.info(f"Saving '{plot_filename}'")
                                plt.savefig(plot_filename, bbox_inches="tight")
                            elif system:
                                plt.show()
                            plt.close()

                        self.logger.info("Visual Rendered Successfully")

                    def gain():

                        display.move_progress()
                        self.logger.info(
                            "Generating predictions / predict_proba on X_test"
                        )
                        with fit_if_not_fitted(
                            pipeline_with_model,
                            data_X,
                            data_y,
                            groups=groups,
                            **fit_kwargs,
                        ) as fitted_pipeline_with_model:
                            y_test__ = test_y
                            predict_proba__ = fitted_pipeline_with_model.predict_proba(
                                test_X
                            )
                        display.move_progress()
                        display.move_progress()
                        display.clear_output()
                        with MatplotlibDefaultDPI(
                            base_dpi=_base_dpi, scale_to_set=scale
                        ):
                            fig = skplt.metrics.plot_cumulative_gain(
                                y_test__, predict_proba__, figsize=(10, 6)
                            )
                            if save:
                                plot_filename = f"{plot_name}.png"
                                if not isinstance(save, bool):
                                    plot_filename = os.path.join(save, plot_filename)
                                self.logger.info(f"Saving '{plot_filename}'")
                                plt.savefig(plot_filename, bbox_inches="tight")
                            elif system:
                                plt.show()
                            plt.close()

                        self.logger.info("Visual Rendered Successfully")

                    def manifold():

                        from yellowbrick.features import Manifold

                        data_X_transformed = data_X.select_dtypes(include="float32")
                        visualizer = Manifold(manifold="tsne", random_state=self.seed)
                        show_yellowbrick_plot(
                            visualizer=visualizer,
                            X_train=data_X_transformed,
                            y_train=data_y,
                            X_test=test_X,
                            y_test=test_y,
                            handle_train="fit_transform",
                            handle_test="",
                            name=plot_name,
                            scale=scale,
                            save=save,
                            fit_kwargs=fit_kwargs,
                            groups=groups,
                            display=display,
                            display_format=display_format,
                        )

                    def tree():

                        from sklearn.base import is_classifier
                        from sklearn.model_selection import check_cv
                        from sklearn.tree import plot_tree

                        is_stacked_model = False
                        is_ensemble_of_forests = False

                        tree_estimator = pipeline_with_model.steps[-1][1]

                        if "final_estimator" in tree_estimator.get_params():
                            tree_estimator = tree_estimator.final_estimator
                            is_stacked_model = True

                        if (
                            "base_estimator" in tree_estimator.get_params()
                            and "n_estimators"
                            in tree_estimator.base_estimator.get_params()
                        ):
                            n_estimators = (
                                tree_estimator.get_params()["n_estimators"]
                                * tree_estimator.base_estimator.get_params()[
                                    "n_estimators"
                                ]
                            )
                            is_ensemble_of_forests = True
                        elif "n_estimators" in tree_estimator.get_params():
                            n_estimators = tree_estimator.get_params()["n_estimators"]
                        else:
                            n_estimators = 1
                        if n_estimators > 10:
                            rows = (n_estimators // 10) + 1
                            cols = 10
                        else:
                            rows = 1
                            cols = n_estimators
                        figsize = (cols * 20, rows * 16)
                        fig, axes = plt.subplots(
                            nrows=rows,
                            ncols=cols,
                            figsize=figsize,
                            dpi=_base_dpi * scale,
                            squeeze=False,
                        )
                        axes = list(axes.flatten())

                        fig.suptitle("Decision Trees")

                        display.move_progress()
                        self.logger.info("Plotting decision trees")
                        with fit_if_not_fitted(
                            pipeline_with_model,
                            data_X,
                            data_y,
                            groups=groups,
                            **fit_kwargs,
                        ) as fitted_pipeline_with_model:
                            trees = []
                            feature_names = list(data_X.columns)
                            if self._ml_usecase == MLUsecase.CLASSIFICATION:
                                class_names = {
                                    v: k
                                    for k, v in self._internal_pipeline.named_steps[
                                        "dtypes"
                                    ].replacement.items()
                                }
                            else:
                                class_names = None
                            fitted_tree_estimator = fitted_pipeline_with_model.steps[
                                -1
                            ][1]
                            if is_stacked_model:
                                stacked_feature_names = []
                                if self._ml_usecase == MLUsecase.CLASSIFICATION:
                                    classes = list(data_y.unique())
                                    if len(classes) == 2:
                                        classes.pop()
                                    for c in classes:
                                        stacked_feature_names.extend(
                                            [
                                                f"{k}_{class_names[c]}"
                                                for k, v in fitted_tree_estimator.estimators
                                            ]
                                        )
                                else:
                                    stacked_feature_names.extend(
                                        [
                                            f"{k}"
                                            for k, v in fitted_tree_estimator.estimators
                                        ]
                                    )
                                if not fitted_tree_estimator.passthrough:
                                    feature_names = stacked_feature_names
                                else:
                                    feature_names = (
                                        stacked_feature_names + feature_names
                                    )
                                fitted_tree_estimator = (
                                    fitted_tree_estimator.final_estimator_
                                )
                            if is_ensemble_of_forests:
                                for estimator in fitted_tree_estimator.estimators_:
                                    trees.extend(estimator.estimators_)
                            else:
                                try:
                                    trees = fitted_tree_estimator.estimators_
                                except:
                                    trees = [fitted_tree_estimator]
                            if self._ml_usecase == MLUsecase.CLASSIFICATION:
                                class_names = list(class_names.values())
                            for i, tree in enumerate(trees):
                                self.logger.info(f"Plotting tree {i}")
                                plot_tree(
                                    tree,
                                    feature_names=feature_names,
                                    class_names=class_names,
                                    filled=True,
                                    rounded=True,
                                    precision=4,
                                    ax=axes[i],
                                )
                                axes[i].set_title(f"Tree {i}")
                        for i in range(len(trees), len(axes)):
                            axes[i].set_visible(False)
                        display.move_progress()

                        display.move_progress()
                        display.clear_output()
                        if save:
                            plot_filename = f"{plot_name}.png"
                            if not isinstance(save, bool):
                                plot_filename = os.path.join(save, plot_filename)
                            self.logger.info(f"Saving '{plot_filename}'")
                            plt.savefig(plot_filename, bbox_inches="tight")
                        elif system:
                            plt.show()
                        plt.close()

                        self.logger.info("Visual Rendered Successfully")

                    def calibration():

                        from sklearn.calibration import calibration_curve

                        plt.figure(figsize=(7, 6), dpi=_base_dpi * scale)
                        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)

                        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
                        display.move_progress()
                        self.logger.info("Scoring test/hold-out set")
                        with fit_if_not_fitted(
                            pipeline_with_model,
                            data_X,
                            data_y,
                            groups=groups,
                            **fit_kwargs,
                        ) as fitted_pipeline_with_model:
                            prob_pos = fitted_pipeline_with_model.predict_proba(test_X)[
                                :, 1
                            ]
                        prob_pos = (prob_pos - prob_pos.min()) / (
                            prob_pos.max() - prob_pos.min()
                        )
                        fraction_of_positives, mean_predicted_value = calibration_curve(
                            test_y, prob_pos, n_bins=10
                        )
                        display.move_progress()
                        ax1.plot(
                            mean_predicted_value,
                            fraction_of_positives,
                            "s-",
                            label=f"{model_name}",
                        )

                        ax1.set_ylabel("Fraction of positives")
                        ax1.set_ylim([0, 1])
                        ax1.set_xlim([0, 1])
                        ax1.legend(loc="lower right")
                        ax1.set_title("Calibration plots (reliability curve)")
                        ax1.set_facecolor("white")
                        ax1.grid(b=True, color="grey", linewidth=0.5, linestyle="-")
                        plt.tight_layout()
                        display.move_progress()
                        display.clear_output()
                        if save:
                            plot_filename = f"{plot_name}.png"
                            if not isinstance(save, bool):
                                plot_filename = os.path.join(save, plot_filename)
                            self.logger.info(f"Saving '{plot_filename}'")
                            plt.savefig(plot_filename, bbox_inches="tight")
                        elif system:
                            plt.show()
                        plt.close()

                        self.logger.info("Visual Rendered Successfully")

                    def vc():

                        self.logger.info("Determining param_name")

                        actual_estimator_label = get_pipeline_estimator_label(
                            pipeline_with_model
                        )
                        actual_estimator = pipeline_with_model.named_steps[
                            actual_estimator_label
                        ]

                        try:
                            try:
                                # catboost special case
                                model_params = actual_estimator.get_all_params()
                            except:
                                model_params = pipeline_with_model.get_params()
                        except:
                            display.clear_output()
                            self.logger.error("VC plot failed. Exception:")
                            self.logger.error(traceback.format_exc())
                            raise TypeError(
                                "Plot not supported for this estimator. Try different estimator."
                            )

                        param_name = ""
                        param_range = None

                        if self._ml_usecase == MLUsecase.CLASSIFICATION:

                            # Catboost
                            if "depth" in model_params:
                                param_name = f"{actual_estimator_label}__depth"
                                param_range = np.arange(1, 8 if self.gpu_param else 11)

                            # SGD Classifier
                            elif f"{actual_estimator_label}__l1_ratio" in model_params:
                                param_name = f"{actual_estimator_label}__l1_ratio"
                                param_range = np.arange(0, 1, 0.01)

                            # tree based models
                            elif f"{actual_estimator_label}__max_depth" in model_params:
                                param_name = f"{actual_estimator_label}__max_depth"
                                param_range = np.arange(1, 11)

                            # knn
                            elif (
                                f"{actual_estimator_label}__n_neighbors" in model_params
                            ):
                                param_name = f"{actual_estimator_label}__n_neighbors"
                                param_range = np.arange(1, 11)

                            # MLP / Ridge
                            elif f"{actual_estimator_label}__alpha" in model_params:
                                param_name = f"{actual_estimator_label}__alpha"
                                param_range = np.arange(0, 1, 0.1)

                            # Logistic Regression
                            elif f"{actual_estimator_label}__C" in model_params:
                                param_name = f"{actual_estimator_label}__C"
                                param_range = np.arange(1, 11)

                            # Bagging / Boosting
                            elif (
                                f"{actual_estimator_label}__n_estimators"
                                in model_params
                            ):
                                param_name = f"{actual_estimator_label}__n_estimators"
                                param_range = np.arange(1, 1000, 10)

                            # Naive Bayes
                            elif (
                                f"{actual_estimator_label}__var_smoothing"
                                in model_params
                            ):
                                param_name = f"{actual_estimator_label}__var_smoothing"
                                param_range = np.arange(0.1, 1, 0.01)

                            # QDA
                            elif f"{actual_estimator_label}__reg_param" in model_params:
                                param_name = f"{actual_estimator_label}__reg_param"
                                param_range = np.arange(0, 1, 0.1)

                            # GPC
                            elif (
                                f"{actual_estimator_label}__max_iter_predict"
                                in model_params
                            ):
                                param_name = (
                                    f"{actual_estimator_label}__max_iter_predict"
                                )
                                param_range = np.arange(100, 1000, 100)

                            else:
                                display.clear_output()
                                raise TypeError(
                                    "Plot not supported for this estimator. Try different estimator."
                                )

                        elif self._ml_usecase == MLUsecase.REGRESSION:

                            # Catboost
                            if "depth" in model_params:
                                param_name = f"{actual_estimator_label}__depth"
                                param_range = np.arange(1, 8 if self.gpu_param else 11)

                            # lasso/ridge/en/llar/huber/kr/mlp/br/ard
                            elif f"{actual_estimator_label}__alpha" in model_params:
                                param_name = f"{actual_estimator_label}__alpha"
                                param_range = np.arange(0, 1, 0.1)

                            elif f"{actual_estimator_label}__alpha_1" in model_params:
                                param_name = f"{actual_estimator_label}__alpha_1"
                                param_range = np.arange(0, 1, 0.1)

                            # par/svm
                            elif f"{actual_estimator_label}__C" in model_params:
                                param_name = f"{actual_estimator_label}__C"
                                param_range = np.arange(1, 11)

                            # tree based models (dt/rf/et)
                            elif f"{actual_estimator_label}__max_depth" in model_params:
                                param_name = f"{actual_estimator_label}__max_depth"
                                param_range = np.arange(1, 11)

                            # knn
                            elif (
                                f"{actual_estimator_label}__n_neighbors" in model_params
                            ):
                                param_name = f"{actual_estimator_label}__n_neighbors"
                                param_range = np.arange(1, 11)

                            # Bagging / Boosting (ada/gbr)
                            elif (
                                f"{actual_estimator_label}__n_estimators"
                                in model_params
                            ):
                                param_name = f"{actual_estimator_label}__n_estimators"
                                param_range = np.arange(1, 1000, 10)

                            # Bagging / Boosting (ada/gbr)
                            elif (
                                f"{actual_estimator_label}__n_nonzero_coefs"
                                in model_params
                            ):
                                param_name = (
                                    f"{actual_estimator_label}__n_nonzero_coefs"
                                )
                                if len(data_X.columns) >= 10:
                                    param_max = 11
                                else:
                                    param_max = len(data_X.columns) + 1
                                param_range = np.arange(1, param_max, 1)

                            elif f"{actual_estimator_label}__eps" in model_params:
                                param_name = f"{actual_estimator_label}__eps"
                                param_range = np.arange(0, 1, 0.1)

                            elif (
                                f"{actual_estimator_label}__max_subpopulation"
                                in model_params
                            ):
                                param_name = (
                                    f"{actual_estimator_label}__max_subpopulation"
                                )
                                param_range = np.arange(1000, 100000, 2000)

                            elif (
                                f"{actual_estimator_label}__min_samples" in model_params
                            ):
                                param_name = (
                                    f"{actual_estimator_label}__max_subpopulation"
                                )
                                param_range = np.arange(0.01, 1, 0.1)

                            else:
                                display.clear_output()
                                raise TypeError(
                                    "Plot not supported for this estimator. Try different estimator."
                                )

                        self.logger.info(f"param_name: {param_name}")

                        display.move_progress()

                        from yellowbrick.model_selection import ValidationCurve

                        viz = ValidationCurve(
                            pipeline_with_model,
                            param_name=param_name,
                            param_range=param_range,
                            cv=cv,
                            random_state=self.seed,
                            n_jobs=self._gpu_n_jobs_param,
                        )
                        show_yellowbrick_plot(
                            visualizer=viz,
                            X_train=data_X,
                            y_train=data_y,
                            X_test=test_X,
                            y_test=test_y,
                            handle_train="fit",
                            handle_test="",
                            name=plot_name,
                            scale=scale,
                            save=save,
                            fit_kwargs=fit_kwargs,
                            groups=groups,
                            display=display,
                            display_format=display_format,
                        )

                    def dimension():

                        from sklearn.decomposition import PCA
                        from sklearn.preprocessing import StandardScaler
                        from yellowbrick.features import RadViz

                        data_X_transformed = data_X.select_dtypes(include="float32")
                        self.logger.info("Fitting StandardScaler()")
                        data_X_transformed = StandardScaler().fit_transform(
                            data_X_transformed
                        )
                        data_y_transformed = np.array(data_y)

                        features = min(round(len(data_X.columns) * 0.3, 0), 5)
                        features = int(features)

                        pca = PCA(n_components=features, random_state=self.seed)
                        self.logger.info("Fitting PCA()")
                        data_X_transformed = pca.fit_transform(data_X_transformed)
                        display.move_progress()
                        classes = data_y.unique().tolist()
                        visualizer = RadViz(classes=classes, alpha=0.25)

                        show_yellowbrick_plot(
                            visualizer=visualizer,
                            X_train=data_X_transformed,
                            y_train=data_y_transformed,
                            X_test=test_X,
                            y_test=test_y,
                            handle_train="fit_transform",
                            handle_test="",
                            name=plot_name,
                            scale=scale,
                            save=save,
                            fit_kwargs=fit_kwargs,
                            groups=groups,
                            display=display,
                            display_format=display_format,
                        )

                    def feature():
                        _feature(10)

                    def feature_all():
                        _feature(len(data_X.columns))

                    def _feature(n: int):
                        variables = None
                        temp_model = pipeline_with_model
                        if hasattr(pipeline_with_model, "steps"):
                            temp_model = pipeline_with_model.steps[-1][1]
                        if hasattr(temp_model, "coef_"):
                            try:
                                coef = temp_model.coef_.flatten()
                                if len(coef) > len(data_X.columns):
                                    coef = coef[: len(data_X.columns)]
                                variables = abs(coef)
                            except:
                                pass
                        if variables is None:
                            self.logger.warning(
                                "No coef_ found. Trying feature_importances_"
                            )
                            variables = abs(temp_model.feature_importances_)
                        coef_df = pd.DataFrame(
                            {"Variable": data_X.columns, "Value": variables}
                        )
                        sorted_df = (
                            coef_df.sort_values(by="Value", ascending=False)
                            .head(n)
                            .sort_values(by="Value")
                        )
                        my_range = range(1, len(sorted_df.index) + 1)
                        display.move_progress()
                        plt.figure(figsize=(8, 5 * (n // 10)), dpi=_base_dpi * scale)
                        plt.hlines(
                            y=my_range, xmin=0, xmax=sorted_df["Value"], color="skyblue"
                        )
                        plt.plot(sorted_df["Value"], my_range, "o")
                        display.move_progress()
                        plt.yticks(my_range, sorted_df["Variable"])
                        plt.title("Feature Importance Plot")
                        plt.xlabel("Variable Importance")
                        plt.ylabel("Features")
                        display.move_progress()
                        display.clear_output()
                        if save:
                            plot_filename = f"{plot_name}.png"
                            if not isinstance(save, bool):
                                plot_filename = os.path.join(save, plot_filename)
                            self.logger.info(f"Saving '{plot_filename}'")
                            plt.savefig(plot_filename, bbox_inches="tight")
                        elif system:
                            plt.show()
                        plt.close()

                        self.logger.info("Visual Rendered Successfully")

                    def parameter():

                        try:
                            params = estimator.get_all_params()
                        except:
                            params = estimator.get_params(deep=False)

                        param_df = pd.DataFrame.from_dict(
                            {str(k): str(v) for k, v in params.items()},
                            orient="index",
                            columns=["Parameters"],
                        )
                        display.display(param_df, clear=True)
                        self.logger.info("Visual Rendered Successfully")

                    def ks():

                        display.move_progress()
                        self.logger.info(
                            "Generating predictions / predict_proba on X_test"
                        )
                        with fit_if_not_fitted(
                            pipeline_with_model,
                            data_X,
                            data_y,
                            groups=groups,
                            **fit_kwargs,
                        ) as fitted_pipeline_with_model:
                            predict_proba__ = fitted_pipeline_with_model.predict_proba(
                                data_X
                            )
                        display.move_progress()
                        display.move_progress()
                        display.clear_output()
                        with MatplotlibDefaultDPI(
                            base_dpi=_base_dpi, scale_to_set=scale
                        ):
                            fig = skplt.metrics.plot_ks_statistic(
                                data_y, predict_proba__, figsize=(10, 6)
                            )
                            if save:
                                plot_filename = f"{plot_name}.png"
                                if not isinstance(save, bool):
                                    plot_filename = os.path.join(save, plot_filename)
                                self.logger.info(f"Saving '{plot_filename}'")
                                plt.savefig(plot_filename, bbox_inches="tight")
                            elif system:
                                plt.show()
                            plt.close()

                        self.logger.info("Visual Rendered Successfully")

                    # execute the plot method
                    ret = locals()[plot]()
                    if ret:
                        plot_filename = ret

                    try:
                        plt.close()
                    except:
                        pass

        gc.collect()

        self.logger.info(
            "plot_model() successfully completed......................................"
        )

        if save:
            return plot_filename

    def evaluate_model(
        self,
        estimator,
        fold: Optional[Union[int, Any]] = None,
        fit_kwargs: Optional[dict] = None,
        feature_name: Optional[str] = None,
        groups: Optional[Union[str, Any]] = None,
        use_train_data: bool = False,
    ):

        """
        This function displays a user interface for all of the available plots for
        a given estimator. It internally uses the plot_model() function.

        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> experiment_name = setup(data = juice,  target = 'Purchase')
        >>> lr = create_model('lr')
        >>> evaluate_model(lr)

        This will display the User Interface for all of the plots for a given
        estimator.

        Parameters
        ----------
        estimator : object, default = none
            A trained model object should be passed as an estimator.

        fold: integer or scikit-learn compatible CV generator, default = None
            Controls cross-validation. If None, will use the CV generator defined in setup().
            If integer, will use KFold CV with that many folds.
            When cross_validation is False, this parameter is ignored.

        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.

        groups: str or array-like, with shape (n_samples,), default = None
            Optional Group labels for the samples used while splitting the dataset into train/test set.
            If string is passed, will use the data column with that name as the groups.
            Only used if a group based cross-validation generator is used (eg. GroupKFold).
            If None, will use the value set in fold_groups parameter in setup().

        Returns
        -------
        User_Interface
            Displays the user interface for plotting.

        """

        function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

        self.logger.info("Initializing evaluate_model()")
        self.logger.info(f"evaluate_model({function_params_str})")

        from ipywidgets import widgets
        from ipywidgets.widgets import fixed, interact

        if not fit_kwargs:
            fit_kwargs = {}

        a = widgets.ToggleButtons(
            options=[(v, k) for k, v in self._available_plots.items()],
            description="Plot Type:",
            disabled=False,
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            icons=[""],
        )

        fold = self._get_cv_splitter(fold)

        groups = self._get_groups(groups)

        interact(
            self.plot_model,
            estimator=fixed(estimator),
            plot=a,
            save=fixed(False),
            verbose=fixed(True),
            scale=fixed(1),
            fold=fixed(fold),
            fit_kwargs=fixed(fit_kwargs),
            feature_name=fixed(feature_name),
            label=fixed(False),
            groups=fixed(groups),
            use_train_data=fixed(use_train_data),
            system=fixed(True),
            display=fixed(None),
            display_format=fixed(None),
        )

    def predict_model(self, *args, **kwargs) -> pd.DataFrame:
        return pd.DataFrame()

    def finalize_model(self) -> None:
        return

    def automl(
        self, optimize: str = "Accuracy", use_holdout: bool = False, turbo: bool = True
    ) -> Any:

        """
        This function returns the best model out of all models created in
        current active environment based on metric defined in optimize parameter.

        Parameters
        ----------
        optimize : str, default = 'Accuracy'
            Other values you can pass in optimize parameter are 'AUC', 'Recall', 'Precision',
            'F1', 'Kappa', and 'MCC'.

        use_holdout: bool, default = False
            When set to True, metrics are evaluated on holdout set instead of CV.

        turbo: bool, default = True
            When set to True and use_holdout is False, only models created with default fold
            parameter will be considered. If set to False, models created with a non-default
            fold parameter will be scored again using default fold settings, so that they can be
            compared.
        """

        function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

        self.logger.info("Initializing automl()")
        self.logger.info(f"automl({function_params_str})")

        # checking optimize parameter
        optimize = self._get_metric_by_name_or_id(optimize)
        if optimize is None:
            raise ValueError(
                f"Optimize method not supported. See docstring for list of available parameters."
            )

        # checking optimize parameter for multiclass
        if self._is_multiclass():
            if not optimize.is_multiclass:
                raise TypeError(
                    f"Optimization metric not supported for multiclass problems. See docstring for list of other optimization parameters."
                )

        compare_dimension = optimize.display_name
        greater_is_better = optimize.greater_is_better
        optimize = optimize.scorer

        best_model = None
        best_score = None

        def compare_score(new, best):
            if not best:
                return True
            if greater_is_better:
                return new > best
            else:
                return new < best

        if use_holdout:
            self.logger.info("Model Selection Basis : Holdout set")
            for i in self.master_model_container:
                self.logger.info(f"Checking model {i}")
                model = i["model"]
                try:
                    pred_holdout = self.predict_model(model, verbose=False)  # type: ignore
                except:
                    self.logger.warning(
                        f"Model {model} is not fitted, running create_model"
                    )
                    model, _ = self.create_model(  # type: ignore
                        estimator=model,
                        system=False,
                        verbose=False,
                        cross_validation=False,
                        predict=False,
                        groups=self.fold_groups_param,
                    )
                    self.pull(pop=True)
                    pred_holdout = self.predict_model(model, verbose=False)  # type: ignore

                p = self.pull(pop=True)
                p = p[compare_dimension][0]
                if compare_score(p, best_score):
                    best_model = model
                    best_score = p

        else:
            self.logger.info("Model Selection Basis : CV Results on Training set")
            for i in range(len(self.master_model_container)):
                model = self.master_model_container[i]
                scores = None
                if model["cv"] is not self.fold_generator:
                    if turbo or self._is_unsupervised():
                        continue
                    self.create_model(  # type: ignore
                        estimator=model["model"],
                        system=False,
                        verbose=False,
                        cross_validation=True,
                        predict=False,
                        groups=self.fold_groups_param,
                    )
                    scores = self.pull(pop=True)
                    self.master_model_container.pop()
                self.logger.info(f"Checking model {i}")
                if scores is None:
                    scores = model["scores"]
                r = scores[compare_dimension][-2:][0]
                if compare_score(r, best_score):
                    best_model = model["model"]
                    best_score = r

        automl_model, _ = self.create_model(  # type: ignore
            estimator=best_model,
            system=False,
            verbose=False,
            cross_validation=False,
            predict=False,
            groups=self.fold_groups_param,
        )

        gc.collect()

        self.logger.info(str(automl_model))
        self.logger.info(
            "automl() successfully completed......................................"
        )

        return automl_model

    def _get_models(self, raise_errors: bool = True) -> Tuple[dict, dict]:
        return ({}, {})

    def _get_metrics(self, raise_errors: bool = True) -> dict:
        return {}

    def models(
        self,
        type: Optional[str] = None,
        internal: bool = False,
        raise_errors: bool = True,
    ) -> pd.DataFrame:

        """
        Returns table of models available in model library.

        Example
        -------
        >>> _all_models = models()

        This will return pandas dataframe with all available
        models and their metadata.

        Parameters
        ----------
        type : str, default = None
            - linear : filters and only return linear models
            - tree : filters and only return tree based models
            - ensemble : filters and only return ensemble models

        internal: bool, default = False
            If True, will return extra columns and rows used internally.

        raise_errors: bool, default = True
            If False, will suppress all exceptions, ignoring models
            that couldn't be created.

        Returns
        -------
        pandas.DataFrame

        """

        self.logger.info(f"gpu_param set to {self.gpu_param}")

        _, model_containers = self._get_models(raise_errors)

        rows = [
            v.get_dict(internal)
            for k, v in model_containers.items()
            if (internal or not v.is_special)
        ]

        df = pd.DataFrame(rows)
        df.set_index("ID", inplace=True, drop=True)

        return df

    def deploy_model(
        self,
        model,
        model_name: str,
        authentication: dict,
        platform: str = "aws",  # added gcp and azure support in pycaret==2.1
    ):

        """
        (In Preview)

        This function deploys the transformation pipeline and trained model object for
        production use. The platform of deployment can be defined under the platform
        parameter along with the applicable authentication tokens which are passed as a
        dictionary to the authentication param.

        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> experiment_name = setup(data = juice,  target = 'Purchase')
        >>> lr = create_model('lr')
        >>> deploy_model(model = lr, model_name = 'deploy_lr', platform = 'aws', authentication = {'bucket' : 'pycaret-test'})

        This will deploy the model on an AWS S3 account under bucket 'pycaret-test'

        Notes
        -----
        For AWS users:
        Before deploying a model to an AWS S3 ('aws'), environment variables must be
        configured using the command line interface. To configure AWS env. variables,
        type aws configure in your python command line. The following information is
        required which can be generated using the Identity and Access Management (IAM)
        portal of your amazon console account:

        - AWS Access Key ID
        - AWS Secret Key Access
        - Default Region Name (can be seen under Global settings on your AWS console)
        - Default output format (must be left blank)

        For GCP users:
        --------------
        Before deploying a model to Google Cloud Platform (GCP), project must be created
        either using command line or GCP console. Once project is created, you must create
        a service account and download the service account key as a JSON file, which is
        then used to set environment variable.

        https://cloud.google.com/docs/authentication/production

        - Google Cloud Project
        - Service Account Authetication

        For Azure users:
        ---------------
        Before deploying a model to Microsoft's Azure (Azure), environment variables
        for connection string must be set. In order to get connection string, user has
        to create account of Azure. Once it is done, create a Storage account. In the settings
        section of storage account, user can get the connection string.

        Read below link for more details.
        https://docs.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python?toc=%2Fpython%2Fazure%2FTOC.json

        - Azure Storage Account

        Parameters
        ----------
        model : object
            A trained model object should be passed as an estimator.

        model_name : str
            Name of model to be passed as a str.

        authentication : dict
            Dictionary of applicable authentication tokens.

            When platform = 'aws':
            {'bucket' : 'Name of Bucket on S3'}

            When platform = 'gcp':
            {'project': 'gcp_pycaret', 'bucket' : 'pycaret-test'}

            When platform = 'azure':
            {'container': 'pycaret-test'}

        platform: str, default = 'aws'
            Name of platform for deployment. Current available options are: 'aws', 'gcp' and 'azure'

        Returns
        -------
        Success_Message

        Warnings
        --------
        - This function uses file storage services to deploy the model on cloud platform.
        As such, this is efficient for batch-use. Where the production objective is to
        obtain prediction at an instance level, this may not be the efficient choice as
        it transmits the binary pickle file between your local python environment and
        the platform.

        """
        return pycaret.internal.persistence.deploy_model(
            model, model_name, authentication, platform, self._internal_pipeline
        )

    def save_model(
        self,
        model,
        model_name: str,
        model_only: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> None:
        """
        This function saves the transformation pipeline and trained model object
        into the current active directory as a pickle file for later use.

        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> experiment_name = setup(data = juice,  target = 'Purchase')
        >>> lr = create_model('lr')
        >>> save_model(lr, 'lr_model_23122019')

        This will save the transformation pipeline and model as a binary pickle
        file in the current active directory.

        Parameters
        ----------
        model : object, default = none
            A trained model object should be passed as an estimator.

        model_name : str, default = none
            Name of pickle file to be passed as a string.

        model_only : bool, default = False
            When set to True, only trained model object is saved and all the
            transformations are ignored.

        verbose: bool, default = True
            Success message is not printed when verbose is set to False.

        Returns
        -------
        Success_Message

        """
        return pycaret.internal.persistence.save_model(
            model, model_name, None if model_only else self._internal_pipeline, verbose, **kwargs
        )

    def load_model(
        self,
        model_name,
        platform: Optional[str] = None,
        authentication: Optional[Dict[str, str]] = None,
        verbose: bool = True,
    ):

        """
        This function loads a previously saved transformation pipeline and model
        from the current active directory into the current python environment.
        Load object must be a pickle file.

        Example
        -------
        >>> saved_lr = load_model('lr_model_23122019')

        This will load the previously saved model in saved_lr variable. The file
        must be in the current directory.

        Parameters
        ----------
        model_name : str, default = none
            Name of pickle file to be passed as a string.

        platform: str, default = None
            Name of platform, if loading model from cloud. Current available options are:
            'aws', 'gcp' and 'azure'.

        authentication : dict
            dictionary of applicable authentication tokens.

            When platform = 'aws':
            {'bucket' : 'Name of Bucket on S3'}

            When platform = 'gcp':
            {'project': 'gcp_pycaret', 'bucket' : 'pycaret-test'}

            When platform = 'azure':
            {'container': 'pycaret-test'}

        verbose: bool, default = True
            Success message is not printed when verbose is set to False.

        Returns
        -------
        Model Object

        """

        return pycaret.internal.persistence.load_model(
            model_name, platform, authentication, verbose
        )
