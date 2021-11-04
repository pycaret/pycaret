from pycaret.internal.pycaret_experiment.utils import MLUsecase
from pycaret.internal.pycaret_experiment.pycaret_experiment import _PyCaretExperiment
from pycaret.internal.meta_estimators import get_estimator_from_meta_estimator
from pycaret.internal.pipeline import (
    get_pipeline_estimator_label,
    estimator_pipeline,
    get_pipeline_fit_kwargs,
    Pipeline as InternalPipeline,
)
from pycaret.internal.utils import (
    mlflow_remove_bad_chars,
    normalize_custom_transformers,
    get_model_name,
)
import pycaret.internal.patches.sklearn
import pycaret.internal.patches.yellowbrick
from pycaret.internal.logging import get_logger, create_logger
from pycaret.internal.plots.yellowbrick import show_yellowbrick_plot
from pycaret.internal.plots.helper import MatplotlibDefaultDPI
from pycaret.internal.Display import Display
from pycaret.internal.distributions import *
from pycaret.internal.validation import *
import pycaret.internal.preprocess
import pycaret.internal.persistence
import pandas as pd  # type ignore
from pandas.io.formats.style import Styler
import numpy as np  # type: ignore
import os
import sys
import datetime
import time
import random
import gc
from copy import deepcopy
from sklearn.model_selection import BaseCrossValidator  # type: ignore
from typing import List, Tuple, Any, Union, Optional, Dict
import warnings
import traceback
from unittest.mock import patch
import plotly.express as px  # type: ignore
import plotly.graph_objects as go  # type: ignore
import scikitplot as skplt  # type: ignore
from packaging import version
import logging


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
                "prep_pipe",
                "experiment__",
                "n_jobs_param",
                "_gpu_n_jobs_param",
                "master_model_container",
                "display_container",
                "exp_name_log",
                "exp_id",
                "logging_param",
                "log_plots_param",
                "data_before_preprocess",
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
        _prep_pipe,
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
                    self.data_before_preprocess.drop([self.target_param], axis=1)
                )
            except Exception:
                self.logger.warning("Couldn't infer MLFlow signature.")
                signature = None
            if not self._is_unsupervised():
                input_example = (
                    self.data_before_preprocess.drop([self.target_param], axis=1)
                    .iloc[0]
                    .to_dict()
                )
            else:
                input_example = self.data_before_preprocess.iloc[0].to_dict()

            # log model as sklearn flavor
            prep_pipe_temp = deepcopy(_prep_pipe)
            prep_pipe_temp.steps.append(["trained_model", model])
            mlflow.sklearn.log_model(
                prep_pipe_temp,
                "model",
                conda_env=default_conda_env,
                signature=signature,
                input_example=input_example,
            )
            del prep_pipe_temp
        gc.collect()

    def _split_data(
        self,
        X_before_preprocess,
        y_before_preprocess,
        target,
        train_data,
        test_data,
        train_size,
        data_split_shuffle,
        dtypes,
        display: Display,
    ) -> None:
        self.X = None
        self.X_train = None
        self.y_train = None
        self.y = None
        self.X_test = None
        self.y_test = None
        return

    def _set_up_mlflow(
        self, functions, runtime, log_profile, profile_kwargs, log_data, display,
    ) -> None:
        return

    def _make_internal_pipeline(
        self, internal_pipeline_steps: list, memory=None
    ) -> InternalPipeline:
        if not internal_pipeline_steps:
            memory = None
            internal_pipeline_steps = [("empty_step", "passthrough")]

        return InternalPipeline(internal_pipeline_steps, memory=memory)

    def setup(
        self,
        data: pd.DataFrame,
        target: str,
        train_size: float = 0.7,
        test_data: Optional[pd.DataFrame] = None,
        preprocess: bool = True,
        imputation_type: str = "simple",
        iterative_imputation_iters: int = 5,
        categorical_features: Optional[List[str]] = None,
        categorical_imputation: str = "mode",
        categorical_iterative_imputer: Union[str, Any] = "lightgbm",
        ordinal_features: Optional[Dict[str, list]] = None,
        high_cardinality_features: Optional[List[str]] = None,
        high_cardinality_method: str = "frequency",
        numeric_features: Optional[List[str]] = None,
        numeric_imputation: str = "mean",  # method 'zero' added in pycaret==2.1
        numeric_iterative_imputer: Union[str, Any] = "lightgbm",
        date_features: Optional[List[str]] = None,
        ignore_features: Optional[List[str]] = None,
        normalize: bool = False,
        normalize_method: str = "zscore",
        transformation: bool = False,
        transformation_method: str = "yeo-johnson",
        handle_unknown_categorical: bool = True,
        unknown_categorical_method: str = "least_frequent",
        pca: bool = False,
        pca_method: str = "linear",
        pca_components: Optional[float] = None,
        ignore_low_variance: bool = False,
        combine_rare_levels: bool = False,
        rare_level_threshold: float = 0.10,
        bin_numeric_features: Optional[List[str]] = None,
        remove_outliers: bool = False,
        outliers_threshold: float = 0.05,
        remove_multicollinearity: bool = False,
        multicollinearity_threshold: float = 0.9,
        remove_perfect_collinearity: bool = True,
        create_clusters: bool = False,
        cluster_iter: int = 20,
        polynomial_features: bool = False,
        polynomial_degree: int = 2,
        trigonometry_features: bool = False,
        polynomial_threshold: float = 0.1,
        group_features: Optional[List[str]] = None,
        group_names: Optional[List[str]] = None,
        feature_selection: bool = False,
        feature_selection_threshold: float = 0.8,
        feature_selection_method: str = "classic",  # boruta algorithm added in pycaret==2.1
        feature_interaction: bool = False,
        feature_ratio: bool = False,
        interaction_threshold: float = 0.01,
        # classification specific
        fix_imbalance: bool = False,
        fix_imbalance_method: Optional[Any] = None,
        # regression specific
        transform_target=False,
        transform_target_method="box-cox",
        data_split_shuffle: bool = True,
        data_split_stratify: Union[bool, List[str]] = False,  # added in pycaret==2.2
        fold_strategy: Union[str, Any] = "kfold",  # added in pycaret==2.2
        fold: int = 10,  # added in pycaret==2.2
        fh: Union[List[int], int, np.array] = 1,
        fold_shuffle: bool = False,
        fold_groups: Optional[Union[str, pd.DataFrame]] = None,
        n_jobs: Optional[int] = -1,
        use_gpu: bool = False,  # added in pycaret==2.1
        custom_pipeline: Union[
            Any, Tuple[str, Any], List[Any], List[Tuple[str, Any]]
        ] = None,
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
        This function initializes the environment in pycaret and creates the transformation
        pipeline to prepare the data for modeling and deployment. setup() must called before
        executing any other function in pycaret. It takes two mandatory parameters:
        data and name of the target column.

        All other parameters are optional.

        """

        function_params_str = ", ".join(
            [f"{k}={v}" for k, v in locals().items() if k != "data"]
        )

        warnings.filterwarnings("ignore")

        from pycaret.utils import __version__

        ver = __version__

        # experiment_name
        if experiment_name:
            if type(experiment_name) is not str:
                raise TypeError(
                    "experiment_name parameter must be a non-empty str if not None."
                )

        if experiment_name:
            self.exp_name_log = experiment_name
        self.logger = create_logger(system_log)

        self.logger.info(f"PyCaret {type(self).__name__}")
        self.logger.info(f"Logging name: {self.exp_name_log}")
        self.logger.info(f"ML Usecase: {self._ml_usecase}")
        self.logger.info(f"version {ver}")
        self.logger.info("Initializing setup()")
        self.logger.info(f"setup({function_params_str})")

        self._check_enviroment()

        # run_time
        runtime_start = time.time()

        self.logger.info("Checking Exceptions")

        # checking data type
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data passed must be of type pandas.DataFrame")
        if data.shape[0] == 0:
            raise ValueError("data passed must be a positive dataframe")

        # checking train size parameter
        if type(train_size) is not float:
            raise TypeError("train_size parameter only accepts float value.")
        if train_size <= 0 or train_size > 1:
            raise ValueError("train_size parameter has to be positive and not above 1.")

        # checking target parameter
        if not self._is_unsupervised() and target not in data.columns:
            raise ValueError(
                f"Target parameter: {target} does not exist in the data provided."
            )

        # checking session_id
        if session_id is not None:
            if type(session_id) is not int:
                raise TypeError("session_id parameter must be an integer.")

        # checking profile parameter
        if type(profile) is not bool:
            raise TypeError("profile parameter only accepts True or False.")

        if profile_kwargs is not None:
            if type(profile_kwargs) is not dict:
                raise TypeError("profile_kwargs can only be a dict.")
        else:
            profile_kwargs = {}

        # checking normalize parameter
        if type(normalize) is not bool:
            raise TypeError("normalize parameter only accepts True or False.")

        # checking transformation parameter
        if type(transformation) is not bool:
            raise TypeError("transformation parameter only accepts True or False.")

        all_cols = list(data.columns)
        if not self._is_unsupervised():
            all_cols.remove(target)

        # checking imputation type
        allowed_imputation_type = ["simple", "iterative"]
        if imputation_type not in allowed_imputation_type:
            raise ValueError(
                f"imputation_type parameter only accepts {', '.join(allowed_imputation_type)}."
            )

        if (
            type(iterative_imputation_iters) is not int
            or iterative_imputation_iters <= 0
        ):
            raise TypeError(
                "iterative_imputation_iters must be an integer greater than 0."
            )

        # checking categorical imputation
        allowed_categorical_imputation = ["constant", "mode"]
        if categorical_imputation not in allowed_categorical_imputation:
            raise ValueError(
                f"categorical_imputation parameter only accepts {', '.join(allowed_categorical_imputation)}."
            )

        # ordinal_features
        if ordinal_features is not None:
            if type(ordinal_features) is not dict:
                raise TypeError(
                    "ordinal_features must be of type dictionary with column name as key and ordered values as list."
                )

        # ordinal features check
        if ordinal_features is not None:
            ordinal_features = ordinal_features.copy()
            data_cols = data.columns.drop(target, errors="ignore")
            ord_keys = ordinal_features.keys()

            for i in ord_keys:
                if i not in data_cols:
                    raise ValueError(
                        "Column name passed as a key in ordinal_features parameter doesnt exist."
                    )

            for k in ord_keys:
                if data[k].nunique() != len(ordinal_features[k]):
                    raise ValueError(
                        "Levels passed in ordinal_features parameter doesnt match with levels in data."
                    )

            for i in ord_keys:
                value_in_keys = ordinal_features.get(i)
                value_in_data = list(data[i].unique().astype(str))
                for j in value_in_keys:
                    if str(j) not in value_in_data:
                        raise ValueError(
                            f"Column name '{i}' doesn't contain any level named '{j}'."
                        )

        # high_cardinality_features
        if high_cardinality_features is not None:
            if type(high_cardinality_features) is not list:
                raise TypeError(
                    "high_cardinality_features parameter only accepts name of columns as a list."
                )

        if high_cardinality_features is not None:
            data_cols = data.columns.drop(target, errors="ignore")
            for high_cardinality_feature in high_cardinality_features:
                if high_cardinality_feature not in data_cols:
                    raise ValueError(
                        f"Item {high_cardinality_feature} in high_cardinality_features parameter is either target column or doesn't exist in the dataset."
                    )

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

        # high_cardinality_methods
        high_cardinality_allowed_methods = ["frequency", "clustering"]
        if high_cardinality_method not in high_cardinality_allowed_methods:
            raise ValueError(
                f"high_cardinality_method parameter only accepts {', '.join(high_cardinality_allowed_methods)}."
            )

        # checking numeric imputation
        allowed_numeric_imputation = ["mean", "median", "zero"]
        if numeric_imputation not in allowed_numeric_imputation:
            raise ValueError(
                f"numeric_imputation parameter only accepts {', '.join(allowed_numeric_imputation)}."
            )

        # checking normalize method
        allowed_normalize_method = ["zscore", "minmax", "maxabs", "robust"]
        if normalize_method not in allowed_normalize_method:
            raise ValueError(
                f"normalize_method parameter only accepts {', '.join(allowed_normalize_method)}."
            )

        # checking transformation method
        allowed_transformation_method = ["yeo-johnson", "quantile"]
        if transformation_method not in allowed_transformation_method:
            raise ValueError(
                f"transformation_method parameter only accepts {', '.join(allowed_transformation_method)}."
            )

        # handle unknown categorical
        if type(handle_unknown_categorical) is not bool:
            raise TypeError(
                "handle_unknown_categorical parameter only accepts True or False."
            )

        # unknown categorical method
        unknown_categorical_method_available = ["least_frequent", "most_frequent"]

        if unknown_categorical_method not in unknown_categorical_method_available:
            raise TypeError(
                f"unknown_categorical_method only accepts {', '.join(unknown_categorical_method_available)}."
            )

        # check pca
        if type(pca) is not bool:
            raise TypeError("PCA parameter only accepts True or False.")

        # pca method check
        allowed_pca_methods = ["linear", "kernel", "incremental"]
        if pca_method not in allowed_pca_methods:
            raise ValueError(
                f"pca method parameter only accepts {', '.join(allowed_pca_methods)}."
            )

        # pca components check
        if pca is True:
            if pca_method != "linear":
                if pca_components is not None:
                    if (type(pca_components)) is not int:
                        raise TypeError(
                            "pca_components parameter must be integer when pca_method is not 'linear'."
                        )

        # pca components check 2
        if pca is True:
            if pca_method != "linear":
                if pca_components is not None:
                    if pca_components > len(data.columns) - 1:
                        raise TypeError(
                            "pca_components parameter cannot be greater than original features space."
                        )

        # pca components check 3
        if pca is True:
            if pca_method == "linear":
                if pca_components is not None:
                    if type(pca_components) is not float:
                        if pca_components > len(data.columns) - 1:
                            raise TypeError(
                                "pca_components parameter cannot be greater than original features space or float between 0 - 1."
                            )

        # check ignore_low_variance
        if type(ignore_low_variance) is not bool:
            raise TypeError("ignore_low_variance parameter only accepts True or False.")

        # check ignore_low_variance
        if type(combine_rare_levels) is not bool:
            raise TypeError("combine_rare_levels parameter only accepts True or False.")

        # check rare_level_threshold
        if (
            type(rare_level_threshold) is not float
            and rare_level_threshold < 0
            or rare_level_threshold > 1
        ):
            raise TypeError(
                "rare_level_threshold parameter must be a float between 0 and 1."
            )

        # bin numeric features
        if bin_numeric_features is not None:
            if type(bin_numeric_features) is not list:
                raise TypeError("bin_numeric_features parameter must be a list.")
            for bin_numeric_feature in bin_numeric_features:
                if type(bin_numeric_feature) is not str:
                    raise TypeError(
                        "bin_numeric_features parameter item must be a string."
                    )
                if bin_numeric_feature not in all_cols:
                    raise ValueError(
                        f"bin_numeric_feature: {bin_numeric_feature} is either target column or does not exist in the dataset."
                    )

        # remove_outliers
        if type(remove_outliers) is not bool:
            raise TypeError("remove_outliers parameter only accepts True or False.")

        # outliers_threshold
        if type(outliers_threshold) is not float:
            raise TypeError("outliers_threshold must be a float between 0 and 1.")

        # remove_multicollinearity
        if type(remove_multicollinearity) is not bool:
            raise TypeError(
                "remove_multicollinearity parameter only accepts True or False."
            )

        # multicollinearity_threshold
        if type(multicollinearity_threshold) is not float:
            raise TypeError(
                "multicollinearity_threshold must be a float between 0 and 1."
            )

        # create_clusters
        if type(create_clusters) is not bool:
            raise TypeError("create_clusters parameter only accepts True or False.")

        # cluster_iter
        if type(cluster_iter) is not int:
            raise TypeError("cluster_iter must be a integer greater than 1.")

        # polynomial_features
        if type(polynomial_features) is not bool:
            raise TypeError("polynomial_features only accepts True or False.")

        # polynomial_degree
        if type(polynomial_degree) is not int:
            raise TypeError("polynomial_degree must be an integer.")

        # polynomial_features
        if type(trigonometry_features) is not bool:
            raise TypeError("trigonometry_features only accepts True or False.")

        # polynomial threshold
        if type(polynomial_threshold) is not float:
            raise TypeError("polynomial_threshold must be a float between 0 and 1.")

        # group features
        if group_features is not None:
            if type(group_features) is not list:
                raise TypeError("group_features must be of type list.")

        if group_names is not None:
            if type(group_names) is not list:
                raise TypeError("group_names must be of type list.")

        # cannot drop target
        if ignore_features is not None:
            if target in ignore_features:
                raise ValueError("cannot drop target column.")

        # feature_selection
        if type(feature_selection) is not bool:
            raise TypeError("feature_selection only accepts True or False.")

        # feature_selection_threshold
        if type(feature_selection_threshold) is not float:
            raise TypeError(
                "feature_selection_threshold must be a float between 0 and 1."
            )

        # feature_selection_method
        feature_selection_methods = ["boruta", "classic"]
        if feature_selection_method not in feature_selection_methods:
            raise TypeError(
                f"feature_selection_method must be one of {', '.join(feature_selection_methods)}"
            )

        # feature_interaction
        if type(feature_interaction) is not bool:
            raise TypeError("feature_interaction only accepts True or False.")

        # feature_ratio
        if type(feature_ratio) is not bool:
            raise TypeError("feature_ratio only accepts True or False.")

        # interaction_threshold
        if type(interaction_threshold) is not float:
            raise TypeError("interaction_threshold must be a float between 0 and 1.")

        # categorical
        if categorical_features is not None:
            for i in categorical_features:
                if i not in all_cols:
                    raise ValueError(
                        "Column type forced is either target column or doesn't exist in the dataset."
                    )

        # numeric
        if numeric_features is not None:
            for i in numeric_features:
                if i not in all_cols:
                    raise ValueError(
                        "Column type forced is either target column or doesn't exist in the dataset."
                    )

        # date features
        if date_features is not None:
            for i in date_features:
                if i not in all_cols:
                    raise ValueError(
                        "Column type forced is either target column or doesn't exist in the dataset."
                    )

        # drop features
        if ignore_features is not None:
            for i in ignore_features:
                if i not in all_cols:
                    raise ValueError(
                        "Feature ignored is either target column or doesn't exist in the dataset."
                    )

        # log_experiment
        if type(log_experiment) is not bool:
            raise TypeError("log_experiment parameter only accepts True or False.")

        # log_profile
        if type(log_profile) is not bool:
            raise TypeError("log_profile parameter only accepts True or False.")

        # silent
        if type(silent) is not bool:
            raise TypeError("silent parameter only accepts True or False.")

        # remove_perfect_collinearity
        if type(remove_perfect_collinearity) is not bool:
            raise TypeError(
                "remove_perfect_collinearity parameter only accepts True or False."
            )

        # html
        if type(html) is not bool:
            raise TypeError("html parameter only accepts True or False.")

        # use_gpu
        if use_gpu != "force" and type(use_gpu) is not bool:
            raise TypeError("use_gpu parameter only accepts 'force', True or False.")

        # data_split_shuffle
        if type(data_split_shuffle) is not bool:
            raise TypeError("data_split_shuffle parameter only accepts True or False.")

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
            if fold_groups not in all_cols:
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

        # fix_imbalance
        if type(fix_imbalance) is not bool:
            raise TypeError("fix_imbalance parameter only accepts True or False.")

        # fix_imbalance_method
        if fix_imbalance:
            if fix_imbalance_method is not None:
                if hasattr(fix_imbalance_method, "fit_resample"):
                    pass
                else:
                    raise TypeError(
                        "fix_imbalance_method must contain resampler with fit_resample method."
                    )

        # check transform_target
        if type(transform_target) is not bool:
            raise TypeError("transform_target parameter only accepts True or False.")

        # transform_target_method
        allowed_transform_target_method = ["box-cox", "yeo-johnson"]
        if transform_target_method not in allowed_transform_target_method:
            raise ValueError(
                f"transform_target_method parameter only accepts {', '.join(allowed_transform_target_method)}."
            )

        if log_plots == True:
            log_plots = self._get_default_plots_to_log()

        # pandas option
        pd.set_option("display.max_columns", 500)
        pd.set_option("display.max_rows", 500)

        # generate USI for mlflow tracking
        import secrets

        # declaring global variables to be accessed by other functions
        self.logger.info("Declaring global variables")
        # global _ml_usecase, USI, html_param, X, y, X_train, X_test, y_train, y_test, seed, prep_pipe, experiment__, fold_shuffle_param, n_jobs_param, _gpu_n_jobs_param, master_model_container, display_container, exp_name_log, logging_param, log_plots_param, fix_imbalance_param, fix_imbalance_method_param, transform_target_param, transform_target_method_param, data_before_preprocess, target_param, gpu_param, _all_models, _all_models_internal, _all_metrics, _internal_pipeline, stratify_param, fold_generator, fold_param, fold_groups_param, imputation_regressor, imputation_classifier, iterative_imputation_iters_param

        self.USI = secrets.token_hex(nbytes=2)
        self.logger.info(f"self.USI: {self.USI}")

        self.logger.info(f"self.variable_keys: {self.variable_keys}")

        # create html_param
        self.html_param = html

        self.logger.info("Preparing display monitor")

        if not display:
            # progress bar
            max_steps = 3

            progress_args = {"max": max_steps}
            timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
            monitor_rows = [
                ["Initiated", ". . . . . . . . . . . . . . . . . .", timestampStr],
                [
                    "Status",
                    ". . . . . . . . . . . . . . . . . .",
                    "Loading Dependencies",
                ],
            ]
            display = Display(
                verbose=verbose,
                html_param=self.html_param,
                progress_args=progress_args,
                monitor_rows=monitor_rows,
            )

            display.display_progress()
            display.display_monitor()

        self.logger.info("Importing libraries")

        # setting sklearn config to print all parameters including default
        import sklearn

        sklearn.set_config(print_changed_only=False)

        self.logger.info("Copying data for preprocessing")

        # copy original data for pandas profiler
        self.data_before_preprocess = data.copy()

        # generate seed to be used globally
        self.seed = random.randint(150, 9000) if session_id is None else session_id

        np.random.seed(self.seed)

        self._internal_pipeline = []

        """
        preprocessing starts here
        """

        display.update_monitor(1, "Preparing Data for Modeling")
        display.display_monitor()

        # define parameters for preprocessor

        self.logger.info("Declaring preprocessing parameters")

        # categorical features
        cat_features_pass = categorical_features or []

        # numeric features
        numeric_features_pass = numeric_features or []

        # drop features
        ignore_features_pass = ignore_features or []

        # date features
        date_features_pass = date_features or []

        # categorical imputation strategy
        cat_dict = {"constant": "not_available", "mode": "most frequent"}
        categorical_imputation_pass = cat_dict[categorical_imputation]

        # transformation method strategy
        trans_dict = {"yeo-johnson": "yj", "quantile": "quantile"}
        trans_method_pass = trans_dict[transformation_method]

        # pass method
        pca_dict = {
            "linear": "pca_liner",
            "kernel": "pca_kernal",
            "incremental": "incremental",
            "pls": "pls",
        }
        pca_method_pass = pca_dict[pca_method]

        # pca components
        if pca is True:
            if pca_components is None:
                if pca_method == "linear":
                    pca_components_pass = 0.99
                else:
                    pca_components_pass = int((len(data.columns) - 1) * 0.5)

            else:
                pca_components_pass = pca_components

        else:
            pca_components_pass = 0.99

        apply_binning_pass = False if bin_numeric_features is None else True
        features_to_bin_pass = bin_numeric_features or []

        # trignometry
        trigonometry_features_pass = (
            ["sin", "cos", "tan"] if trigonometry_features else []
        )

        # group features
        # =============#

        # apply grouping
        apply_grouping_pass = True if group_features is not None else False

        # group features listing
        if apply_grouping_pass is True:

            if type(group_features[0]) is str:
                group_features_pass = []
                group_features_pass.append(group_features)
            else:
                group_features_pass = group_features

        else:

            group_features_pass = [[]]

        # group names
        if apply_grouping_pass is True:

            if (group_names is None) or (len(group_names) != len(group_features_pass)):
                group_names_pass = list(np.arange(len(group_features_pass)))
                group_names_pass = [f"group_{i}" for i in group_names_pass]

            else:
                group_names_pass = group_names

        else:
            group_names_pass = []

        # feature interactions

        apply_feature_interactions_pass = (
            True if feature_interaction or feature_ratio else False
        )

        interactions_to_apply_pass = []

        if feature_interaction:
            interactions_to_apply_pass.append("multiply")

        if feature_ratio:
            interactions_to_apply_pass.append("divide")

        # unknown categorical
        unkn_dict = {
            "least_frequent": "least frequent",
            "most_frequent": "most frequent",
        }
        unknown_categorical_method_pass = unkn_dict[unknown_categorical_method]

        # ordinal_features
        apply_ordinal_encoding_pass = True if ordinal_features is not None else False

        ordinal_columns_and_categories_pass = (
            ordinal_features if apply_ordinal_encoding_pass else {}
        )

        apply_cardinality_reduction_pass = (
            True if high_cardinality_features is not None else False
        )

        hi_card_dict = {"frequency": "count", "clustering": "cluster"}
        cardinal_method_pass = hi_card_dict[high_cardinality_method]

        cardinal_features_pass = (
            high_cardinality_features if apply_cardinality_reduction_pass else []
        )

        display_dtypes_pass = False if silent else True

        # transform target method
        self.transform_target_param = transform_target
        self.transform_target_method_param = transform_target_method

        # create n_jobs_param
        self.n_jobs_param = n_jobs

        cuml_version = None
        if use_gpu:
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

        # create gpu_param var
        self.gpu_param = use_gpu

        self.iterative_imputation_iters_param = iterative_imputation_iters

        # creating variables to be used later in the function
        train_data = self.data_before_preprocess.copy()
        if self._is_unsupervised():
            target = "UNSUPERVISED_DUMMY_TARGET"
            train_data[target] = 2
            # just to add diversified values to target
            train_data[target][0:3] = 3
        X_before_preprocess = train_data.drop(target, axis=1)
        y_before_preprocess = train_data[target]

        self.imputation_regressor = numeric_iterative_imputer
        self.imputation_classifier = categorical_iterative_imputer
        imputation_regressor_name = "Bayesian Ridge"  # todo change
        imputation_classifier_name = "Random Forest Classifier"

        if imputation_type == "iterative":
            self.logger.info("Setting up iterative imputation")

            iterative_imputer_models_globals = self.variables.copy()
            iterative_imputer_models_globals["y_train"] = y_before_preprocess
            iterative_imputer_models_globals["X_train"] = X_before_preprocess
            iterative_imputer_classification_models = {
                k: v
                for k, v in pycaret.containers.models.classification.get_all_model_containers(
                    iterative_imputer_models_globals, raise_errors=True
                ).items()
                if not v.is_special
            }
            iterative_imputer_regression_models = {
                k: v
                for k, v in pycaret.containers.models.regression.get_all_model_containers(
                    iterative_imputer_models_globals, raise_errors=True
                ).items()
                if not v.is_special
            }

            if not (
                (
                    isinstance(self.imputation_regressor, str)
                    and self.imputation_regressor in iterative_imputer_regression_models
                )
                or hasattr(self.imputation_regressor, "predict")
            ):
                raise ValueError(
                    f"numeric_iterative_imputer parameter must be either a scikit-learn estimator or a string - one of {', '.join(iterative_imputer_regression_models.keys())}."
                )

            if not (
                (
                    isinstance(self.imputation_classifier, str)
                    and self.imputation_classifier
                    in iterative_imputer_classification_models
                )
                or hasattr(self.imputation_classifier, "predict")
            ):
                raise ValueError(
                    f"categorical_iterative_imputer parameter must be either a scikit-learn estimator or a string - one of {', '.join(iterative_imputer_classification_models.keys())}."
                )

            if isinstance(self.imputation_regressor, str):
                self.imputation_regressor = iterative_imputer_regression_models[
                    self.imputation_regressor
                ]
                imputation_regressor_name = self.imputation_regressor.name
                self.imputation_regressor = self.imputation_regressor.class_def(
                    **self.imputation_regressor.args
                )
            else:
                imputation_regressor_name = type(self.imputation_regressor).__name__

            if isinstance(self.imputation_classifier, str):
                self.imputation_classifier = iterative_imputer_classification_models[
                    self.imputation_classifier
                ]
                imputation_classifier_name = self.imputation_classifier.name
                self.imputation_classifier = self.imputation_classifier.class_def(
                    **self.imputation_classifier.args
                )
            else:
                imputation_classifier_name = type(self.imputation_classifier).__name__

        self.logger.info("Creating preprocessing pipeline")

        self.prep_pipe = pycaret.internal.preprocess.Preprocess_Path_One(
            train_data=train_data,
            ml_usecase="classification"
            if self._ml_usecase == MLUsecase.CLASSIFICATION
            else "regression",
            imputation_type=imputation_type,
            target_variable=target,
            imputation_regressor=self.imputation_regressor,
            imputation_classifier=self.imputation_classifier,
            imputation_max_iter=self.iterative_imputation_iters_param,
            categorical_features=cat_features_pass,
            apply_ordinal_encoding=apply_ordinal_encoding_pass,
            ordinal_columns_and_categories=ordinal_columns_and_categories_pass,
            apply_cardinality_reduction=apply_cardinality_reduction_pass,
            cardinal_method=cardinal_method_pass,
            cardinal_features=cardinal_features_pass,
            numerical_features=numeric_features_pass,
            time_features=date_features_pass,
            features_todrop=ignore_features_pass,
            numeric_imputation_strategy=numeric_imputation,
            categorical_imputation_strategy=categorical_imputation_pass,
            scale_data=normalize,
            scaling_method=normalize_method,
            Power_transform_data=transformation,
            Power_transform_method=trans_method_pass,
            apply_untrained_levels_treatment=handle_unknown_categorical,
            untrained_levels_treatment_method=unknown_categorical_method_pass,
            apply_pca=pca,
            pca_method=pca_method_pass,
            pca_variance_retained_or_number_of_components=pca_components_pass,
            apply_zero_nearZero_variance=ignore_low_variance,
            club_rare_levels=combine_rare_levels,
            rara_level_threshold_percentage=rare_level_threshold,
            apply_binning=apply_binning_pass,
            features_to_binn=features_to_bin_pass,
            remove_outliers=remove_outliers,
            outlier_contamination_percentage=outliers_threshold,
            outlier_methods=["pca"],
            remove_multicollinearity=remove_multicollinearity,
            maximum_correlation_between_features=multicollinearity_threshold,
            remove_perfect_collinearity=remove_perfect_collinearity,
            cluster_entire_data=create_clusters,
            range_of_clusters_to_try=cluster_iter,
            apply_polynomial_trigonometry_features=polynomial_features,
            max_polynomial=polynomial_degree,
            trigonometry_calculations=trigonometry_features_pass,
            top_poly_trig_features_to_select_percentage=polynomial_threshold,
            apply_grouping=apply_grouping_pass,
            features_to_group_ListofList=group_features_pass,
            group_name=group_names_pass,
            apply_feature_selection=feature_selection,
            feature_selection_top_features_percentage=feature_selection_threshold,
            feature_selection_method=feature_selection_method,
            apply_feature_interactions=apply_feature_interactions_pass,
            feature_interactions_to_apply=interactions_to_apply_pass,
            feature_interactions_top_features_to_select_percentage=interaction_threshold,
            display_types=display_dtypes_pass,  # this is for inferred input box
            random_state=self.seed,
            float_dtype="float64"
            if self._ml_usecase == MLUsecase.TIME_SERIES
            else "float32",
        )

        dtypes = self.prep_pipe.named_steps["dtypes"]
        try:
            fix_perfect_removed_columns = (
                self.prep_pipe.named_steps["fix_perfect"].columns_to_drop
                if remove_perfect_collinearity
                else []
            )
        except:
            fix_perfect_removed_columns = []
        try:
            fix_multi_removed_columns = (
                (
                    self.prep_pipe.named_steps["fix_multi"].to_drop
                    + self.prep_pipe.named_steps["fix_multi"].to_drop_taret_correlation
                )
                if remove_multicollinearity
                else []
            )
        except:
            fix_multi_removed_columns = []
        multicollinearity_removed_columns = (
            fix_perfect_removed_columns + fix_multi_removed_columns
        )

        display.move_progress()
        self.logger.info("Preprocessing pipeline created successfully")

        try:
            res_type = [
                "quit",
                "Quit",
                "exit",
                "EXIT",
                "q",
                "Q",
                "e",
                "E",
                "QUIT",
                "Exit",
            ]
            res = dtypes.response

            if res in res_type:
                sys.exit(
                    "(Process Exit): setup has been interupted with user command 'quit'. setup must rerun."
                )

        except:
            self.logger.error(
                "(Process Exit): setup has been interupted with user command 'quit'. setup must rerun."
            )

        if not preprocess:
            self.prep_pipe.steps = self.prep_pipe.steps[:1]

        """
        preprocessing ends here
        """

        # reset pandas option
        pd.reset_option("display.max_rows")
        pd.reset_option("display.max_columns")

        self.logger.info("Creating global containers")

        # create an empty list for pickling later.
        self.experiment__ = []

        # CV params
        self.fold_param = fold
        self.fold_groups_param = None
        self.fold_groups_param_full = None
        if fold_groups is not None:
            if isinstance(fold_groups, str):
                self.fold_groups_param = X_before_preprocess[fold_groups]
            else:
                self.fold_groups_param = fold_groups
            if pd.isnull(self.fold_groups_param).any():
                raise ValueError(f"fold_groups cannot contain NaNs.")
        self.fold_shuffle_param = fold_shuffle

        # create master_model_container
        self.master_model_container = []

        # create display container
        self.display_container = []

        # create logging parameter
        self.logging_param = log_experiment

        # create an empty log_plots_param
        if not log_plots:
            self.log_plots_param = False
        else:
            self.log_plots_param = log_plots

        # add custom transformers to prep pipe
        if custom_pipeline:
            custom_steps = normalize_custom_transformers(custom_pipeline)
            self._internal_pipeline.extend(custom_steps)

        # create a fix_imbalance_param and fix_imbalance_method_param
        self.fix_imbalance_param = fix_imbalance and preprocess
        self.fix_imbalance_method_param = fix_imbalance_method

        fix_imbalance_model_name = "SMOTE"

        if self.fix_imbalance_param:
            if self.fix_imbalance_method_param is None:
                import six

                sys.modules["sklearn.externals.six"] = six
                from imblearn.over_sampling import SMOTE

                fix_imbalance_resampler = SMOTE(random_state=self.seed)
            else:
                fix_imbalance_model_name = str(self.fix_imbalance_method_param).split(
                    "("
                )[0]
                fix_imbalance_resampler = self.fix_imbalance_method_param
            self._internal_pipeline.append(("fix_imbalance", fix_imbalance_resampler))

        for x in self._internal_pipeline:
            if x[0] in self.prep_pipe.named_steps:
                raise ValueError(f"Step named {x[0]} already present in pipeline.")

        self._internal_pipeline = self._make_internal_pipeline(self._internal_pipeline)

        self.logger.info(f"Internal pipeline: {self._internal_pipeline}")

        # create target_param var
        self.target_param = target

        # create stratify_param var
        self.stratify_param = data_split_stratify

        display.move_progress()

        display.update_monitor(1, "Preprocessing Data")
        display.display_monitor()

        self._split_data(
            X_before_preprocess,
            y_before_preprocess,
            target,
            train_data,
            test_data,
            train_size,
            data_split_shuffle,
            dtypes,
            display,
            fh,
        )

        if not self._is_unsupervised():
            from sklearn.model_selection import (
                StratifiedKFold,
                KFold,
                GroupKFold,
                TimeSeriesSplit,
            )

            from sktime.forecasting.model_selection import (
                ExpandingWindowSplitter,
                SlidingWindowSplitter,
            )

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

                    # Set the strategy from the cv object
                    if isinstance(self.fold_generator, ExpandingWindowSplitter):
                        self.fold_strategy = "expanding"
                    if isinstance(self.fold_generator, SlidingWindowSplitter):
                        self.fold_strategy = "sliding"

        # we do just the fitting so that it will be fitted when saved/deployed,
        # but we don't want to modify the data
        self._internal_pipeline.fit(
            self.X, y=self.y if not self._is_unsupervised() else None
        )

        self.prep_pipe.steps = self.prep_pipe.steps + [
            (step[0], deepcopy(step[1]))
            for step in self._internal_pipeline.steps
            if hasattr(step[1], "transform")
        ]

        try:
            dtypes.final_training_columns.remove(target)
        except ValueError:
            pass

        # determining target type
        if self._is_multiclass():
            target_type = "Multiclass"
        else:
            target_type = "Binary"

        self._all_models, self._all_models_internal = self._get_models()
        self._all_metrics = self._get_metrics()

        """
        Final display Starts
        """
        self.logger.info("Creating grid variables")

        if hasattr(dtypes, "replacement"):
            label_encoded = dtypes.replacement
            label_encoded = (
                str(label_encoded).replace("'", "").replace("{", "").replace("}", "")
            )

        else:
            label_encoded = "None"

        # generate values for grid show
        missing_values = self.data_before_preprocess.isna().sum().sum()
        missing_flag = True if missing_values > 0 else False

        normalize_grid = normalize_method if normalize else "None"

        transformation_grid = transformation_method if transformation else "None"

        pca_method_grid = pca_method if pca else "None"

        pca_components_grid = pca_components_pass if pca else "None"

        rare_level_threshold_grid = (
            rare_level_threshold if combine_rare_levels else "None"
        )

        numeric_bin_grid = False if bin_numeric_features is None else True

        outliers_threshold_grid = outliers_threshold if remove_outliers else None

        multicollinearity_threshold_grid = (
            multicollinearity_threshold if remove_multicollinearity else None
        )

        cluster_iter_grid = cluster_iter if create_clusters else None

        polynomial_degree_grid = polynomial_degree if polynomial_features else None

        polynomial_threshold_grid = (
            polynomial_threshold
            if polynomial_features or trigonometry_features
            else None
        )

        feature_selection_threshold_grid = (
            feature_selection_threshold if feature_selection else None
        )

        interaction_threshold_grid = (
            interaction_threshold if feature_interaction or feature_ratio else None
        )

        ordinal_features_grid = False if ordinal_features is None else True

        unknown_categorical_method_grid = (
            unknown_categorical_method if handle_unknown_categorical else None
        )

        group_features_grid = False if group_features is None else True

        high_cardinality_features_grid = (
            False if high_cardinality_features is None else True
        )

        high_cardinality_method_grid = (
            high_cardinality_method if high_cardinality_features_grid else None
        )

        learned_types = dtypes.learned_dtypes
        learned_types.drop(target, inplace=True)

        float_type = 0
        cat_type = 0

        for i in dtypes.learned_dtypes:
            if "float" in str(i):
                float_type += 1
            elif "object" in str(i):
                cat_type += 1
            elif "int" in str(i):
                float_type += 1

        if profile:
            print("Setup Successfully Completed! Loading Profile Now... Please Wait!")
        else:
            if verbose:
                print("Setup Successfully Completed!")

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

        self.preprocess = preprocess
        functions = self._get_setup_display(
            label_encoded=label_encoded,
            target_type=target_type,
            missing_flag=missing_flag,
            float_type=float_type,
            cat_type=cat_type,
            ordinal_features_grid=ordinal_features_grid,
            high_cardinality_features_grid=high_cardinality_features_grid,
            high_cardinality_method_grid=high_cardinality_method_grid,
            data_split_shuffle=data_split_shuffle,
            data_split_stratify=data_split_stratify,
            imputation_type=imputation_type,
            numeric_imputation=numeric_imputation,
            imputation_regressor_name=imputation_regressor_name,
            categorical_imputation=categorical_imputation,
            imputation_classifier_name=imputation_classifier_name,
            unknown_categorical_method_grid=unknown_categorical_method_grid,
            normalize=normalize,
            normalize_grid=normalize_grid,
            transformation=transformation,
            transformation_grid=transformation_grid,
            pca=pca,
            pca_method_grid=pca_method_grid,
            pca_components_grid=pca_components_grid,
            ignore_low_variance=ignore_low_variance,
            combine_rare_levels=combine_rare_levels,
            rare_level_threshold_grid=rare_level_threshold_grid,
            numeric_bin_grid=numeric_bin_grid,
            remove_outliers=remove_outliers,
            outliers_threshold_grid=outliers_threshold_grid,
            remove_perfect_collinearity=remove_perfect_collinearity,
            remove_multicollinearity=remove_multicollinearity,
            multicollinearity_threshold_grid=multicollinearity_threshold_grid,
            multicollinearity_removed_columns=multicollinearity_removed_columns,
            create_clusters=create_clusters,
            cluster_iter_grid=cluster_iter_grid,
            polynomial_features=polynomial_features,
            polynomial_degree_grid=polynomial_degree_grid,
            trigonometry_features=trigonometry_features,
            polynomial_threshold_grid=polynomial_threshold_grid,
            group_features_grid=group_features_grid,
            feature_selection=feature_selection,
            feature_selection_method=feature_selection_method,
            feature_selection_threshold_grid=feature_selection_threshold_grid,
            feature_interaction=feature_interaction,
            feature_ratio=feature_ratio,
            interaction_threshold_grid=interaction_threshold_grid,
            fix_imbalance_model_name=fix_imbalance_model_name,
        )

        self.display_container.append(functions)

        display.display(functions, clear=True)

        if profile:
            try:
                import pandas_profiling

                pf = pandas_profiling.ProfileReport(
                    self.data_before_preprocess, **profile_kwargs
                )
                display.display(pf, clear=True)
            except:
                print(
                    "Data Profiler Failed. No output to show, please continue with Modeling."
                )
                self.logger.error(
                    "Data Profiler Failed. No output to show, please continue with Modeling."
                )

        """
        Final display Ends
        """

        # end runtime
        runtime_end = time.time()
        runtime = np.array(runtime_end - runtime_start).round(2)

        self._set_up_mlflow(
            functions, runtime, log_profile, profile_kwargs, log_data, display,
        )

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

        self.logger.info(str(self.prep_pipe))
        self.logger.info(
            "setup() successfully completed......................................"
        )

        gc.collect()

        return self

    def create_model(self, *args, **kwargs):
        return

    @staticmethod
    def plot_model_check_display_format_(display_format: Optional[str]):
        """Checks if the display format is in the allowed list"""
        plot_formats = [None, "streamlit"]

        if display_format not in plot_formats:
            raise ValueError("display_format can only be None or 'streamlit'.")

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
        self.plot_model_check_display_format_(display_format=display_format)

        # Import required libraries ----
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
            from sklearn.tree import BaseDecisionTree
            from sklearn.ensemble._forest import BaseForest

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
        import yellowbrick.utils.types
        import yellowbrick.utils.helpers

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
                            pca_["Feature"] = self.data_before_preprocess[feature_name]
                        else:
                            pca_["Feature"] = self.data_before_preprocess[
                                self.data_before_preprocess.columns[0]
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
                            df["Feature"] = self.data_before_preprocess[feature_name]
                        else:
                            df["Feature"] = self.data_before_preprocess[
                                self.data_before_preprocess.columns[0]
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
                            X["Feature"] = self.data_before_preprocess[feature_name]
                        else:
                            X["Feature"] = self.data_before_preprocess[
                                self.data_before_preprocess.columns[0]
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
                            X_embedded["Feature"] = self.data_before_preprocess[
                                feature_name
                            ]
                        else:
                            X_embedded["Feature"] = self.data_before_preprocess[
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

                        from sklearn.preprocessing import StandardScaler
                        from sklearn.decomposition import PCA
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

                        from sklearn.tree import plot_tree
                        from sklearn.base import is_classifier
                        from sklearn.model_selection import check_cv

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
                                    for k, v in self.prep_pipe.named_steps[
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

                        from yellowbrick.features import RadViz
                        from sklearn.preprocessing import StandardScaler
                        from sklearn.decomposition import PCA

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
        from ipywidgets.widgets import interact, fixed

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
            model, model_name, authentication, platform, self.prep_pipe
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
            model, model_name, None if model_only else self.prep_pipe, verbose, **kwargs
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
