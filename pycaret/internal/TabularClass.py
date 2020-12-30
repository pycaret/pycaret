from enum import Enum, auto
import math
from pycaret.internal.meta_estimators import (
    PowerTransformedTargetRegressor,
    get_estimator_from_meta_estimator,
)
from pycaret.internal.patches.tune_sklearn import (
    get_tune_sklearn_tunegridsearchcv,
    get_tune_sklearn_tunesearchcv,
)
from pycaret.internal.pipeline import (
    add_estimator_to_pipeline,
    get_pipeline_estimator_label,
    make_internal_pipeline,
    estimator_pipeline,
    merge_pipelines,
    Pipeline as InternalPipeline,
)
from pycaret.internal.utils import (
    color_df,
    normalize_custom_transformers,
    nullcontext,
    true_warm_start,
    can_early_stop,
    infer_ml_usecase,
    get_columns_to_stratify_by,
    get_model_name,
)
import pycaret.internal.patches.sklearn
import pycaret.internal.patches.yellowbrick
from pycaret.internal.logging import get_logger, create_logger
from pycaret.internal.plotting import show_yellowbrick_plot, MatplotlibDefaultDPI
from pycaret.internal.Display import Display
from pycaret.internal.distributions import *
from pycaret.internal.validation import *
import pycaret.containers.metrics.classification
import pycaret.containers.metrics.regression
import pycaret.containers.metrics.clustering
import pycaret.containers.metrics.anomaly
import pycaret.containers.models.classification
import pycaret.containers.models.regression
import pycaret.containers.models.clustering
import pycaret.containers.models.anomaly
import pycaret.internal.preprocess
import pycaret.internal.persistence
from pycaret.internal.pipeline import get_pipeline_fit_kwargs
import pandas as pd
import numpy as np
import os
import sys
import datetime
import time
import random
import gc
import multiprocessing
from copy import deepcopy
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Any, Union, Optional, Dict
import warnings
from IPython.utils import io
import traceback
from unittest.mock import patch
import plotly.express as px
import plotly.graph_objects as go
import scikitplot as skplt

warnings.filterwarnings("ignore")
LOGGER = get_logger()


class MLUsecase(Enum):
    CLASSIFICATION = auto()
    REGRESSION = auto()
    CLUSTERING = auto()
    ANOMALY = auto()


def get_ml_task(y):
    c1 = y.dtype == "int64"
    c2 = y.nunique() <= 20
    c3 = y.dtype.name in ["object", "bool", "category"]
    if ((c1) & (c2)) | (c3):
        ml_usecase = MLUsecase.CLASSIFICATION
    else:
        ml_usecase = MLUsecase.REGRESSION
    return ml_usecase


class _PyCaretExperiment:
    def __init__(self) -> None:
        self._ml_usecase = None
        self._available_plots = {}
        self.variable_keys = set()
        self._setup_ran = False
        self.display_container = []
        self.exp_id = None
        self.gpu_param = False
        self.n_jobs_param = -1
        self.logger = LOGGER
        return

    @property
    def _gpu_n_jobs_param(self) -> int:
        return self.n_jobs_param if not self.gpu_param else 1

    @property
    def variables(self) -> dict:
        return {
            k: (vars(self)[k] if k in vars(self) else None) for k in self.variable_keys
        }

    def _is_multiclass(self) -> bool:
        """
        Method to check if the problem is multiclass.
        """
        return False

    def _check_enviroment(self) -> None:
        # logging environment and libraries
        self.logger.info("Checking environment")

        from platform import python_version, platform, python_build, machine

        self.logger.info(f"python_version: {python_version()}")
        self.logger.info(f"python_build: {python_build()}")
        self.logger.info(f"machine: {machine()}")
        self.logger.info(f"platform: {platform()}")

        try:
            import psutil

            self.logger.info(f"Memory: {psutil.virtual_memory()}")
            self.logger.info(f"Physical Core: {psutil.cpu_count(logical=False)}")
            self.logger.info(f"Logical Core: {psutil.cpu_count(logical=True)}")
        except:
            self.logger.warning(
                "cannot find psutil installation. memory not traceable. Install psutil using pip to enable memory logging."
            )

        self.logger.info("Checking libraries")

        try:
            from pandas import __version__

            self.logger.info(f"pd=={__version__}")
        except:
            self.logger.warning("pandas not found")

        try:
            from numpy import __version__

            self.logger.info(f"numpy=={__version__}")
        except:
            self.logger.warning("numpy not found")

        try:
            from sklearn import __version__

            self.logger.info(f"sklearn=={__version__}")
        except:
            self.logger.warning("sklearn not found")

        try:
            from xgboost import __version__

            self.logger.info(f"xgboost=={__version__}")
        except:
            self.logger.warning("xgboost not found")

        try:
            from lightgbm import __version__

            self.logger.info(f"lightgbm=={__version__}")
        except:
            self.logger.warning("lightgbm not found")

        try:
            from catboost import __version__

            self.logger.info(f"catboost=={__version__}")
        except:
            self.logger.warning("catboost not found")

        try:
            from mlflow.version import VERSION

            warnings.filterwarnings("ignore")
            self.logger.info(f"mlflow=={VERSION}")
        except:
            self.logger.warning("mlflow not found")

    def setup(self) -> None:
        return

    def deploy_model(
        self,
        model,
        model_name: str,
        authentication: dict,
        platform: str = "aws",  # added gcp and azure support in pycaret==2.1
    ):
        return None

    def save_model(
        self, model, model_name: str, model_only: bool = False, verbose: bool = True
    ):
        return None

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

    def get_logs(
        self, experiment_name: Optional[str] = None, save: bool = False
    ) -> pd.DataFrame:

        """
        Returns a table with experiment logs consisting
        run details, parameter, metrics and tags. 

        Example
        -------
        >>> logs = get_logs()

        This will return pandas dataframe.

        Parameters
        ----------
        experiment_name : str, default = None
            When set to None current active run is used.

        save : bool, default = False
            When set to True, csv file is saved in current directory.

        Returns
        -------
        pandas.DataFrame

        """

        import mlflow
        from mlflow.tracking import MlflowClient

        client = MlflowClient()

        if experiment_name is None:
            exp_id = self.exp_id
            experiment = client.get_experiment(exp_id)
            if experiment is None:
                raise ValueError(
                    "No active run found. Check logging parameter in setup or to get logs for inactive run pass experiment_name."
                )

            exp_name_log_ = experiment.name
        else:
            exp_name_log_ = experiment_name
            experiment = client.get_experiment_by_name(exp_name_log_)
            if experiment is None:
                raise ValueError(
                    "No active run found. Check logging parameter in setup or to get logs for inactive run pass experiment_name."
                )

            exp_id = client.get_experiment_by_name(exp_name_log_).experiment_id

        runs = mlflow.search_runs(exp_id)

        if save:
            file_name = f"{exp_name_log_}_logs.csv"
            runs.to_csv(file_name, index=False)

        return runs

    def get_config(self, variable: str) -> Any:
        """
        This function is used to access global environment variables.

        Example
        -------
        >>> X_train = get_config('X_train') 

        This will return X_train transformed dataset.

        Returns
        -------
        variable

        """

        function_params_str = ", ".join(
            [f"{k}={v}" for k, v in locals().items() if not k == "globals_d"]
        )

        self.logger.info("Initializing get_config()")
        self.logger.info(f"get_config({function_params_str})")

        if not variable in self.variables:
            raise ValueError(
                f"Variable {variable} not found. Possible variables are: {list(self.variables)}"
            )
        var = getattr(self, variable)

        self.logger.info(f"Variable: {variable} returned as {var}")
        self.logger.info(
            "get_config() succesfully completed......................................"
        )

        return var

    def set_config(
        self, variable: Optional[str] = None, value: Optional[Any] = None, **kwargs
    ) -> None:
        """
        This function is used to reset global environment variables.

        Example
        -------
        >>> set_config('seed', 123) 

        This will set the global seed to '123'.

        """
        function_params_str = ", ".join(
            [f"{k}={v}" for k, v in locals().items() if not k == "globals_d"]
        )

        self.logger.info("Initializing set_config()")
        self.logger.info(f"set_config({function_params_str})")

        if kwargs and variable:
            raise ValueError(
                "variable parameter cannot be used together with keyword arguments."
            )

        variables = kwargs if kwargs else {variable: value}

        for k, v in variables.items():
            if k.startswith("_"):
                raise ValueError(f"Variable {k} is read only ('_' prefix).")

            if not k in self.variables:
                raise ValueError(
                    f"Variable {k} not found. Possible variables are: {list(self.variables)}"
                )

            setattr(self, k, v)
            self.logger.info(f"Global variable: {k} updated to {v}")
        self.logger.info(
            "set_config() succesfully completed......................................"
        )
        return

    def save_config(self, file_name: str) -> None:
        function_params_str = ", ".join(
            [f"{k}={v}" for k, v in locals().items() if not k == "globals_d"]
        )

        self.logger.info("Initializing save_config()")
        self.logger.info(f"save_config({function_params_str})")

        globals_to_ignore = {
            "_all_models",
            "_all_models_internal",
            "_all_metrics",
            "create_model_container",
            "master_model_container",
            "display_container",
        }

        globals_to_dump = {
            k: v
            for k, v in self.variables.items()
            if k not in globals_to_ignore
        }

        import joblib

        joblib.dump(globals_to_dump, file_name)

        self.logger.info(f"Global variables dumped to {file_name}")
        self.logger.info(
            "save_config() succesfully completed......................................"
        )
        return

    def load_config(self, file_name: str) -> None:
        function_params_str = ", ".join(
            [f"{k}={v}" for k, v in locals().items() if not k == "globals_d"]
        )

        self.logger.info("Initializing load_config()")
        self.logger.info(f"load_config({function_params_str})")

        import joblib

        loaded_globals = joblib.load(file_name)

        self.logger.info(f"Global variables loaded from {file_name}")

        for k, v in loaded_globals.items():
            setattr(self, k, v)
            self.logger.info(f"Global variable: {k} updated to {v}")

        self.logger.info(f"Global variables set to match those in {file_name}")

        self.logger.info(
            "load_config() succesfully completed......................................"
        )
        return

    def pull(self, pop=False) -> pd.DataFrame:  # added in pycaret==2.2.0
        """
        Returns latest displayed table.

        Parameters
        ----------
        pop : bool, default = False
            If true, will pop (remove) the returned dataframe from the
            display container.

        Returns
        -------
        pandas.DataFrame
            Equivalent to get_config('display_container')[-1]

        """
        if not self.display_container:
            return None
        return self.display_container.pop(-1) if pop else self.display_container[-1]


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
                "X",
                "seed",
                "prep_pipe",
                "experiment__",
                "n_jobs_param",
                "_gpu_n_jobs_param",
                "create_model_container",
                "master_model_container",
                "display_container",
                "exp_name_log",
                "exp_id",
                "logging_param",
                "log_plots_param",
                "transform_target_param",
                "transform_target_method_param",
                "data_before_preprocess",
                "target_param",
                "gpu_param",
                "_all_models",
                "_all_models_internal",
                "_all_metrics",
                "_internal_pipeline",
                "imputation_regressor",
                "imputation_classifier",
                "iterative_imputation_iters_param",
                "fold_shuffle_param",
                "fix_imbalance_param",
                "fix_imbalance_method_param",
                "stratify_param",
                "fold_generator",
                "fold_param",
                "fold_groups_param",
            }
        )
        return

    def _get_groups(self, groups, ml_usecase: Optional[MLUsecase] = None):
        import pycaret.internal.utils

        return pycaret.internal.utils.get_groups(
            groups, self.X_train, self.fold_groups_param
        )

    def _get_cv_splitter(self, fold, ml_usecase: Optional[MLUsecase] = None):
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
        except:
            pass

        try:
            metric = next(
                v for k, v in metrics.items() if name_or_id in (v.display_name, v.name)
            )
            return metric
        except:
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

        with mlflow.start_run(run_name=full_name) as run:

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
                except:
                    params = params.get_params()
            except:
                self.logger.warning("Couldn't get params for model. Exception:")
                self.logger.warning(traceback.format_exc())
                params = {}

            for i in list(params):
                v = params.get(i)
                if len(str(v)) > 250:
                    params.pop(i)

            self.logger.info(f"logged params: {params}")
            mlflow.log_params(params)

            # Log metrics
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
                except:
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
                    except:
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
            except:
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
        self,
        functions,
        functions_,
        runtime,
        log_profile,
        profile_kwargs,
        log_data,
        display,
    ) -> None:
        return

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
        fold_shuffle: bool = False,
        fold_groups: Optional[Union[str, pd.DataFrame]] = None,
        n_jobs: Optional[int] = -1,
        use_gpu: bool = False,  # added in pycaret==2.1
        custom_pipeline: Union[
            Any, Tuple[str, Any], List[Any], List[Tuple[str, Any]]
        ] = None,
        html: bool = True,
        session_id: Optional[int] = None,
        log_experiment: bool = False,
        experiment_name: Optional[str] = None,
        log_plots: Union[bool, list] = False,
        log_profile: bool = False,
        log_data: bool = False,
        silent: bool = False,
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
            self.logger = create_logger(experiment_name)
        else:
            # create exp_name_log param incase logging is False
            self.exp_name_log = "no_logging"

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
        if hasattr(data, "shape") is False:
            raise TypeError("data passed must be of type pandas.DataFrame")

        # checking train size parameter
        if type(train_size) is not float:
            raise TypeError("train_size parameter only accepts float value.")

        # checking target parameter
        if not self._is_unsupervised() and target not in data.columns:
            raise ValueError("Target parameter doesnt exist in the data provided.")

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
                "imputation_type param only accepts 'simple' or 'iterative'"
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
                "categorical_imputation param only accepts 'constant' or 'mode'"
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
                        "Column name passed as a key in ordinal_features param doesnt exist."
                    )

            for k in ord_keys:
                if data[k].nunique() != len(ordinal_features[k]):
                    raise ValueError(
                        "Levels passed in ordinal_features param doesnt match with levels in data."
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
                    "high_cardinality_features param only accepts name of columns as a list."
                )

        if high_cardinality_features is not None:
            data_cols = data.columns.drop(target, errors="ignore")
            for i in high_cardinality_features:
                if i not in data_cols:
                    raise ValueError(
                        "Column type forced is either target column or doesn't exist in the dataset."
                    )

        # stratify
        if data_split_stratify:
            if (
                type(data_split_stratify) is not list
                and type(data_split_stratify) is not bool
            ):
                raise TypeError(
                    "data_split_stratify param only accepts a bool or a list of strings."
                )

            if not data_split_shuffle:
                raise TypeError(
                    "data_split_stratify param requires data_split_shuffle to be set to True."
                )

        # high_cardinality_methods
        high_cardinality_allowed_methods = ["frequency", "clustering"]
        if high_cardinality_method not in high_cardinality_allowed_methods:
            raise ValueError(
                "high_cardinality_method param only accepts 'frequency' or 'clustering'"
            )

        # checking numeric imputation
        allowed_numeric_imputation = ["mean", "median", "zero"]
        if numeric_imputation not in allowed_numeric_imputation:
            raise ValueError(
                f"numeric_imputation param only accepts {', '.join(allowed_numeric_imputation)}."
            )

        # checking normalize method
        allowed_normalize_method = ["zscore", "minmax", "maxabs", "robust"]
        if normalize_method not in allowed_normalize_method:
            raise ValueError(
                f"normalize_method param only accepts {', '.join(allowed_normalize_method)}."
            )

        # checking transformation method
        allowed_transformation_method = ["yeo-johnson", "quantile"]
        if transformation_method not in allowed_transformation_method:
            raise ValueError(
                f"transformation_method param only accepts {', '.join(allowed_transformation_method)}."
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
                f"pca method param only accepts {', '.join(allowed_pca_methods)}."
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
        if type(rare_level_threshold) is not float:
            raise TypeError("rare_level_threshold must be a float between 0 and 1.")

        # bin numeric features
        if bin_numeric_features is not None:
            for i in bin_numeric_features:
                if i not in all_cols:
                    raise ValueError(
                        "Column type forced is either target column or doesn't exist in the dataset."
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
        if not (
            fold_strategy in possible_fold_strategy
            or is_sklearn_cv_generator(fold_strategy)
        ):
            raise TypeError(
                f"fold_strategy parameter must be either a scikit-learn compatible CV generator object or one of {', '.join(possible_fold_strategy)}."
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
        if type(fold) is not int:
            raise TypeError("fold parameter only accepts integer value.")

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
                f"transform_target_method param only accepts {', '.join(allowed_transform_target_method)}."
            )

        # pandas option
        pd.set_option("display.max_columns", 500)
        pd.set_option("display.max_rows", 500)

        # generate USI for mlflow tracking
        import secrets

        # declaring global variables to be accessed by other functions
        self.logger.info("Declaring global variables")
        # global _ml_usecase, USI, html_param, X, y, X_train, X_test, y_train, y_test, seed, prep_pipe, experiment__, fold_shuffle_param, n_jobs_param, _gpu_n_jobs_param, create_model_container, master_model_container, display_container, exp_name_log, logging_param, log_plots_param, fix_imbalance_param, fix_imbalance_method_param, transform_target_param, transform_target_method_param, data_before_preprocess, target_param, gpu_param, _all_models, _all_models_internal, _all_metrics, _internal_pipeline, stratify_param, fold_generator, fold_param, fold_groups_param, imputation_regressor, imputation_classifier, iterative_imputation_iters_param

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

        # define highlight function for function grid to display
        def highlight_max(s):
            is_max = s == True
            return ["background-color: lightgreen" if v else "" for v in is_max]

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
            except:
                self.logger.warning(f"cuML not found")

            if cuml_version is None or not cuml_version >= (0, 15):
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
            train_data.loc[0:3, target] = 3
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
                    f"numeric_iterative_imputer param must be either a scikit-learn estimator or a string - one of {', '.join(iterative_imputer_regression_models.keys())}."
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
                    f"categorical_iterative_imputer param must be either a scikit-learn estimator or a string - one of {', '.join(iterative_imputer_classification_models.keys())}."
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
        if fold_groups is not None:
            if isinstance(fold_groups, str):
                self.fold_groups_param = X_before_preprocess[fold_groups]
            else:
                self.fold_groups_param = fold_groups
            if pd.isnull(self.fold_groups_param).any():
                raise ValueError(f"fold_groups cannot contain NaNs.")
        self.fold_shuffle_param = fold_shuffle

        from sklearn.model_selection import (
            StratifiedKFold,
            KFold,
            GroupKFold,
            TimeSeriesSplit,
        )

        if fold_strategy == "kfold":
            self.fold_generator = KFold(
                self.fold_param, random_state=self.seed, shuffle=self.fold_shuffle_param
            )
        elif fold_strategy == "stratifiedkfold":
            self.fold_generator = StratifiedKFold(
                self.fold_param, random_state=self.seed, shuffle=self.fold_shuffle_param
            )
        elif fold_strategy == "groupkfold":
            self.fold_generator = GroupKFold(self.fold_param)
        elif fold_strategy == "timeseries":
            self.fold_generator = TimeSeriesSplit(self.fold_param)
        else:
            self.fold_generator = fold_strategy

        # create create_model_container
        self.create_model_container = []

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

        if self.fix_imbalance_method_param is None:
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

        self._internal_pipeline = make_internal_pipeline(self._internal_pipeline)

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
        )

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
            print("Setup Succesfully Completed! Loading Profile Now... Please Wait!")
        else:
            if verbose:
                print("Setup Succesfully Completed!")

        functions = pd.DataFrame(
            [["session_id", self.seed],]
            + ([["Target", target]] if not self._is_unsupervised() else [])
            + (
                [["Target Type", target_type], ["Label Encoded", label_encoded],]
                if self._ml_usecase == MLUsecase.CLASSIFICATION
                else []
            )
            + [
                ["Original Data", self.data_before_preprocess.shape],
                ["Missing Values", missing_flag],
                ["Numeric Features", str(float_type)],
                ["Categorical Features", str(cat_type)],
            ]
            + (
                [
                    ["Ordinal Features", ordinal_features_grid],
                    ["High Cardinality Features", high_cardinality_features_grid],
                    ["High Cardinality Method", high_cardinality_method_grid],
                ]
                if preprocess
                else []
            )
            + (
                [
                    ["Transformed Train Set", self.X_train.shape],
                    ["Transformed Test Set", self.X_test.shape],
                    ["Shuffle Train-Test", str(data_split_shuffle)],
                    ["Stratify Train-Test", str(data_split_stratify)],
                    ["Fold Generator", type(self.fold_generator).__name__],
                    ["Fold Number", self.fold_param],
                ]
                if not self._is_unsupervised()
                else [["Transformed Data", self.X.shape]]
            )
            + [
                ["CPU Jobs", self.n_jobs_param],
                ["Use GPU", self.gpu_param],
                ["Log Experiment", self.logging_param],
                ["Experiment Name", self.exp_name_log],
                ["USI", self.USI],
            ]
            + (
                [
                    ["Imputation Type", imputation_type],
                    [
                        "Iterative Imputation Iteration",
                        self.iterative_imputation_iters_param
                        if imputation_type == "iterative"
                        else "None",
                    ],
                    ["Numeric Imputer", numeric_imputation],
                    [
                        "Iterative Imputation Numeric Model",
                        imputation_regressor_name
                        if imputation_type == "iterative"
                        else "None",
                    ],
                    ["Categorical Imputer", categorical_imputation],
                    [
                        "Iterative Imputation Categorical Model",
                        imputation_classifier_name
                        if imputation_type == "iterative"
                        else "None",
                    ],
                    ["Unknown Categoricals Handling", unknown_categorical_method_grid],
                    ["Normalize", normalize],
                    ["Normalize Method", normalize_grid],
                    ["Transformation", transformation],
                    ["Transformation Method", transformation_grid],
                    ["PCA", pca],
                    ["PCA Method", pca_method_grid],
                    ["PCA Components", pca_components_grid],
                    ["Ignore Low Variance", ignore_low_variance],
                    ["Combine Rare Levels", combine_rare_levels],
                    ["Rare Level Threshold", rare_level_threshold_grid],
                    ["Numeric Binning", numeric_bin_grid],
                    ["Remove Outliers", remove_outliers],
                    ["Outliers Threshold", outliers_threshold_grid],
                    ["Remove Perfect Collinearity", remove_perfect_collinearity],
                    ["Remove Multicollinearity", remove_multicollinearity],
                    ["Multicollinearity Threshold", multicollinearity_threshold_grid],
                    [
                        "Columns Removed Due to Multicollinearity",
                        multicollinearity_removed_columns,
                    ],
                    ["Clustering", create_clusters],
                    ["Clustering Iteration", cluster_iter_grid],
                    ["Polynomial Features", polynomial_features],
                    ["Polynomial Degree", polynomial_degree_grid],
                    ["Trignometry Features", trigonometry_features],
                    ["Polynomial Threshold", polynomial_threshold_grid],
                    ["Group Features", group_features_grid],
                    ["Feature Selection", feature_selection],
                    ["Features Selection Threshold", feature_selection_threshold_grid],
                    ["Feature Interaction", feature_interaction],
                    ["Feature Ratio", feature_ratio],
                    ["Interaction Threshold", interaction_threshold_grid],
                ]
                if preprocess
                else []
            )
            + (
                [
                    ["Fix Imbalance", self.fix_imbalance_param],
                    ["Fix Imbalance Method", fix_imbalance_model_name],  # type: ignore
                ]
                if self._ml_usecase == MLUsecase.CLASSIFICATION
                else []
            )
            + (
                [
                    ["Transform Target", self.transform_target_param],
                    ["Transform Target Method", self.transform_target_method_param],
                ]
                if self._ml_usecase == MLUsecase.REGRESSION
                else []
            ),
            columns=["Description", "Value"],
        )
        functions_ = functions.style.apply(highlight_max)

        self.display_container.append(functions_)

        display.display(functions_, clear=True)

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
            functions,
            functions_,
            runtime,
            log_profile,
            profile_kwargs,
            log_data,
            display,
        )

        self._setup_ran = True

        self.logger.info(
            f"self.create_model_container: {len(self.create_model_container)}"
        )
        self.logger.info(
            f"self.master_model_container: {len(self.master_model_container)}"
        )
        self.logger.info(f"self.display_container: {len(self.display_container)}")

        self.logger.info(str(self.prep_pipe))
        self.logger.info(
            "setup() succesfully completed......................................"
        )

        gc.collect()

        return self

    def create_model(self, *args, **kwargs):
        return

    def plot_model(
        self,
        estimator,
        plot: str = "auc",
        scale: float = 1,  # added in pycaret==2.1.0
        save: bool = False,
        fold: Optional[Union[int, Any]] = None,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        feature_name: Optional[str] = None,
        label: bool = False,
        use_train_data: bool = False,
        verbose: bool = True,
        system: bool = True,
        display: Optional[Display] = None,  # added in pycaret==2.2.0
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

        save: bool, default = False
            When set to True, Plot is saved as a 'png' file in current working directory.

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
            If None, will use the value set in fold_groups param in setup().

        verbose: bool, default = True
            Progress bar not shown when verbose set to False. 

        system: bool, default = True
            Must remain True all times. Only to be changed by internal functions.

        Returns
        -------
        Visual_Plot
            Prints the visual plot. 
        str:
            If save param is True, will return the name of the saved file.

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

        if plot not in self._available_plots:
            raise ValueError(
                "Plot Not Available. Please see docstring for list of available Plots."
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
            raise TypeError("Label param only accepts True or False.")

        if type(use_train_data) is not bool:
            raise TypeError("use_train_data param only accepts True or False.")

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
                            fig.write_html(plot_filename)
                            self.logger.info(
                                f"Saving '{plot_filename}' in current active directory"
                            )

                        elif system:
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
                            fig.write_html(f"{plot_filename}")
                            self.logger.info(
                                f"Saving '{plot_filename}' in current active directory"
                            )
                        elif system:
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
                            fig.write_html(f"{plot_filename}")
                            self.logger.info(
                                f"Saving '{plot_filename}' in current active directory"
                            )
                        elif system:
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
                            fig.write_html(f"{plot_filename}")
                            self.logger.info(
                                f"Saving '{plot_filename}' in current active directory"
                            )
                        elif system:
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
                            fig.write_html(f"{plot_filename}")
                            self.logger.info(
                                f"Saving '{plot_filename}' in current active directory"
                            )
                        elif system:
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
                            y_test__ = fitted_pipeline_with_model.predict(test_X)
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
                                self.logger.info(
                                    f"Saving '{plot_name}.png' in current active directory"
                                )
                                plt.savefig(f"{plot_name}.png")
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
                            y_test__ = fitted_pipeline_with_model.predict(test_X)
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
                                self.logger.info(
                                    f"Saving '{plot_name}.png' in current active directory"
                                )
                                plt.savefig(f"{plot_name}.png")
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
                            self.logger.info(
                                f"Saving '{plot_name}.png' in current active directory"
                            )
                            plt.savefig(f"{plot_name}.png", bbox_inches="tight")
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
                            self.logger.info(
                                f"Saving '{plot_name}.png' in current active directory"
                            )
                            plt.savefig(f"{plot_name}.png")
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
                            self.logger.info(
                                f"Saving '{plot_name}.png' in current active directory"
                            )
                            plt.savefig(f"{plot_name}.png")
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
            "plot_model() succesfully completed......................................"
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
            If None, will use the value set in fold_groups param in setup().

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
        )

    def predict_model(self, *args, **kwargs) -> pd.DataFrame:
        return pd.DataFrame()

    def finalize_model(self) -> None:
        return

    def automl(self, optimize: str = "Accuracy", use_holdout: bool = False) -> Any:

        """
        This function returns the best model out of all models created in 
        current active environment based on metric defined in optimize parameter. 

        Parameters
        ----------
        optimize : str, default = 'Accuracy'
            Other values you can pass in optimize param are 'AUC', 'Recall', 'Precision',
            'F1', 'Kappa', and 'MCC'.

        use_holdout: bool, default = False
            When set to True, metrics are evaluated on holdout set instead of CV.

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

        scorer = []

        if use_holdout:
            self.logger.info("Model Selection Basis : Holdout set")
            for i in self.master_model_container:
                try:
                    pred_holdout = self.predict_model(i, verbose=False)  # type: ignore
                except:
                    self.logger.warning(
                        f"Model {i} is not fitted, running create_model"
                    )
                    i, _ = self.create_model(  # type: ignore
                        estimator=i,
                        system=False,
                        verbose=False,
                        cross_validation=False,
                        predict=False,
                        groups=self.fold_groups_param,
                    )
                    self.pull(pop=True)
                    pred_holdout = self.predict_model(i, verbose=False)  # type: ignore

                p = self.pull(pop=True)
                p = p[compare_dimension][0]
                scorer.append(p)

        else:
            self.logger.info("Model Selection Basis : CV Results on Training set")
            for i in self.create_model_container:
                r = i[compare_dimension][-2:][0]
                scorer.append(r)

        # returning better model
        if greater_is_better:
            index_scorer = scorer.index(max(scorer))
        else:
            index_scorer = scorer.index(min(scorer))

        automl_result = self.master_model_container[index_scorer]

        automl_model, _ = self.create_model(  # type: ignore
            estimator=automl_result,
            system=False,
            verbose=False,
            cross_validation=False,
            predict=False,
            groups=self.fold_groups_param,
        )

        self.logger.info(str(automl_model))
        self.logger.info(
            "automl() succesfully completed......................................"
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
        param along with the applicable authentication tokens which are passed as a
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
        self, model, model_name: str, model_only: bool = False, verbose: bool = True
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
            model, model_name, None if model_only else self.prep_pipe, verbose
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


class _SupervisedExperiment(_TabularExperiment):
    def __init__(self) -> None:
        super().__init__()
        self.variable_keys = self.variable_keys.union(
            {"y", "X_train", "X_test", "y_train", "y_test",}
        )
        return

    def _calculate_metrics(
        self, y_test, pred, pred_prob, weights: Optional[list] = None,
    ) -> dict:
        """
        Calculate all metrics in _all_metrics.
        """
        from pycaret.internal.utils import calculate_metrics

        try:
            return calculate_metrics(
                metrics=self._all_metrics,
                y_test=y_test,
                pred=pred,
                pred_proba=pred_prob,
                weights=weights,
            )
        except:
            ml_usecase = get_ml_task(y_test)
            if ml_usecase == MLUsecase.CLASSIFICATION:
                metrics = pycaret.containers.metrics.classification.get_all_metric_containers(
                    self.variables, True
                )
            elif ml_usecase == MLUsecase.REGRESSION:
                metrics = pycaret.containers.metrics.regression.get_all_metric_containers(
                    self.variables, True
                )
            return calculate_metrics(
                metrics=metrics,  # type: ignore
                y_test=y_test,
                pred=pred,
                pred_proba=pred_prob,
                weights=weights,
            )

    def _is_unsupervised(self) -> bool:
        return False

    def _choose_better(
        self,
        models_and_results: list,
        compare_dimension: str,
        fold: int,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        display: Optional[Display] = None,
    ):
        """
        When choose_better is set to True, optimize metric in scoregrid is
        compared with base model created using create_model so that the
        functions return the model with better score only. This will ensure 
        model performance is at least equivalent to what is seen in compare_models 
        """

        self.logger.info("choose_better activated")
        display.update_monitor(1, "Compiling Final Results")
        display.display_monitor()

        if not fit_kwargs:
            fit_kwargs = {}

        for i, x in enumerate(models_and_results):
            if not isinstance(x, tuple):
                models_and_results[i] = (x, None)
            elif isinstance(x[0], str):
                models_and_results[i] = (x[1], None)
            elif len(x) != 2:
                raise ValueError(f"{x} must have lenght 2 but has {len(x)}")

        metric = self._get_metric_by_name_or_id(compare_dimension)

        best_result = None
        best_model = None
        for model, result in models_and_results:
            if result is not None and is_fitted(model):
                result = result.loc["Mean"][compare_dimension]
            else:
                self.logger.info(
                    "SubProcess create_model() called =================================="
                )
                model, _ = self.create_model(
                    model,
                    verbose=False,
                    system=False,
                    fold=fold,
                    fit_kwargs=fit_kwargs,
                    groups=groups,
                )
                self.logger.info(
                    "SubProcess create_model() end =================================="
                )
                result = self.pull(pop=True).loc["Mean"][compare_dimension]
            self.logger.info(f"{model} result for {compare_dimension} is {result}")
            if not metric.greater_is_better:
                result *= -1
            if best_result is None or best_result < result:
                best_result = result
                best_model = model

        self.logger.info(f"{best_model} is best model")

        self.logger.info("choose_better completed")
        return best_model

    def _get_cv_n_folds(self, fold, X, y=None, groups=None):
        import pycaret.internal.utils

        return pycaret.internal.utils.get_cv_n_folds(
            fold, default=self.fold_generator, X=X, y=y, groups=groups
        )

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
        _stratify_columns = get_columns_to_stratify_by(
            X_before_preprocess, y_before_preprocess, self.stratify_param, target
        )
        if test_data is None:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_before_preprocess,
                y_before_preprocess,
                test_size=1 - train_size,
                stratify=_stratify_columns,
                random_state=self.seed,
                shuffle=data_split_shuffle,
            )
            train_data = pd.concat([self.X_train, self.y_train], axis=1)
            test_data = pd.concat([self.X_test, self.y_test], axis=1)

        train_data = self.prep_pipe.fit_transform(train_data)
        # workaround to also transform target
        dtypes.final_training_columns.append(target)
        test_data = self.prep_pipe.transform(test_data)

        self.X_train = train_data.drop(target, axis=1)
        self.y_train = train_data[target]

        self.X_test = test_data.drop(target, axis=1)
        self.y_test = test_data[target]

        if self.fold_groups_param is not None:
            self.fold_groups_param = self.fold_groups_param[
                self.fold_groups_param.index.isin(self.X_train.index)
            ]

        display.move_progress()
        self._internal_pipeline.fit(train_data.drop(target, axis=1), train_data[target])
        data = self.prep_pipe.transform(self.data_before_preprocess.copy())
        self.X = data.drop(target, axis=1)
        self.y = data[target]
        return

    def _set_up_mlflow(
        self,
        functions,
        functions_,
        runtime,
        log_profile,
        profile_kwargs,
        log_data,
        display,
    ) -> None:
        # log into experiment
        self.experiment__.append(("Setup Config", functions))
        self.experiment__.append(("X_training Set", self.X_train))
        self.experiment__.append(("y_training Set", self.y_train))
        self.experiment__.append(("X_test Set", self.X_test))
        self.experiment__.append(("y_test Set", self.y_test))
        self.experiment__.append(("Transformation Pipeline", self.prep_pipe))

        if self.logging_param:

            self.logger.info("Logging experiment in MLFlow")

            import mlflow

            try:
                self.exp_id = mlflow.create_experiment(self.exp_name_log)
            except:
                self.exp_id = None
                self.logger.warning("Couldn't create mlflow experiment. Exception:")
                self.logger.warning(traceback.format_exc())

            # mlflow logging
            mlflow.set_experiment(self.exp_name_log)

            run_name_ = f"Session Initialized {self.USI}"

            with mlflow.start_run(run_name=run_name_) as run:

                # Get active run to log as tag
                RunID = mlflow.active_run().info.run_id

                k = functions.copy()
                k.set_index("Description", drop=True, inplace=True)
                kdict = k.to_dict()
                params = kdict.get("Value")
                mlflow.log_params(params)

                # set tag of compare_models
                mlflow.set_tag("Source", "setup")

                import secrets

                URI = secrets.token_hex(nbytes=4)
                mlflow.set_tag("URI", URI)
                mlflow.set_tag("USI", self.USI)
                mlflow.set_tag("Run Time", runtime)
                mlflow.set_tag("Run ID", RunID)

                # Log the transformation pipeline
                self.logger.info(
                    "SubProcess save_model() called =================================="
                )
                self.save_model(
                    self.prep_pipe, "Transformation Pipeline", verbose=False
                )
                self.logger.info(
                    "SubProcess save_model() end =================================="
                )
                mlflow.log_artifact("Transformation Pipeline.pkl")
                os.remove("Transformation Pipeline.pkl")

                # Log pandas profile
                if log_profile:
                    import pandas_profiling

                    pf = pandas_profiling.ProfileReport(
                        self.data_before_preprocess, **profile_kwargs
                    )
                    pf.to_file("Data Profile.html")
                    mlflow.log_artifact("Data Profile.html")
                    os.remove("Data Profile.html")
                    display.display(functions_, clear=True)

                # Log training and testing set
                if log_data:
                    self.X_train.join(self.y_train).to_csv("Train.csv")
                    self.X_test.join(self.y_test).to_csv("Test.csv")
                    mlflow.log_artifact("Train.csv")
                    mlflow.log_artifact("Test.csv")
                    os.remove("Train.csv")
                    os.remove("Test.csv")
        return

    def compare_models(
        self,
        include: Optional[
            List[Union[str, Any]]
        ] = None,  # changed whitelist to include in pycaret==2.1
        exclude: Optional[
            List[str]
        ] = None,  # changed blacklist to exclude in pycaret==2.1
        fold: Optional[Union[int, Any]] = None,
        round: int = 4,
        cross_validation: bool = True,
        sort: str = "Accuracy",
        n_select: int = 1,
        budget_time: Optional[float] = None,  # added in pycaret==2.1.0
        turbo: bool = True,
        errors: str = "ignore",
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        verbose: bool = True,
        display: Optional[Display] = None,
    ) -> List[Any]:

        """
        This function train all the models available in the model library and scores them 
        using Cross Validation. The output prints a score grid with Accuracy, 
        AUC, Recall, Precision, F1, Kappa and MCC (averaged across folds).
        
        This function returns all of the models compared, sorted by the value of the selected metric.

        When turbo is set to True ('rbfsvm', 'gpc' and 'mlp') are excluded due to longer
        training time. By default turbo param is set to True.

        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> experiment_name = setup(data = juice,  target = 'Purchase')
        >>> best_model = compare_models() 

        This will return the averaged score grid of all the models except 'rbfsvm', 'gpc' 
        and 'mlp'. When turbo param is set to False, all models including 'rbfsvm', 'gpc' 
        and 'mlp' are used but this may result in longer training time.
        
        >>> best_model = compare_models( exclude = [ 'knn', 'gbc' ] , turbo = False) 

        This will return a comparison of all models except K Nearest Neighbour and
        Gradient Boosting Classifier.
        
        >>> best_model = compare_models( exclude = [ 'knn', 'gbc' ] , turbo = True) 

        This will return comparison of all models except K Nearest Neighbour, 
        Gradient Boosting Classifier, SVM (RBF), Gaussian Process Classifier and
        Multi Level Perceptron.
            

        >>> tuned_model = tune_model(create_model('lr'))
        >>> best_model = compare_models( include = [ 'lr', tuned_model ]) 

        This will compare a tuned Linear Regression model with an untuned one.

        Parameters
        ----------
        exclude: list of strings, default = None
            In order to omit certain models from the comparison model ID's can be passed as 
            a list of strings in exclude param. 

        include: list of strings or objects, default = None
            In order to run only certain models for the comparison, the model ID's can be 
            passed as a list of strings in include param. The list can also include estimator
            objects to be compared.

        fold: integer or scikit-learn compatible CV generator, default = None
            Controls cross-validation. If None, will use the CV generator defined in setup().
            If integer, will use KFold CV with that many folds.
            When cross_validation is False, this parameter is ignored.

        round: integer, default = 4
            Number of decimal places the metrics in the score grid will be rounded to.
    
        cross_validation: bool, default = True
            When cross_validation set to False fold parameter is ignored and models are trained
            on entire training dataset, returning metrics calculated using the train (holdout) set.

        sort: str, default = 'Accuracy'
            The scoring measure specified is used for sorting the average score grid
            Other options are 'AUC', 'Recall', 'Precision', 'F1', 'Kappa' and 'MCC'.

        n_select: int, default = 1
            Number of top_n models to return. use negative argument for bottom selection.
            for example, n_select = -3 means bottom 3 models.

        budget_time: int or float, default = None
            If not 0 or None, will terminate execution of the function after budget_time 
            minutes have passed and return results up to that point.

        turbo: bool, default = True
            When turbo is set to True, it excludes estimators that have longer
            training time.

        errors: str, default = 'ignore'
            If 'ignore', will suppress model exceptions and continue.
            If 'raise', will allow exceptions to be raised.

        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model. The parameters will be applied to all models,
            therefore it is recommended to set errors parameter to 'ignore'.

        groups: str or array-like, with shape (n_samples,), default = None
            Optional Group labels for the samples used while splitting the dataset into train/test set.
            If string is passed, will use the data column with that name as the groups.
            Only used if a group based cross-validation generator is used (eg. GroupKFold).
            If None, will use the value set in fold_groups param in setup().

        verbose: bool, default = True
            Score grid is not printed when verbose is set to False.
        
        Returns
        -------
        score_grid
            A table containing the scores of the model across the kfolds. 
            Scoring metrics used are Accuracy, AUC, Recall, Precision, F1, 
            Kappa and MCC. Mean and standard deviation of the scores across 
            the folds are also returned.

        list
            List of fitted model objects that were compared.

        Warnings
        --------
        - compare_models() though attractive, might be time consuming with large 
        datasets. By default turbo is set to True, which excludes models that
        have longer training times. Changing turbo parameter to False may result 
        in very high training times with datasets where number of samples exceed 
        10,000.

        - If target variable is multiclass (more than 2 classes), AUC will be 
        returned as zero (0.0)

        - If cross_validation param is set to False, no models will be logged with MLFlow.

        """

        function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

        self.logger.info("Initializing compare_models()")
        self.logger.info(f"compare_models({function_params_str})")

        self.logger.info("Checking exceptions")

        if not fit_kwargs:
            fit_kwargs = {}

        # checking error for exclude (string)
        available_estimators = self._all_models

        if exclude != None:
            for i in exclude:
                if i not in available_estimators:
                    raise ValueError(
                        f"Estimator Not Available {i}. Please see docstring for list of available estimators."
                    )

        if include != None:
            for i in include:
                if isinstance(i, str):
                    if i not in available_estimators:
                        raise ValueError(
                            f"Estimator {i} Not Available. Please see docstring for list of available estimators."
                        )
                elif not hasattr(i, "fit"):
                    raise ValueError(
                        f"Estimator {i} does not have the required fit() method."
                    )

        # include and exclude together check
        if include is not None and exclude is not None:
            raise TypeError(
                "Cannot use exclude parameter when include is used to compare models."
            )

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

        # checking budget_time parameter
        if (
            budget_time
            and type(budget_time) is not int
            and type(budget_time) is not float
        ):
            raise TypeError(
                "budget_time parameter only accepts integer or float values."
            )

        # checking sort parameter
        if not (isinstance(sort, str) and (sort == "TT" or sort == "TT (Sec)")):
            sort = self._get_metric_by_name_or_id(sort)
            if sort is None:
                raise ValueError(
                    f"Sort method not supported. See docstring for list of available parameters."
                )

        # checking errors parameter
        possible_errors = ["ignore", "raise"]
        if errors not in possible_errors:
            raise ValueError(
                f"errors parameter must be one of: {', '.join(possible_errors)}."
            )

        # checking optimize parameter for multiclass
        if self._is_multiclass():
            if not sort.is_multiclass:
                raise TypeError(
                    f"{sort} metric not supported for multiclass problems. See docstring for list of other optimization parameters."
                )

        """
        
        ERROR HANDLING ENDS HERE
        
        """

        fold = self._get_cv_splitter(fold)

        groups = self._get_groups(groups)

        pd.set_option("display.max_columns", 500)

        self.logger.info("Preparing display monitor")

        len_mod = (
            len({k: v for k, v in self._all_models.items() if v.is_turbo})
            if turbo
            else len(self._all_models)
        )

        if include:
            len_mod = len(include)
        elif exclude:
            len_mod -= len(exclude)

        if not display:
            progress_args = {"max": (4 * len_mod) + 4 + len_mod}
            master_display_columns = (
                ["Model"]
                + [v.display_name for k, v in self._all_metrics.items()]
                + ["TT (Sec)"]
            )
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

        greater_is_worse_columns = {
            v.display_name
            for k, v in self._all_metrics.items()
            if not v.greater_is_better
        }
        greater_is_worse_columns.add("TT (Sec)")

        np.random.seed(self.seed)

        display.move_progress()

        # defining sort parameter (making Precision equivalent to Prec. )

        if not (isinstance(sort, str) and (sort == "TT" or sort == "TT (Sec)")):
            sort_ascending = not sort.greater_is_better
            sort = sort.display_name
        else:
            sort_ascending = True
            sort = "TT (Sec)"

        """
        MONITOR UPDATE STARTS
        """

        display.update_monitor(1, "Loading Estimator")
        display.display_monitor()

        """
        MONITOR UPDATE ENDS
        """

        if include:
            model_library = include
        else:
            if turbo:
                model_library = self._all_models
                model_library = [k for k, v in self._all_models.items() if v.is_turbo]
            else:
                model_library = list(self._all_models.keys())
            if exclude:
                model_library = [x for x in model_library if x not in exclude]

        display.move_progress()

        # create URI (before loop)
        import secrets

        URI = secrets.token_hex(nbytes=4)

        master_display = None
        master_display_ = None

        total_runtime_start = time.time()
        total_runtime = 0
        over_time_budget = False
        if budget_time and budget_time > 0:
            self.logger.info(f"Time budget is {budget_time} minutes")

        for i, model in enumerate(model_library):

            model_id = (
                model
                if (
                    isinstance(model, str)
                    and all(isinstance(m, str) for m in model_library)
                )
                else str(i)
            )
            model_name = self._get_model_name(model)

            if isinstance(model, str):
                self.logger.info(f"Initializing {model_name}")
            else:
                self.logger.info(f"Initializing custom model {model_name}")

            # run_time
            runtime_start = time.time()
            total_runtime += (runtime_start - total_runtime_start) / 60
            self.logger.info(f"Total runtime is {total_runtime} minutes")
            over_time_budget = (
                budget_time and budget_time > 0 and total_runtime > budget_time
            )
            if over_time_budget:
                self.logger.info(
                    f"Total runtime {total_runtime} is over time budget by {total_runtime - budget_time}, breaking loop"
                )
                break
            total_runtime_start = runtime_start

            display.move_progress()

            """
            MONITOR UPDATE STARTS
            """

            display.update_monitor(2, model_name)
            display.display_monitor()

            """
            MONITOR UPDATE ENDS
            """
            display.replace_master_display(None)

            self.logger.info(
                "SubProcess create_model() called =================================="
            )
            if errors == "raise":
                model, model_fit_time = self.create_model(
                    estimator=model,
                    system=False,
                    verbose=False,
                    display=display,
                    fold=fold,
                    round=round,
                    cross_validation=cross_validation,
                    fit_kwargs=fit_kwargs,
                    groups=groups,
                    refit=False,
                )
                model_results = self.pull(pop=True)
            else:
                try:
                    model, model_fit_time = self.create_model(
                        estimator=model,
                        system=False,
                        verbose=False,
                        display=display,
                        fold=fold,
                        round=round,
                        cross_validation=cross_validation,
                        fit_kwargs=fit_kwargs,
                        groups=groups,
                        refit=False,
                    )
                    model_results = self.pull(pop=True)
                    assert np.sum(model_results.iloc[0]) != 0.0
                except:
                    self.logger.warning(
                        f"create_model() for {model} raised an exception or returned all 0.0, trying without fit_kwargs:"
                    )
                    self.logger.warning(traceback.format_exc())
                    try:
                        model, model_fit_time = self.create_model(
                            estimator=model,
                            system=False,
                            verbose=False,
                            display=display,
                            fold=fold,
                            round=round,
                            cross_validation=cross_validation,
                            groups=groups,
                            refit=False,
                        )
                        model_results = self.pull(pop=True)
                    except:
                        self.logger.error(
                            f"create_model() for {model} raised an exception:"
                        )
                        self.logger.error(traceback.format_exc())
                        continue
            self.logger.info(
                "SubProcess create_model() end =================================="
            )

            if model is None:
                over_time_budget = True
                self.logger.info(
                    f"Time budged exceeded in create_model(), breaking loop"
                )
                break

            runtime_end = time.time()
            runtime = np.array(runtime_end - runtime_start).round(2)

            self.logger.info("Creating metrics dataframe")
            if cross_validation:
                compare_models_ = pd.DataFrame(model_results.loc["Mean"]).T
            else:
                compare_models_ = pd.DataFrame(model_results.iloc[0]).T
            compare_models_.insert(
                len(compare_models_.columns), "TT (Sec)", model_fit_time
            )
            compare_models_.insert(0, "Model", model_name)
            compare_models_.insert(0, "Object", [model])
            compare_models_.insert(0, "runtime", runtime)
            compare_models_.index = [model_id]
            if master_display is None:
                master_display = compare_models_
            else:
                master_display = pd.concat(
                    [master_display, compare_models_], ignore_index=False
                )
            master_display = master_display.round(round)
            master_display = master_display.sort_values(
                by=sort, ascending=sort_ascending
            )

            master_display_ = master_display.drop(
                ["Object", "runtime"], axis=1, errors="ignore"
            ).style.set_precision(round)
            master_display_ = master_display_.set_properties(**{"text-align": "left"})
            master_display_ = master_display_.set_table_styles(
                [dict(selector="th", props=[("text-align", "left")])]
            )

            display.replace_master_display(master_display_)

            display.display_master_display()

        display.move_progress()

        def highlight_max(s):
            to_highlight = s == s.max()
            return ["background-color: yellow" if v else "" for v in to_highlight]

        def highlight_min(s):
            to_highlight = s == s.min()
            return ["background-color: yellow" if v else "" for v in to_highlight]

        def highlight_cols(s):
            color = "lightgrey"
            return f"background-color: {color}"

        if master_display_ is not None:
            compare_models_ = (
                master_display_.apply(
                    highlight_max,
                    subset=[
                        x
                        for x in master_display_.columns[1:]
                        if x not in greater_is_worse_columns
                    ],
                )
                .apply(
                    highlight_min,
                    subset=[
                        x
                        for x in master_display_.columns[1:]
                        if x in greater_is_worse_columns
                    ],
                )
                .applymap(highlight_cols, subset=["TT (Sec)"])
            )
        else:
            compare_models_ = pd.DataFrame().style

        display.update_monitor(1, "Compiling Final Models")
        display.display_monitor()

        display.move_progress()

        sorted_models = []

        if master_display is not None:
            if n_select < 0:
                n_select_range = range(
                    len(master_display) - n_select, len(master_display)
                )
            else:
                n_select_range = range(0, n_select)

            for index, row in enumerate(master_display.iterrows()):
                _, row = row
                model = row["Object"]

                results = row.to_frame().T.drop(
                    ["Object", "Model", "runtime", "TT (Sec)"], errors="ignore", axis=1
                )

                avgs_dict_log = {k: v for k, v in results.iloc[0].items()}

                full_logging = False

                if index in n_select_range:
                    display.update_monitor(2, self._get_model_name(model))
                    display.display_monitor()
                    model, model_fit_time = self.create_model(
                        estimator=model,
                        system=False,
                        verbose=False,
                        fold=fold,
                        round=round,
                        cross_validation=False,
                        predict=False,
                        fit_kwargs=fit_kwargs,
                        groups=groups,
                    )
                    sorted_models.append(model)
                    full_logging = True

                if self.logging_param and cross_validation:

                    try:
                        self._mlflow_log_model(
                            model=model,
                            model_results=results,
                            score_dict=avgs_dict_log,
                            source="compare_models",
                            runtime=row["runtime"],
                            model_fit_time=row["TT (Sec)"],
                            _prep_pipe=self.prep_pipe,
                            log_plots=self.log_plots_param if full_logging else False,
                            log_holdout=full_logging,
                            URI=URI,
                            display=display,
                        )
                    except:
                        self.logger.error(
                            f"_mlflow_log_model() for {model} raised an exception:"
                        )
                        self.logger.error(traceback.format_exc())

        if len(sorted_models) == 1:
            sorted_models = sorted_models[0]

        display.display(compare_models_, clear=True)

        pd.reset_option("display.max_columns")

        # store in display container
        self.display_container.append(compare_models_.data)

        self.logger.info(f"create_model_container: {len(self.create_model_container)}")
        self.logger.info(f"master_model_container: {len(self.master_model_container)}")
        self.logger.info(f"display_container: {len(self.display_container)}")

        self.logger.info(str(sorted_models))
        self.logger.info(
            "compare_models() succesfully completed......................................"
        )

        return sorted_models

    def create_model(
        self,
        estimator,
        fold: Optional[Union[int, Any]] = None,
        round: int = 4,
        cross_validation: bool = True,
        predict: bool = True,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        refit: bool = True,
        verbose: bool = True,
        system: bool = True,
        X_train_data: Optional[pd.DataFrame] = None,  # added in pycaret==2.2.0
        y_train_data: Optional[pd.DataFrame] = None,  # added in pycaret==2.2.0
        metrics=None,
        display: Optional[Display] = None,  # added in pycaret==2.2.0
        **kwargs,
    ) -> Any:

        """  
        This is an internal version of the create_model function.

        This function creates a model and scores it using Cross Validation. 
        The output prints a score grid that shows Accuracy, AUC, Recall, Precision, 
        F1, Kappa and MCC by fold (default = 10 Fold). 

        This function returns a trained model object. 

        setup() function must be called before using create_model()

        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> experiment_name = setup(data = juice,  target = 'Purchase')
        >>> lr = create_model('lr')

        This will create a trained Logistic Regression model.

        Parameters
        ----------
        estimator : str / object, default = None
            Enter ID of the estimators available in model library or pass an untrained model 
            object consistent with fit / predict API to train and evaluate model. All 
            estimators support binary or multiclass problem. List of estimators in model 
            library (ID - Name):

            * 'lr' - Logistic Regression             
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

        fold: integer or scikit-learn compatible CV generator, default = None
            Controls cross-validation. If None, will use the CV generator defined in setup().
            If integer, will use KFold CV with that many folds.
            When cross_validation is False, this parameter is ignored.

        round: integer, default = 4
            Number of decimal places the metrics in the score grid will be rounded to. 

        cross_validation: bool, default = True
            When cross_validation set to False fold parameter is ignored and model is trained
            on entire training dataset.

        predict: bool, default = True
            Whether to predict model on holdout if cross_validation == False.

        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.

        groups: str or array-like, with shape (n_samples,), default = None
            Optional Group labels for the samples used while splitting the dataset into train/test set.
            If string is passed, will use the data column with that name as the groups.
            Only used if a group based cross-validation generator is used (eg. GroupKFold).
            If None, will use the value set in fold_groups param in setup().

        refit: bool, default = True
            Whether to refit the model on the entire dataset after CV. Ignored if cross_validation == False.

        verbose: bool, default = True
            Score grid is not printed when verbose is set to False.

        system: bool, default = True
            Must remain True all times. Only to be changed by internal functions.
            If False, method will return a tuple of model and the model fit time.

        X_train_data: pandas.DataFrame, default = None
            If not None, will use this dataframe as training features.
            Intended to be only changed by internal functions.

        y_train_data: pandas.DataFrame, default = None
            If not None, will use this dataframe as training target.
            Intended to be only changed by internal functions.

        **kwargs: 
            Additional keyword arguments to pass to the estimator.

        Returns
        -------
        score_grid
            A table containing the scores of the model across the kfolds. 
            Scoring metrics used are Accuracy, AUC, Recall, Precision, F1, 
            Kappa and MCC. Mean and standard deviation of the scores across 
            the folds are highlighted in yellow.

        model
            trained model object

        Warnings
        --------
        - 'svm' and 'ridge' doesn't support predict_proba method. As such, AUC will be
        returned as zero (0.0)
        
        - If target variable is multiclass (more than 2 classes), AUC will be returned 
        as zero (0.0)

        - 'rbfsvm' and 'gpc' uses non-linear kernel and hence the fit time complexity is 
        more than quadratic. These estimators are hard to scale on datasets with more 
        than 10,000 samples.

        - If cross_validation param is set to False, model will not be logged with MLFlow.

        """

        function_params_str = ", ".join(
            [
                f"{k}={v}"
                for k, v in locals().items()
                if k not in ("X_train_data", "y_train_data")
            ]
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

        # checking system parameter
        if type(system) is not bool:
            raise TypeError("System parameter can only take argument as True or False.")

        # checking cross_validation parameter
        if type(cross_validation) is not bool:
            raise TypeError(
                "cross_validation parameter can only take argument as True or False."
            )

        """
        
        ERROR HANDLING ENDS HERE
        
        """

        groups = self._get_groups(groups)

        if not display:
            progress_args = {"max": 4}
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

        self.logger.info("Importing libraries")

        # general dependencies

        np.random.seed(self.seed)

        self.logger.info("Copying training dataset")

        # Storing X_train and y_train in data_X and data_y parameter
        data_X = self.X_train.copy() if X_train_data is None else X_train_data.copy()
        data_y = self.y_train.copy() if y_train_data is None else y_train_data.copy()

        # reset index
        data_X.reset_index(drop=True, inplace=True)
        data_y.reset_index(drop=True, inplace=True)

        if metrics is None:
            metrics = self._all_metrics

        display.move_progress()

        self.logger.info("Defining folds")

        # cross validation setup starts here
        cv = self._get_cv_splitter(fold)

        self.logger.info("Declaring metric variables")

        """
        MONITOR UPDATE STARTS
        """
        display.update_monitor(1, "Selecting Estimator")
        display.display_monitor()
        """
        MONITOR UPDATE ENDS
        """

        self.logger.info("Importing untrained model")

        if isinstance(estimator, str) and estimator in available_estimators:
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

        # workaround for an issue with set_params in cuML
        model = clone(model)

        display.update_monitor(2, full_name)
        display.display_monitor()

        if self.transform_target_param and not isinstance(
            model, TransformedTargetRegressor
        ):
            model = PowerTransformedTargetRegressor(
                regressor=model,
                power_transformer_method=self.transform_target_method_param,
            )

        self.logger.info(f"{full_name} Imported succesfully")

        display.move_progress()

        """
        MONITOR UPDATE STARTS
        """
        if not cross_validation:
            display.update_monitor(1, f"Fitting {str(full_name)}")
        else:
            display.update_monitor(1, "Initializing CV")

        display.display_monitor()
        """
        MONITOR UPDATE ENDS
        """

        if not cross_validation:

            with estimator_pipeline(
                self._internal_pipeline, model
            ) as pipeline_with_model:
                fit_kwargs = get_pipeline_fit_kwargs(pipeline_with_model, fit_kwargs)
                self.logger.info("Cross validation set to False")

                self.logger.info("Fitting Model")
                model_fit_start = time.time()
                with io.capture_output():
                    pipeline_with_model.fit(data_X, data_y, **fit_kwargs)
                model_fit_end = time.time()

                model_fit_time = np.array(model_fit_end - model_fit_start).round(2)

                display.move_progress()

                if predict:
                    self.predict_model(pipeline_with_model, verbose=False)
                    model_results = self.pull(pop=True).drop("Model", axis=1)

                    self.display_container.append(model_results)

                    display.display(
                        model_results,
                        clear=system,
                        override=False if not system else None,
                    )

                    self.logger.info(
                        f"display_container: {len(self.display_container)}"
                    )

            display.move_progress()

            self.logger.info(str(model))
            self.logger.info(
                "create_models() succesfully completed......................................"
            )

            gc.collect()

            if not system:
                return (model, model_fit_time)
            return model

        """
        MONITOR UPDATE STARTS
        """
        display.update_monitor(
            1,
            f"Fitting {self._get_cv_n_folds(fold, data_X, y=data_y, groups=groups)} Folds",
        )
        display.display_monitor()
        """
        MONITOR UPDATE ENDS
        """

        from sklearn.model_selection import cross_validate

        metrics_dict = dict([(k, v.scorer) for k, v in metrics.items()])

        self.logger.info("Starting cross validation")

        n_jobs = self._gpu_n_jobs_param
        from sklearn.gaussian_process import (
            GaussianProcessClassifier,
            GaussianProcessRegressor,
        )

        # special case to prevent running out of memory
        if isinstance(model, (GaussianProcessClassifier, GaussianProcessRegressor)):
            n_jobs = 1

        with estimator_pipeline(self._internal_pipeline, model) as pipeline_with_model:
            fit_kwargs = get_pipeline_fit_kwargs(pipeline_with_model, fit_kwargs)
            self.logger.info(f"Cross validating with {cv}, n_jobs={n_jobs}")

            model_fit_start = time.time()
            scores = cross_validate(
                pipeline_with_model,
                data_X,
                data_y,
                cv=cv,
                groups=groups,
                scoring=metrics_dict,
                fit_params=fit_kwargs,
                n_jobs=n_jobs,
                return_train_score=False,
                error_score=0,
            )
            model_fit_end = time.time()
            model_fit_time = np.array(model_fit_end - model_fit_start).round(2)

            score_dict = {
                v.display_name: scores[f"test_{k}"] * (1 if v.greater_is_better else -1)
                for k, v in metrics.items()
            }

            self.logger.info("Calculating mean and std")

            avgs_dict = {k: [np.mean(v), np.std(v)] for k, v in score_dict.items()}

            display.move_progress()

            self.logger.info("Creating metrics dataframe")

            model_results = pd.DataFrame(score_dict)
            model_avgs = pd.DataFrame(avgs_dict, index=["Mean", "SD"],)

            model_results = model_results.append(model_avgs)
            model_results = model_results.round(round)

            # yellow the mean
            model_results = color_df(model_results, "yellow", ["Mean"], axis=1)
            model_results = model_results.set_precision(round)

            if refit:
                # refitting the model on complete X_train, y_train
                display.update_monitor(1, "Finalizing Model")
                display.display_monitor()
                model_fit_start = time.time()
                self.logger.info("Finalizing model")
                with io.capture_output():
                    pipeline_with_model.fit(data_X, data_y, **fit_kwargs)
                model_fit_end = time.time()

                model_fit_time = np.array(model_fit_end - model_fit_start).round(2)
            else:
                model_fit_time /= self._get_cv_n_folds(
                    cv, data_X, y=data_y, groups=groups
                )

            # end runtime
            runtime_end = time.time()
            runtime = np.array(runtime_end - runtime_start).round(2)

        # mlflow logging
        if self.logging_param and system and refit:

            avgs_dict_log = avgs_dict.copy()
            avgs_dict_log = {k: v[0] for k, v in avgs_dict_log.items()}

            try:
                self._mlflow_log_model(
                    model=model,
                    model_results=model_results,
                    score_dict=avgs_dict_log,
                    source="create_model",
                    runtime=runtime,
                    model_fit_time=model_fit_time,
                    _prep_pipe=self.prep_pipe,
                    log_plots=self.log_plots_param,
                    display=display,
                )
            except:
                self.logger.error(
                    f"_mlflow_log_model() for {model} raised an exception:"
                )
                self.logger.error(traceback.format_exc())

        display.move_progress()

        self.logger.info("Uploading results into container")

        # storing results in create_model_container
        self.create_model_container.append(model_results.data)
        self.display_container.append(model_results.data)

        # storing results in master_model_container
        self.logger.info("Uploading model into container now")
        self.master_model_container.append(model)

        display.display(
            model_results, clear=system, override=False if not system else None
        )

        self.logger.info(f"create_model_container: {len(self.create_model_container)}")
        self.logger.info(f"master_model_container: {len(self.master_model_container)}")
        self.logger.info(f"display_container: {len(self.display_container)}")

        self.logger.info(str(model))
        self.logger.info(
            "create_model() succesfully completed......................................"
        )
        gc.collect()

        if not system:
            return (model, model_fit_time)

        return model

    def tune_model(
        self,
        estimator,
        fold: Optional[Union[int, Any]] = None,
        round: int = 4,
        n_iter: int = 10,
        custom_grid: Optional[Union[Dict[str, list], Any]] = None,
        optimize: str = "Accuracy",
        custom_scorer=None,  # added in pycaret==2.1 - depreciated
        search_library: str = "scikit-learn",
        search_algorithm: Optional[str] = None,
        early_stopping: Any = False,
        early_stopping_max_iters: int = 10,
        choose_better: bool = False,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        return_tuner: bool = False,
        verbose: bool = True,
        tuner_verbose: Union[int, bool] = True,
        display: Optional[Display] = None,
        **kwargs,
    ) -> Any:

        """
        This function tunes the hyperparameters of a model and scores it using Cross Validation.
        The output prints a score grid that shows Accuracy, AUC, Recall
        Precision, F1, Kappa and MCC by fold (by default = 10 Folds).

        This function returns a trained model object.

        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> experiment_name = setup(data = juice,  target = 'Purchase')
        >>> xgboost = create_model('xgboost')
        >>> tuned_xgboost = tune_model(xgboost) 

        This will tune the hyperparameters of Extreme Gradient Boosting Classifier.


        Parameters
        ----------
        estimator : object, default = None

        fold: integer or scikit-learn compatible CV generator, default = None
            Controls cross-validation. If None, will use the CV generator defined in setup().
            If integer, will use KFold CV with that many folds.
            When cross_validation is False, this parameter is ignored.

        round: integer, default = 4
            Number of decimal places the metrics in the score grid will be rounded to. 

        n_iter: integer, default = 10
            Number of iterations within the Random Grid Search. For every iteration, 
            the model randomly selects one value from the pre-defined grid of 
            hyperparameters.

        custom_grid: dictionary, default = None
            To use custom hyperparameters for tuning pass a dictionary with parameter name
            and values to be iterated. When set to None it uses pre-defined tuning grid.
            Custom grids must be in a format supported by the chosen search library.

        optimize: str, default = 'Accuracy'
            Measure used to select the best model through hyperparameter tuning.
            Can be either a string representing a metric or a custom scorer object
            created using sklearn.make_scorer. 

        custom_scorer: object, default = None
            Will be eventually depreciated.
            custom_scorer can be passed to tune hyperparameters of the model. It must be
            created using sklearn.make_scorer. 

        search_library: str, default = 'scikit-learn'
            The search library used to tune hyperparameters.
            Possible values:

            - 'scikit-learn' - default, requires no further installation
            - 'scikit-optimize' - scikit-optimize. ``pip install scikit-optimize`` https://scikit-optimize.github.io/stable/
            - 'tune-sklearn' - Ray Tune scikit API. Does not support GPU models.
            ``pip install tune-sklearn ray[tune]`` https://github.com/ray-project/tune-sklearn
            - 'optuna' - Optuna. ``pip install optuna`` https://optuna.org/

        search_algorithm: str, default = None
            The search algorithm to be used for finding the best hyperparameters.
            Selection of search algorithms depends on the search_library parameter.
            Some search algorithms require additional libraries to be installed.
            If None, will use search library-specific default algorith.
            'scikit-learn' possible values:

            - 'random' - random grid search (default)
            - 'grid' - grid search

            'scikit-optimize' possible values:

            - 'bayesian' - Bayesian search (default)

            'tune-sklearn' possible values:

            - 'random' - random grid search (default)
            - 'grid' - grid search
            - 'bayesian' - Bayesian search using scikit-optimize
            ``pip install scikit-optimize``
            - 'hyperopt' - Tree-structured Parzen Estimator search using Hyperopt 
            ``pip install hyperopt``
            - 'bohb' - Bayesian search using HpBandSter 
            ``pip install hpbandster ConfigSpace``

            'optuna' possible values:

            - 'random' - randomized search
            - 'tpe' - Tree-structured Parzen Estimator search (default)

        early_stopping: bool or str or object, default = False
            Use early stopping to stop fitting to a hyperparameter configuration 
            if it performs poorly. Ignored if search_library is ``scikit-learn``, or
            if the estimator doesn't have partial_fit attribute.
            If False or None, early stopping will not be used.
            Can be either an object accepted by the search library or one of the
            following:

            - 'asha' for Asynchronous Successive Halving Algorithm
            - 'hyperband' for Hyperband
            - 'median' for median stopping rule
            - If False or None, early stopping will not be used.

            More info for Optuna - https://optuna.readthedocs.io/en/stable/reference/pruners.html
            More info for Ray Tune (tune-sklearn) - https://docs.ray.io/en/master/tune/api_docs/schedulers.html

        early_stopping_max_iters: int, default = 10
            Maximum number of epochs to run for each sampled configuration.
            Ignored if early_stopping is False or None.

        choose_better: bool, default = False
            When set to set to True, base estimator is returned when the performance doesn't 
            improve by tune_model. This gurantees the returned object would perform atleast 
            equivalent to base estimator created using create_model or model returned by 
            compare_models.

        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the tuner.

        groups: str or array-like, with shape (n_samples,), default = None
            Optional Group labels for the samples used while splitting the dataset into train/test set.
            If string is passed, will use the data column with that name as the groups.
            Only used if a group based cross-validation generator is used (eg. GroupKFold).
            If None, will use the value set in fold_groups param in setup().

        return_tuner: bool, default = False
            If True, will reutrn a tuple of (model, tuner_object). Otherwise,
            will return just the best model.

        verbose: bool, default = True
            Score grid is not printed when verbose is set to False.

        tuner_verbose: bool or in, default = True
            If True or above 0, will print messages from the tuner. Higher values
            print more messages. Ignored if verbose param is False.

        **kwargs: 
            Additional keyword arguments to pass to the optimizer.

        Returns
        -------
        score_grid
            A table containing the scores of the model across the kfolds. 
            Scoring metrics used are Accuracy, AUC, Recall, Precision, F1, 
            Kappa and MCC. Mean and standard deviation of the scores across 
            the folds are also returned.

        model
            Trained and tuned model object.

        tuner_object
            Only if return_tuner param is True. The object used for tuning.

        Notes
        -----

        - If a StackingClassifier is passed, the hyperparameters of the meta model (final_estimator)
        will be tuned.
        
        - If a VotingClassifier is passed, the weights will be tuned.

        Warnings
        --------

        - Using 'Grid' search algorithm with default parameter grids may result in very
        long computation.


        """
        function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

        self.logger.info("Initializing tune_model()")
        self.logger.info(f"tune_model({function_params_str})")

        self.logger.info("Checking exceptions")

        # run_time
        runtime_start = time.time()

        if not fit_kwargs:
            fit_kwargs = {}

        # checking estimator if string
        if type(estimator) is str:
            raise TypeError(
                "The behavior of tune_model in version 1.0.1 is changed. Please pass trained model object."
            )

        # Check for estimator
        if not hasattr(estimator, "fit"):
            raise ValueError(
                f"Estimator {estimator} does not have the required fit() method."
            )

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

        # checking n_iter parameter
        if type(n_iter) is not int:
            raise TypeError("n_iter parameter only accepts integer value.")

        # checking early_stopping parameter
        possible_early_stopping = ["asha", "Hyperband", "Median"]
        if (
            isinstance(early_stopping, str)
            and early_stopping not in possible_early_stopping
        ):
            raise TypeError(
                f"early_stopping parameter must be one of {', '.join(possible_early_stopping)}"
            )

        # checking early_stopping_max_iters parameter
        if type(early_stopping_max_iters) is not int:
            raise TypeError(
                "early_stopping_max_iters parameter only accepts integer value."
            )

        # checking search_library parameter
        possible_search_libraries = [
            "scikit-learn",
            "scikit-optimize",
            "tune-sklearn",
            "optuna",
        ]
        search_library = search_library.lower()
        if search_library not in possible_search_libraries:
            raise ValueError(
                f"search_library parameter must be one of {', '.join(possible_search_libraries)}"
            )

        if search_library == "scikit-optimize":
            try:
                import skopt
            except ImportError:
                raise ImportError(
                    "'scikit-optimize' requires scikit-optimize package to be installed. Do: pip install scikit-optimize"
                )

            if not search_algorithm:
                search_algorithm = "bayesian"

            possible_search_algorithms = ["bayesian"]
            if search_algorithm not in possible_search_algorithms:
                raise ValueError(
                    f"For 'scikit-optimize' search_algorithm parameter must be one of {', '.join(possible_search_algorithms)}"
                )

        elif search_library == "tune-sklearn":
            try:
                import tune_sklearn
            except ImportError:
                raise ImportError(
                    "'tune-sklearn' requires tune_sklearn package to be installed. Do: pip install tune-sklearn ray[tune]"
                )

            if not search_algorithm:
                search_algorithm = "random"

            possible_search_algorithms = [
                "random",
                "grid",
                "bayesian",
                "hyperopt",
                "bohb",
            ]
            if search_algorithm not in possible_search_algorithms:
                raise ValueError(
                    f"For 'tune-sklearn' search_algorithm parameter must be one of {', '.join(possible_search_algorithms)}"
                )

            if search_algorithm == "bohb":
                try:
                    from ray.tune.suggest.bohb import TuneBOHB
                    from ray.tune.schedulers import HyperBandForBOHB
                    import ConfigSpace as CS
                    import hpbandster
                except ImportError:
                    raise ImportError(
                        "It appears that either HpBandSter or ConfigSpace is not installed. Do: pip install hpbandster ConfigSpace"
                    )
            elif search_algorithm == "hyperopt":
                try:
                    from ray.tune.suggest.hyperopt import HyperOptSearch
                    from hyperopt import hp
                except ImportError:
                    raise ImportError(
                        "It appears that hyperopt is not installed. Do: pip install hyperopt"
                    )
            elif search_algorithm == "bayesian":
                try:
                    import skopt
                except ImportError:
                    raise ImportError(
                        "It appears that scikit-optimize is not installed. Do: pip install scikit-optimize"
                    )

        elif search_library == "optuna":
            try:
                import optuna
            except ImportError:
                raise ImportError(
                    "'optuna' requires optuna package to be installed. Do: pip install optuna"
                )

            if not search_algorithm:
                search_algorithm = "tpe"

            possible_search_algorithms = ["random", "tpe"]
            if search_algorithm not in possible_search_algorithms:
                raise ValueError(
                    f"For 'optuna' search_algorithm parameter must be one of {', '.join(possible_search_algorithms)}"
                )
        else:
            if not search_algorithm:
                search_algorithm = "random"

            possible_search_algorithms = ["random", "grid"]
            if search_algorithm not in possible_search_algorithms:
                raise ValueError(
                    f"For 'scikit-learn' search_algorithm parameter must be one of {', '.join(possible_search_algorithms)}"
                )

        if custom_scorer is not None:
            optimize = custom_scorer
            warnings.warn(
                "custom_scorer parameter will be depreciated, use optimize instead",
                DeprecationWarning,
                stacklevel=2,
            )

        if isinstance(optimize, str):
            # checking optimize parameter
            optimize = self._get_metric_by_name_or_id(optimize)
            if optimize is None:
                raise ValueError(
                    "Optimize method not supported. See docstring for list of available parameters."
                )

            # checking optimize parameter for multiclass
            if self._is_multiclass():
                if not optimize.is_multiclass:
                    raise TypeError(
                        "Optimization metric not supported for multiclass problems. See docstring for list of other optimization parameters."
                    )
        else:
            self.logger.info(f"optimize set to user defined function {optimize}")

        # checking verbose parameter
        if type(verbose) is not bool:
            raise TypeError(
                "verbose parameter can only take argument as True or False."
            )

        # checking verbose parameter
        if type(return_tuner) is not bool:
            raise TypeError(
                "return_tuner parameter can only take argument as True or False."
            )

        if not verbose:
            tuner_verbose = 0

        if type(tuner_verbose) not in (bool, int):
            raise TypeError("tuner_verbose parameter must be a bool or an int.")

        tuner_verbose = int(tuner_verbose)

        if tuner_verbose < 0:
            tuner_verbose = 0
        elif tuner_verbose > 2:
            tuner_verbose = 2

        """
        
        ERROR HANDLING ENDS HERE
        
        """

        fold = self._get_cv_splitter(fold)

        groups = self._get_groups(groups)

        if not display:
            progress_args = {"max": 3 + 4}
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

        # ignore warnings

        warnings.filterwarnings("ignore")

        import logging

        np.random.seed(self.seed)

        self.logger.info("Copying training dataset")
        # Storing X_train and y_train in data_X and data_y parameter
        data_X = self.X_train.copy()
        data_y = self.y_train.copy()

        # reset index
        data_X.reset_index(drop=True, inplace=True)
        data_y.reset_index(drop=True, inplace=True)

        display.move_progress()

        # setting optimize parameter

        compare_dimension = optimize.display_name
        optimize = optimize.scorer

        # convert trained estimator into string name for grids

        self.logger.info("Checking base model")

        model = clone(estimator)
        is_stacked_model = False

        base_estimator = model

        if hasattr(base_estimator, "final_estimator"):
            self.logger.info("Model is stacked, using the definition of the meta-model")
            is_stacked_model = True
            base_estimator = base_estimator.final_estimator

        estimator_id = self._get_model_id(base_estimator)

        estimator_definition = self._all_models_internal[estimator_id]
        estimator_name = estimator_definition.name
        self.logger.info(f"Base model : {estimator_name}")

        display.update_monitor(2, estimator_name)
        display.display_monitor()

        display.move_progress()

        self.logger.info("Declaring metric variables")

        """
        MONITOR UPDATE STARTS
        """

        display.update_monitor(1, "Searching Hyperparameters")
        display.display_monitor()

        """
        MONITOR UPDATE ENDS
        """

        self.logger.info("Defining Hyperparameters")

        from pycaret.internal.tunable import VotingClassifier, VotingRegressor

        def total_combintaions_in_grid(grid):
            nc = 1

            def get_iter(x):
                if isinstance(x, dict):
                    return x.values()
                return x

            for v in get_iter(grid):
                if isinstance(v, dict):
                    for v2 in get_iter(v):
                        nc *= len(v2)
                else:
                    nc *= len(v)
            return nc

        def get_ccp_alphas(estimator):
            path = estimator.cost_complexity_pruning_path(data_X, data_y)
            ccp_alphas, _ = path.ccp_alphas, path.impurities
            return list(ccp_alphas[:-1])

        if custom_grid is not None:
            if not isinstance(custom_grid, dict):
                raise TypeError(f"custom_grid must be a dict, got {type(custom_grid)}.")
            param_grid = custom_grid
            if not (
                search_library == "scikit-learn"
                or (
                    search_library == "tune-sklearn"
                    and (search_algorithm == "grid" or search_algorithm == "random")
                )
            ):
                param_grid = {
                    k: CategoricalDistribution(v)
                    if not isinstance(v, Distribution)
                    else v
                    for k, v in param_grid.items()
                }
            elif any(isinstance(v, Distribution) for k, v in param_grid.items()):
                raise TypeError(
                    f"For the combination of search_library {search_library} and search_algorithm {search_algorithm}, PyCaret Distribution objects are not supported. Pass a list or other object supported by the search library (in most cases, an object with a 'rvs' function)."
                )
        elif search_library == "scikit-learn" or (
            search_library == "tune-sklearn"
            and (search_algorithm == "grid" or search_algorithm == "random")
        ):
            param_grid = estimator_definition.tune_grid
            if isinstance(base_estimator, (VotingClassifier, VotingRegressor)):
                # special case to handle VotingClassifier, as weights need to be
                # generated dynamically
                param_grid = {
                    f"weight_{i}": np.arange(0.01, 1, 0.01)
                    for i, e in enumerate(base_estimator.estimators)
                }
            # if hasattr(base_estimator, "cost_complexity_pruning_path"):
            #     # special case for Tree-based models
            #     param_grid["ccp_alpha"] = get_ccp_alphas(base_estimator)
            #     if "min_impurity_decrease" in param_grid:
            #         param_grid.pop("min_impurity_decrease")

            if search_algorithm != "grid":
                tc = total_combintaions_in_grid(param_grid)
                if tc <= n_iter:
                    self.logger.info(
                        f"{n_iter} is bigger than total combinations {tc}, setting search algorithm to grid"
                    )
                    search_algorithm = "grid"
        else:
            param_grid = estimator_definition.tune_distribution

            if isinstance(base_estimator, (VotingClassifier, VotingRegressor)):
                # special case to handle VotingClassifier, as weights need to be
                # generated dynamically
                param_grid = {
                    f"weight_{i}": UniformDistribution(0.000000001, 1)
                    for i, e in enumerate(base_estimator.estimators)
                }
            # if hasattr(base_estimator, "cost_complexity_pruning_path"):
            #     # special case for Tree-based models
            #     param_grid["ccp_alpha"] = CategoricalDistribution(
            #         get_ccp_alphas(base_estimator)
            #     )
            #     if "min_impurity_decrease" in param_grid:
            #         param_grid.pop("min_impurity_decrease")

        if not param_grid:
            raise ValueError(
                "parameter grid for tuning is empty. If passing custom_grid, make sure that it is not empty. If not passing custom_grid, the passed estimator does not have a built-in tuning grid."
            )

        suffixes = []

        if is_stacked_model:
            self.logger.info(
                "Stacked model passed, will tune meta model hyperparameters"
            )
            suffixes.append("final_estimator")

        gc.collect()

        with estimator_pipeline(self._internal_pipeline, model) as pipeline_with_model:
            extra_params = {}

            fit_kwargs = get_pipeline_fit_kwargs(pipeline_with_model, fit_kwargs)

            actual_estimator_label = get_pipeline_estimator_label(pipeline_with_model)

            suffixes.append(actual_estimator_label)

            suffixes = "__".join(reversed(suffixes))

            param_grid = {f"{suffixes}__{k}": v for k, v in param_grid.items()}

            search_kwargs = {**estimator_definition.tune_args, **kwargs}

            if custom_grid is not None:
                self.logger.info(f"custom_grid: {param_grid}")

            n_jobs = (
                self._gpu_n_jobs_param
                if estimator_definition.is_gpu_enabled
                else self.n_jobs_param
            )

            from sklearn.gaussian_process import GaussianProcessClassifier

            # special case to prevent running out of memory
            if isinstance(pipeline_with_model.steps[-1][1], GaussianProcessClassifier):
                n_jobs = 1

            self.logger.info(f"Tuning with n_jobs={n_jobs}")

            if search_library == "optuna":
                # suppress output
                logging.getLogger("optuna").setLevel(logging.WARNING)

                pruner_translator = {
                    "asha": optuna.pruners.SuccessiveHalvingPruner(),  # type: ignore
                    "hyperband": optuna.pruners.HyperbandPruner(),  # type: ignore
                    "median": optuna.pruners.MedianPruner(),  # type: ignore
                    False: optuna.pruners.NopPruner(),  # type: ignore
                    None: optuna.pruners.NopPruner(),  # type: ignore
                }
                pruner = early_stopping
                if pruner in pruner_translator:
                    pruner = pruner_translator[early_stopping]

                sampler_translator = {
                    "tpe": optuna.samplers.TPESampler(seed=self.seed),  # type: ignore
                    "random": optuna.samplers.RandomSampler(seed=self.seed),  # type: ignore
                }
                sampler = sampler_translator[search_algorithm]

                try:
                    param_grid = get_optuna_distributions(param_grid)
                except:
                    self.logger.warning(
                        "Couldn't convert param_grid to specific library distributions. Exception:"
                    )
                    self.logger.warning(traceback.format_exc())

                study = optuna.create_study(  # type: ignore
                    direction="maximize", sampler=sampler, pruner=pruner
                )

                self.logger.info("Initializing optuna.integration.OptunaSearchCV")
                model_grid = optuna.integration.OptunaSearchCV(  # type: ignore
                    estimator=pipeline_with_model,
                    param_distributions=param_grid,
                    cv=fold,
                    enable_pruning=early_stopping
                    and can_early_stop(
                        pipeline_with_model, True, False, False, param_grid
                    ),
                    max_iter=early_stopping_max_iters,
                    n_jobs=n_jobs,
                    n_trials=n_iter,
                    random_state=self.seed,
                    scoring=optimize,
                    study=study,
                    refit=False,
                    verbose=tuner_verbose,
                    error_score="raise",
                    **search_kwargs,
                )

            elif search_library == "tune-sklearn":

                early_stopping_translator = {
                    "asha": "ASHAScheduler",
                    "hyperband": "HyperBandScheduler",
                    "median": "MedianStoppingRule",
                }
                if early_stopping in early_stopping_translator:
                    early_stopping = early_stopping_translator[early_stopping]

                do_early_stop = early_stopping and can_early_stop(
                    pipeline_with_model, True, True, True, param_grid
                )

                if not do_early_stop and search_algorithm == "bohb":
                    raise ValueError(
                        "'bohb' requires early_stopping = True and the estimator to support early stopping (has partial_fit, warm_start or is an XGBoost model)."
                    )

                elif early_stopping and can_early_stop(
                    pipeline_with_model, False, True, False, param_grid
                ):
                    if "actual_estimator__n_estimators" in param_grid:
                        if custom_grid is None:
                            extra_params[
                                "actual_estimator__n_estimators"
                            ] = pipeline_with_model.get_params()[
                                "actual_estimator__n_estimators"
                            ]
                            param_grid.pop("actual_estimator__n_estimators")
                        else:
                            raise ValueError(
                                "Param grid cannot contain n_estimators or max_iter if early_stopping is True and the model is warm started. Use early_stopping_max_iters params to set the upper bound of n_estimators or max_iter."
                            )
                    if "actual_estimator__max_iter" in param_grid:
                        if custom_grid is None:
                            param_grid.pop("actual_estimator__max_iter")
                        else:
                            raise ValueError(
                                "Param grid cannot contain n_estimators or max_iter if early_stopping is True and the model is warm started. Use early_stopping_max_iters params to set the upper bound of n_estimators or max_iter."
                            )

                if not do_early_stop:
                    # enable ray local mode
                    n_jobs = 1
                elif n_jobs == -1:
                    n_jobs = int(math.ceil(multiprocessing.cpu_count() / 2))

                TuneSearchCV = get_tune_sklearn_tunesearchcv()
                TuneGridSearchCV = get_tune_sklearn_tunegridsearchcv()

                with true_warm_start(
                    pipeline_with_model
                ) if do_early_stop else nullcontext():
                    if search_algorithm == "grid":

                        self.logger.info("Initializing tune_sklearn.TuneGridSearchCV")
                        model_grid = TuneGridSearchCV(
                            estimator=pipeline_with_model,
                            param_grid=param_grid,
                            early_stopping=do_early_stop,
                            scoring=optimize,
                            cv=fold,
                            max_iters=early_stopping_max_iters,
                            n_jobs=n_jobs,
                            use_gpu=self.gpu_param,
                            refit=True,
                            verbose=tuner_verbose,
                            # pipeline_detection=False,
                            **search_kwargs,
                        )
                    elif search_algorithm == "hyperopt":
                        try:
                            param_grid = get_hyperopt_distributions(param_grid)
                        except:
                            self.logger.warning(
                                "Couldn't convert param_grid to specific library distributions. Exception:"
                            )
                            self.logger.warning(traceback.format_exc())
                        self.logger.info(
                            "Initializing tune_sklearn.TuneSearchCV, hyperopt"
                        )
                        model_grid = TuneSearchCV(
                            estimator=pipeline_with_model,
                            search_optimization="hyperopt",
                            param_distributions=param_grid,
                            n_trials=n_iter,
                            early_stopping=do_early_stop,
                            scoring=optimize,
                            cv=fold,
                            random_state=self.seed,
                            max_iters=early_stopping_max_iters,
                            n_jobs=n_jobs,
                            use_gpu=self.gpu_param,
                            refit=True,
                            verbose=tuner_verbose,
                            # pipeline_detection=False,
                            **search_kwargs,
                        )
                    elif search_algorithm == "bayesian":
                        try:
                            param_grid = get_skopt_distributions(param_grid)
                        except:
                            self.logger.warning(
                                "Couldn't convert param_grid to specific library distributions. Exception:"
                            )
                            self.logger.warning(traceback.format_exc())
                        self.logger.info(
                            "Initializing tune_sklearn.TuneSearchCV, bayesian"
                        )
                        model_grid = TuneSearchCV(
                            estimator=pipeline_with_model,
                            search_optimization="bayesian",
                            param_distributions=param_grid,
                            n_trials=n_iter,
                            early_stopping=do_early_stop,
                            scoring=optimize,
                            cv=fold,
                            random_state=self.seed,
                            max_iters=early_stopping_max_iters,
                            n_jobs=n_jobs,
                            use_gpu=self.gpu_param,
                            refit=True,
                            verbose=tuner_verbose,
                            # pipeline_detection=False,
                            **search_kwargs,
                        )
                    elif search_algorithm == "bohb":
                        try:
                            param_grid = get_CS_distributions(param_grid)
                        except:
                            self.logger.warning(
                                "Couldn't convert param_grid to specific library distributions. Exception:"
                            )
                            self.logger.warning(traceback.format_exc())
                        self.logger.info("Initializing tune_sklearn.TuneSearchCV, bohb")
                        model_grid = TuneSearchCV(
                            estimator=pipeline_with_model,
                            search_optimization="bohb",
                            param_distributions=param_grid,
                            n_trials=n_iter,
                            early_stopping=do_early_stop,
                            scoring=optimize,
                            cv=fold,
                            random_state=self.seed,
                            max_iters=early_stopping_max_iters,
                            n_jobs=n_jobs,
                            use_gpu=self.gpu_param,
                            refit=True,
                            verbose=tuner_verbose,
                            # pipeline_detection=False,
                            **search_kwargs,
                        )
                    else:
                        self.logger.info(
                            "Initializing tune_sklearn.TuneSearchCV, random"
                        )
                        model_grid = TuneSearchCV(
                            estimator=pipeline_with_model,
                            param_distributions=param_grid,
                            early_stopping=do_early_stop,
                            n_trials=n_iter,
                            scoring=optimize,
                            cv=fold,
                            random_state=self.seed,
                            max_iters=early_stopping_max_iters,
                            n_jobs=n_jobs,
                            use_gpu=self.gpu_param,
                            refit=True,
                            verbose=tuner_verbose,
                            # pipeline_detection=False,
                            **search_kwargs,
                        )

            elif search_library == "scikit-optimize":
                import skopt

                try:
                    param_grid = get_skopt_distributions(param_grid)
                except:
                    self.logger.warning(
                        "Couldn't convert param_grid to specific library distributions. Exception:"
                    )
                    self.logger.warning(traceback.format_exc())

                self.logger.info("Initializing skopt.BayesSearchCV")
                model_grid = skopt.BayesSearchCV(
                    estimator=pipeline_with_model,
                    search_spaces=param_grid,
                    scoring=optimize,
                    n_iter=n_iter,
                    cv=fold,
                    random_state=self.seed,
                    refit=False,
                    n_jobs=n_jobs,
                    verbose=tuner_verbose,
                    **search_kwargs,
                )
            else:
                # needs to be imported like that for the monkeypatch
                import sklearn.model_selection._search

                if search_algorithm == "grid":
                    self.logger.info("Initializing GridSearchCV")
                    model_grid = sklearn.model_selection._search.GridSearchCV(
                        estimator=pipeline_with_model,
                        param_grid=param_grid,
                        scoring=optimize,
                        cv=fold,
                        refit=False,
                        n_jobs=n_jobs,
                        verbose=tuner_verbose,
                        **search_kwargs,
                    )
                else:
                    self.logger.info("Initializing RandomizedSearchCV")
                    model_grid = sklearn.model_selection._search.RandomizedSearchCV(
                        estimator=pipeline_with_model,
                        param_distributions=param_grid,
                        scoring=optimize,
                        n_iter=n_iter,
                        cv=fold,
                        random_state=self.seed,
                        refit=False,
                        n_jobs=n_jobs,
                        verbose=tuner_verbose,
                        **search_kwargs,
                    )

            # with io.capture_output():
            if search_library == "scikit-learn":
                # monkey patching to fix overflows on Windows
                with patch(
                    "sklearn.model_selection._search.sample_without_replacement",
                    pycaret.internal.patches.sklearn._mp_sample_without_replacement,
                ):
                    with patch(
                        "sklearn.model_selection._search.ParameterGrid.__getitem__",
                        pycaret.internal.patches.sklearn._mp_ParameterGrid_getitem,
                    ):
                        model_grid.fit(data_X, data_y, groups=groups, **fit_kwargs)
            else:
                model_grid.fit(data_X, data_y, groups=groups, **fit_kwargs)
            best_params = model_grid.best_params_
            self.logger.info(f"best_params: {best_params}")
            best_params = {**best_params, **extra_params}
            best_params = {
                k.replace(f"{actual_estimator_label}__", ""): v
                for k, v in best_params.items()
            }
            cv_results = None
            try:
                cv_results = model_grid.cv_results_
            except:
                self.logger.warning(
                    "Couldn't get cv_results from model_grid. Exception:"
                )
                self.logger.warning(traceback.format_exc())

        display.move_progress()

        self.logger.info("Random search completed")

        self.logger.info(
            "SubProcess create_model() called =================================="
        )
        best_model, model_fit_time = self.create_model(
            estimator=model,
            system=False,
            display=display,
            fold=fold,
            round=round,
            groups=groups,
            fit_kwargs=fit_kwargs,
            **best_params,
        )
        model_results = self.pull()
        self.logger.info(
            "SubProcess create_model() end =================================="
        )

        if choose_better:
            best_model = self._choose_better(
                [estimator, (best_model, model_results)],
                compare_dimension,
                fold,
                groups=groups,
                fit_kwargs=fit_kwargs,
                display=display,
            )

        # end runtime
        runtime_end = time.time()
        runtime = np.array(runtime_end - runtime_start).round(2)

        # mlflow logging
        if self.logging_param:

            avgs_dict_log = {k: v for k, v in model_results.loc["Mean"].items()}

            try:
                self._mlflow_log_model(
                    model=best_model,
                    model_results=model_results,
                    score_dict=avgs_dict_log,
                    source="tune_model",
                    runtime=runtime,
                    model_fit_time=model_fit_time,
                    _prep_pipe=self.prep_pipe,
                    log_plots=self.log_plots_param,
                    tune_cv_results=cv_results,
                    display=display,
                )
            except:
                self.logger.error(
                    f"_mlflow_log_model() for {best_model} raised an exception:"
                )
                self.logger.error(traceback.format_exc())

        model_results = color_df(model_results, "yellow", ["Mean"], axis=1)
        model_results = model_results.set_precision(round)
        display.display(model_results, clear=True)

        self.logger.info(f"create_model_container: {len(self.create_model_container)}")
        self.logger.info(f"master_model_container: {len(self.master_model_container)}")
        self.logger.info(f"display_container: {len(self.display_container)}")

        self.logger.info(str(best_model))
        self.logger.info(
            "tune_model() succesfully completed......................................"
        )

        gc.collect()
        if return_tuner:
            return (best_model, model_grid)
        return best_model

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
        verbose: bool = True,
        display: Optional[Display] = None,  # added in pycaret==2.2.0
    ) -> Any:
        """
        This function ensembles the trained base estimator using the method defined in 
        'method' param (default = 'Bagging'). The output prints a score grid that shows 
        Accuracy, AUC, Recall, Precision, F1, Kappa and MCC by fold (default = 10 Fold). 

        This function returns a trained model object.  

        Model must be created using create_model() or tune_model().

        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> experiment_name = setup(data = juice,  target = 'Purchase')
        >>> dt = create_model('dt')
        >>> ensembled_dt = ensemble_model(dt)

        This will return an ensembled Decision Tree model using 'Bagging'.
        
        Parameters
        ----------
        estimator : object, default = None

        method: str, default = 'Bagging'
            Bagging method will create an ensemble meta-estimator that fits base 
            classifiers each on random subsets of the original dataset. The other
            available method is 'Boosting' which will create a meta-estimators by
            fitting a classifier on the original dataset and then fits additional 
            copies of the classifier on the same dataset but where the weights of 
            incorrectly classified instances are adjusted such that subsequent 
            classifiers focus more on difficult cases.
        
        fold: integer or scikit-learn compatible CV generator, default = None
            Controls cross-validation. If None, will use the CV generator defined in setup().
            If integer, will use KFold CV with that many folds.
            When cross_validation is False, this parameter is ignored.
        
        n_estimators: integer, default = 10
            The number of base estimators in the ensemble.
            In case of perfect fit, the learning procedure is stopped early.

        round: integer, default = 4
            Number of decimal places the metrics in the score grid will be rounded to.

        choose_better: bool, default = False
            When set to set to True, base estimator is returned when the metric doesn't 
            improve by ensemble_model. This gurantees the returned object would perform 
            atleast equivalent to base estimator created using create_model or model 
            returned by compare_models.

        optimize: str, default = 'Accuracy'
            Only used when choose_better is set to True. optimize parameter is used
            to compare emsembled model with base estimator. Values accepted in 
            optimize parameter are 'Accuracy', 'AUC', 'Recall', 'Precision', 'F1', 
            'Kappa', 'MCC'.

        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.

        groups: str or array-like, with shape (n_samples,), default = None
            Optional Group labels for the samples used while splitting the dataset into train/test set.
            If string is passed, will use the data column with that name as the groups.
            Only used if a group based cross-validation generator is used (eg. GroupKFold).
            If None, will use the value set in fold_groups param in setup().

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
            Trained ensembled model object.

        Warnings
        --------  
        - If target variable is multiclass (more than 2 classes), AUC will be returned 
        as zero (0.0).
            
        
        """

        function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

        self.logger.info("Initializing ensemble_model()")
        self.logger.info(f"ensemble_model({function_params_str})")

        self.logger.info("Checking exceptions")

        # run_time
        runtime_start = time.time()

        if not fit_kwargs:
            fit_kwargs = {}

        # Check for estimator
        if not hasattr(estimator, "fit"):
            raise ValueError(
                f"Estimator {estimator} does not have the required fit() method."
            )

        # Check for allowed method
        available_method = ["Bagging", "Boosting"]
        if method not in available_method:
            raise ValueError(
                "Method parameter only accepts two values 'Bagging' or 'Boosting'."
            )

        # check boosting conflict
        if method == "Boosting":

            boosting_model_definition = self._all_models_internal["ada"]

            check_model = estimator

            try:
                check_model = boosting_model_definition.class_def(
                    check_model,
                    n_estimators=n_estimators,
                    **boosting_model_definition.args,
                )
                with io.capture_output():
                    check_model.fit(self.X_train.values, self.y_train.values)
            except:
                raise TypeError(
                    "Estimator not supported for the Boosting method. Change the estimator or method to 'Bagging'."
                )

        # checking fold parameter
        if fold is not None and not (
            type(fold) is int or is_sklearn_cv_generator(fold)
        ):
            raise TypeError(
                "fold parameter must be either None, an integer or a scikit-learn compatible CV generator object."
            )

        # checking n_estimators parameter
        if type(n_estimators) is not int:
            raise TypeError("n_estimators parameter only accepts integer value.")

        # checking round parameter
        if type(round) is not int:
            raise TypeError("Round parameter only accepts integer value.")

        # checking verbose parameter
        if type(verbose) is not bool:
            raise TypeError(
                "Verbose parameter can only take argument as True or False."
            )

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

        """
        
        ERROR HANDLING ENDS HERE
        
        """

        fold = self._get_cv_splitter(fold)

        groups = self._get_groups(groups)

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

        self.logger.info("Importing libraries")

        np.random.seed(self.seed)

        self.logger.info("Copying training dataset")

        display.move_progress()

        # setting optimize parameter

        compare_dimension = optimize.display_name
        optimize = optimize.scorer

        self.logger.info("Checking base model")

        _estimator_ = estimator

        estimator_id = self._get_model_id(estimator)

        estimator_definition = self._all_models_internal[estimator_id]
        estimator_name = estimator_definition.name
        self.logger.info(f"Base model : {estimator_name}")

        display.update_monitor(2, estimator_name)
        display.display_monitor()

        """
        MONITOR UPDATE STARTS
        """

        display.update_monitor(1, "Selecting Estimator")
        display.display_monitor()

        """
        MONITOR UPDATE ENDS
        """

        model = get_estimator_from_meta_estimator(_estimator_)

        self.logger.info("Importing untrained ensembler")

        if method == "Bagging":
            self.logger.info("Ensemble method set to Bagging")
            bagging_model_definition = self._all_models_internal["Bagging"]

            model = bagging_model_definition.class_def(
                model,
                bootstrap=True,
                n_estimators=n_estimators,
                **bagging_model_definition.args,
            )

        else:
            self.logger.info("Ensemble method set to Boosting")
            boosting_model_definition = self._all_models_internal["ada"]
            model = boosting_model_definition.class_def(
                model, n_estimators=n_estimators, **boosting_model_definition.args
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
        )
        best_model = model
        model_results = self.pull()
        self.logger.info(
            "SubProcess create_model() end =================================="
        )

        # end runtime
        runtime_end = time.time()
        runtime = np.array(runtime_end - runtime_start).round(2)

        # mlflow logging
        if self.logging_param:

            avgs_dict_log = {k: v for k, v in model_results.loc["Mean"].items()}

            try:
                self._mlflow_log_model(
                    model=best_model,
                    model_results=model_results,
                    score_dict=avgs_dict_log,
                    source="ensemble_model",
                    runtime=runtime,
                    model_fit_time=model_fit_time,
                    _prep_pipe=self.prep_pipe,
                    log_plots=self.log_plots_param,
                    display=display,
                )
            except:
                self.logger.error(
                    f"_mlflow_log_model() for {best_model} raised an exception:"
                )
                self.logger.error(traceback.format_exc())

        if choose_better:
            model = self._choose_better(
                [_estimator_, (best_model, model_results)],
                compare_dimension,
                fold,
                groups=groups,
                fit_kwargs=fit_kwargs,
                display=display,
            )

        model_results = color_df(model_results, "yellow", ["Mean"], axis=1)
        model_results = model_results.set_precision(round)
        display.display(model_results, clear=True)

        self.logger.info(f"create_model_container: {len(self.create_model_container)}")
        self.logger.info(f"master_model_container: {len(self.master_model_container)}")
        self.logger.info(f"display_container: {len(self.display_container)}")

        self.logger.info(str(model))
        self.logger.info(
            "ensemble_model() succesfully completed......................................"
        )

        gc.collect()
        return model

    def blend_models(
        self,
        estimator_list: list,
        fold: Optional[Union[int, Any]] = None,
        round: int = 4,
        choose_better: bool = False,
        optimize: str = "Accuracy",
        method: str = "auto",
        weights: Optional[List[float]] = None,  # added in pycaret==2.2.0
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        verbose: bool = True,
        display: Optional[Display] = None,  # added in pycaret==2.2.0
    ) -> Any:

        """
        This function creates a Soft Voting / Majority Rule classifier for all the 
        estimators in the model library (excluding the few when turbo is True) or 
        for specific trained estimators passed as a list in estimator_list param.
        It scores it using Cross Validation. The output prints a score
        grid that shows Accuracy, AUC, Recall, Precision, F1, Kappa and MCC by 
        fold (default CV = 10 Folds).

        This function returns a trained model object.

        Example
        -------
        >>> lr = create_model('lr')
        >>> rf = create_model('rf')
        >>> knn = create_model('knn')
        >>> blend_three = blend_models(estimator_list = [lr,rf,knn])

        This will create a VotingClassifier of lr, rf and knn.

        Parameters
        ----------
        estimator_list : list of objects

        fold: integer or scikit-learn compatible CV generator, default = None
            Controls cross-validation. If None, will use the CV generator defined in setup().
            If integer, will use KFold CV with that many folds.
            When cross_validation is False, this parameter is ignored.

        round: integer, default = 4
            Number of decimal places the metrics in the score grid will be rounded to.

        choose_better: bool, default = False
            When set to set to True, base estimator is returned when the metric doesn't 
            improve by ensemble_model. This gurantees the returned object would perform 
            atleast equivalent to base estimator created using create_model or model 
            returned by compare_models.

        optimize: str, default = 'Accuracy'
            Only used when choose_better is set to True. optimize parameter is used
            to compare emsembled model with base estimator. Values accepted in 
            optimize parameter are 'Accuracy', 'AUC', 'Recall', 'Precision', 'F1', 
            'Kappa', 'MCC'.

        method: str, default = 'auto'
            'hard' uses predicted class labels for majority rule voting. 'soft', predicts 
            the class label based on the argmax of the sums of the predicted probabilities, 
            which is recommended for an ensemble of well-calibrated classifiers. Default value,
            'auto', will try to use 'soft' and fall back to 'hard' if the former is not supported.

        weights: list, default = None
            Sequence of weights (float or int) to weight the occurrences of predicted class labels (hard voting)
            or class probabilities before averaging (soft voting). Uses uniform weights if None.

        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.

        groups: str or array-like, with shape (n_samples,), default = None
            Optional Group labels for the samples used while splitting the dataset into train/test set.
            If string is passed, will use the data column with that name as the groups.
            Only used if a group based cross-validation generator is used (eg. GroupKFold).
            If None, will use the value set in fold_groups param in setup().

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
            Trained Voting Classifier model object. 

        Warnings
        --------
        - When passing estimator_list with method set to 'soft'. All the models in the
        estimator_list must support predict_proba function. 'svm' and 'ridge' doesnt
        support the predict_proba and hence an exception will be raised.
        
        - When estimator_list is set to 'All' and method is forced to 'soft', estimators
        that doesnt support the predict_proba function will be dropped from the estimator
        list.
            
        - If target variable is multiclass (more than 2 classes), AUC will be returned as
        zero (0.0).
            
        
    
        """

        function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

        self.logger.info("Initializing blend_models()")
        self.logger.info(f"blend_models({function_params_str})")

        self.logger.info("Checking exceptions")

        # run_time
        runtime_start = time.time()

        if not fit_kwargs:
            fit_kwargs = {}

        # checking method parameter
        available_method = ["auto", "soft", "hard"]
        if method not in available_method:
            raise ValueError(
                "Method parameter only accepts 'auto', 'soft' or 'hard' as a parameter. See Docstring for details."
            )

        # checking error for estimator_list
        for i in estimator_list:
            if not hasattr(i, "fit"):
                raise ValueError(
                    f"Estimator {i} does not have the required fit() method."
                )
            if self._ml_usecase == MLUsecase.CLASSIFICATION:
                # checking method param with estimator list
                if method != "hard":

                    for i in estimator_list:
                        if not hasattr(i, "predict_proba"):
                            if method != "auto":
                                raise TypeError(
                                    f"Estimator list contains estimator {i} that doesn't support probabilities and method is forced to 'soft'. Either change the method or drop the estimator."
                                )
                            else:
                                self.logger.info(
                                    f"Estimator {i} doesn't support probabilities, falling back to 'hard'."
                                )
                                method = "hard"
                                break

                    if method == "auto":
                        method = "soft"

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

        if weights is not None:
            num_estimators = len(estimator_list)
            # checking weights parameter
            if len(weights) != num_estimators:
                raise ValueError(
                    "weights parameter must have the same length as the estimator_list."
                )
            if not all((isinstance(x, int) or isinstance(x, float)) for x in weights):
                raise TypeError("weights must contain only ints or floats.")

        # checking verbose parameter
        if type(verbose) is not bool:
            raise TypeError(
                "Verbose parameter can only take argument as True or False."
            )

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

        """
        
        ERROR HANDLING ENDS HERE
        
        """

        fold = self._get_cv_splitter(fold)

        groups = self._get_groups(groups)

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

        self.logger.info("Importing libraries")

        np.random.seed(self.seed)

        self.logger.info("Copying training dataset")

        # setting optimize parameter
        compare_dimension = optimize.display_name
        optimize = optimize.scorer

        display.move_progress()

        """
        MONITOR UPDATE STARTS
        """

        display.update_monitor(1, "Compiling Estimators")
        display.display_monitor()

        """
        MONITOR UPDATE ENDS
        """

        self.logger.info("Getting model names")
        estimator_dict = {}
        for x in estimator_list:
            x = get_estimator_from_meta_estimator(x)
            name = self._get_model_id(x)
            suffix = 1
            original_name = name
            while name in estimator_dict:
                name = f"{original_name}_{suffix}"
                suffix += 1
            estimator_dict[name] = x

        estimator_list = list(estimator_dict.items())

        voting_model_definition = self._all_models_internal["Voting"]
        if self._ml_usecase == MLUsecase.CLASSIFICATION:
            model = voting_model_definition.class_def(
                estimators=estimator_list, voting=method, n_jobs=self._gpu_n_jobs_param
            )
        else:
            model = voting_model_definition.class_def(
                estimators=estimator_list, n_jobs=self._gpu_n_jobs_param
            )

        display.update_monitor(2, voting_model_definition.name)
        display.display_monitor()

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
        )
        model_results = self.pull()
        self.logger.info(
            "SubProcess create_model() end =================================="
        )

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
                    source="blend_models",
                    runtime=runtime,
                    model_fit_time=model_fit_time,
                    _prep_pipe=self.prep_pipe,
                    log_plots=self.log_plots_param,
                    display=display,
                )
            except:
                self.logger.error(
                    f"_mlflow_log_model() for {model} raised an exception:"
                )
                self.logger.error(traceback.format_exc())

        if choose_better:
            model = self._choose_better(
                [(model, model_results)] + estimator_list,
                compare_dimension,
                fold,
                groups=groups,
                fit_kwargs=fit_kwargs,
                display=display,
            )

        model_results = color_df(model_results, "yellow", ["Mean"], axis=1)
        model_results = model_results.set_precision(round)
        display.display(model_results, clear=True)

        self.logger.info(f"create_model_container: {len(self.create_model_container)}")
        self.logger.info(f"master_model_container: {len(self.master_model_container)}")
        self.logger.info(f"display_container: {len(self.display_container)}")

        self.logger.info(str(model))
        self.logger.info(
            "blend_models() succesfully completed......................................"
        )

        gc.collect()
        return model

    def stack_models(
        self,
        estimator_list: list,
        meta_model=None,
        fold: Optional[Union[int, Any]] = None,
        round: int = 4,
        method: str = "auto",
        restack: bool = True,
        choose_better: bool = False,
        optimize: str = "Accuracy",
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        verbose: bool = True,
        display: Optional[Display] = None,
    ) -> Any:

        """
        This function trains a meta model and scores it using Cross Validation.
        The predictions from the base level models as passed in the estimator_list param 
        are used as input features for the meta model. The restacking parameter controls
        the ability to expose raw features to the meta model when set to True
        (default = False).

        The output prints the score grid that shows Accuracy, AUC, Recall, Precision, 
        F1, Kappa and MCC by fold (default = 10 Folds). 
        
        This function returns a trained model object. 

        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> experiment_name = setup(data = juice,  target = 'Purchase')
        >>> dt = create_model('dt')
        >>> rf = create_model('rf')
        >>> ada = create_model('ada')
        >>> ridge = create_model('ridge')
        >>> knn = create_model('knn')
        >>> stacked_models = stack_models(estimator_list=[dt,rf,ada,ridge,knn])

        This will create a meta model that will use the predictions of all the 
        models provided in estimator_list param. By default, the meta model is 
        Logistic Regression but can be changed with meta_model param.

        Parameters
        ----------
        estimator_list : list of objects

        meta_model : object, default = None
            If set to None, Logistic Regression is used as a meta model.

        fold: integer or scikit-learn compatible CV generator, default = None
            Controls cross-validation. If None, will use the CV generator defined in setup().
            If integer, will use KFold CV with that many folds.
            When cross_validation is False, this parameter is ignored.

        round: integer, default = 4
            Number of decimal places the metrics in the score grid will be rounded to.

        method: string, default = 'auto'
            - if ‘auto’, it will try to invoke, for each estimator, 'predict_proba', 
            'decision_function' or 'predict' in that order.
            - otherwise, one of 'predict_proba', 'decision_function' or 'predict'. 
            If the method is not implemented by the estimator, it will raise an error.

        restack: bool, default = True
            When restack is set to True, raw data will be exposed to meta model when
            making predictions, otherwise when False, only the predicted label or
            probabilities is passed to meta model when making final predictions.

        choose_better: bool, default = False
            When set to set to True, base estimator is returned when the metric doesn't 
            improve by ensemble_model. This gurantees the returned object would perform 
            atleast equivalent to base estimator created using create_model or model 
            returned by compare_models.

        optimize: str, default = 'Accuracy'
            Only used when choose_better is set to True. optimize parameter is used
            to compare emsembled model with base estimator. Values accepted in 
            optimize parameter are 'Accuracy', 'AUC', 'Recall', 'Precision', 'F1', 
            'Kappa', 'MCC'.

        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.

        groups: str or array-like, with shape (n_samples,), default = None
            Optional Group labels for the samples used while splitting the dataset into train/test set.
            If string is passed, will use the data column with that name as the groups.
            Only used if a group based cross-validation generator is used (eg. GroupKFold).
            If None, will use the value set in fold_groups param in setup().

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
            Trained model object.

        Warnings
        --------
        -  If target variable is multiclass (more than 2 classes), AUC will be returned 
        as zero (0.0).

        """

        function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

        self.logger.info("Initializing stack_models()")
        self.logger.info(f"stack_models({function_params_str})")

        self.logger.info("Checking exceptions")

        # run_time
        runtime_start = time.time()

        if not fit_kwargs:
            fit_kwargs = {}

        # checking error for estimator_list
        for i in estimator_list:
            if not hasattr(i, "fit"):
                raise ValueError(
                    f"Estimator {i} does not have the required fit() method."
                )

        # checking meta model
        if meta_model is not None:
            if not hasattr(meta_model, "fit"):
                raise ValueError(
                    f"Meta Model {meta_model} does not have the required fit() method."
                )

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

        # checking method parameter
        available_method = ["auto", "predict_proba", "decision_function", "predict"]
        if method not in available_method:
            raise ValueError(
                "Method parameter not acceptable. It only accepts 'auto', 'predict_proba', 'decision_function', 'predict'."
            )

        # checking restack parameter
        if type(restack) is not bool:
            raise TypeError(
                "Restack parameter can only take argument as True or False."
            )

        # checking verbose parameter
        if type(verbose) is not bool:
            raise TypeError(
                "Verbose parameter can only take argument as True or False."
            )

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

        """
        
        ERROR HANDLING ENDS HERE
        
        """

        fold = self._get_cv_splitter(fold)

        groups = self._get_groups(groups)

        self.logger.info("Defining meta model")
        if meta_model == None:
            estimator = "lr"
            meta_model_definition = self._all_models_internal[estimator]
            meta_model_args = meta_model_definition.args
            meta_model = meta_model_definition.class_def(**meta_model_args)
        else:
            meta_model = clone(get_estimator_from_meta_estimator(meta_model))

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

        # setting optimize parameter
        compare_dimension = optimize.display_name
        optimize = optimize.scorer

        display.move_progress()

        """
        MONITOR UPDATE STARTS
        """

        display.update_monitor(1, "Compiling Estimators")
        display.display_monitor()

        """
        MONITOR UPDATE ENDS
        """

        self.logger.info("Getting model names")
        estimator_dict = {}
        for x in estimator_list:
            x = get_estimator_from_meta_estimator(x)
            name = self._get_model_id(x)
            suffix = 1
            original_name = name
            while name in estimator_dict:
                name = f"{original_name}_{suffix}"
                suffix += 1
            estimator_dict[name] = x

        estimator_list = list(estimator_dict.items())

        self.logger.info(estimator_list)

        stacking_model_definition = self._all_models_internal["Stacking"]
        if self._ml_usecase == MLUsecase.CLASSIFICATION:
            model = stacking_model_definition.class_def(
                estimators=estimator_list,
                final_estimator=meta_model,
                cv=fold,
                stack_method=method,
                n_jobs=self._gpu_n_jobs_param,
                passthrough=restack,
            )
        else:
            model = stacking_model_definition.class_def(
                estimators=estimator_list,
                final_estimator=meta_model,
                cv=fold,
                n_jobs=self._gpu_n_jobs_param,
                passthrough=restack,
            )

        display.update_monitor(2, stacking_model_definition.name)
        display.display_monitor()

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
        )
        model_results = self.pull()
        self.logger.info(
            "SubProcess create_model() end =================================="
        )

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
                    source="stack_models",
                    runtime=runtime,
                    model_fit_time=model_fit_time,
                    _prep_pipe=self.prep_pipe,
                    log_plots=self.log_plots_param,
                    display=display,
                )
            except:
                self.logger.error(
                    f"_mlflow_log_model() for {model} raised an exception:"
                )
                self.logger.error(traceback.format_exc())

        if choose_better:
            model = self._choose_better(
                [(model, model_results)] + estimator_list,
                compare_dimension,
                fold,
                groups=groups,
                fit_kwargs=fit_kwargs,
                display=display,
            )

        model_results = color_df(model_results, "yellow", ["Mean"], axis=1)
        model_results = model_results.set_precision(round)
        display.display(model_results, clear=True)

        self.logger.info(f"create_model_container: {len(self.create_model_container)}")
        self.logger.info(f"master_model_container: {len(self.master_model_container)}")
        self.logger.info(f"display_container: {len(self.display_container)}")

        self.logger.info(str(model))
        self.logger.info(
            "stack_models() succesfully completed......................................"
        )

        gc.collect()
        return model

    def interpret_model(
        self,
        estimator,
        plot: str = "summary",
        feature: Optional[str] = None,
        observation: Optional[int] = None,
        use_train_data: bool = False,
        **kwargs,  # added in pycaret==2.1
    ):

        """
        This function takes a trained model object and returns an interpretation plot 
        based on the test / hold-out set. It only supports tree based algorithms. 

        This function is implemented based on the SHAP (SHapley Additive exPlanations),
        which is a unified approach to explain the output of any machine learning model. 
        SHAP connects game theory with local explanations.

        For more information : https://shap.readthedocs.io/en/latest/

        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> experiment_name = setup(data = juice,  target = 'Purchase')
        >>> dt = create_model('dt')
        >>> interpret_model(dt)

        This will return a summary interpretation plot of Decision Tree model.

        Parameters
        ----------
        estimator : object, default = none
            A trained tree based model object should be passed as an estimator. 

        plot : str, default = 'summary'
            Other available options are 'correlation' and 'reason'.

        feature: str, default = None
            This parameter is only needed when plot = 'correlation'. By default feature is 
            set to None which means the first column of the dataset will be used as a 
            variable. A feature parameter must be passed to change this.

        observation: integer, default = None
            This parameter only comes into effect when plot is set to 'reason'. If no 
            observation number is provided, it will return an analysis of all observations 
            with the option to select the feature on x and y axes through drop down 
            interactivity. For analysis at the sample level, an observation parameter must
            be passed with the index value of the observation in test / hold-out set. 

        **kwargs: 
            Additional keyword arguments to pass to the plot.

        Returns
        -------
        Visual_Plot
            Returns the visual plot.
            Returns the interactive JS plot when plot = 'reason'.

        Warnings
        -------- 
        - interpret_model doesn't support multiclass problems.

        """

        function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

        self.logger.info("Initializing interpret_model()")
        self.logger.info(f"interpret_model({function_params_str})")

        self.logger.info("Checking exceptions")

        # checking if shap available
        try:
            import shap
        except ImportError:
            self.logger.error(
                "shap library not found. pip install shap to use interpret_model function."
            )
            raise ImportError(
                "shap library not found. pip install shap to use interpret_model function."
            )

        # get estimator from meta estimator
        estimator = get_estimator_from_meta_estimator(estimator)

        # allowed models
        model_id = self._get_model_id(estimator)

        shap_models = {k: v for k, v in self._all_models_internal.items() if v.shap}
        shap_models_ids = set(shap_models.keys())

        if model_id not in shap_models_ids:
            raise TypeError(
                f"This function only supports tree based models for binary classification: {', '.join(shap_models_ids)}."
            )

        # plot type
        allowed_types = ["summary", "correlation", "reason"]
        if plot not in allowed_types:
            raise ValueError(
                "type parameter only accepts 'summary', 'correlation' or 'reason'."
            )

        """
        Error Checking Ends here
        
        """

        # Storing X_train and y_train in data_X and data_y parameter
        test_X = self.X_train if use_train_data else self.X_test

        np.random.seed(self.seed)

        # storing estimator in model variable
        model = estimator

        # defining type of classifier
        shap_models_type1 = {k for k, v in shap_models.items() if v.shap == "type1"}
        shap_models_type2 = {k for k, v in shap_models.items() if v.shap == "type2"}

        self.logger.info(f"plot type: {plot}")

        shap_plot = None

        def summary():

            self.logger.info("Creating TreeExplainer")
            explainer = shap.TreeExplainer(model)
            self.logger.info("Compiling shap values")
            shap_values = explainer.shap_values(test_X)
            shap_plot = shap.summary_plot(shap_values, test_X, **kwargs)
            return shap_plot

        def correlation():

            if feature == None:

                self.logger.warning(
                    f"No feature passed. Default value of feature used for correlation plot: {test_X.columns[0]}"
                )
                dependence = test_X.columns[0]

            else:

                self.logger.warning(
                    f"feature value passed. Feature used for correlation plot: {test_X.columns[0]}"
                )
                dependence = feature

            self.logger.info("Creating TreeExplainer")
            explainer = shap.TreeExplainer(model)
            self.logger.info("Compiling shap values")
            shap_values = explainer.shap_values(test_X)

            if model_id in shap_models_type1:
                self.logger.info("model type detected: type 1")
                shap.dependence_plot(dependence, shap_values[1], test_X, **kwargs)
            elif model_id in shap_models_type2:
                self.logger.info("model type detected: type 2")
                shap.dependence_plot(dependence, shap_values, test_X, **kwargs)
            return None

        def reason():
            shap_plot = None
            if model_id in shap_models_type1:
                self.logger.info("model type detected: type 1")

                self.logger.info("Creating TreeExplainer")
                explainer = shap.TreeExplainer(model)
                self.logger.info("Compiling shap values")

                if observation is None:
                    self.logger.warning(
                        "Observation set to None. Model agnostic plot will be rendered."
                    )
                    shap_values = explainer.shap_values(test_X)
                    shap.initjs()
                    shap_plot = shap.force_plot(
                        explainer.expected_value[1], shap_values[1], test_X, **kwargs
                    )

                else:
                    row_to_show = observation
                    data_for_prediction = test_X.iloc[row_to_show]

                    if model_id == "lightgbm":
                        self.logger.info("model type detected: LGBMClassifier")
                        shap_values = explainer.shap_values(test_X)
                        shap.initjs()
                        shap_plot = shap.force_plot(
                            explainer.expected_value[1],
                            shap_values[0][row_to_show],
                            data_for_prediction,
                            **kwargs,
                        )

                    else:
                        self.logger.info("model type detected: Unknown")

                        shap_values = explainer.shap_values(data_for_prediction)
                        shap.initjs()
                        shap_plot = shap.force_plot(
                            explainer.expected_value[1],
                            shap_values[1],
                            data_for_prediction,
                            **kwargs,
                        )

            elif model_id in shap_models_type2:
                self.logger.info("model type detected: type 2")

                self.logger.info("Creating TreeExplainer")
                explainer = shap.TreeExplainer(model)
                self.logger.info("Compiling shap values")
                shap_values = explainer.shap_values(test_X)
                shap.initjs()

                if observation is None:
                    self.logger.warning(
                        "Observation set to None. Model agnostic plot will be rendered."
                    )

                    shap_plot = shap.force_plot(
                        explainer.expected_value, shap_values, test_X, **kwargs
                    )

                else:

                    row_to_show = observation
                    data_for_prediction = test_X.iloc[row_to_show]

                    shap_plot = shap.force_plot(
                        explainer.expected_value,
                        shap_values[row_to_show, :],
                        test_X.iloc[row_to_show, :],
                        **kwargs,
                    )
            return shap_plot

        shap_plot = locals()[plot]()

        self.logger.info("Visual Rendered Successfully")

        self.logger.info(
            "interpret_model() succesfully completed......................................"
        )

        gc.collect()
        return shap_plot

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

        model_type = {
            "linear": [
                "lr",
                "ridge",
                "svm",
                "lasso",
                "en",
                "lar",
                "llar",
                "omp",
                "br",
                "ard",
                "par",
                "ransac",
                "tr",
                "huber",
                "kr",
            ],
            "tree": ["dt"],
            "ensemble": [
                "rf",
                "et",
                "gbc",
                "gbr",
                "xgboost",
                "lightgbm",
                "catboost",
                "ada",
            ],
        }

        def filter_model_df_by_type(df):
            if not type:
                return df
            return df[df.index.isin(model_type[type])]

        # Check if type is valid
        if type not in list(model_type) + [None]:
            raise ValueError(
                f"type param only accepts {', '.join(list(model_type) + str(None))}."
            )

        self.logger.info(f"gpu_param set to {self.gpu_param}")

        _, model_containers = self._get_models(raise_errors)

        rows = [
            v.get_dict(internal)
            for k, v in model_containers.items()
            if (internal or not v.is_special)
        ]

        df = pd.DataFrame(rows)
        df.set_index("ID", inplace=True, drop=True)

        return filter_model_df_by_type(df)

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
        >>> metrics = get_metrics()

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
        multiclass: bool = True,
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

        multiclass: bool, default = True
            Whether the metric supports multiclass problems.

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

        if self._ml_usecase == MLUsecase.CLASSIFICATION:
            new_metric = pycaret.containers.metrics.classification.ClassificationMetricContainer(
                id=id,
                name=name,
                score_func=score_func,
                target=target,
                args=kwargs,
                display_name=name,
                greater_is_better=greater_is_better,
                is_multiclass=bool(multiclass),
                is_custom=True,
            )
        else:
            new_metric = pycaret.containers.metrics.regression.RegressionMetricContainer(
                id=id,
                name=name,
                score_func=score_func,
                args=kwargs,
                display_name=name,
                greater_is_better=greater_is_better,
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

    def finalize_model(
        self,
        estimator,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        model_only: bool = True,
        display: Optional[Display] = None,
    ) -> Any:  # added in pycaret==2.2.0

        """
        This function fits the estimator onto the complete dataset passed during the
        setup() stage. The purpose of this function is to prepare for final model
        deployment after experimentation. 
        
        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> experiment_name = setup(data = juice,  target = 'Purchase')
        >>> lr = create_model('lr')
        >>> final_lr = finalize_model(lr)
        
        This will return the final model object fitted to complete dataset. 

        Parameters
        ----------
        estimator : object, default = none
            A trained model object should be passed as an estimator. 

        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.

        groups: str or array-like, with shape (n_samples,), default = None
            Optional Group labels for the samples used while splitting the dataset into train/test set.
            If string is passed, will use the data column with that name as the groups.
            Only used if a group based cross-validation generator is used (eg. GroupKFold).
            If None, will use the value set in fold_groups param in setup().

        model_only : bool, default = True
            When set to True, only trained model object is saved and all the 
            transformations are ignored.

        Returns
        -------
        model
            Trained model object fitted on complete dataset.

        Warnings
        --------
        - If the model returned by finalize_model(), is used on predict_model() without 
        passing a new unseen dataset, then the information grid printed is misleading 
        as the model is trained on the complete dataset including test / hold-out sample. 
        Once finalize_model() is used, the model is considered ready for deployment and
        should be used on new unseens dataset only.
        
            
        """

        function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

        self.logger.info("Initializing finalize_model()")
        self.logger.info(f"finalize_model({function_params_str})")

        # run_time
        runtime_start = time.time()

        if not fit_kwargs:
            fit_kwargs = {}

        groups = self._get_groups(groups)

        if not display:
            display = Display(verbose=False, html_param=self.html_param,)

        np.random.seed(self.seed)

        self.logger.info(f"Finalizing {estimator}")
        display.clear_output()
        model_final, model_fit_time = self.create_model(
            estimator=estimator,
            verbose=False,
            system=False,
            X_train_data=self.X,
            y_train_data=self.y,
            fit_kwargs=fit_kwargs,
            groups=groups,
        )
        model_results = self.pull(pop=True)

        # end runtime
        runtime_end = time.time()
        runtime = np.array(runtime_end - runtime_start).round(2)

        # mlflow logging
        if self.logging_param:

            avgs_dict_log = {k: v for k, v in model_results.loc["Mean"].items()}

            try:
                self._mlflow_log_model(
                    model=model_final,
                    model_results=model_results,
                    score_dict=avgs_dict_log,
                    source="finalize_model",
                    runtime=runtime,
                    model_fit_time=model_fit_time,
                    _prep_pipe=self.prep_pipe,
                    log_plots=self.log_plots_param,
                    display=display,
                )
            except:
                self.logger.error(
                    f"_mlflow_log_model() for {model_final} raised an exception:"
                )
                self.logger.error(traceback.format_exc())

        model_results = color_df(model_results, "yellow", ["Mean"], axis=1)
        model_results = model_results.set_precision(round)
        display.display(model_results, clear=True)

        self.logger.info(f"create_model_container: {len(self.create_model_container)}")
        self.logger.info(f"master_model_container: {len(self.master_model_container)}")
        self.logger.info(f"display_container: {len(self.display_container)}")

        self.logger.info(str(model_final))
        self.logger.info(
            "finalize_model() succesfully completed......................................"
        )

        gc.collect()
        if not model_only:
            pipeline_final = deepcopy(self.prep_pipe)
            pipeline_final.steps.append(["trained_model", model_final])
            return pipeline_final

        return model_final

    def predict_model(
        self,
        estimator,
        data: Optional[pd.DataFrame] = None,
        probability_threshold: Optional[float] = None,
        encoded_labels: bool = False,  # added in pycaret==2.1.0
        round: int = 4,  # added in pycaret==2.2.0
        verbose: bool = True,
        ml_usecase: Optional[MLUsecase] = None,
        display: Optional[Display] = None,  # added in pycaret==2.2.0
    ) -> pd.DataFrame:

        """
        This function is used to predict label and probability score on the new dataset
        using a trained estimator. New unseen data can be passed to data param as pandas 
        Dataframe. If data is not passed, the test / hold-out set separated at the time of 
        setup() is used to generate predictions. 
        
        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> experiment_name = setup(data = juice,  target = 'Purchase')
        >>> lr = create_model('lr')
        >>> lr_predictions_holdout = predict_model(lr)
            
        Parameters
        ----------
        estimator : object, default = none
            A trained model object / pipeline should be passed as an estimator. 
        
        data : pandas.DataFrame
            Shape (n_samples, n_features) where n_samples is the number of samples 
            and n_features is the number of features. All features used during training 
            must be present in the new dataset.
        
        probability_threshold : float, default = None
            Threshold used to convert probability values into binary outcome. By default 
            the probability threshold for all binary classifiers is 0.5 (50%). This can be 
            changed using probability_threshold param.

        encoded_labels: Boolean, default = False
            If True, will return labels encoded as an integer.

        round: integer, default = 4
            Number of decimal places the metrics in the score grid will be rounded to. 

        verbose: bool, default = True
            Holdout score grid is not printed when verbose is set to False.

        Returns
        -------
        Predictions
            Predictions (Label and Score) column attached to the original dataset
            and returned as pandas dataframe.

        score_grid
            A table containing the scoring metrics on hold-out / test set.

        Warnings
        --------
        - The behavior of the predict_model is changed in version 2.1 without backward compatibility.
        As such, the pipelines trained using the version (<= 2.0), may not work for inference 
        with version >= 2.1. You can either retrain your models with a newer version or downgrade
        the version for inference.
        
        
        """

        function_params_str = ", ".join(
            [f"{k}={v}" for k, v in locals().items() if k != "data"]
        )

        self.logger.info("Initializing predict_model()")
        self.logger.info(f"predict_model({function_params_str})")

        self.logger.info("Checking exceptions")

        """
        exception checking starts here
        """

        if ml_usecase is None:
            ml_usecase = self._ml_usecase

        if data is None and not self._setup_ran:
            raise ValueError(
                "data parameter may not be None without running setup() first."
            )

        if probability_threshold is not None:
            # probability_threshold allowed types
            allowed_types = [int, float]
            if type(probability_threshold) not in allowed_types:
                raise TypeError(
                    "probability_threshold parameter only accepts value between 0 to 1."
                )

            if probability_threshold > 1:
                raise TypeError(
                    "probability_threshold parameter only accepts value between 0 to 1."
                )

            if probability_threshold < 0:
                raise TypeError(
                    "probability_threshold parameter only accepts value between 0 to 1."
                )

        """
        exception checking ends here
        """

        self.logger.info("Preloading libraries")

        # general dependencies
        from sklearn import metrics

        try:
            np.random.seed(self.seed)
            if not display:
                display = Display(verbose=verbose, html_param=self.html_param,)
        except:
            display = Display(verbose=False, html_param=False,)

        dtypes = None

        # dataset
        if data is None:

            if is_sklearn_pipeline(estimator):
                estimator = estimator.steps[-1][1]

            X_test_ = self.X_test.copy()
            y_test_ = self.y_test.copy()

            dtypes = self.prep_pipe.named_steps["dtypes"]

            X_test_.reset_index(drop=True, inplace=True)
            y_test_.reset_index(drop=True, inplace=True)

        else:

            if is_sklearn_pipeline(estimator) and hasattr(estimator, "predict"):
                dtypes = estimator.named_steps["dtypes"]
            else:
                try:
                    dtypes = self.prep_pipe.named_steps["dtypes"]

                    estimator_ = deepcopy(self.prep_pipe)
                    if is_sklearn_pipeline(estimator):
                        merge_pipelines(estimator_, estimator)
                        estimator_.steps[-1] = (
                            "trained_model",
                            estimator_.steps[-1][1],
                        )
                    else:
                        add_estimator_to_pipeline(
                            estimator_, estimator, name="trained_model"
                        )
                    estimator = estimator_

                except:
                    self.logger.error("Pipeline not found. Exception:")
                    self.logger.error(traceback.format_exc())
                    raise ValueError("Pipeline not found")

            X_test_ = data.copy()

        # function to replace encoded labels with their original values
        # will not run if categorical_labels is false
        def replace_lables_in_column(label_column):
            if dtypes and hasattr(dtypes, "replacement"):
                replacement_mapper = {int(v): k for k, v in dtypes.replacement.items()}
                label_column.replace(replacement_mapper, inplace=True)

        # prediction starts here

        pred = np.nan_to_num(estimator.predict(X_test_))

        try:
            score = estimator.predict_proba(X_test_)

            if len(np.unique(pred)) <= 2:
                pred_prob = score[:, 1]
            else:
                pred_prob = score

        except:
            score = None
            pred_prob = None

        if probability_threshold is not None and pred_prob is not None:
            try:
                pred = (pred_prob >= probability_threshold).astype(int)
            except:
                pass

        if pred_prob is None:
            pred_prob = pred

        df_score = None

        if data is None:
            # model name
            full_name = self._get_model_name(estimator)
            metrics = self._calculate_metrics(y_test_, pred, pred_prob)  # type: ignore
            df_score = pd.DataFrame(metrics, index=[0])
            df_score.insert(0, "Model", full_name)
            df_score = df_score.round(round)
            display.display(df_score.style.set_precision(round), clear=False)

        label = pd.DataFrame(pred)
        label.columns = ["Label"]
        if not encoded_labels:
            replace_lables_in_column(label["Label"])
        if ml_usecase == MLUsecase.CLASSIFICATION:
            try:
                label["Label"] = label["Label"].astype(int)
            except:
                pass

        if data is None:
            if not encoded_labels:
                replace_lables_in_column(y_test_)  # type: ignore
            X_test_ = pd.concat([X_test_, y_test_, label], axis=1)  # type: ignore
        else:
            X_test_ = data.copy()
            X_test_["Label"] = label["Label"].values

        if score is not None:
            d = []
            for i in range(0, len(score)):
                d.append(score[i][pred[i]])

            score = d
            try:
                score = pd.DataFrame(score)
                score.columns = ["Score"]
                score = score.round(round)
                X_test_["Score"] = score["Score"].values
            except:
                pass

        # store predictions on hold-out in display_container
        if df_score is not None:
            self.display_container.append(df_score)

        gc.collect()
        return X_test_


class _UnsupervisedExperiment(_TabularExperiment):
    def __init__(self) -> None:
        super().__init__()
        return

    def _calculate_metrics(self, X, labels, ground_truth=None, ml_usecase=None) -> dict:
        """
        Calculate all metrics in _all_metrics.
        """
        from pycaret.internal.utils import calculate_unsupervised_metrics

        if ml_usecase is None:
            ml_usecase = self._ml_usecase

        try:
            return calculate_unsupervised_metrics(
                metrics=self._all_metrics, X=X, labels=labels, ground_truth=ground_truth
            )
        except:
            if ml_usecase == MLUsecase.CLUSTERING:
                metrics = pycaret.containers.metrics.clustering.get_all_metric_containers(
                    self.variables, True
                )
            return calculate_unsupervised_metrics(
                metrics=metrics,  # type: ignore
                X=X,
                labels=labels,
                ground_truth=ground_truth,
            )

    def _is_unsupervised(self) -> bool:
        return True

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
        display.move_progress()
        self.X = self.prep_pipe.fit_transform(train_data).drop(target, axis=1)
        self.X_train = self.X

    def _set_up_mlflow(
        self,
        functions,
        functions_,
        runtime,
        log_profile,
        profile_kwargs,
        log_data,
        display,
    ) -> None:
        # log into experiment
        self.experiment__.append(("Setup Config", functions))
        self.experiment__.append(("Transformed Data", self.X))
        self.experiment__.append(("Transformation Pipeline", self.prep_pipe))

        if self.logging_param:

            self.logger.info("Logging experiment in MLFlow")

            import mlflow

            try:
                mlflow.create_experiment(self.exp_name_log)
            except:
                self.logger.warning("Couldn't create mlflow experiment. Exception:")
                self.logger.warning(traceback.format_exc())

            # mlflow logging
            mlflow.set_experiment(self.exp_name_log)

            run_name_ = f"Session Initialized {self.USI}"

            with mlflow.start_run(run_name=run_name_) as run:

                # Get active run to log as tag
                RunID = mlflow.active_run().info.run_id

                k = functions.copy()
                k.set_index("Description", drop=True, inplace=True)
                kdict = k.to_dict()
                params = kdict.get("Value")
                mlflow.log_params(params)

                # set tag of compare_models
                mlflow.set_tag("Source", "setup")

                import secrets

                URI = secrets.token_hex(nbytes=4)
                mlflow.set_tag("URI", URI)
                mlflow.set_tag("USI", self.USI)
                mlflow.set_tag("Run Time", runtime)
                mlflow.set_tag("Run ID", RunID)

                # Log the transformation pipeline
                self.logger.info(
                    "SubProcess save_model() called =================================="
                )
                self.save_model(
                    self.prep_pipe, "Transformation Pipeline", verbose=False
                )
                self.logger.info(
                    "SubProcess save_model() end =================================="
                )
                mlflow.log_artifact("Transformation Pipeline.pkl")
                os.remove("Transformation Pipeline.pkl")

                # Log pandas profile
                if log_profile:
                    import pandas_profiling

                    pf = pandas_profiling.ProfileReport(
                        self.data_before_preprocess, **profile_kwargs
                    )
                    pf.to_file("Data Profile.html")
                    mlflow.log_artifact("Data Profile.html")
                    os.remove("Data Profile.html")
                    display.display(functions_, clear=True)

                # Log training and testing set
                if log_data:
                    self.X.to_csv("Dataset.csv")
                    mlflow.log_artifact("Dataset.csv")
                    os.remove("Dataset.csv")
        return

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
        display: Optional[Display] = None,
        **kwargs,
    ):

        function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

        self.logger.info("Initializing tune_model()")
        self.logger.info(f"tune_model({function_params_str})")

        self.logger.info("Checking exceptions")

        # run_time
        runtime_start = time.time()

        if not fit_kwargs:
            fit_kwargs = {}

        if supervised_target not in self.data_before_preprocess.columns:
            raise ValueError(
                f"{supervised_target} is not present as a column in the dataset."
            )

        warnings.filterwarnings("ignore")

        np.random.seed(self.seed)

        cols_to_drop = [x for x in self.X.columns if x.startswith(supervised_target)]
        data_X = self.X.drop(cols_to_drop, axis=1)
        data_y = self.data_before_preprocess[[supervised_target]]
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
            metrics = pycaret.containers.metrics.classification.get_all_metric_containers(
                temp_globals, raise_errors=True
            )
            available_estimators = pycaret.containers.models.classification.get_all_model_containers(
                temp_globals, raise_errors=True
            )
            ml_usecase = MLUsecase.CLASSIFICATION
        elif supervised_type == "regression":
            metrics = pycaret.containers.metrics.regression.get_all_metric_containers(
                temp_globals, raise_errors=True
            )
            available_estimators = pycaret.containers.models.regression.get_all_model_containers(
                temp_globals, raise_errors=True
            )
            ml_usecase = MLUsecase.REGRESSION
        else:
            raise ValueError(
                f"supervised_type param must be either 'classification' or 'regression'."
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
            raise ValueError(f"custom_grid param must be a list.")

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

        if not display:
            progress_args = {"max": len(param_grid) * 3 + (len(param_grid) + 1) * 4}
            master_display_columns = None
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

        unsupervised_models = {}
        unsupervised_models_results = {}
        unsupervised_grids = {0: data_X}

        self.logger.info("Fitting unsupervised models")

        for k in param_grid:
            if self._ml_usecase == MLUsecase.CLUSTERING:
                try:
                    new_model, _ = self.create_model(
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
                        f"Model {model} cannot be used in this function as its number of clusters cannot be set (n_clusters param required)."
                    )
            else:
                new_model, _ = self.create_model(
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
                    unsupervised_grids[k], columns=["Cluster"],
                )
            elif method == "drop":
                unsupervised_grids[k] = unsupervised_grids[k][
                    unsupervised_grids[k]["Anomaly"] == 0
                ].drop(["Anomaly", "Anomaly_Score"], axis=1)

        results = {}

        self.logger.info("Fitting supervised estimator")

        for k, v in unsupervised_grids.items():
            self.create_model(
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

        results = results.style.apply(
            highlight_max,
            subset=[x for x in results.columns if x not in greater_is_worse_columns],
        ).apply(
            highlight_min,
            subset=[x for x in results.columns if x in greater_is_worse_columns],
        )

        # end runtime
        runtime_end = time.time()
        runtime = np.array(runtime_end - runtime_start).round(2)

        if self._ml_usecase == MLUsecase.CLUSTERING:
            best_model, best_model_fit_time = self.create_model(
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
            best_model, best_model_fit_time = self.create_model(
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

            try:
                self._mlflow_log_model(
                    model=model,
                    model_results=None,
                    score_dict=metrics_log,
                    source="tune_model",
                    runtime=runtime,
                    model_fit_time=best_model_fit_time,
                    _prep_pipe=self.prep_pipe,
                    log_plots=self.log_plots_param,
                    display=display,
                )
            except:
                self.logger.error(
                    f"_mlflow_log_model() for {model} raised an exception:"
                )
                self.logger.error(traceback.format_exc())

        results = results.set_precision(round)
        self.display_container.append(results)

        display.display(results, clear=True)

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

        self.logger.info(f"create_model_container: {len(self.create_model_container)}")
        self.logger.info(f"master_model_container: {len(self.master_model_container)}")
        self.logger.info(f"display_container: {len(self.display_container)}")

        self.logger.info(str(best_model))
        self.logger.info(
            "tune_model() succesfully completed......................................"
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
        This function assigns each of the data point in the dataset passed during setup
        stage to one of the clusters using trained model object passed as model param.
        create_model() function must be called before using assign_model().
        
        This function returns a pandas.DataFrame.

        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> jewellery = get_data('jewellery')
        >>> experiment_name = setup(data = jewellery, normalize = True)
        >>> kmeans = create_model('kmeans')
        >>> kmeans_df = assign_model(kmeans)

        This will return a pandas.DataFrame with inferred clusters using trained model.

        Parameters
        ----------
        model: trained model object, default = None
        
        transformation: bool, default = False
            When set to True, assigned clusters are returned on transformed dataset instead 
            of original dataset passed during setup().
        
        verbose: Boolean, default = True
            Status update is not printed when verbose is set to False.

        Returns
        -------
        pandas.DataFrame
            Returns a DataFrame with assigned clusters using a trained model.
    
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
            data = self.X.copy()
            self.logger.info(
                "Transformation param set to True. Assigned clusters are attached on transformed dataset."
            )
        else:
            data = self.data_before_preprocess.copy()

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
            "assign_model() succesfully completed......................................"
        )

        return data

    def predict_model(
        self, estimator, data: pd.DataFrame, ml_usecase: Optional[MLUsecase] = None,
    ) -> pd.DataFrame:
        function_params_str = ", ".join(
            [f"{k}={v}" for k, v in locals().items() if k != "data"]
        )

        self.logger.info("Initializing predict_model()")
        self.logger.info(f"predict_model({function_params_str})")

        if ml_usecase is None:
            ml_usecase = self._ml_usecase

        # copy data and model
        data_transformed = data.copy()

        # exception checking for predict param
        if hasattr(estimator, "predict"):
            pass
        else:
            raise TypeError("Model doesn't support predict parameter.")

        pred_score = None

        # predictions start here
        if is_sklearn_pipeline(estimator):
            pred = estimator.predict(data_transformed)
            if ml_usecase == MLUsecase.ANOMALY:
                pred_score = estimator.decision_function(data_transformed)
        else:
            pred = estimator.predict(self.prep_pipe.transform(data_transformed))
            if ml_usecase == MLUsecase.ANOMALY:
                pred_score = estimator.decision_function(
                    self.prep_pipe.transform(data_transformed)
                )

        if ml_usecase == MLUsecase.CLUSTERING:
            pred_list = [f"Cluster {i}" for i in pred]

            data_transformed["Cluster"] = pred_list
        else:
            data_transformed["Anomaly"] = pred
            data_transformed["Anomaly_Score"] = pred_score

        return data_transformed

    def create_model(
        self,
        estimator,
        num_clusters: int = 4,
        fraction: float = 0.05,
        ground_truth: Optional[str] = None,
        round: int = 4,
        fit_kwargs: Optional[dict] = None,
        verbose: bool = True,
        system: bool = True,
        raise_num_clusters: bool = False,
        X_data: Optional[pd.DataFrame] = None,  # added in pycaret==2.2.0
        display: Optional[Display] = None,  # added in pycaret==2.2.0
        **kwargs,
    ) -> Any:

        """  
        This is an internal version of the create_model function.

        This function creates a model and scores it using Cross Validation. 
        The output prints a score grid that shows Accuracy, AUC, Recall, Precision, 
        F1, Kappa and MCC by fold (default = 10 Fold). 

        This function returns a trained model object. 

        setup() function must be called before using create_model()

        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> experiment_name = setup(data = juice,  target = 'Purchase')
        >>> lr = create_model('lr')

        This will create a trained Logistic Regression model.

        Parameters
        ----------
        model : string / object, default = None
            Enter ID of the models available in model library or pass an untrained model 
            object consistent with fit / predict API to train and evaluate model. List of 
            models available in model library (ID - Model):

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
            Number of clusters to be generated with the dataset.

        ground_truth: string, default = None
            When ground_truth is provided, Homogeneity Score, Rand Index, and 
            Completeness Score is evaluated and printer along with other metrics.

        round: integer, default = 4
            Number of decimal places the metrics in the score grid will be rounded to. 

        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.

        verbose: bool, default = True
            Score grid is not printed when verbose is set to False.

        system: bool, default = True
            Must remain True all times. Only to be changed by internal functions.
            If False, method will return a tuple of model and the model fit time.

        **kwargs: 
            Additional keyword arguments to pass to the estimator.

        Returns
        -------
        score_grid
            A table containing the Silhouette, Calinski-Harabasz,  
            Davies-Bouldin, Homogeneity Score, Rand Index, and 
            Completeness Score. Last 3 are only evaluated when
            ground_truth param is provided.

        model
            trained model object

        Warnings
        --------
        - num_clusters not required for Affinity Propagation ('ap'), Mean shift 
        clustering ('meanshift'), Density-Based Spatial Clustering ('dbscan')
        and OPTICS Clustering ('optics'). num_clusters param for these models 
        are automatically determined.
        
        - When fit doesn't converge in Affinity Propagation ('ap') model, all 
        datapoints are labelled as -1.
        
        - Noisy samples are given the label -1, when using Density-Based Spatial 
        ('dbscan') or OPTICS Clustering ('optics'). 
        
        - OPTICS ('optics') clustering may take longer training times on large 
        datasets.

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
            if ground_truth not in self.data_before_preprocess.columns:
                raise ValueError(
                    f"ground_truth {ground_truth} doesn't exist in the dataset."
                )

        """
        
        ERROR HANDLING ENDS HERE
        
        """

        if not display:
            progress_args = {"max": 3}
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

        self.logger.info("Importing libraries")

        # general dependencies

        np.random.seed(self.seed)

        # Storing X_train and y_train in data_X and data_y parameter
        data_X = self.X if X_data is None else X_data

        """
        MONITOR UPDATE STARTS
        """
        display.update_monitor(1, "Selecting Estimator")
        display.display_monitor()
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
        display.display_monitor()

        if self._ml_usecase == MLUsecase.CLUSTERING:
            if raise_num_clusters:
                model.set_params(n_clusters=num_clusters)
            else:
                try:
                    model.set_params(n_clusters=num_clusters)
                except:
                    pass
        else:
            model.set_params(contamination=fraction)

        # workaround for an issue with set_params in cuML
        try:
            model = clone(model)
        except:
            self.logger.warning(
                f"create_model_unsupervised() for {model} raised an exception when cloning:"
            )
            self.logger.warning(traceback.format_exc())

        self.logger.info(f"{full_name} Imported succesfully")

        display.move_progress()

        """
        MONITOR UPDATE STARTS
        """
        if self._ml_usecase == MLUsecase.CLUSTERING:
            display.update_monitor(1, f"Fitting {num_clusters} Clusters")
        else:
            display.update_monitor(1, f"Fitting {fraction} Fraction")
        display.display_monitor()
        """
        MONITOR UPDATE ENDS
        """

        with estimator_pipeline(self._internal_pipeline, model) as pipeline_with_model:
            fit_kwargs = get_pipeline_fit_kwargs(pipeline_with_model, fit_kwargs)

            self.logger.info("Fitting Model")
            model_fit_start = time.time()
            with io.capture_output():
                if is_cblof and "n_clusters" not in kwargs:
                    try:
                        pipeline_with_model.fit(data_X, **fit_kwargs)
                    except:
                        try:
                            pipeline_with_model.set_params(
                                actual_estimator__n_clusters=12
                            )
                            model_fit_start = time.time()
                            pipeline_with_model.fit(data_X, **fit_kwargs)
                        except:
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

            gt = np.array(self.data_before_preprocess[ground_truth])
        else:
            gt = None

        if self._ml_usecase == MLUsecase.CLUSTERING:
            metrics = self._calculate_metrics(data_X, model.labels_, ground_truth=gt)
        else:
            metrics = {}

        self.logger.info(str(model))
        self.logger.info(
            "create_models() succesfully completed......................................"
        )

        runtime = time.time() - runtime_start

        # mlflow logging
        if self.logging_param and system:

            metrics_log = {k: v for k, v in metrics.items()}

            try:
                self._mlflow_log_model(
                    model=model,
                    model_results=None,
                    score_dict=metrics_log,
                    source="create_model",
                    runtime=runtime,
                    model_fit_time=model_fit_time,
                    _prep_pipe=self.prep_pipe,
                    log_plots=self.log_plots_param,
                    display=display,
                )
            except:
                self.logger.error(
                    f"_mlflow_log_model() for {model} raised an exception:"
                )
                self.logger.error(traceback.format_exc())

        display.move_progress()

        self.logger.info("Uploading results into container")

        model_results = pd.DataFrame(metrics, index=[0])
        model_results = model_results.round(round)

        # storing results in create_model_container
        self.create_model_container.append(model_results)
        self.display_container.append(model_results)

        # storing results in master_model_container
        self.logger.info("Uploading model into container now")
        self.master_model_container.append(model)

        if self._ml_usecase == MLUsecase.CLUSTERING:
            display.display(
                model_results, clear=system, override=False if not system else None
            )
        elif system:
            display.clear_output()

        self.logger.info(f"create_model_container: {len(self.create_model_container)}")
        self.logger.info(f"master_model_container: {len(self.master_model_container)}")
        self.logger.info(f"display_container: {len(self.display_container)}")

        self.logger.info(str(model))
        self.logger.info(
            "create_model() succesfully completed......................................"
        )
        gc.collect()

        if not system:
            return (model, model_fit_time)

        return model


class RegressionExperiment(_SupervisedExperiment):
    def __init__(self) -> None:
        super().__init__()
        self._ml_usecase = MLUsecase.REGRESSION
        self.exp_name_log = "reg-default-name"
        self._available_plots = {
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
        }
        return

    def _get_models(self, raise_errors: bool = True) -> Tuple[dict, dict]:
        all_models = {
            k: v
            for k, v in pycaret.containers.models.regression.get_all_model_containers(
                self.variables, raise_errors=raise_errors
            ).items()
            if not v.is_special
        }
        all_models_internal = pycaret.containers.models.regression.get_all_model_containers(
            self.variables, raise_errors=raise_errors
        )
        return all_models, all_models_internal

    def _get_metrics(self, raise_errors: bool = True) -> dict:
        return pycaret.containers.metrics.regression.get_all_metric_containers(
            self.variables, raise_errors=raise_errors
        )

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
        categorical_imputation: str = "constant",
        categorical_iterative_imputer: Union[str, Any] = "lightgbm",
        ordinal_features: Optional[Dict[str, list]] = None,
        high_cardinality_features: Optional[List[str]] = None,
        high_cardinality_method: str = "frequency",
        numeric_features: Optional[List[str]] = None,
        numeric_imputation: str = "mean",
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
        feature_selection_method: str = "classic",
        feature_interaction: bool = False,
        feature_ratio: bool = False,
        interaction_threshold: float = 0.01,
        transform_target: bool = False,
        transform_target_method: str = "box-cox",
        data_split_shuffle: bool = True,
        data_split_stratify: Union[bool, List[str]] = False,
        fold_strategy: Union[str, Any] = "kfold",
        fold: int = 10,
        fold_shuffle: bool = False,
        fold_groups: Optional[Union[str, pd.DataFrame]] = None,
        n_jobs: Optional[int] = -1,
        use_gpu: bool = False,
        custom_pipeline: Union[
            Any, Tuple[str, Any], List[Any], List[Tuple[str, Any]]
        ] = None,
        html: bool = True,
        session_id: Optional[int] = None,
        log_experiment: bool = False,
        experiment_name: Optional[str] = None,
        log_plots: Union[bool, list] = False,
        log_profile: bool = False,
        log_data: bool = False,
        silent: bool = False,
        verbose: bool = True,
        profile: bool = False,
        profile_kwargs: Dict[str, Any] = None,
    ):
        """
        This function initializes the training environment and creates the transformation 
        pipeline. Setup function must be called before executing any other function. It takes 
        two mandatory parameters: ``data`` and ``target``. All the other parameters are
        optional.

        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> boston = get_data('boston')
        >>> from pycaret.regression import *
        >>> exp_name = setup(data = boston,  target = 'medv')


        data : pandas.DataFrame
            Shape (n_samples, n_features), where n_samples is the number of samples and 
            n_features is the number of features.


        target: str
            Name of the target column to be passed in as a string. The target variable can 
            be either binary or multiclass.


        train_size: float, default = 0.7
            Proportion of the dataset to be used for training and validation. Should be 
            between 0.0 and 1.0.


        test_data: pandas.DataFrame, default = None
            If not None, test_data is used as a hold-out set and ``train_size`` parameter is 
            ignored. test_data must be labelled and the shape of data and test_data must 
            match. 


        preprocess: bool, default = True
            When set to False, no transformations are applied except for train_test_split 
            and custom transformations passed in ``custom_pipeline`` param. Data must be 
            ready for modeling (no missing values, no dates, categorical data encoding), 
            when preprocess is set to False. 


        imputation_type: str, default = 'simple'
            The type of imputation to use. Can be either 'simple' or 'iterative'.


        iterative_imputation_iters: int, default = 5
            Number of iterations. Ignored when ``imputation_type`` is not 'iterative'.	


        categorical_features: list of str, default = None
            If the inferred data types are not correct or the silent param is set to True,
            categorical_features param can be used to overwrite or define the data types. 
            It takes a list of strings with column names that are categorical.


        categorical_imputation: str, default = 'constant'
            Missing values in categorical features are imputed with a constant 'not_available'
            value. The other available option is 'mode'.


        categorical_iterative_imputer: str, default = 'lightgbm'
            Estimator for iterative imputation of missing values in categorical features.
            Ignored when ``imputation_type`` is not 'iterative'. 


        ordinal_features: dict, default = None
            Encode categorical features as ordinal. For example, a categorical feature with 
            'low', 'medium', 'high' values where low < medium < high can be passed as  
            ordinal_features = { 'column_name' : ['low', 'medium', 'high'] }. 


        high_cardinality_features: list of str, default = None
            When categorical features contains many levels, it can be compressed into fewer
            levels using this parameter. It takes a list of strings with column names that 
            are categorical.


        high_cardinality_method: str, default = 'frequency'
            Categorical features with high cardinality are replaced with the frequency of
            values in each level occurring in the training dataset. Other available method
            is 'clustering' which trains the K-Means clustering algorithm on the statistical
            attribute of the training data and replaces the original value of feature with the 
            cluster label. The number of clusters is determined by optimizing Calinski-Harabasz 
            and Silhouette criterion. 


        numeric_features: list of str, default = None
            If the inferred data types are not correct or the silent param is set to True,
            numeric_features param can be used to overwrite or define the data types. 
            It takes a list of strings with column names that are numeric.


        numeric_imputation: str, default = 'mean'
            Missing values in numeric features are imputed with 'mean' value of the feature 
            in the training dataset. The other available option is 'median' or 'zero'.


        numeric_iterative_imputer: str, default = 'lightgbm'
            Estimator for iterative imputation of missing values in numeric features.
            Ignored when ``imputation_type`` is set to 'simple'. 


        date_features: list of str, default = None
            If the inferred data types are not correct or the silent param is set to True,
            date_features param can be used to overwrite or define the data types. It takes 
            a list of strings with column names that are DateTime.


        ignore_features: list of str, default = None
            ignore_features param can be used to ignore features during model training.
            It takes a list of strings with column names that are to be ignored.


        normalize: bool, default = False
            When set to True, it transforms the numeric features by scaling them to a given
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


        transformation: bool, default = False
            When set to True, it applies the power transform to make data more Gaussian-like.
            Type of transformation is defined by the ``transformation_method`` parameter.


        transformation_method: str, default = 'yeo-johnson'
            Defines the method for transformation. By default, the transformation method is 
            set to 'yeo-johnson'. The other available option for transformation is 'quantile'. 
            Ignored when ``transformation`` is not True.

        
        handle_unknown_categorical: bool, default = True
            When set to True, unknown categorical levels in unseen data are replaced by the
            most or least frequent level as learned in the training dataset. 


        unknown_categorical_method: str, default = 'least_frequent'
            Method used to replace unknown categorical levels in unseen data. Method can be
            set to 'least_frequent' or 'most_frequent'.


        pca: bool, default = False
            When set to True, dimensionality reduction is applied to project the data into 
            a lower dimensional space using the method defined in ``pca_method`` parameter. 
            

        pca_method: str, default = 'linear'
            The 'linear' method performs uses Singular Value  Decomposition. Other options are:
            
            - kernel: dimensionality reduction through the use of RVF kernel.
            - incremental: replacement for 'linear' pca when the dataset is too large.


        pca_components: int or float, default = None
            Number of components to keep. if pca_components is a float, it is treated as a 
            target percentage for information retention. When pca_components is an integer
            it is treated as the number of features to be kept. pca_components must be less
            than the original number of features. Ignored when ``pca`` is not True.


        ignore_low_variance: bool, default = False
            When set to True, all categorical features with insignificant variances are 
            removed from the data. The variance is calculated using the ratio of unique 
            values to the number of samples, and the ratio of the most common value to the 
            frequency of the second most common value.

        
        combine_rare_levels: bool, default = False
            When set to True, frequency percentile for levels in categorical features below 
            a certain threshold is combined into a single level.

        
        rare_level_threshold: float, default = 0.1
            Percentile distribution below which rare categories are combined. Ignored when
            ``combine_rare_levels`` is not True.

        
        bin_numeric_features: list of str, default = None
            To convert numeric features into categorical, bin_numeric_features parameter can 
            be used. It takes a list of strings with column names to be discretized. It does
            so by using 'sturges' rule to determine the number of clusters and then apply
            KMeans algorithm. Original values of the feature are then replaced by the
            cluster label.


        remove_outliers: bool, default = False
            When set to True, outliers from the training data are removed using the Singular 
            Value Decomposition.


        outliers_threshold: float, default = 0.05
            The percentage outliers to be removed from the training dataset. Ignored when 
            ``remove_outliers`` is not True.


        remove_multicollinearity: bool, default = False
            When set to True, features with the inter-correlations higher than the defined 
            threshold are removed. When two features are highly correlated with each other, 
            the feature that is less correlated with the target variable is removed. 


        multicollinearity_threshold: float, default = 0.9
            Threshold for correlated features. Ignored when ``remove_multicollinearity``
            is not True.

        
        remove_perfect_collinearity: bool, default = True
            When set to True, perfect collinearity (features with correlation = 1) is removed
            from the dataset, when two features are 100% correlated, one of it is randomly 
            removed from the dataset.


        create_clusters: bool, default = False
            When set to True, an additional feature is created in training dataset where each 
            instance is assigned to a cluster. The number of clusters is determined by 
            optimizing Calinski-Harabasz and Silhouette criterion.


        cluster_iter: int, default = 20
            Number of iterations for creating cluster. Each iteration represents cluster 
            size. Ignored when ``create_clusters`` is not True. 


        polynomial_features: bool, default = False
            When set to True, new features are derived using existing numeric features. 


        polynomial_degree: int, default = 2
            Degree of polynomial features. For example, if an input sample is two dimensional 
            and of the form [a, b], the polynomial features with degree = 2 are: 
            [1, a, b, a^2, ab, b^2]. Ignored when ``polynomial_features`` is not True.


        trigonometry_features: bool, default = False
            When set to True, new features are derived using existing numeric features.


        polynomial_threshold: float, default = 0.1
            When ``polynomial_features`` or ``trigonometry_features`` is True, new features
            are derived from the existing numeric features. This may sometimes result in too 
            large feature space. polynomial_threshold parameter can be used to deal with this  
            problem. It does so by using combination of Random Forest, AdaBoost and Linear 
            correlation. All derived features that falls within the percentile distribution 
            are kept and rest of the features are removed.


        group_features: list or list of list, default = None
            When the dataset contains features with related characteristics, group_features
            parameter can be used for feature extraction. It takes a list of strings with 
            column names that are related.

            
        group_names: list, default = None
            Group names to be used in naming new features. When the length of group_names 
            does not match with the length of ``group_features``, new features are named 
            sequentially group_1, group_2, etc. It is ignored when ``group_features`` is
            None.

        
        feature_selection: bool, default = False
            When set to True, a subset of features are selected using a combination of 
            various permutation importance techniques including Random Forest, Adaboost 
            and Linear correlation with target variable. The size of the subset is 
            dependent on the ``feature_selection_threshold`` parameter. 


        feature_selection_threshold: float, default = 0.8
            Threshold value used for feature selection. When ``polynomial_features`` or 
            ``feature_interaction`` is True, it is recommended to keep the threshold low
            to avoid large feature spaces. Setting a very low value may be efficient but 
            could result in under-fitting.

        
        feature_selection_method: str, default = 'classic'
            Algorithm for feature selection. 'classic' method uses permutation feature
            importance techniques. Other possible value is 'boruta' which uses boruta
            algorithm for feature selection. 

        
        feature_interaction: bool, default = False 
            When set to True, new features are created by interacting (a * b) all the 
            numeric variables in the dataset. This feature is not scalable and may not
            work as expected on datasets with large feature space.

        
        feature_ratio: bool, default = False
            When set to True, new features are created by calculating the ratios (a / b) 
            between all numeric variables in the dataset. This feature is not scalable and 
            may not work as expected on datasets with large feature space.

        
        interaction_threshold: bool, default = 0.01
            Similar to polynomial_threshold, It is used to compress a sparse matrix of newly 
            created features through interaction. Features whose importance based on the 
            combination  of  Random Forest, AdaBoost and Linear correlation falls within the 
            percentile of the  defined threshold are kept in the dataset. Remaining features 
            are dropped before further processing.


        transform_target: bool, default = False
            When set to True, target variable is transformed using the method defined in
            ``transform_target_method`` param. Target transformation is applied separately
            from feature transformations. 


        transform_target_method: str, default = 'box-cox'
            'Box-cox' and 'yeo-johnson' methods are supported. Box-Cox requires input data to 
            be strictly positive, while Yeo-Johnson supports both positive or negative data.
            When transform_target_method is 'box-cox' and target variable contains negative
            values, method is internally forced to 'yeo-johnson' to avoid exceptions.
            

        data_split_shuffle: bool, default = True
            When set to False, prevents shuffling of rows during 'train_test_split'.


        data_split_stratify: bool or list, default = False
            Controls stratification during 'train_test_split'. When set to True, will 
            stratify by target column. To stratify on any other columns, pass a list of 
            column names. Ignored when ``data_split_shuffle`` is False.


        fold_strategy: str or sklearn CV generator object, default = 'kfold'
            Choice of cross validation strategy. Possible values are:

            * 'kfold'
            * 'stratifiedkfold'
            * 'groupkfold'
            * 'timeseries'
            * a custom CV generator object compatible with scikit-learn.


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

            - CatBoost Regressor, requires no further installation
            (GPU is only enabled when data > 50,000 rows)
            
            - Light Gradient Boosting Machine, requires GPU installation
            https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html

            - Linear Regression, Lasso Regression, Ridge Regression, K Neighbors Regressor,
            Random Forest, Support Vector Regression, Elastic Net requires cuML >= 0.15 
            https://github.com/rapidsai/cuml


        custom_pipeline: (str, transformer) or list of (str, transformer), default = None
            When passed, will append the custom transformers in the preprocessing pipeline
            and are applied on each CV fold separately and on the final fit. All the custom
            transformations are applied after 'train_test_split' and before pycaret's internal 
            transformations. 


        html: bool, default = True
            When set to False, prevents runtime display of monitor. This must be set to False
            when the environment does not support IPython. For example, command line terminal,
            Databricks Notebook, Spyder and other similar IDEs. 


        session_id: int, default = None
            Controls the randomness of experiment. It is equivalent to 'random_state' in
            scikit-learn. When None, a pseudo random number is generated. This can be used 
            for later reproducibility of the entire experiment.


        log_experiment: bool, default = False
            When set to True, all metrics and parameters are logged on the ``MLFlow`` server.


        experiment_name: str, default = None
            Name of the experiment for logging. Ignored when ``log_experiment`` is not True.


        log_plots: bool or list, default = False
            When set to True, certain plots are logged automatically in the ``MLFlow`` server. 
            To change the type of plots to be logged, pass a list containing plot IDs. Refer
            to documentation of ``plot_model``. Ignored when ``log_experiment`` is not True.


        log_profile: bool, default = False
            When set to True, data profile is logged on the ``MLflow`` server as a html file.
            Ignored when ``log_experiment`` is not True. 


        log_data: bool, default = False
            When set to True, dataset is logged on the ``MLflow`` server as a csv file.
            Ignored when ``log_experiment`` is not True.
            

        silent: bool, default = False
            Controls the confirmation input of data types when ``setup`` is executed. When
            executing in completely automated mode or on a remote kernel, this must be True.

        
        verbose: bool, default = True
            When set to False, Information grid is not printed.


        profile: bool, default = False
            When set to True, an interactive EDA report is displayed. 


        profile_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the ProfileReport method used
            to create the EDA report. Ignored if ``profile`` is False.


        Returns:
            Global variables that can be changed using the ``set_config`` function.
        
        """
        if log_plots == True:
            log_plots = ["residuals", "error", "feature"]
        return super().setup(
            data=data,
            target=target,
            train_size=train_size,
            test_data=test_data,
            preprocess=preprocess,
            imputation_type=imputation_type,
            iterative_imputation_iters=iterative_imputation_iters,
            categorical_features=categorical_features,
            categorical_imputation=categorical_imputation,
            categorical_iterative_imputer=categorical_iterative_imputer,
            ordinal_features=ordinal_features,
            high_cardinality_features=high_cardinality_features,
            high_cardinality_method=high_cardinality_method,
            numeric_features=numeric_features,
            numeric_imputation=numeric_imputation,
            numeric_iterative_imputer=numeric_iterative_imputer,
            date_features=date_features,
            ignore_features=ignore_features,
            normalize=normalize,
            normalize_method=normalize_method,
            transformation=transformation,
            transformation_method=transformation_method,
            handle_unknown_categorical=handle_unknown_categorical,
            unknown_categorical_method=unknown_categorical_method,
            pca=pca,
            pca_method=pca_method,
            pca_components=pca_components,
            ignore_low_variance=ignore_low_variance,
            combine_rare_levels=combine_rare_levels,
            rare_level_threshold=rare_level_threshold,
            bin_numeric_features=bin_numeric_features,
            remove_outliers=remove_outliers,
            outliers_threshold=outliers_threshold,
            remove_multicollinearity=remove_multicollinearity,
            multicollinearity_threshold=multicollinearity_threshold,
            remove_perfect_collinearity=remove_perfect_collinearity,
            create_clusters=create_clusters,
            cluster_iter=cluster_iter,
            polynomial_features=polynomial_features,
            polynomial_degree=polynomial_degree,
            trigonometry_features=trigonometry_features,
            polynomial_threshold=polynomial_threshold,
            group_features=group_features,
            group_names=group_names,
            feature_selection=feature_selection,
            feature_selection_threshold=feature_selection_threshold,
            feature_selection_method=feature_selection_method,
            feature_interaction=feature_interaction,
            feature_ratio=feature_ratio,
            interaction_threshold=interaction_threshold,
            transform_target=transform_target,
            transform_target_method=transform_target_method,
            data_split_shuffle=data_split_shuffle,
            data_split_stratify=data_split_stratify,
            fold_strategy=fold_strategy,
            fold=fold,
            fold_shuffle=fold_shuffle,
            fold_groups=fold_groups,
            n_jobs=n_jobs,
            use_gpu=use_gpu,
            custom_pipeline=custom_pipeline,
            html=html,
            session_id=session_id,
            log_experiment=log_experiment,
            experiment_name=experiment_name,
            log_plots=log_plots,
            log_profile=log_profile,
            log_data=log_data,
            silent=silent,
            verbose=verbose,
            profile=profile,
            profile_kwargs=profile_kwargs,
        )

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
        verbose: bool = True,
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


        verbose: bool, default = True
            Score grid is not printed when verbose is set to False.
        
        
        Returns:
            Trained model or list of trained models, depending on the ``n_select`` param.


        Warnings
        --------
        - Changing turbo parameter to False may result in very high training times with 
        datasets exceeding 10,000 rows.

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
            verbose=verbose,
        )

    def create_model(
        self,
        estimator: Union[str, Any],
        fold: Optional[Union[int, Any]] = None,
        round: int = 4,
        cross_validation: bool = True,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        verbose: bool = True,
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


        verbose: bool, default = True
            Score grid is not printed when verbose is set to False.


        **kwargs: 
            Additional keyword arguments to pass to the estimator.


        Returns:
            Trained Model


        Warnings
        --------
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
            **kwargs,
        )

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
        choose_better: bool = False,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        return_tuner: bool = False,
        verbose: bool = True,
        tuner_verbose: Union[int, bool] = True,
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


        choose_better: bool, default = False
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
        optimize: str = "R2",
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        verbose: bool = True,
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
        )

    def stack_models(
        self,
        estimator_list: list,
        meta_model=None,
        fold: Optional[Union[int, Any]] = None,
        round: int = 4,
        restack: bool = True,
        choose_better: bool = False,
        optimize: str = "R2",
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        verbose: bool = True,
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


        fold: int or scikit-learn compatible CV generator, default = None
            Controls cross-validation. If None, the CV generator in the ``fold_strategy`` 
            parameter of the ``setup`` function is used. When an integer is passed, 
            it is interpreted as the 'n_splits' parameter of the CV generator in the 
            ``setup`` function.


        round: int, default = 4
            Number of decimal places the metrics in the score grid will be rounded to.


        restack: bool, default = True
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


        Returns:
            Trained Model

        """

        return super().stack_models(
            estimator_list=estimator_list,
            meta_model=meta_model,
            fold=fold,
            round=round,
            method="auto",
            restack=restack,
            choose_better=choose_better,
            optimize=optimize,
            fit_kwargs=fit_kwargs,
            groups=groups,
            verbose=verbose,
        )

    def plot_model(
        self,
        estimator,
        plot: str = "residuals",
        scale: float = 1,
        save: bool = False,
        fold: Optional[Union[int, Any]] = None,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        use_train_data: bool = False,
        verbose: bool = True,
    ) -> str:

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


        Returns:
            None

        """

        return super().plot_model(
            estimator=estimator,
            plot=plot,
            scale=scale,
            save=save,
            fold=fold,
            fit_kwargs=fit_kwargs,
            groups=groups,
            verbose=verbose,
            use_train_data=use_train_data,
            system=True,
        )

    def evaluate_model(
        self,
        estimator,
        fold: Optional[Union[int, Any]] = None,
        fit_kwargs: Optional[dict] = None,
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
        **kwargs,
    ):

        """
        This function analyzes the predictions generated from a tree-based model. It is
        implemented based on the SHAP (SHapley Additive exPlanations). For more info on
        this, please see https://shap.readthedocs.io/en/latest/


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


        plot: str, default = 'summary'
            Type of plot. Available options are: 'summary', 'correlation', and 'reason'.


        feature: str, default = None
            Feature to check correlation with. This parameter is only required when ``plot``
            type is 'correlation'. When set to None, it uses the first column in the train
            dataset.


        observation: int, default = None
            Observation index number in holdout set to explain. When ``plot`` is not
            'reason', this parameter is ignored. 


        use_train_data: bool, default = False
            When set to true, train data will be used for plots, instead
            of test data.


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
            **kwargs,
        )

    def predict_model(
        self,
        estimator,
        data: Optional[pd.DataFrame] = None,
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
            encoded_labels=True,
            round=round,
            verbose=verbose,
            ml_usecase=MLUsecase.REGRESSION,
        )

    def finalize_model(
        self,
        estimator,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        model_only: bool = True,
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
        )

    def deploy_model(
        self, model, model_name: str, authentication: dict, platform: str = "aws",
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
        >>> deploy_model(model = lr, model_name = 'lr-for-deployment', platform = 'aws', authentication = {'bucket' : 'S3-bucket-name'})
            

        Amazon Web Service (AWS) users:
            To deploy a model on AWS S3 ('aws'), environment variables must be set in your
            local environment. To configure AWS environment variables, type ``aws configure`` 
            in the command line. Following information from the IAM portal of amazon console 
            account is required:

            - AWS Access Key ID
            - AWS Secret Key Access
            - Default Region Name (can be seen under Global settings on your AWS console)

            More info: https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-envvars.html


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

            More info: https://docs.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python?toc=%2Fpython%2Fazure%2FTOC.json


        model: scikit-learn compatible object
            Trained model object
        

        model_name: str
            Name of model.
        

        authentication: dict
            Dictionary of applicable authentication tokens.

            When platform = 'aws':
            {'bucket' : 'S3-bucket-name'}

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
        self, model, model_name: str, model_only: bool = False, verbose: bool = True
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


        verbose: bool, default = True
            Success message is not printed when verbose is set to False.


        Returns:
            Tuple of the model object and the filename.

        """

        return super().save_model(
            model=model, model_name=model_name, model_only=model_only, verbose=verbose
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

    def automl(self, optimize: str = "R2", use_holdout: bool = False) -> Any:

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
        

        Returns:
            Trained Model


        """

        return super().automl(optimize=optimize, use_holdout=use_holdout)

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
        Returns table of available metrics used for CV.


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
            reset=reset, include_custom=include_custom, raise_errors=raise_errors,
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
        Adds a custom metric to be used for CV.


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
        Removes a metric from CV.


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


class ClassificationExperiment(_SupervisedExperiment):
    def __init__(self) -> None:
        super().__init__()
        self._ml_usecase = MLUsecase.CLASSIFICATION
        self.exp_name_log = "clf-default-name"
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
        }
        return

    def _get_models(self, raise_errors: bool = True) -> Tuple[dict, dict]:
        all_models = {
            k: v
            for k, v in pycaret.containers.models.classification.get_all_model_containers(
                self.variables, raise_errors=raise_errors
            ).items()
            if not v.is_special
        }
        all_models_internal = pycaret.containers.models.classification.get_all_model_containers(
            self.variables, raise_errors=raise_errors
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
        except:
            return False

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
        categorical_imputation: str = "constant",
        categorical_iterative_imputer: Union[str, Any] = "lightgbm",
        ordinal_features: Optional[Dict[str, list]] = None,
        high_cardinality_features: Optional[List[str]] = None,
        high_cardinality_method: str = "frequency",
        numeric_features: Optional[List[str]] = None,
        numeric_imputation: str = "mean",
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
        feature_selection_method: str = "classic",
        feature_interaction: bool = False,
        feature_ratio: bool = False,
        interaction_threshold: float = 0.01,
        fix_imbalance: bool = False,
        fix_imbalance_method: Optional[Any] = None,
        data_split_shuffle: bool = True,
        data_split_stratify: Union[bool, List[str]] = False,
        fold_strategy: Union[str, Any] = "stratifiedkfold",
        fold: int = 10,
        fold_shuffle: bool = False,
        fold_groups: Optional[Union[str, pd.DataFrame]] = None,
        n_jobs: Optional[int] = -1,
        use_gpu: bool = False,
        custom_pipeline: Union[
            Any, Tuple[str, Any], List[Any], List[Tuple[str, Any]]
        ] = None,
        html: bool = True,
        session_id: Optional[int] = None,
        log_experiment: bool = False,
        experiment_name: Optional[str] = None,
        log_plots: Union[bool, list] = False,
        log_profile: bool = False,
        log_data: bool = False,
        silent: bool = False,
        verbose: bool = True,
        profile: bool = False,
        profile_kwargs: Dict[str, Any] = None,
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


        data: pandas.DataFrame
            Shape (n_samples, n_features), where n_samples is the number of samples and 
            n_features is the number of features.


        target: str
            Name of the target column to be passed in as a string. The target variable can 
            be either binary or multiclass.


        train_size: float, default = 0.7
            Proportion of the dataset to be used for training and validation. Should be 
            between 0.0 and 1.0.


        test_data: pandas.DataFrame, default = None
            If not None, test_data is used as a hold-out set and ``train_size`` parameter is 
            ignored. test_data must be labelled and the shape of data and test_data must 
            match. 


        preprocess: bool, default = True
            When set to False, no transformations are applied except for train_test_split 
            and custom transformations passed in ``custom_pipeline`` param. Data must be 
            ready for modeling (no missing values, no dates, categorical data encoding), 
            when preprocess is set to False. 


        imputation_type: str, default = 'simple'
            The type of imputation to use. Can be either 'simple' or 'iterative'.


        iterative_imputation_iters: int, default = 5
            Number of iterations. Ignored when ``imputation_type`` is not 'iterative'.	


        categorical_features: list of str, default = None
            If the inferred data types are not correct or the silent param is set to True,
            categorical_features param can be used to overwrite or define the data types. 
            It takes a list of strings with column names that are categorical.


        categorical_imputation: str, default = 'constant'
            Missing values in categorical features are imputed with a constant 'not_available'
            value. The other available option is 'mode'.


        categorical_iterative_imputer: str, default = 'lightgbm'
            Estimator for iterative imputation of missing values in categorical features.
            Ignored when ``imputation_type`` is not 'iterative'. 


        ordinal_features: dict, default = None
            Encode categorical features as ordinal. For example, a categorical feature with 
            'low', 'medium', 'high' values where low < medium < high can be passed as  
            ordinal_features = { 'column_name' : ['low', 'medium', 'high'] }. 


        high_cardinality_features: list of str, default = None
            When categorical features contains many levels, it can be compressed into fewer
            levels using this parameter. It takes a list of strings with column names that 
            are categorical.


        high_cardinality_method: str, default = 'frequency'
            Categorical features with high cardinality are replaced with the frequency of
            values in each level occurring in the training dataset. Other available method
            is 'clustering' which trains the K-Means clustering algorithm on the statistical
            attribute of the training data and replaces the original value of feature with the 
            cluster label. The number of clusters is determined by optimizing Calinski-Harabasz 
            and Silhouette criterion. 


        numeric_features: list of str, default = None
            If the inferred data types are not correct or the silent param is set to True,
            numeric_features param can be used to overwrite or define the data types. 
            It takes a list of strings with column names that are numeric.


        numeric_imputation: str, default = 'mean'
            Missing values in numeric features are imputed with 'mean' value of the feature 
            in the training dataset. The other available option is 'median' or 'zero'.


        numeric_iterative_imputer: str, default = 'lightgbm'
            Estimator for iterative imputation of missing values in numeric features.
            Ignored when ``imputation_type`` is set to 'simple'. 


        date_features: list of str, default = None
            If the inferred data types are not correct or the silent param is set to True,
            date_features param can be used to overwrite or define the data types. It takes 
            a list of strings with column names that are DateTime.


        ignore_features: list of str, default = None
            ignore_features param can be used to ignore features during model training.
            It takes a list of strings with column names that are to be ignored.


        normalize: bool, default = False
            When set to True, it transforms the numeric features by scaling them to a given
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


        transformation: bool, default = False
            When set to True, it applies the power transform to make data more Gaussian-like.
            Type of transformation is defined by the ``transformation_method`` parameter.


        transformation_method: str, default = 'yeo-johnson'
            Defines the method for transformation. By default, the transformation method is 
            set to 'yeo-johnson'. The other available option for transformation is 'quantile'. 
            Ignored when ``transformation`` is not True.

        
        handle_unknown_categorical: bool, default = True
            When set to True, unknown categorical levels in unseen data are replaced by the
            most or least frequent level as learned in the training dataset. 


        unknown_categorical_method: str, default = 'least_frequent'
            Method used to replace unknown categorical levels in unseen data. Method can be
            set to 'least_frequent' or 'most_frequent'.


        pca: bool, default = False
            When set to True, dimensionality reduction is applied to project the data into 
            a lower dimensional space using the method defined in ``pca_method`` parameter. 
            

        pca_method: str, default = 'linear'
            The 'linear' method performs uses Singular Value  Decomposition. Other options are:
            
            - kernel: dimensionality reduction through the use of RVF kernel.
            - incremental: replacement for 'linear' pca when the dataset is too large.


        pca_components: int or float, default = None
            Number of components to keep. if pca_components is a float, it is treated as a 
            target percentage for information retention. When pca_components is an integer
            it is treated as the number of features to be kept. pca_components must be less
            than the original number of features. Ignored when ``pca`` is not True.


        ignore_low_variance: bool, default = False
            When set to True, all categorical features with insignificant variances are 
            removed from the data. The variance is calculated using the ratio of unique 
            values to the number of samples, and the ratio of the most common value to the 
            frequency of the second most common value.

        
        combine_rare_levels: bool, default = False
            When set to True, frequency percentile for levels in categorical features below 
            a certain threshold is combined into a single level.

        
        rare_level_threshold: float, default = 0.1
            Percentile distribution below which rare categories are combined. Ignored when
            ``combine_rare_levels`` is not True.

        
        bin_numeric_features: list of str, default = None
            To convert numeric features into categorical, bin_numeric_features parameter can 
            be used. It takes a list of strings with column names to be discretized. It does
            so by using 'sturges' rule to determine the number of clusters and then apply
            KMeans algorithm. Original values of the feature are then replaced by the
            cluster label.


        remove_outliers: bool, default = False
            When set to True, outliers from the training data are removed using the Singular 
            Value Decomposition.


        outliers_threshold: float, default = 0.05
            The percentage outliers to be removed from the training dataset. Ignored when 
            ``remove_outliers`` is not True.


        remove_multicollinearity: bool, default = False
            When set to True, features with the inter-correlations higher than the defined 
            threshold are removed. When two features are highly correlated with each other, 
            the feature that is less correlated with the target variable is removed. 


        multicollinearity_threshold: float, default = 0.9
            Threshold for correlated features. Ignored when ``remove_multicollinearity``
            is not True.

        
        remove_perfect_collinearity: bool, default = True
            When set to True, perfect collinearity (features with correlation = 1) is removed
            from the dataset, when two features are 100% correlated, one of it is randomly 
            removed from the dataset.


        create_clusters: bool, default = False
            When set to True, an additional feature is created in training dataset where each 
            instance is assigned to a cluster. The number of clusters is determined by 
            optimizing Calinski-Harabasz and Silhouette criterion.


        cluster_iter: int, default = 20
            Number of iterations for creating cluster. Each iteration represents cluster 
            size. Ignored when ``create_clusters`` is not True. 


        polynomial_features: bool, default = False
            When set to True, new features are derived using existing numeric features. 


        polynomial_degree: int, default = 2
            Degree of polynomial features. For example, if an input sample is two dimensional 
            and of the form [a, b], the polynomial features with degree = 2 are: 
            [1, a, b, a^2, ab, b^2]. Ignored when ``polynomial_features`` is not True.


        trigonometry_features: bool, default = False
            When set to True, new features are derived using existing numeric features.


        polynomial_threshold: float, default = 0.1
            When ``polynomial_features`` or ``trigonometry_features`` is True, new features
            are derived from the existing numeric features. This may sometimes result in too 
            large feature space. polynomial_threshold parameter can be used to deal with this  
            problem. It does so by using combination of Random Forest, AdaBoost and Linear 
            correlation. All derived features that falls within the percentile distribution 
            are kept and rest of the features are removed.


        group_features: list or list of list, default = None
            When the dataset contains features with related characteristics, group_features
            parameter can be used for feature extraction. It takes a list of strings with 
            column names that are related.

            
        group_names: list, default = None
            Group names to be used in naming new features. When the length of group_names 
            does not match with the length of ``group_features``, new features are named 
            sequentially group_1, group_2, etc. It is ignored when ``group_features`` is
            None.

        
        feature_selection: bool, default = False
            When set to True, a subset of features are selected using a combination of 
            various permutation importance techniques including Random Forest, Adaboost 
            and Linear correlation with target variable. The size of the subset is 
            dependent on the ``feature_selection_threshold`` parameter. 


        feature_selection_threshold: float, default = 0.8
            Threshold value used for feature selection. When ``polynomial_features`` or 
            ``feature_interaction`` is True, it is recommended to keep the threshold low
            to avoid large feature spaces. Setting a very low value may be efficient but 
            could result in under-fitting.

        
        feature_selection_method: str, default = 'classic'
            Algorithm for feature selection. 'classic' method uses permutation feature
            importance techniques. Other possible value is 'boruta' which uses boruta
            algorithm for feature selection. 

        
        feature_interaction: bool, default = False 
            When set to True, new features are created by interacting (a * b) all the 
            numeric variables in the dataset. This feature is not scalable and may not
            work as expected on datasets with large feature space.

        
        feature_ratio: bool, default = False
            When set to True, new features are created by calculating the ratios (a / b) 
            between all numeric variables in the dataset. This feature is not scalable and 
            may not work as expected on datasets with large feature space.

        
        interaction_threshold: bool, default = 0.01
            Similar to polynomial_threshold, It is used to compress a sparse matrix of newly 
            created features through interaction. Features whose importance based on the 
            combination  of  Random Forest, AdaBoost and Linear correlation falls within the 
            percentile of the  defined threshold are kept in the dataset. Remaining features 
            are dropped before further processing.

        
        fix_imbalance: bool, default = False
            When training dataset has unequal distribution of target class it can be balanced 
            using this parameter. When set to True, SMOTE (Synthetic Minority Over-sampling 
            Technique) is applied by default to create synthetic datapoints for minority class.


        fix_imbalance_method: obj, default = None
            When ``fix_imbalance`` is True, 'imblearn' compatible object with 'fit_resample'
            method can be passed. When set to None, 'imblearn.over_sampling.SMOTE' is used.  
            

        data_split_shuffle: bool, default = True
            When set to False, prevents shuffling of rows during 'train_test_split'.


        data_split_stratify: bool or list, default = False
            Controls stratification during 'train_test_split'. When set to True, will 
            stratify by target column. To stratify on any other columns, pass a list of 
            column names. Ignored when ``data_split_shuffle`` is False.


        fold_strategy: str or sklearn CV generator object, default = 'stratifiedkfold'
            Choice of cross validation strategy. Possible values are:

            * 'kfold'
            * 'stratifiedkfold'
            * 'groupkfold'
            * 'timeseries'
            * a custom CV generator object compatible with scikit-learn.


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

            - Logistic Regression, Ridge Classifier, Random Forest, K Neighbors Classifier,
            Support Vector Machine, requires cuML >= 0.15 
            https://github.com/rapidsai/cuml


        custom_pipeline: (str, transformer) or list of (str, transformer), default = None
            When passed, will append the custom transformers in the preprocessing pipeline
            and are applied on each CV fold separately and on the final fit. All the custom
            transformations are applied after 'train_test_split' and before pycaret's internal 
            transformations. 


        html: bool, default = True
            When set to False, prevents runtime display of monitor. This must be set to False
            when the environment does not support IPython. For example, command line terminal,
            Databricks Notebook, Spyder and other similar IDEs. 


        session_id: int, default = None
            Controls the randomness of experiment. It is equivalent to 'random_state' in
            scikit-learn. When None, a pseudo random number is generated. This can be used 
            for later reproducibility of the entire experiment.


        log_experiment: bool, default = False
            When set to True, all metrics and parameters are logged on the ``MLFlow`` server.


        experiment_name: str, default = None
            Name of the experiment for logging. Ignored when ``log_experiment`` is not True.


        log_plots: bool or list, default = False
            When set to True, certain plots are logged automatically in the ``MLFlow`` server. 
            To change the type of plots to be logged, pass a list containing plot IDs. Refer
            to documentation of ``plot_model``. Ignored when ``log_experiment`` is not True.


        log_profile: bool, default = False
            When set to True, data profile is logged on the ``MLflow`` server as a html file.
            Ignored when ``log_experiment`` is not True. 


        log_data: bool, default = False
            When set to True, dataset is logged on the ``MLflow`` server as a csv file.
            Ignored when ``log_experiment`` is not True.
            

        silent: bool, default = False
            Controls the confirmation input of data types when ``setup`` is executed. When
            executing in completely automated mode or on a remote kernel, this must be True.

        
        verbose: bool, default = True
            When set to False, Information grid is not printed.


        profile: bool, default = False
            When set to True, an interactive EDA report is displayed. 


        profile_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the ProfileReport method used
            to create the EDA report. Ignored if ``profile`` is False.


        Returns:
            Global variables that can be changed using the ``set_config`` function.
            
        """
        if log_plots == True:
            log_plots = ["auc", "confusion_matrix", "feature"]
        return super().setup(
            data=data,
            target=target,
            train_size=train_size,
            test_data=test_data,
            preprocess=preprocess,
            imputation_type=imputation_type,
            iterative_imputation_iters=iterative_imputation_iters,
            categorical_features=categorical_features,
            categorical_imputation=categorical_imputation,
            categorical_iterative_imputer=categorical_iterative_imputer,
            ordinal_features=ordinal_features,
            high_cardinality_features=high_cardinality_features,
            high_cardinality_method=high_cardinality_method,
            numeric_features=numeric_features,
            numeric_imputation=numeric_imputation,
            numeric_iterative_imputer=numeric_iterative_imputer,
            date_features=date_features,
            ignore_features=ignore_features,
            normalize=normalize,
            normalize_method=normalize_method,
            transformation=transformation,
            transformation_method=transformation_method,
            handle_unknown_categorical=handle_unknown_categorical,
            unknown_categorical_method=unknown_categorical_method,
            pca=pca,
            pca_method=pca_method,
            pca_components=pca_components,
            ignore_low_variance=ignore_low_variance,
            combine_rare_levels=combine_rare_levels,
            rare_level_threshold=rare_level_threshold,
            bin_numeric_features=bin_numeric_features,
            remove_outliers=remove_outliers,
            outliers_threshold=outliers_threshold,
            remove_multicollinearity=remove_multicollinearity,
            multicollinearity_threshold=multicollinearity_threshold,
            remove_perfect_collinearity=remove_perfect_collinearity,
            create_clusters=create_clusters,
            cluster_iter=cluster_iter,
            polynomial_features=polynomial_features,
            polynomial_degree=polynomial_degree,
            trigonometry_features=trigonometry_features,
            polynomial_threshold=polynomial_threshold,
            group_features=group_features,
            group_names=group_names,
            feature_selection=feature_selection,
            feature_selection_threshold=feature_selection_threshold,
            feature_selection_method=feature_selection_method,
            feature_interaction=feature_interaction,
            feature_ratio=feature_ratio,
            interaction_threshold=interaction_threshold,
            fix_imbalance=fix_imbalance,
            fix_imbalance_method=fix_imbalance_method,
            data_split_shuffle=data_split_shuffle,
            data_split_stratify=data_split_stratify,
            fold_strategy=fold_strategy,
            fold=fold,
            fold_shuffle=fold_shuffle,
            fold_groups=fold_groups,
            n_jobs=n_jobs,
            use_gpu=use_gpu,
            custom_pipeline=custom_pipeline,
            html=html,
            session_id=session_id,
            log_experiment=log_experiment,
            experiment_name=experiment_name,
            log_plots=log_plots,
            log_profile=log_profile,
            log_data=log_data,
            silent=silent,
            verbose=verbose,
            profile=profile,
            profile_kwargs=profile_kwargs,
        )

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
            verbose=verbose,
        )

    def create_model(
        self,
        estimator: Union[str, Any],
        fold: Optional[Union[int, Any]] = None,
        round: int = 4,
        cross_validation: bool = True,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
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
        choose_better: bool = False,
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


        choose_better: bool, default = False
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
        )

    def stack_models(
        self,
        estimator_list: list,
        meta_model=None,
        fold: Optional[Union[int, Any]] = None,
        round: int = 4,
        method: str = "auto",
        restack: bool = True,
        choose_better: bool = False,
        optimize: str = "Accuracy",
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
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
            
            
        restack: bool, default = True
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
            fold=fold,
            round=round,
            method=method,
            restack=restack,
            choose_better=choose_better,
            optimize=optimize,
            fit_kwargs=fit_kwargs,
            groups=groups,
            verbose=verbose,
        )

    def plot_model(
        self,
        estimator,
        plot: str = "auc",
        scale: float = 1,
        save: bool = False,
        fold: Optional[Union[int, Any]] = None,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        use_train_data: bool = False,
        verbose: bool = True,
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
            groups=groups,
            verbose=verbose,
            use_train_data=use_train_data,
            system=True,
        )

    def evaluate_model(
        self,
        estimator,
        fold: Optional[Union[int, Any]] = None,
        fit_kwargs: Optional[dict] = None,
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
        **kwargs,
    ):

        """ 
        This function analyzes the predictions generated from a tree-based model. It is
        implemented based on the SHAP (SHapley Additive exPlanations). For more info on
        this, please see https://shap.readthedocs.io/en/latest/

        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase')
        >>> xgboost = create_model('xgboost')
        >>> interpret_model(xgboost)


        estimator: scikit-learn compatible object
            Trained model object


        plot: str, default = 'summary'
            Type of plot. Available options are: 'summary', 'correlation', and 'reason'.


        feature: str, default = None
            Feature to check correlation with. This parameter is only required when ``plot``
            type is 'correlation'. When set to None, it uses the first column in the train
            dataset.


        observation: int, default = None
            Observation index number in holdout set to explain. When ``plot`` is not
            'reason', this parameter is ignored. 


        use_train_data: bool, default = False
            When set to true, train data will be used for plots, instead
            of test data.


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
            **kwargs,
        )

    def calibrate_model(
        self,
        estimator,
        method: str = "sigmoid",
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
            If None, will use the value set in fold_groups param in setup().

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
            cv=fold,
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
                    _prep_pipe=self.prep_pipe,
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

        self.logger.info(f"create_model_container: {len(self.create_model_container)}")
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
        true_positive: int = 0,
        true_negative: int = 0,
        false_positive: int = 0,
        false_negative: int = 0,
    ):

        """
        This function optimizes probability threshold for a trained model using custom cost
        function that can be defined using combination of True Positives, True Negatives,
        False Positives (also known as Type I error), and False Negatives (Type II error).
        
        This function returns a plot of optimized cost as a function of probability 
        threshold between 0 to 100. 

        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> experiment_name = setup(data = juice,  target = 'Purchase')
        >>> lr = create_model('lr')
        >>> optimize_threshold(lr, true_negative = 10, false_negative = -100)

        This will return a plot of optimized cost as a function of probability threshold.

        Parameters
        ----------
        estimator : object
            A trained model object should be passed as an estimator. 
        
        true_positive : int, default = 0
            Cost function or returns when prediction is true positive.  
        
        true_negative : int, default = 0
            Cost function or returns when prediction is true negative.
        
        false_positive : int, default = 0
            Cost function or returns when prediction is false positive.    
        
        false_negative : int, default = 0
            Cost function or returns when prediction is false negative.       
        
        
        Returns
        -------
        Visual_Plot
            Prints the visual plot. 

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

        # check cost function type
        allowed_types = [int, float]

        if type(true_positive) not in allowed_types:
            raise TypeError(
                "true_positive parameter only accepts float or integer value."
            )

        if type(true_negative) not in allowed_types:
            raise TypeError(
                "true_negative parameter only accepts float or integer value."
            )

        if type(false_positive) not in allowed_types:
            raise TypeError(
                "false_positive parameter only accepts float or integer value."
            )

        if type(false_negative) not in allowed_types:
            raise TypeError(
                "false_negative parameter only accepts float or integer value."
            )

        """
        ERROR HANDLING ENDS HERE
        """

        # define model as estimator
        model = estimator

        model_name = self._get_model_name(model)

        # generate predictions and store actual on y_test in numpy array
        actual = self.y_test.values

        predicted = model.predict_proba(self.X_test.values)
        predicted = predicted[:, 1]

        """
        internal function to calculate loss starts here
        """

        self.logger.info("Defining loss function")

        def calculate_loss(
            actual,
            predicted,
            tp_cost=true_positive,
            tn_cost=true_negative,
            fp_cost=false_positive,
            fn_cost=false_negative,
        ):

            # true positives
            tp = predicted + actual
            tp = np.where(tp == 2, 1, 0)
            tp = tp.sum()

            # true negative
            tn = predicted + actual
            tn = np.where(tn == 0, 1, 0)
            tn = tn.sum()

            # false positive
            fp = (predicted > actual).astype(int)
            fp = np.where(fp == 1, 1, 0)
            fp = fp.sum()

            # false negative
            fn = (predicted < actual).astype(int)
            fn = np.where(fn == 1, 1, 0)
            fn = fn.sum()

            total_cost = (
                (tp_cost * tp) + (tn_cost * tn) + (fp_cost * fp) + (fn_cost * fn)
            )

            return total_cost

        """
        internal function to calculate loss ends here
        """

        grid = np.arange(0, 1, 0.0001)

        # loop starts here

        cost = []
        # global optimize_results

        self.logger.info("Iteration starts at 0")

        for i in grid:

            pred_prob = (predicted >= i).astype(int)
            cost.append(calculate_loss(actual, pred_prob))

        optimize_results = pd.DataFrame(
            {"Probability Threshold": grid, "Cost Function": cost}
        )
        fig = px.line(
            optimize_results,
            x="Probability Threshold",
            y="Cost Function",
            line_shape="linear",
        )
        fig.update_layout(plot_bgcolor="rgb(245,245,245)")
        title = f"{model_name} Probability Threshold Optimization"

        # calculate vertical line
        y0 = optimize_results["Cost Function"].min()
        y1 = optimize_results["Cost Function"].max()
        x0 = optimize_results.sort_values(by="Cost Function", ascending=False).iloc[0][
            0
        ]
        x1 = x0

        t = x0
        if self.html_param:

            fig.add_shape(
                dict(
                    type="line",
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    line=dict(color="red", width=2),
                )
            )
            fig.update_layout(
                title={
                    "text": title,
                    "y": 0.95,
                    "x": 0.45,
                    "xanchor": "center",
                    "yanchor": "top",
                }
            )
            self.logger.info("Figure ready for render")
            fig.show()
        print(f"Optimized Probability Threshold: {t} | Optimized Cost Function: {y1}")
        self.logger.info(
            "optimize_threshold() succesfully completed......................................"
        )

        return float(t)

    def predict_model(
        self,
        estimator,
        data: Optional[pd.DataFrame] = None,
        probability_threshold: Optional[float] = None,
        encoded_labels: bool = False,
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
            round=round,
            verbose=verbose,
            ml_usecase=MLUsecase.CLASSIFICATION,
        )

    def finalize_model(
        self,
        estimator,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        model_only: bool = True,
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
        )

    def deploy_model(
        self, model, model_name: str, authentication: dict, platform: str = "aws",
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
        >>> deploy_model(model = lr, model_name = 'lr-for-deployment', platform = 'aws', authentication = {'bucket' : 'S3-bucket-name'})
            

        Amazon Web Service (AWS) users:
            To deploy a model on AWS S3 ('aws'), environment variables must be set in your
            local environment. To configure AWS environment variables, type ``aws configure`` 
            in the command line. Following information from the IAM portal of amazon console 
            account is required:

            - AWS Access Key ID
            - AWS Secret Key Access
            - Default Region Name (can be seen under Global settings on your AWS console)

            More info: https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-envvars.html


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

            More info: https://docs.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python?toc=%2Fpython%2Fazure%2FTOC.json


        model: scikit-learn compatible object
            Trained model object
        

        model_name: str
            Name of model.
        

        authentication: dict
            Dictionary of applicable authentication tokens.

            When platform = 'aws':
            {'bucket' : 'S3-bucket-name'}

            When platform = 'gcp':
            {'project': 'gcp-project-name', 'bucket' : 'gcp-bucket-name'}

            When platform = 'azure':
            {'container': 'azure-container-name'}
        

        platform: str, default = 'aws'
            Name of the cloud platform. Currently supported platforms: 'aws', 'gcp' and 'azure'.
        

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
        self, model, model_name: str, model_only: bool = False, verbose: bool = True
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
            model=model, model_name=model_name, model_only=model_only, verbose=verbose
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

    def automl(self, optimize: str = "Accuracy", use_holdout: bool = False) -> Any:

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
        

        Returns:
            Trained Model

        """
        return super().automl(optimize=optimize, use_holdout=use_holdout)

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
            reset=reset, include_custom=include_custom, raise_errors=raise_errors,
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


class AnomalyExperiment(_UnsupervisedExperiment):
    def __init__(self) -> None:
        super().__init__()
        self._ml_usecase = MLUsecase.ANOMALY
        self.exp_name_log = "anomaly-default-name"
        return

    def _get_models(self, raise_errors: bool = True) -> Tuple[dict, dict]:
        all_models = {
            k: v
            for k, v in pycaret.containers.models.anomaly.get_all_model_containers(
                self.variables, raise_errors=raise_errors
            ).items()
            if not v.is_special
        }
        all_models_internal = pycaret.containers.models.anomaly.get_all_model_containers(
            self.variables, raise_errors=raise_errors
        )
        return all_models, all_models_internal

    def _get_metrics(self, raise_errors: bool = True) -> dict:
        return pycaret.containers.metrics.anomaly.get_all_metric_containers(
            self.variables, raise_errors=raise_errors
        )


class ClusteringExperiment(_UnsupervisedExperiment):
    def __init__(self) -> None:
        super().__init__()
        self._ml_usecase = MLUsecase.CLUSTERING
        self.exp_name_log = "cluster-default-name"
        return

    def _get_models(self, raise_errors: bool = True) -> Tuple[dict, dict]:
        all_models = {
            k: v
            for k, v in pycaret.containers.models.clustering.get_all_model_containers(
                self.variables, raise_errors=raise_errors
            ).items()
            if not v.is_special
        }
        all_models_internal = pycaret.containers.models.clustering.get_all_model_containers(
            self.variables, raise_errors=raise_errors
        )
        return all_models, all_models_internal

    def _get_metrics(self, raise_errors: bool = True) -> dict:
        return pycaret.containers.metrics.clustering.get_all_metric_containers(
            self.variables, raise_errors=raise_errors
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


def experiment_factory(usecase: MLUsecase):
    switch = {
        MLUsecase.CLASSIFICATION: ClassificationExperiment,
        MLUsecase.REGRESSION: RegressionExperiment,
        MLUsecase.CLUSTERING: ClusteringExperiment,
        MLUsecase.ANOMALY: AnomalyExperiment
    }
    return switch[usecase]()