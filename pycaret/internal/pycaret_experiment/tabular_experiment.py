import gc
import logging
import os
import random
import secrets
import traceback
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import patch

import numpy as np  # type: ignore
import pandas as pd
import plotly.express as px  # type: ignore
import scikitplot as skplt  # type: ignore
from IPython.display import display as ipython_display
from joblib.memory import Memory
from packaging import version
from pandas.io.formats.style import Styler
from sklearn.model_selection import BaseCrossValidator  # type: ignore
from sklearn.pipeline import Pipeline

import pycaret.internal.patches.sklearn
import pycaret.internal.patches.yellowbrick
import pycaret.internal.persistence
import pycaret.internal.preprocess
import pycaret.loggers
from pycaret.internal.display import CommonDisplay
from pycaret.internal.logging import create_logger, get_logger, redirect_output
from pycaret.internal.pipeline import Pipeline as InternalPipeline
from pycaret.internal.pipeline import get_memory
from pycaret.internal.plots.helper import MatplotlibDefaultDPI
from pycaret.internal.plots.yellowbrick import show_yellowbrick_plot
from pycaret.internal.pycaret_experiment.pycaret_experiment import _PyCaretExperiment
from pycaret.internal.validation import is_sklearn_cv_generator
from pycaret.loggers.base_logger import BaseLogger
from pycaret.loggers.dagshub_logger import DagshubLogger
from pycaret.loggers.mlflow_logger import MlflowLogger
from pycaret.loggers.wandb_logger import WandbLogger
from pycaret.utils._dependencies import _check_soft_dependencies
from pycaret.utils.generic import (
    MLUsecase,
    get_allowed_engines,
    get_label_encoder,
    get_model_name,
)

LOGGER = get_logger()


class _TabularExperiment(_PyCaretExperiment):
    def __init__(self) -> None:
        super().__init__()
        self.all_allowed_engines = None
        self.fold_shuffle_param = False
        self.fold_groups_param = None
        self.exp_model_engines = {}
        self._variable_keys = self._variable_keys.union(
            {
                "_ml_usecase",
                "_available_plots",
                "USI",
                "html_param",
                "seed",
                "pipeline",
                "n_jobs_param",
                "gpu_n_jobs_param",
                "exp_name_log",
                "exp_id",
                "logging_param",
                "log_plots_param",
                "data",
                "idx",
                "gpu_param",
                "memory",
            }
        )
        return

    def _pack_for_remote(self) -> dict:
        pack = super()._pack_for_remote()
        for k in ["_all_metrics", "seed"]:
            if hasattr(self, k):
                pack[k] = getattr(self, k)
        return pack

    def _get_setup_display(self, **kwargs) -> Styler:
        return pd.DataFrame().style

    def _get_default_plots_to_log(self) -> List[str]:
        return []

    def _get_groups(
        self,
        groups,
        data: Optional[pd.DataFrame] = None,
        fold_groups=None,
    ):
        import pycaret.utils.generic

        data = data if data is not None else self.X_train
        fold_groups = fold_groups if fold_groups is not None else self.fold_groups_param
        return pycaret.utils.generic.get_groups(groups, data, fold_groups)

    def _get_cv_splitter(
        self, fold, ml_usecase: Optional[MLUsecase] = None
    ) -> BaseCrossValidator:
        """Returns the cross validator object used to perform cross validation"""
        if not ml_usecase:
            ml_usecase = self._ml_usecase

        import pycaret.utils.generic

        return pycaret.utils.generic.get_cv_splitter(
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

        return pycaret.utils.generic.get_model_id(e, models)

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
            models = getattr(self, "_all_models_internal", None)

        return get_model_name(e, models, deep=deep)

    def _log_model(
        self,
        model,
        model_results,
        score_dict: dict,
        source: str,
        runtime: float,
        model_fit_time: float,
        pipeline,
        log_holdout: bool = True,
        log_plots: Optional[List[str]] = None,
        tune_cv_results=None,
        URI=None,
        experiment_custom_tags=None,
        display: Optional[CommonDisplay] = None,
    ):
        log_plots = log_plots or []
        try:
            self.logging_param.log_model(
                experiment=self,
                model=model,
                model_results=model_results,
                pipeline=pipeline,
                score_dict=score_dict,
                source=source,
                runtime=runtime,
                model_fit_time=model_fit_time,
                log_plots=log_plots,
                experiment_custom_tags=experiment_custom_tags,
                log_holdout=log_holdout,
                tune_cv_results=tune_cv_results,
                URI=URI,
                display=display,
            )
        except Exception:
            self.logger.error(
                f"_log_model() for {model} raised an exception:\n"
                f"{traceback.format_exc()}"
            )

    def _profile(self, profile, profile_kwargs):
        """Create a profile report"""
        if profile:
            profile_kwargs = profile_kwargs or {}

            if self.verbose:
                print("Loading profile... Please Wait!")
            try:
                import pandas_profiling

                self.report = pandas_profiling.ProfileReport(
                    self.data, **profile_kwargs
                )
            except Exception as ex:
                print("Profiler Failed. No output to show, continue with modeling.")
                self.logger.error(
                    f"Data Failed with exception:\n {ex}\n"
                    "No output to show, continue with modeling."
                )

        return self

    def _validate_log_experiment(self, obj: Any) -> None:
        return isinstance(obj, (bool, BaseLogger)) or (
            isinstance(obj, str) and obj.lower() in ["mlflow", "wandb", "dagshub"]
        )

    def _convert_log_experiment(
        self, log_experiment: Any
    ) -> Union[bool, pycaret.loggers.DashboardLogger]:
        if not (
            (
                isinstance(log_experiment, list)
                and all(self._validate_log_experiment(x) for x in log_experiment)
            )
            or self._validate_log_experiment(log_experiment)
        ):
            raise TypeError(
                "log_experiment parameter must be a bool, BaseLogger, one of 'mlflow', 'wandb', 'dagshub'; or a list of the former."
            )

        def convert_logging_param(obj):
            if isinstance(obj, BaseLogger):
                return obj
            obj = obj.lower()
            if obj == "mlflow":
                return MlflowLogger()
            if obj == "wandb":
                return WandbLogger()
            if obj == "dagshub":
                return DagshubLogger(os.getenv("MLFLOW_TRACKING_URI"))

        if log_experiment:
            if log_experiment is True:
                loggers_list = [MlflowLogger()]
            else:
                if not isinstance(log_experiment, list):
                    log_experiment = [log_experiment]
                loggers_list = [convert_logging_param(x) for x in log_experiment]

            if loggers_list:
                return pycaret.loggers.DashboardLogger(loggers_list)
        return False

    def _initialize_setup(
        self,
        n_jobs: Optional[int] = -1,
        use_gpu: bool = False,
        html: bool = True,
        session_id: Optional[int] = None,
        system_log: Union[bool, str, logging.Logger] = True,
        log_experiment: Union[
            bool, str, BaseLogger, List[Union[str, BaseLogger]]
        ] = False,
        experiment_name: Optional[str] = None,
        memory: Union[bool, str, Memory] = True,
        verbose: bool = True,
    ):
        """
        This function initializes the environment in pycaret. setup()
        must be called before executing any other function in pycaret.
        It takes only two mandatory parameters: data and name of the
        target column.

        """
        from pycaret.utils import __version__

        # Parameter attrs
        self.n_jobs_param = n_jobs
        self.gpu_param = use_gpu
        self.html_param = html
        self.logging_param = self._convert_log_experiment(log_experiment)
        self.memory = get_memory(memory)
        self.verbose = verbose

        # Global attrs
        self.USI = secrets.token_hex(nbytes=2)
        self.seed = random.randint(150, 9000) if session_id is None else session_id
        np.random.seed(self.seed)

        # Initialization =========================================== >>

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
        self.logger.info(f"self.USI: {self.USI}")

        self.logger.info(f"self._variable_keys: {self._variable_keys}")

        self._check_environment()

        # Set up GPU usage ========================================= >>

        if self.gpu_param != "force" and type(self.gpu_param) is not bool:
            raise TypeError(
                f"Invalid value for the use_gpu parameter, got {self.gpu_param}. "
                "Possible values are: 'force', True or False."
            )

        cuml_version = None
        if self.gpu_param:
            self.logger.info("Set up GPU usage.")

            if _check_soft_dependencies("cuml", extra=None, severity="warning"):
                from cuml import __version__

                cuml_version = __version__
                self.logger.info(f"cuml=={cuml_version}")

            if cuml_version is None or not version.parse(cuml_version) >= version.parse(
                "22.10"
            ):
                message = f"cuML is outdated or not found. Required version is >=22.10, got {__version__}"
                if use_gpu == "force":
                    raise ImportError(message)
                else:
                    self.logger.warning(message)

        return self

    @staticmethod
    def plot_model_check_display_format_(display_format: Optional[str]):
        """Checks if the display format is in the allowed list"""
        plot_formats = [None, "streamlit"]

        if display_format not in plot_formats:
            raise ValueError("display_format can only be None or 'streamlit'.")

    def _plot_model(
        self,
        estimator,
        plot: str = "auc",
        scale: float = 1,  # added in pycaret==2.1.0
        save: Union[str, bool] = False,
        fold: Optional[Union[int, Any]] = None,
        fit_kwargs: Optional[dict] = None,
        plot_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        feature_name: Optional[str] = None,
        label: bool = False,
        use_train_data: bool = False,
        verbose: bool = True,
        system: bool = True,
        display: Optional[CommonDisplay] = None,  # added in pycaret==2.2.0
        display_format: Optional[str] = None,
    ) -> str:

        """Internal version of ``plot_model`` with ``system`` arg."""
        self._check_setup_ran()

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
            _check_soft_dependencies("streamlit", extra=None, severity="error")
            import streamlit as st

        # multiclass plot exceptions:
        multiclass_not_available = ["calibration", "threshold", "manifold", "rfe"]
        if self.is_multiclass:
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
            display = CommonDisplay(verbose=verbose, html_param=self.html_param)

        plot_kwargs = plot_kwargs or {}

        self.logger.info("Preloading libraries")
        # pre-load libraries
        import matplotlib.pyplot as plt

        np.random.seed(self.seed)

        # defining estimator as model locally
        # deepcopy instead of clone so we have a fitted estimator
        if isinstance(estimator, InternalPipeline):
            estimator = estimator.steps[-1][1]
        estimator = deepcopy(estimator)
        model = estimator

        # plots used for logging (controlled through plots_log_param)
        # AUC, #Confusion Matrix and #Feature Importance

        self.logger.info("Copying training dataset")

        self.logger.info(f"Plot type: {plot}")
        plot_name = self._available_plots[plot]

        # yellowbrick workaround start

        # yellowbrick workaround end

        model_name = self._get_model_name(model)
        base_plot_filename = f"{plot_name}.png"
        with patch(
            "yellowbrick.utils.types.is_estimator",
            pycaret.internal.patches.yellowbrick.is_estimator,
        ):
            with patch(
                "yellowbrick.utils.helpers.is_estimator",
                pycaret.internal.patches.yellowbrick.is_estimator,
            ):
                _base_dpi = 100

                def pipeline():

                    from schemdraw import Drawing
                    from schemdraw.flow import Arrow, Data, RoundBox, Subroutine

                    # Create schematic drawing
                    d = Drawing(backend="matplotlib")
                    d.config(fontsize=plot_kwargs.get("fontsize", 14))
                    d += Subroutine(w=10, h=5, s=1).label("Raw data").drop("E")
                    for est in self.pipeline:
                        name = getattr(est, "transformer", est).__class__.__name__
                        d += Arrow().right()
                        d += RoundBox(w=max(len(name), 7), h=5, cornerradius=1).label(
                            name
                        )

                    # Add the model box
                    name = estimator.__class__.__name__
                    d += Arrow().right()
                    d += Data(w=max(len(name), 7), h=5).label(name)

                    display.clear_output()

                    with MatplotlibDefaultDPI(base_dpi=_base_dpi, scale_to_set=scale):
                        fig, ax = plt.subplots(
                            figsize=((2 + len(self.pipeline) * 5), 6)
                        )

                        d.draw(ax=ax, showframe=False, show=False)
                        ax.set_aspect("equal")
                        plt.axis("off")
                        plt.tight_layout()

                    if save:
                        if not isinstance(save, bool):
                            plot_filename = os.path.join(save, base_plot_filename)
                        else:
                            plot_filename = base_plot_filename
                        self.logger.info(f"Saving '{plot_filename}'")
                        plt.savefig(plot_filename, bbox_inches="tight")
                    elif system:
                        plt.show()
                    plt.close()

                    self.logger.info("Visual Rendered Successfully")

                def residuals_interactive():
                    from pycaret.internal.plots.residual_plots import (
                        InteractiveResidualsPlot,
                    )

                    resplots = InteractiveResidualsPlot(
                        x=self.X_train_transformed,
                        y=self.y_train_transformed,
                        x_test=self.X_test_transformed,
                        y_test=self.y_test_transformed,
                        model=estimator,
                    )

                    # display.clear_output()
                    if system:
                        resplots.show()

                    plot_filename = None
                    if save:
                        if not isinstance(save, bool):
                            plot_filename = os.path.join(save, base_plot_filename)
                        else:
                            plot_filename = base_plot_filename
                        self.logger.info(f"Saving '{plot_filename}'")
                        resplots.write_html(plot_filename)

                    self.logger.info("Visual Rendered Successfully")
                    return plot_filename

                def cluster():
                    self.logger.info(
                        "SubProcess assign_model() called =================================="
                    )
                    b = self.assign_model(  # type: ignore
                        estimator, verbose=False, transformation=True
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
                        pca_["Feature"] = self.data[self.data.columns[0]]

                    if label:
                        pca_["Label"] = pca_["Feature"]

                    """
                    sorting
                    """

                    self.logger.info("Sorting dataframe")

                    clus_num = [int(i.split()[1]) for i in pca_["Cluster"]]

                    pca_["cnum"] = clus_num
                    pca_.sort_values(by="cnum", inplace=True)

                    """
                    sorting ends
                    """

                    # display.clear_output()

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

                    plot_filename = None
                    if save:
                        if not isinstance(save, bool):
                            plot_filename = os.path.join(save, base_plot_filename)
                        else:
                            plot_filename = base_plot_filename
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

                    _check_soft_dependencies(
                        "umap",
                        extra="analysis",
                        severity="error",
                        install_name="umap-learn",
                    )
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
                        df["Feature"] = self.data[self.data.columns[0]]

                    # display.clear_output()

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

                    plot_filename = None
                    if save:
                        if not isinstance(save, bool):
                            plot_filename = os.path.join(save, base_plot_filename)
                        else:
                            plot_filename = base_plot_filename
                        self.logger.info(f"Saving '{plot_filename}'")
                        fig.write_html(plot_filename)

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

                    self.logger.info("Getting dummies to cast categorical variables")

                    from sklearn.manifold import TSNE

                    self.logger.info("Fitting TSNE()")
                    X_embedded = TSNE(n_components=3).fit_transform(b)

                    X = pd.DataFrame(X_embedded)
                    X["Anomaly"] = cluster
                    if feature_name is not None:
                        X["Feature"] = self.data[feature_name]
                    else:
                        X["Feature"] = self.data[self.data.columns[0]]

                    df = X

                    # display.clear_output()

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

                    plot_filename = None
                    if save:
                        if not isinstance(save, bool):
                            plot_filename = os.path.join(save, base_plot_filename)
                        else:
                            plot_filename = base_plot_filename
                        self.logger.info(f"Saving '{plot_filename}'")
                        fig.write_html(plot_filename)

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
                        estimator,
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
                        X_embedded["Feature"] = self.data[feature_name]
                    else:
                        X_embedded["Feature"] = self.data[self.data.columns[0]]

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

                    # display.clear_output()

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

                    plot_filename = None
                    if save:
                        if not isinstance(save, bool):
                            plot_filename = os.path.join(save, base_plot_filename)
                        else:
                            plot_filename = base_plot_filename
                        self.logger.info(f"Saving '{plot_filename}'")
                        fig.write_html(plot_filename)

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
                        estimator, verbose=False
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

                    # display.clear_output()

                    self.logger.info("Rendering Visual")

                    fig = px.histogram(
                        d,
                        x=x_col,
                        color="Cluster",
                        marginal="box",
                        opacity=0.7,
                        hover_data=d.columns,
                    )

                    fig.update_layout(
                        height=600 * scale,
                    )

                    plot_filename = None
                    if save:
                        if not isinstance(save, bool):
                            plot_filename = os.path.join(save, base_plot_filename)
                        else:
                            plot_filename = base_plot_filename
                        self.logger.info(f"Saving '{plot_filename}'")
                        fig.write_html(plot_filename)

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
                            estimator, timings=False, **plot_kwargs
                        )
                        return show_yellowbrick_plot(
                            visualizer=visualizer,
                            X_train=self.X_train_transformed,
                            y_train=None,
                            X_test=None,
                            y_test=None,
                            name=plot_name,
                            handle_test="",
                            scale=scale,
                            save=save,
                            fit_kwargs=fit_kwargs,
                            display_format=display_format,
                        )

                    except Exception:
                        self.logger.error("Elbow plot failed. Exception:")
                        self.logger.error(traceback.format_exc())
                        raise TypeError("Plot Type not supported for this model.")

                def silhouette():
                    from yellowbrick.cluster import SilhouetteVisualizer

                    try:
                        visualizer = SilhouetteVisualizer(
                            estimator, colors="yellowbrick", **plot_kwargs
                        )
                        return show_yellowbrick_plot(
                            visualizer=visualizer,
                            X_train=self.X_train_transformed,
                            y_train=None,
                            X_test=None,
                            y_test=None,
                            name=plot_name,
                            handle_test="",
                            scale=scale,
                            save=save,
                            fit_kwargs=fit_kwargs,
                            display_format=display_format,
                        )
                    except Exception:
                        self.logger.error("Silhouette plot failed. Exception:")
                        self.logger.error(traceback.format_exc())
                        raise TypeError("Plot Type not supported for this model.")

                def distance():
                    from yellowbrick.cluster import InterclusterDistance

                    try:
                        visualizer = InterclusterDistance(estimator, **plot_kwargs)
                        return show_yellowbrick_plot(
                            visualizer=visualizer,
                            X_train=self.X_train_transformed,
                            y_train=None,
                            X_test=None,
                            y_test=None,
                            name=plot_name,
                            handle_test="",
                            scale=scale,
                            save=save,
                            fit_kwargs=fit_kwargs,
                            display_format=display_format,
                        )
                    except Exception:
                        self.logger.error("Distance plot failed. Exception:")
                        self.logger.error(traceback.format_exc())
                        raise TypeError("Plot Type not supported for this model.")

                def residuals():

                    from yellowbrick.regressor import ResidualsPlot

                    visualizer = ResidualsPlot(estimator, **plot_kwargs)
                    return show_yellowbrick_plot(
                        visualizer=visualizer,
                        X_train=self.X_train_transformed,
                        y_train=self.y_train_transformed,
                        X_test=self.X_test_transformed,
                        y_test=self.y_test_transformed,
                        name=plot_name,
                        scale=scale,
                        save=save,
                        fit_kwargs=fit_kwargs,
                        display_format=display_format,
                    )

                def auc():

                    from yellowbrick.classifier import ROCAUC

                    visualizer = ROCAUC(estimator, **plot_kwargs)
                    return show_yellowbrick_plot(
                        visualizer=visualizer,
                        X_train=self.X_train_transformed,
                        y_train=self.y_train_transformed,
                        X_test=self.X_test_transformed,
                        y_test=self.y_test_transformed,
                        name=plot_name,
                        scale=scale,
                        save=save,
                        fit_kwargs=fit_kwargs,
                        display_format=display_format,
                    )

                def threshold():

                    from yellowbrick.classifier import DiscriminationThreshold

                    visualizer = DiscriminationThreshold(
                        estimator, random_state=self.seed, **plot_kwargs
                    )
                    return show_yellowbrick_plot(
                        visualizer=visualizer,
                        X_train=self.X_train_transformed,
                        y_train=self.y_train_transformed,
                        X_test=self.X_test_transformed,
                        y_test=self.y_test_transformed,
                        name=plot_name,
                        scale=scale,
                        save=save,
                        fit_kwargs=fit_kwargs,
                        display_format=display_format,
                    )

                def pr():

                    from yellowbrick.classifier import PrecisionRecallCurve

                    visualizer = PrecisionRecallCurve(
                        estimator, random_state=self.seed, **plot_kwargs
                    )
                    return show_yellowbrick_plot(
                        visualizer=visualizer,
                        X_train=self.X_train_transformed,
                        y_train=self.y_train_transformed,
                        X_test=self.X_test_transformed,
                        y_test=self.y_test_transformed,
                        name=plot_name,
                        scale=scale,
                        save=save,
                        fit_kwargs=fit_kwargs,
                        display_format=display_format,
                    )

                def confusion_matrix():

                    from yellowbrick.classifier import ConfusionMatrix

                    plot_kwargs.setdefault("fontsize", 15)
                    plot_kwargs.setdefault("cmap", "Greens")

                    visualizer = ConfusionMatrix(
                        estimator, random_state=self.seed, **plot_kwargs
                    )
                    return show_yellowbrick_plot(
                        visualizer=visualizer,
                        X_train=self.X_train_transformed,
                        y_train=self.y_train_transformed,
                        X_test=self.X_test_transformed,
                        y_test=self.y_test_transformed,
                        name=plot_name,
                        scale=scale,
                        save=save,
                        fit_kwargs=fit_kwargs,
                        display_format=display_format,
                    )

                def error():

                    if self._ml_usecase == MLUsecase.CLASSIFICATION:
                        from yellowbrick.classifier import ClassPredictionError

                        visualizer = ClassPredictionError(
                            estimator, random_state=self.seed, **plot_kwargs
                        )

                    elif self._ml_usecase == MLUsecase.REGRESSION:
                        from yellowbrick.regressor import PredictionError

                        visualizer = PredictionError(
                            estimator, random_state=self.seed, **plot_kwargs
                        )

                    return show_yellowbrick_plot(
                        visualizer=visualizer,  # type: ignore
                        X_train=self.X_train_transformed,
                        y_train=self.y_train_transformed,
                        X_test=self.X_test_transformed,
                        y_test=self.y_test_transformed,
                        name=plot_name,
                        scale=scale,
                        save=save,
                        fit_kwargs=fit_kwargs,
                        display_format=display_format,
                    )

                def cooks():

                    from yellowbrick.regressor import CooksDistance

                    visualizer = CooksDistance()
                    return show_yellowbrick_plot(
                        visualizer=visualizer,
                        X_train=self.X_train_transformed,
                        y_train=self.y_train_transformed,
                        X_test=self.X_test_transformed,
                        y_test=self.y_test_transformed,
                        name=plot_name,
                        scale=scale,
                        save=save,
                        fit_kwargs=fit_kwargs,
                        handle_test="",
                        display_format=display_format,
                    )

                def class_report():

                    from yellowbrick.classifier import ClassificationReport

                    visualizer = ClassificationReport(
                        estimator, random_state=self.seed, support=True, **plot_kwargs
                    )
                    return show_yellowbrick_plot(
                        visualizer=visualizer,
                        X_train=self.X_train_transformed,
                        y_train=self.y_train_transformed,
                        X_test=self.X_test_transformed,
                        y_test=self.y_test_transformed,
                        name=plot_name,
                        scale=scale,
                        save=save,
                        fit_kwargs=fit_kwargs,
                        display_format=display_format,
                    )

                def boundary():

                    from sklearn.decomposition import PCA
                    from sklearn.preprocessing import StandardScaler
                    from yellowbrick.contrib.classifier import DecisionViz

                    data_X_transformed = self.X_train_transformed.select_dtypes(
                        include="number"
                    )
                    test_X_transformed = self.X_test_transformed.select_dtypes(
                        include="number"
                    )
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

                    viz_ = DecisionViz(estimator, **plot_kwargs)
                    return show_yellowbrick_plot(
                        visualizer=viz_,
                        X_train=data_X_transformed,
                        y_train=np.array(self.y_train_transformed),
                        X_test=test_X_transformed,
                        y_test=np.array(self.y_test_transformed),
                        name=plot_name,
                        scale=scale,
                        handle_test="draw",
                        save=save,
                        fit_kwargs=fit_kwargs,
                        features=["Feature One", "Feature Two"],
                        classes=["A", "B"],
                        display_format=display_format,
                    )

                def rfe():

                    from yellowbrick.model_selection import RFECV

                    visualizer = RFECV(estimator, cv=cv, groups=groups, **plot_kwargs)
                    return show_yellowbrick_plot(
                        visualizer=visualizer,
                        X_train=self.X_train_transformed,
                        y_train=self.y_train_transformed,
                        X_test=self.X_test_transformed,
                        y_test=self.y_test_transformed,
                        handle_test="",
                        name=plot_name,
                        scale=scale,
                        save=save,
                        fit_kwargs=fit_kwargs,
                        display_format=display_format,
                    )

                def learning():

                    from yellowbrick.model_selection import LearningCurve

                    sizes = np.linspace(0.3, 1.0, 10)
                    visualizer = LearningCurve(
                        estimator,
                        cv=cv,
                        train_sizes=sizes,
                        groups=groups,
                        n_jobs=self.gpu_n_jobs_param,
                        random_state=self.seed,
                    )
                    return show_yellowbrick_plot(
                        visualizer=visualizer,
                        X_train=self.X_train_transformed,
                        y_train=self.y_train_transformed,
                        X_test=self.X_test_transformed,
                        y_test=self.y_test_transformed,
                        handle_test="",
                        name=plot_name,
                        scale=scale,
                        save=save,
                        fit_kwargs=fit_kwargs,
                        display_format=display_format,
                    )

                def lift():

                    self.logger.info("Generating predictions / predict_proba on X_test")
                    y_test__ = self.y_test_transformed
                    predict_proba__ = estimator.predict_proba(self.X_test_transformed)
                    # display.clear_output()
                    with MatplotlibDefaultDPI(base_dpi=_base_dpi, scale_to_set=scale):
                        skplt.metrics.plot_lift_curve(
                            y_test__, predict_proba__, figsize=(10, 6)
                        )
                        plot_filename = None
                        if save:
                            if not isinstance(save, bool):
                                plot_filename = os.path.join(save, base_plot_filename)
                            else:
                                plot_filename = base_plot_filename
                            self.logger.info(f"Saving '{plot_filename}'")
                            plt.savefig(plot_filename, bbox_inches="tight")
                        elif system:
                            plt.show()
                        plt.close()

                    self.logger.info("Visual Rendered Successfully")
                    return plot_filename

                def gain():

                    self.logger.info("Generating predictions / predict_proba on X_test")
                    y_test__ = self.y_test_transformed
                    predict_proba__ = estimator.predict_proba(self.X_test_transformed)
                    # display.clear_output()
                    with MatplotlibDefaultDPI(base_dpi=_base_dpi, scale_to_set=scale):
                        skplt.metrics.plot_cumulative_gain(
                            y_test__, predict_proba__, figsize=(10, 6)
                        )
                        plot_filename = None
                        if save:
                            if not isinstance(save, bool):
                                plot_filename = os.path.join(save, base_plot_filename)
                            else:
                                plot_filename = base_plot_filename
                            self.logger.info(f"Saving '{plot_filename}'")
                            plt.savefig(plot_filename, bbox_inches="tight")
                        elif system:
                            plt.show()
                        plt.close()

                    self.logger.info("Visual Rendered Successfully")
                    return plot_filename

                def manifold():

                    from yellowbrick.features import Manifold

                    data_X_transformed = self.X_train_transformed.select_dtypes(
                        include="number"
                    )
                    visualizer = Manifold(
                        manifold="tsne", random_state=self.seed, **plot_kwargs
                    )
                    return show_yellowbrick_plot(
                        visualizer=visualizer,
                        X_train=data_X_transformed,
                        y_train=self.y_train_transformed,
                        X_test=self.X_test_transformed,
                        y_test=self.y_test_transformed,
                        handle_train="fit_transform",
                        handle_test="",
                        name=plot_name,
                        scale=scale,
                        save=save,
                        fit_kwargs=fit_kwargs,
                        display_format=display_format,
                    )

                def tree():

                    from sklearn.tree import plot_tree

                    is_stacked_model = False
                    is_ensemble_of_forests = False

                    if isinstance(estimator, Pipeline):
                        fitted_estimator = estimator._final_estimator
                    else:
                        fitted_estimator = estimator

                    if "final_estimator" in fitted_estimator.get_params():
                        tree_estimator = fitted_estimator.final_estimator
                        is_stacked_model = True
                    else:
                        tree_estimator = fitted_estimator

                    if (
                        "base_estimator" in tree_estimator.get_params()
                        and "n_estimators" in tree_estimator.base_estimator.get_params()
                    ):
                        n_estimators = (
                            tree_estimator.get_params()["n_estimators"]
                            * tree_estimator.base_estimator.get_params()["n_estimators"]
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

                    self.logger.info("Plotting decision trees")
                    trees = []
                    feature_names = list(self.X_train_transformed.columns)
                    if self._ml_usecase == MLUsecase.CLASSIFICATION:
                        class_names = {
                            i: class_name
                            for i, class_name in enumerate(
                                get_label_encoder(self.pipeline).classes_
                            )
                        }
                    else:
                        class_names = None
                    fitted_estimator = tree_estimator
                    if is_stacked_model:
                        stacked_feature_names = []
                        if self._ml_usecase == MLUsecase.CLASSIFICATION:
                            classes = list(self.y_train_transformed.unique())
                            if len(classes) == 2:
                                classes.pop()
                            for c in classes:
                                stacked_feature_names.extend(
                                    [
                                        f"{k}_{class_names[c]}"
                                        for k, v in fitted_estimator.estimators
                                    ]
                                )
                        else:
                            stacked_feature_names.extend(
                                [f"{k}" for k, v in fitted_estimator.estimators]
                            )
                        if not fitted_estimator.passthrough:
                            feature_names = stacked_feature_names
                        else:
                            feature_names = stacked_feature_names + feature_names
                        fitted_estimator = fitted_estimator.final_estimator_
                    if is_ensemble_of_forests:
                        for tree_estimator in fitted_estimator.estimators_:
                            trees.extend(tree_estimator.estimators_)
                    else:
                        try:
                            trees = fitted_estimator.estimators_
                        except Exception:
                            trees = [fitted_estimator]
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

                    # display.clear_output()
                    plot_filename = None
                    if save:
                        if not isinstance(save, bool):
                            plot_filename = os.path.join(save, base_plot_filename)
                        else:
                            plot_filename = base_plot_filename
                        self.logger.info(f"Saving '{plot_filename}'")
                        plt.savefig(plot_filename, bbox_inches="tight")
                    elif system:
                        plt.show()
                    plt.close()

                    self.logger.info("Visual Rendered Successfully")
                    return plot_filename

                def calibration():

                    from sklearn.calibration import calibration_curve

                    plt.figure(figsize=(7, 6), dpi=_base_dpi * scale)
                    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)

                    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
                    self.logger.info("Scoring test/hold-out set")
                    prob_pos = estimator.predict_proba(self.X_test_transformed)[:, 1]
                    prob_pos = (prob_pos - prob_pos.min()) / (
                        prob_pos.max() - prob_pos.min()
                    )
                    (
                        fraction_of_positives,
                        mean_predicted_value,
                    ) = calibration_curve(self.y_test_transformed, prob_pos, n_bins=10)
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
                    # display.clear_output()
                    plot_filename = None
                    if save:
                        if not isinstance(save, bool):
                            plot_filename = os.path.join(save, base_plot_filename)
                        else:
                            plot_filename = base_plot_filename
                        self.logger.info(f"Saving '{plot_filename}'")
                        plt.savefig(plot_filename, bbox_inches="tight")
                    elif system:
                        plt.show()
                    plt.close()

                    self.logger.info("Visual Rendered Successfully")
                    return plot_filename

                def vc():

                    self.logger.info("Determining param_name")

                    try:
                        try:
                            # catboost special case
                            model_params = estimator.get_all_params()
                        except Exception:
                            model_params = estimator.get_params()
                    except Exception:
                        # display.clear_output()
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
                            param_name = "depth"
                            param_range = np.arange(1, 8 if self.gpu_param else 11)

                        # SGD Classifier
                        elif "l1_ratio" in model_params:
                            param_name = "l1_ratio"
                            param_range = np.arange(0, 1, 0.01)

                        # tree based models
                        elif "max_depth" in model_params:
                            param_name = "max_depth"
                            param_range = np.arange(1, 11)

                        # knn
                        elif "n_neighbors" in model_params:
                            param_name = "n_neighbors"
                            param_range = np.arange(1, 11)

                        # MLP / Ridge
                        elif "alpha" in model_params:
                            param_name = "alpha"
                            param_range = np.arange(0, 1, 0.1)

                        # Logistic Regression
                        elif "C" in model_params:
                            param_name = "C"
                            param_range = np.arange(1, 11)

                        # Bagging / Boosting
                        elif "n_estimators" in model_params:
                            param_name = "n_estimators"
                            param_range = np.arange(1, 1000, 10)

                        # Naive Bayes
                        elif "var_smoothing" in model_params:
                            param_name = "var_smoothing"
                            param_range = np.arange(0.1, 1, 0.01)

                        # QDA
                        elif "reg_param" in model_params:
                            param_name = "reg_param"
                            param_range = np.arange(0, 1, 0.1)

                        # GPC
                        elif "max_iter_predict" in model_params:
                            param_name = "max_iter_predict"
                            param_range = np.arange(100, 1000, 100)

                        else:
                            # display.clear_output()
                            raise TypeError(
                                "Plot not supported for this estimator. Try different estimator."
                            )

                    elif self._ml_usecase == MLUsecase.REGRESSION:

                        # Catboost
                        if "depth" in model_params:
                            param_name = "depth"
                            param_range = np.arange(1, 8 if self.gpu_param else 11)

                        # lasso/ridge/en/llar/huber/kr/mlp/br/ard
                        elif "alpha" in model_params:
                            param_name = "alpha"
                            param_range = np.arange(0, 1, 0.1)

                        elif "alpha_1" in model_params:
                            param_name = "alpha_1"
                            param_range = np.arange(0, 1, 0.1)

                        # par/svm
                        elif "C" in model_params:
                            param_name = "C"
                            param_range = np.arange(1, 11)

                        # tree based models (dt/rf/et)
                        elif "max_depth" in model_params:
                            param_name = "max_depth"
                            param_range = np.arange(1, 11)

                        # knn
                        elif "n_neighbors" in model_params:
                            param_name = "n_neighbors"
                            param_range = np.arange(1, 11)

                        # Bagging / Boosting (ada/gbr)
                        elif "n_estimators" in model_params:
                            param_name = "n_estimators"
                            param_range = np.arange(1, 1000, 10)

                        # Bagging / Boosting (ada/gbr)
                        elif "n_nonzero_coefs" in model_params:
                            param_name = "n_nonzero_coefs"
                            if len(self.X_train_transformed.columns) >= 10:
                                param_max = 11
                            else:
                                param_max = len(self.X_train_transformed.columns) + 1
                            param_range = np.arange(1, param_max, 1)

                        elif "eps" in model_params:
                            param_name = "eps"
                            param_range = np.arange(0, 1, 0.1)

                        elif "max_subpopulation" in model_params:
                            param_name = "max_subpopulation"
                            param_range = np.arange(1000, 100000, 2000)

                        elif "min_samples" in model_params:
                            param_name = "min_samples"
                            param_range = np.arange(0.01, 1, 0.1)

                        else:
                            # display.clear_output()
                            raise TypeError(
                                "Plot not supported for this estimator. Try different estimator."
                            )

                    self.logger.info(f"param_name: {param_name}")

                    from yellowbrick.model_selection import ValidationCurve

                    viz = ValidationCurve(
                        estimator,
                        param_name=param_name,
                        param_range=param_range,
                        cv=cv,
                        groups=groups,
                        random_state=self.seed,
                        n_jobs=self.gpu_n_jobs_param,
                    )
                    return show_yellowbrick_plot(
                        visualizer=viz,
                        X_train=self.X_train_transformed,
                        y_train=self.y_train_transformed,
                        X_test=self.X_test_transformed,
                        y_test=self.y_test_transformed,
                        handle_train="fit",
                        handle_test="",
                        name=plot_name,
                        scale=scale,
                        save=save,
                        fit_kwargs=fit_kwargs,
                        display_format=display_format,
                    )

                def dimension():

                    from sklearn.decomposition import PCA
                    from sklearn.preprocessing import StandardScaler
                    from yellowbrick.features import RadViz

                    data_X_transformed = self.X_train_transformed.select_dtypes(
                        include="number"
                    )
                    self.logger.info("Fitting StandardScaler()")
                    data_X_transformed = StandardScaler().fit_transform(
                        data_X_transformed
                    )

                    features = min(
                        round(len(self.X_train_transformed.columns) * 0.3, 0), 5
                    )
                    features = int(features)

                    pca = PCA(n_components=features, random_state=self.seed)
                    self.logger.info("Fitting PCA()")
                    data_X_transformed = pca.fit_transform(data_X_transformed)
                    classes = self.y_train_transformed.unique().tolist()
                    visualizer = RadViz(classes=classes, alpha=0.25, **plot_kwargs)

                    return show_yellowbrick_plot(
                        visualizer=visualizer,
                        X_train=data_X_transformed,
                        y_train=np.array(self.y_train_transformed),
                        X_test=self.X_test_transformed,
                        y_test=self.y_test_transformed,
                        handle_train="fit_transform",
                        handle_test="",
                        name=plot_name,
                        scale=scale,
                        save=save,
                        fit_kwargs=fit_kwargs,
                        display_format=display_format,
                    )

                def feature():
                    return _feature(10)

                def feature_all():
                    return _feature(len(self.X_train_transformed.columns))

                def _feature(n: int):
                    variables = None
                    temp_model = estimator
                    if hasattr(estimator, "steps"):
                        temp_model = estimator.steps[-1][1]
                    if hasattr(temp_model, "coef_"):
                        try:
                            coef = temp_model.coef_.flatten()
                            if len(coef) > len(self.X_train_transformed.columns):
                                coef = coef[: len(self.X_train_transformed.columns)]
                            variables = abs(coef)
                        except Exception:
                            pass
                    if variables is None:
                        self.logger.warning(
                            "No coef_ found. Trying feature_importances_"
                        )
                        variables = abs(temp_model.feature_importances_)
                    coef_df = pd.DataFrame(
                        {
                            "Variable": self.X_train_transformed.columns,
                            "Value": variables,
                        }
                    )
                    sorted_df = (
                        coef_df.sort_values(by="Value", ascending=False)
                        .head(n)
                        .sort_values(by="Value")
                    )
                    my_range = range(1, len(sorted_df.index) + 1)
                    plt.figure(figsize=(8, 5 * (n // 10)), dpi=_base_dpi * scale)
                    plt.hlines(
                        y=my_range,
                        xmin=0,
                        xmax=sorted_df["Value"],
                        color="skyblue",
                    )
                    plt.plot(sorted_df["Value"], my_range, "o")
                    plt.yticks(my_range, sorted_df["Variable"])
                    plt.title("Feature Importance Plot")
                    plt.xlabel("Variable Importance")
                    plt.ylabel("Features")
                    # display.clear_output()
                    plot_filename = None
                    if save:
                        if not isinstance(save, bool):
                            plot_filename = os.path.join(save, base_plot_filename)
                        else:
                            plot_filename = base_plot_filename
                        self.logger.info(f"Saving '{plot_filename}'")
                        plt.savefig(plot_filename, bbox_inches="tight")
                    elif system:
                        plt.show()
                    plt.close()

                    self.logger.info("Visual Rendered Successfully")
                    return plot_filename

                def parameter():

                    try:
                        params = estimator.get_all_params()
                    except Exception:
                        params = estimator.get_params(deep=False)

                    param_df = pd.DataFrame.from_dict(
                        {str(k): str(v) for k, v in params.items()},
                        orient="index",
                        columns=["Parameters"],
                    )
                    # use ipython directly to show it in the widget
                    ipython_display(param_df)
                    self.logger.info("Visual Rendered Successfully")

                def ks():

                    self.logger.info("Generating predictions / predict_proba on X_test")
                    predict_proba__ = estimator.predict_proba(self.X_train_transformed)
                    # display.clear_output()
                    with MatplotlibDefaultDPI(base_dpi=_base_dpi, scale_to_set=scale):
                        skplt.metrics.plot_ks_statistic(
                            self.y_train_transformed, predict_proba__, figsize=(10, 6)
                        )
                        plot_filename = None
                        if save:
                            if not isinstance(save, bool):
                                plot_filename = os.path.join(save, base_plot_filename)
                            else:
                                plot_filename = base_plot_filename
                            self.logger.info(f"Saving '{plot_filename}'")
                            plt.savefig(plot_filename, bbox_inches="tight")
                        elif system:
                            plt.show()
                        plt.close()

                    self.logger.info("Visual Rendered Successfully")
                    return plot_filename

                # execute the plot method
                with redirect_output(self.logger):
                    ret = locals()[plot]()
                if ret:
                    plot_filename = ret
                else:
                    plot_filename = base_plot_filename

                try:
                    plt.close()
                except Exception:
                    pass

        gc.collect()

        self.logger.info(
            "plot_model() successfully completed......................................"
        )

        if save:
            return plot_filename

    def plot_model(
        self,
        estimator,
        plot: str = "auc",
        scale: float = 1,  # added in pycaret==2.1.0
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

            * 'pipeline' - Schematic drawing of the preprocessing pipeline
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
        return self._plot_model(
            estimator=estimator,
            plot=plot,
            scale=scale,
            save=save,
            fold=fold,
            fit_kwargs=fit_kwargs,
            plot_kwargs=plot_kwargs,
            groups=groups,
            feature_name=feature_name,
            label=label,
            use_train_data=use_train_data,
            verbose=verbose,
            display_format=display_format,
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
            self._plot_model,
            estimator=fixed(estimator),
            plot=a,
            save=fixed(False),
            verbose=fixed(False),
            scale=fixed(1),
            fold=fixed(fold),
            fit_kwargs=fixed(fit_kwargs),
            plot_kwargs=fixed(plot_kwargs),
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
        self._check_setup_ran()

        return pycaret.internal.persistence.deploy_model(
            model, model_name, authentication, platform, self.pipeline
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
        if self._ml_usecase == MLUsecase.TIME_SERIES:
            pipeline_to_use = self._get_pipeline_to_use(estimator=model)
        else:
            pipeline_to_use = self.pipeline

        model_, model_filename = pycaret.internal.persistence.save_model(
            model=model,
            model_name=model_name,
            prep_pipe_=None if model_only else pipeline_to_use,
            verbose=verbose,
            use_case=self._ml_usecase,
            **kwargs,
        )
        if self.logging_param:
            [
                logger.log_artifact(file=model_filename, type="model")
                for logger in self.logging_param.loggers
                if hasattr(logger, "remote")
            ]

        return model_, model_filename

    def load_model(
        self,
        model_name: str,
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

    @staticmethod
    def convert_model(estimator, language: str = "python") -> str:
        """
        This function transpiles trained machine learning models into native
        inference script in different programming languages (Python, C, Java,
        Go, JavaScript, Visual Basic, C#, PowerShell, R, PHP, Dart, Haskell,
        Ruby, F#). This functionality is very useful if you want to deploy models
        into environments where you can't install your normal Python stack to
        support model inference.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase')
        >>> lr = create_model('lr')
        >>> lr_java = convert_model(lr, 'java')


        estimator: scikit-learn compatible object
            Trained model object


        language: str, default = 'python'
            Language in which inference script to be generated. Following
            options are available:

            * 'python'
            * 'java'
            * 'javascript'
            * 'c'
            * 'c#'
            * 'f#'
            * 'go'
            * 'haskell'
            * 'php'
            * 'powershell'
            * 'r'
            * 'ruby'
            * 'vb'
            * 'dart'


        Returns:
            str

        """

        _check_soft_dependencies("m2cgen", extra=None, severity="error")
        import m2cgen as m2c

        if language == "python":
            return m2c.export_to_python(estimator)
        elif language == "java":
            return m2c.export_to_java(estimator)
        elif language == "c":
            return m2c.export_to_c(estimator)
        elif language == "c#":
            return m2c.export_to_c_sharp(estimator)
        elif language == "dart":
            return m2c.export_to_dart(estimator)
        elif language == "f#":
            return m2c.export_to_f_sharp(estimator)
        elif language == "go":
            return m2c.export_to_go(estimator)
        elif language == "haskell":
            return m2c.export_to_haskell(estimator)
        elif language == "javascript":
            return m2c.export_to_javascript(estimator)
        elif language == "php":
            return m2c.export_to_php(estimator)
        elif language == "powershell":
            return m2c.export_to_powershell(estimator)
        elif language == "r":
            return m2c.export_to_r(estimator)
        elif language == "ruby":
            return m2c.export_to_ruby(estimator)
        elif language == "vb":
            return m2c.export_to_visual_basic(estimator)
        else:
            raise ValueError(
                f"Wrong language {language}. Expected one of 'python', 'java', 'c', 'c#', 'dart', "
                "'f#', 'go', 'haskell', 'javascript', 'php', 'powershell', 'r', 'ruby', 'vb'."
            )

    def create_api(self, estimator, api_name, host="127.0.0.1", port=8000):

        """
        This function takes an input ``estimator`` and creates a POST API for
        inference. It only creates the API and doesn't run it automatically.
        To run the API, you must run the Python file using ``!python``.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase')
        >>> lr = create_model('lr')
        >>> create_api(lr, 'lr_api')
        >>> !python lr_api.py


        estimator: scikit-learn compatible object
            Trained model object


        api_name: str
            Name of the model.


        host: str, default = '127.0.0.1'
            API host address.


        port: int, default = 8000
            port for API.


        Returns:
            None
        """
        _check_soft_dependencies("fastapi", extra="mlops", severity="error")
        _check_soft_dependencies("uvicorn", extra="mlops", severity="error")
        _check_soft_dependencies("pydantic", extra="mlops", severity="error")

        self.save_model(estimator, model_name=api_name, verbose=False)
        target = f"{self.target_param}_prediction"

        query = f"""# -*- coding: utf-8 -*-

import pandas as pd
from pycaret.{self._ml_usecase.name.lower()} import load_model, predict_model
from fastapi import FastAPI
import uvicorn
from pydantic import create_model

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model("{api_name}")

# Create input/output pydantic models
input_model = create_model("{api_name}_input", **{self.X.iloc[0].to_dict()})
output_model = create_model("{api_name}_output", {target}={repr(self.y[0])})


# Define predict function
@app.post("/predict", response_model=output_model)
def predict(data: input_model):
    data = pd.DataFrame([data.dict()])
    predictions = predict_model(model, data=data)
    return {{"{target}": predictions["prediction_label"].iloc[0]}}


if __name__ == "__main__":
    uvicorn.run(app, host="{host}", port={port})
"""

        file_name = str(api_name) + ".py"

        f = open(file_name, "w")
        f.write(query)
        f.close()

        print(
            "API successfully created. This function only creates a POST API, "
            "it doesn't run it automatically. To run your API, please run this "
            f"command --> !python {api_name}.py"
        )

    def eda(self, display_format: str = "bokeh", **kwargs):
        """
        This function generates AutoEDA using AutoVIZ library. You must
        install Autoviz separately ``pip install autoviz`` to use this
        function.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase')
        >>> eda(display_format = 'bokeh')

        display_format: str, default = 'bokeh'
            When set to 'bokeh' the plots are interactive. Other option is ``svg`` for static
            plots that are generated using matplotlib and seaborn.


        **kwargs:
            Additional keyword arguments to pass to the AutoVIZ class.


        Returns:
            None
        """

        _check_soft_dependencies("autoviz", extra="mlops", severity="error")
        from autoviz.AutoViz_Class import AutoViz_Class

        AV = AutoViz_Class()
        AV.AutoViz(
            filename="",
            dfte=self.dataset_transformed,
            depVar=self.target_param,
            chart_format=display_format,
            **kwargs,
        )

    def create_docker(
        self,
        api_name: str,
        base_image: str = "python:3.8-slim",
        expose_port: int = 8000,
    ):
        """
        This function creates a ``Dockerfile`` and ``requirements.txt`` for
        productionalizing API end-point.


        Example
        -------
        >>> from pycaret.datasets import get_data
        >>> juice = get_data('juice')
        >>> from pycaret.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase')
        >>> lr = create_model('lr')
        >>> create_api(lr, 'lr_api')
        >>> create_docker('lr_api')


        api_name: str
            Name of API. Must be saved as a .py file in the same folder.


        base_image: str, default = "python:3.8-slim"
            Name of the base image for Dockerfile.


        expose_port: int, default = 8000
            port for expose for API in the Dockerfile.


        Returns:
            None
        """

        requirements = """
pycaret
fastapi
uvicorn
"""
        print("Writing requirements.txt")
        f = open("requirements.txt", "w")
        f.write(requirements)
        f.close()

        print("Writing Dockerfile")
        docker = """

FROM {BASE_IMAGE}

WORKDIR /app

ADD . /app

RUN apt-get update && apt-get install -y libgomp1

RUN pip install -r requirements.txt

EXPOSE {PORT}

CMD ["python", "{API_NAME}.py"]
""".format(
            BASE_IMAGE=base_image, PORT=expose_port, API_NAME=api_name
        )

        with open("Dockerfile", "w") as f:
            f.write(docker)

        print(
            """Dockerfile and requirements.txt successfully created.
    To build image you have to run --> !docker image build -f "Dockerfile" -t IMAGE_NAME:IMAGE_TAG .
            """
        )

    def _set_all_models(self) -> "_TabularExperiment":
        """Set all available models

        Returns
        -------
        _TabularExperiment
            The experiment object to allow chaining of methods
        """
        self._all_models, self._all_models_internal = self._get_models()
        return self

    def get_allowed_engines(self, estimator: str) -> Optional[List[str]]:
        """Get all the allowed engines for the specified estimator

        Parameters
        ----------
        estimator : str
            Identifier for the model for which the engines should be retrieved,
            e.g. "auto_arima"

        Returns
        -------
        Optional[List[str]]
            The allowed engines for the model. If the model only supports the
            default engine, then it return `None`.
        """
        allowed_engines = get_allowed_engines(
            estimator=estimator, all_allowed_engines=self.all_allowed_engines
        )
        return allowed_engines

    def get_engine(self, estimator: str) -> Optional[str]:
        """Gets the model engine currently set in the experiment for the specified
        model.

        Parameters
        ----------
        estimator : str
            Identifier for the model for which the engine should be retrieved,
            e.g. "auto_arima"

        Returns
        -------
        Optional[str]
            The engine for the model. If the model only supports the default
            engine, then it returns `None`.
        """
        engine = self.exp_model_engines.get(estimator, None)
        if engine is None:
            msg = (
                f"Engine for model '{estimator}' has not been set explicitly, "
                "hence returning None."
            )
            self.logger.info(msg)

        return engine

    def _set_engine(self, estimator: str, engine: str, severity: str = "error"):
        """Sets the engine to use for a particular model.

        Parameters
        ----------
        estimator : str
            Identifier for the model for which the engine should be set, e.g.
            "auto_arima"
        engine : str
            Engine to set for the model. All available engines for the model
            can be retrieved using get_allowed_engines()
        severity : str, optional
            How to handle incorrectly specified engines. Allowed values are "error"
            and "warning". If set to "warning", the existing engine is left
            unchanged if the specified engine is not correct., by default "error".

        Raises
        ------
        ValueError
            (1) If specified engine is not in the allowed list of engines and
                severity is set to "error"
            (2) If the value of "severity" is not one of the allowed values
        """
        if severity not in ("error", "warning"):
            raise ValueError(
                "Error in calling set_engine, severity "
                f'argument must be "error" or "warning", got "{severity}".'
            )

        allowed_engines = self.get_allowed_engines(estimator=estimator)
        if allowed_engines is None:
            msg = (
                f"Either model '{estimator}' has only 1 engine and hence can not be changed, "
                "or the model is not in the allowed list of models for this setup."
            )

            if severity == "error":
                raise ValueError(msg)
            elif severity == "warning":
                self.logger.warning(msg)
                print(msg)

        elif engine not in allowed_engines:
            msg = (
                f"Engine '{engine}' for estimator '{estimator}' is not allowed."
                f" Allowed values are: {', '.join(allowed_engines)}."
            )

            if severity == "error":
                raise ValueError(msg)
            elif severity == "warning":
                self.logger.warning(msg)
                print(msg)

        else:
            self.exp_model_engines[estimator] = engine
            self.logger.info(
                f"Engine successfully changes for model '{estimator}' to '{engine}'."
            )

        # Need to do this, else internal class variables are not reset with new engine.
        self._set_all_models()

    def _set_exp_model_engines(
        self,
        container_default_engines: Dict[str, str],
        engine: Optional[Dict[str, str]] = None,
    ) -> "_TabularExperiment":
        """Set all the model engines for the experiment.

        container_default_model_engines : Dict[str, str]
            Default engines obtained from the model containers

        engine: Optional[Dict[str, str]] = None
            The engine to use for the models, e.g. for auto_arima, users can
            switch between "pmdarima" and "statsforecast" by specifying
            engine={"auto_arima": "statsforecast"}

            If model ID is not present in key, default value will be obtained
            from the model container (i.e. container_default_model_engines).

            If a model container does not define the engines (means that the
            container does not support multiple engines), but the model's ID is
            present in the "engines" argument, it is simply ignored.

        Returns
        -------
        _TabularExperiment
            The experiment object to allow chaining of methods
        """
        # If user provides their own value, override the container defaults
        engine = engine or {}
        for key in container_default_engines:
            # If provided by user, then use that, else get from the defaults
            eng = engine.get(key, container_default_engines.get(key))
            self._set_engine(estimator=key, engine=eng, severity="error")

        return self

    def _set_all_metrics(self) -> "_TabularExperiment":
        """Set all available metrics

        Returns
        -------
        _TabularExperiment
            The experiment object to allow chaining of methods
        """
        self._all_metrics = self._get_metrics()
        return self
