import gc
import os
import tempfile
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import pandas as pd

from pycaret.internal.meta_estimators import get_estimator_from_meta_estimator
from pycaret.internal.pipeline import get_pipeline_estimator_label

from .base_logger import SETUP_TAG, BaseLogger

if TYPE_CHECKING:
    from pycaret.internal.pycaret_experiment.tabular_experiment import (
        _TabularExperiment,
    )


class DashboardLogger:
    def __init__(self, logger_list: List[BaseLogger]) -> None:
        self.loggers = logger_list

    def __repr__(self) -> str:
        return ", ".join([str(logger) for logger in self.loggers])

    def init_loggers(self, exp_name_log, full_name=None):
        for logger in self.loggers:
            logger.init_experiment(exp_name_log, full_name)

    def log_params(self, params):
        for logger in self.loggers:
            logger.log_params(params)

    def log_model(
        self,
        experiment: "_TabularExperiment",
        model,
        model_results,
        pipeline,
        score_dict: dict,
        source: str,
        runtime: float,
        model_fit_time: float,
        log_holdout: bool = True,
        log_plots: Optional[List[str]] = None,
        experiment_custom_tags: Optional[Dict[str, Any]] = None,
        tune_cv_results=None,
        URI=None,
        display=None,
    ):
        log_plots = log_plots or []
        console = experiment.logger
        console.info("Creating Dashboard logs")

        # Creating Logs message monitor
        if display:
            display.update_monitor(1, "Creating Logs")

        full_name = experiment._get_model_name(model)
        console.info(f"Model: {full_name}")
        self.init_loggers(experiment.exp_name_log, full_name)

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
            except AttributeError:
                params = params.get_params()
        except Exception:
            console.warning(
                f"Couldn't get params for model. Exception:\n{traceback.format_exc()}"
            )
            params = {}

        for i in list(params):
            v = params.get(i)
            if len(str(v)) > 250:
                params.pop(i)

        console.info(f"Logged params: {params}")
        score_dict["TT"] = model_fit_time

        # Log metrics
        def try_make_float(val):
            try:
                return np.float64(val)
            except Exception:
                return np.nan

        score_dict = {k: try_make_float(v) for k, v in score_dict.items()}

        for logger in self.loggers:
            logger.log_params(params, model_name=full_name)
            logger.log_metrics(score_dict, source)
            logger.set_tags(source, experiment_custom_tags, runtime, USI=experiment.USI)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Log the CV results as model_results.html artifact
            if not experiment._is_unsupervised() and model_results is not None:
                results_path = os.path.join(tmpdir, "Results.html")
                try:
                    model_results.data.to_html(
                        results_path, col_space=65, justify="left"
                    )
                except Exception:
                    model_results.to_html(results_path, col_space=65, justify="left")
                [
                    logger.log_artifact(results_path, "Results")
                    for logger in self.loggers
                ]

                if log_holdout:
                    # Generate hold-out predictions and save as html
                    holdout_path = os.path.join(tmpdir, "Holdout.html")
                    try:
                        experiment.predict_model(model, verbose=False)  # type: ignore
                        holdout_score = experiment.pull(pop=True)
                        holdout_score.to_html(
                            holdout_path, col_space=65, justify="left"
                        )
                        [
                            logger.log_artifact(holdout_path, "Holdout")
                            for logger in self.loggers
                        ]
                    except Exception:
                        console.warning(
                            "Couldn't create holdout prediction for model, exception below:\n"
                            f"{traceback.format_exc()}"
                        )

            # Log AUC and Confusion Matrix plot

            if log_plots:
                console.info(
                    "SubProcess plot_model() called =================================="
                )

                def _log_plot(plot):
                    try:
                        plot_name = experiment._plot_model(
                            model, plot=plot, verbose=False, save=tmpdir, system=False
                        )
                        [
                            logger.log_plot(plot_name, Path(plot_name).stem)
                            for logger in self.loggers
                        ]
                    except Exception:
                        console.warning(
                            f"Couldn't create plot {plot} for model, exception below:\n"
                            f"{traceback.format_exc()}"
                        )

                for plot in log_plots:
                    _log_plot(plot)

                console.info(
                    "SubProcess plot_model() end =================================="
                )

            # Log hyperparameter tuning grid
            if tune_cv_results:
                iterations_path = os.path.join(tmpdir, "Iterations.html")
                d1 = tune_cv_results.get("params")
                dd = pd.DataFrame.from_dict(d1)
                dd["Score"] = tune_cv_results.get("mean_test_score")
                dd.to_html(iterations_path, col_space=75, justify="left")
                [
                    logger.log_hpram_grid(iterations_path, "Hyperparameter-grid")
                    for logger in self.loggers
                ]

            [
                logger.log_sklearn_pipeline(experiment, pipeline, model, path=tmpdir)
                for logger in self.loggers
            ]

        self.finish()
        gc.collect()

    def log_experiment(
        self,
        experiment: "_TabularExperiment",
        log_profile,
        log_data,
        experiment_custom_tags,
        runtime,
    ):
        console = experiment.logger
        console.info("Logging experiment in loggers")

        k = experiment._display_container[0].copy()
        k.set_index("Description", drop=True, inplace=True)
        kdict = k.to_dict()
        params = kdict.get("Value")
        for logger in self.loggers:
            logger.init_experiment(
                experiment.exp_name_log, f"{SETUP_TAG} {experiment.USI}"
            )
            logger.log_params(params, "setup")
            logger.set_tags("setup", experiment_custom_tags, runtime)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Log the transformation pipeline
            console.info(
                "SubProcess save_model() called =================================="
            )
            experiment.save_model(
                experiment.pipeline,
                os.path.join(tmpdir, "Transformation Pipeline"),
                verbose=False,
            )
            console.info(
                "SubProcess save_model() end =================================="
            )
            [
                logger.log_artifact(
                    os.path.join(tmpdir, "Transformation Pipeline.pkl"),
                    "transformation_pipe",
                )
                for logger in self.loggers
            ]

            # Log pandas profile
            if log_profile:
                profile_path = os.path.join(tmpdir, "Data Profile.html")
                experiment.report.to_file(profile_path)
                [
                    logger.log_artifact(profile_path, "data_profile")
                    for logger in self.loggers
                ]

            # Log training and testing set
            if log_data:
                if not experiment._is_unsupervised():
                    train_path = os.path.join(tmpdir, "Train.csv")
                    test_path = os.path.join(tmpdir, "Test.csv")

                    experiment.train.to_csv(train_path)
                    experiment.test.to_csv(test_path)
                    [
                        logger.log_artifact(train_path, "train_data")
                        for logger in self.loggers
                    ]
                    [
                        logger.log_artifact(test_path, "test_data")
                        for logger in self.loggers
                    ]
                    # upload data to remote server
                    [
                        logger.log_artifact(train_path, "data")
                        for logger in self.loggers
                        if hasattr(logger, "remote")
                    ]
                    [
                        logger.log_artifact(test_path, "data")
                        for logger in self.loggers
                        if hasattr(logger, "remote")
                    ]
                    if experiment.transform_target_param:
                        train_transform_path = os.path.join(
                            tmpdir, "Train_transform.csv"
                        )
                        test_transform_path = os.path.join(tmpdir, "Test_transform.csv")
                        experiment.train_transformed.to_csv(train_transform_path)
                        experiment.test_transformed.to_csv(test_transform_path)
                        [
                            logger.log_artifact(train_transform_path, "data")
                            for logger in self.loggers
                            if hasattr(logger, "remote")
                        ]
                        [
                            logger.log_artifact(test_transform_path, "data")
                            for logger in self.loggers
                            if hasattr(logger, "remote")
                        ]
                else:
                    train_path = os.path.join(tmpdir, "Dataset.csv")
                    experiment.train.to_csv(train_path)
                    [logger.log_artifact(train_path, "data") for logger in self.loggers]
                    if experiment.transform_target_param:
                        train_transform_path = os.path.join(
                            tmpdir, "Dataset_transform.csv"
                        )
                        experiment.train_transformed.to_csv(train_transform_path)
                        [
                            logger.log_artifact(train_transform_path, "data")
                            for logger in self.loggers
                            if hasattr(logger, "remote")
                        ]
                [
                    logger.log_artifact(file="", type="data_commit")
                    for logger in self.loggers
                    if hasattr(logger, "remote")
                ]

    def log_model_comparison(self, results, source):
        for logger in self.loggers:
            logger.log_model_comparison(results, source)

    def finish(self):
        for logger in self.loggers:
            logger.finish_experiment()
