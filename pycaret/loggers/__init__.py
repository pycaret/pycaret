import os
import traceback
import gc
import traceback
import pandas as pd
from typing import List, Optional, Dict, Any
from .base_logger import BaseLogger
from pycaret.internal.logging import get_logger
from pycaret.internal.meta_estimators import get_estimator_from_meta_estimator

logger = get_logger()


class DashboardLogger:
    def __init__(self, logger_list: List[BaseLogger]) -> None:
        self.loggers = logger_list

    def init_loggers(self, exp_name_log, full_name=None):
        for logger in self.loggers:
            logger.init_experiment(exp_name_log, full_name)

    def log_params(self, params):
        for logger in self.loggers:
            logger.log_params(params)

    def log_model(
        self,
        model,
        model_results,
        ml_usecase,
        _prep_pipe,
        score_dict: dict,
        source: str,
        runtime: float,
        model_fit_time: float,
        log_holdout: bool = True,
        log_plots: bool = False,
        experiment_custom_tags: Optional[Dict[str, Any]] = None,
        tune_cv_results=None,
        URI=None,
        display=None,
    ):
        from pycaret.internal.tabular import (
            get_pipeline_estimator_label,
            plot_model,
            pull,
            _get_model_name,
            exp_name_log,
            predict_model,
            _is_unsupervised,
        )

        console = get_logger()
        console.info("Creating Dashboard logs")

        # Creating Logs message monitor
        if display:
            display.update_monitor(1, "Creating Logs")
            display.display_monitor()

        full_name = _get_model_name(model)
        console.info(f"Model: {full_name}")
        self.init_loggers(exp_name_log, full_name)

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
            console.warning("Couldn't get params for model. Exception:")
            console.warning(traceback.format_exc())
            params = {}

        for i in list(params):
            v = params.get(i)
            if len(str(v)) > 250:
                params.pop(i)

        console.info(f"logged params: {params}")
        score_dict["TT"] = model_fit_time
        for logger in self.loggers:
            logger.log_params(params, model_name=full_name)
            logger.log_metrics(score_dict, source)
            logger.set_tags(source, experiment_custom_tags, runtime)

        # Log the CV results as model_results.html artifact
        if not _is_unsupervised(ml_usecase):
            try:
                model_results.data.to_html("Results.html", col_space=65, justify="left")
            except:
                model_results.to_html("Results.html", col_space=65, justify="left")

            # [logger.log_artifact("Results.html", "Results") for logger in self.loggers]
            os.remove("Results.html")

            if log_holdout:
                # Generate hold-out predictions and save as html
                try:
                    holdout = predict_model(model, verbose=False)
                    holdout_score = pull(pop=True)
                    del holdout
                    holdout_score.to_html("Holdout.html", col_space=65, justify="left")
                    [
                        logger.log_artifact("Holdout.html", "Holdout")
                        for logger in self.loggers
                    ]
                    os.remove("Holdout.html")
                except:
                    console.warning(
                        "Couldn't create holdout prediction for model, exception below:"
                    )
                    console.warning(traceback.format_exc())

        # Log AUC and Confusion Matrix plot

        if log_plots:
            console.info(
                "SubProcess plot_model() called =================================="
            )

            def _log_plot(plot):
                try:
                    plot_name = plot_model(
                        model, plot=plot, verbose=False, save=True, system=False
                    )
                    [
                        logger.log_plot(plot_name, plot_name.split(".")[0])
                        for logger in self.loggers
                    ]
                    os.remove(plot_name)
                except Exception as e:
                    console.warning(e)

            for plot in log_plots:
                _log_plot(plot)

            console.info(
                "SubProcess plot_model() end =================================="
            )

        # Log hyperparameter tuning grid
        if tune_cv_results:
            d1 = tune_cv_results.get("params")
            dd = pd.DataFrame.from_dict(d1)
            dd["Score"] = tune_cv_results.get("mean_test_score")
            dd.to_html("Iterations.html", col_space=75, justify="left")
            [
                logger.log_hpram_grid("Iterations.html", "Hyperparameter-grid")
                for logger in self.loggers
            ]
            os.remove("Iterations.html")

        [logger.log_sklearn_pipeline(_prep_pipe, model) for logger in self.loggers]

        gc.collect()

    def log_experiment(
        self,
        log_profile,
        profile_kwargs,
        log_data,
        ml_usecase,
        functions,
        experiment_custom_tags,
        runtime,
        display,
    ):
        from pycaret.internal.tabular import (
            exp_name_log,
            _is_unsupervised,
            data_before_preprocess,
            USI,
            save_model,
            prep_pipe,
            X,
        )

        console = get_logger()
        console.info("Logging experiment in MLFlow")

        k = functions.copy()
        k.set_index("Description", drop=True, inplace=True)
        kdict = k.to_dict()
        params = kdict.get("Value")
        for logger in self.loggers:
            logger.init_experiment(exp_name_log)
            logger.log_params(params, "setup")
            logger.set_tags("setup", experiment_custom_tags, runtime)

        # Log the transformation pipeline
        console.info(
            "SubProcess save_model() called =================================="
        )
        save_model(prep_pipe, "Transformation Pipeline", verbose=False)
        console.info("SubProcess save_model() end ==================================")
        [
            logger.log_artifact("Transformation Pipeline.pkl", "transformation_pipe")
            for logger in self.loggers
        ]
        os.remove("Transformation Pipeline.pkl")

        # Log pandas profile
        if log_profile:
            import pandas_profiling

            pf = pandas_profiling.ProfileReport(
                data_before_preprocess, **profile_kwargs
            )
            pf.to_file("Data Profile.html")
            [
                logger.log_artifact("Data Profile.html", "data_profile")
                for logger in self.loggers
            ]
            os.remove("Data Profile.html")
            display.display(functions, clear=True)

        # Log training and testing set
        if log_data:
            if not _is_unsupervised(ml_usecase):
                from pycaret.internal.tabular import X_train, X_test, y_train, y_test

                X_train.join(y_train).to_csv("Train.csv")
                X_test.join(y_test).to_csv("Test.csv")
                [
                    logger.log_artifact("Train.csv", "train_data")
                    for logger in self.loggers
                ]
                [
                    logger.log_artifact("Test.csv", "test_data")
                    for logger in self.loggers
                ]
                os.remove("Train.csv")
                os.remove("Test.csv")
            else:
                X.to_csv("Dataset.csv")
                [logger.log_artifact("Dataset.csv", "data") for logger in self.loggers]
                os.remove("Dataset.csv")

    def log_model_comparison(self, results, source):
        for logger in self.loggers:
            logger.log_model_comparison(results, source)

    def finish(self):
        for logger in self.loggers:
            logger.finish_experiment()
