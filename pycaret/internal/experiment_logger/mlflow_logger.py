try:
    import mlflow
    import mlflow.sklearn
    import mlflow.models.signature
    from mlflow.tracking import MlflowClient
except ImportError:
    mlflow = None

from pycaret.internal.meta_estimators import get_estimator_from_meta_estimator
from pycaret.internal.pipeline import get_pipeline_estimator_label
import pandas as pd
from pandas.io.formats.style import Styler
from typing import Optional, Union

import os
import gc
import traceback
import secrets
from copy import deepcopy

from pycaret.internal.Display import Display
from pycaret.internal.experiment_logger.experiment_logger import ExperimentLogger

# TODO separate logging for Loggers (don't use experiment.logger)
class MLFlowLogger(ExperimentLogger):
    name: str = "MLFlow Logger"
    id: str = "mlflow"

    def __init__(self, *, errors="ignore") -> None:
        if mlflow is None:
            raise ImportError(
                f"It appears that mlflow is not installed, which is required for {self.name}. Do: pip install mlflow"
            )
        self.exp_id: Optional[str] = None
        self.exp_name_log: Optional[str] = None
        self.USI: Optional[str] = None
        assert errors in ["raise", "ignore"]
        self.errors = errors
        super().__init__()

    def _setup_supervised(
        self,
        experiment: "_PyCaretExperiment",
        functions: Union[pd.DataFrame, Styler],
        runtime: float,
        log_profile: bool,
        profile_kwargs: dict,
        log_data: bool,
        display: Optional[Display],
    ):
        functions_styler = functions
        if isinstance(functions, Styler):
            functions = functions.data

        # log into experiment
        experiment.experiment__.append(("Setup Config", functions))
        experiment.experiment__.append(("X_training Set", experiment.X_train))
        experiment.experiment__.append(("y_training Set", experiment.y_train))
        experiment.experiment__.append(("X_test Set", experiment.X_test))
        experiment.experiment__.append(("y_test Set", experiment.y_test))
        experiment.experiment__.append(
            ("Transformation Pipeline", experiment.prep_pipe)
        )

        if experiment.logging_param:

            experiment.logger.info("Logging experiment in MLFlow")

            self.exp_name_log = experiment.exp_name_log
            self.USI = experiment.USI

            try:
                self.exp_id = mlflow.create_experiment(self.exp_name_log)
            except Exception:
                self.exp_id = None
                experiment.logger.warning(
                    "Couldn't create mlflow experiment. Exception:"
                )
                experiment.logger.warning(traceback.format_exc())

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

                URI = secrets.token_hex(nbytes=4)
                mlflow.set_tag("URI", URI)
                mlflow.set_tag("USI", self.USI)
                mlflow.set_tag("Run Time", runtime)
                mlflow.set_tag("Run ID", RunID)

                # Log the transformation pipeline
                experiment.logger.info(
                    "SubProcess save_model() called =================================="
                )
                experiment.save_model(
                    experiment.prep_pipe, "Transformation Pipeline", verbose=False
                )
                experiment.logger.info(
                    "SubProcess save_model() end =================================="
                )
                mlflow.log_artifact("Transformation Pipeline.pkl")
                os.remove("Transformation Pipeline.pkl")

                # Log pandas profile
                if log_profile:
                    # TODO decouple, make it a separate logger?
                    import pandas_profiling

                    pf = pandas_profiling.ProfileReport(
                        experiment.data_before_preprocess, **profile_kwargs
                    )
                    pf.to_file("Data Profile.html")
                    mlflow.log_artifact("Data Profile.html")
                    os.remove("Data Profile.html")
                    display.display(functions_styler, clear=True)

                # Log training and testing set
                if log_data:
                    experiment.X_train.join(experiment.y_train).to_csv("Train.csv")
                    experiment.X_test.join(experiment.y_test).to_csv("Test.csv")
                    mlflow.log_artifact("Train.csv")
                    mlflow.log_artifact("Test.csv")
                    os.remove("Train.csv")
                    os.remove("Test.csv")
        return

    def _setup_unsupervised(
        self,
        experiment: "_PyCaretExperiment",
        functions: Union[pd.DataFrame, Styler],
        runtime: float,
        log_profile: bool,
        profile_kwargs: dict,
        log_data: bool,
        display: Optional[Display],
    ):
        functions_styler = functions
        if isinstance(functions, Styler):
            functions = functions.data

        # log into experiment
        experiment.experiment__.append(("Setup Config", functions))
        experiment.experiment__.append(("Transformed Data", self.X))

        if experiment.logging_param:

            experiment.logger.info("Logging experiment in MLFlow")

            self.exp_name_log = experiment.exp_name_log
            self.USI = experiment.USI

            try:
                self.exp_id = mlflow.create_experiment(self.exp_name_log)
            except Exception:
                self.exp_id = None
                experiment.logger.warning(
                    "Couldn't create mlflow experiment. Exception:"
                )
                experiment.logger.warning(traceback.format_exc())

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

                URI = secrets.token_hex(nbytes=4)
                mlflow.set_tag("URI", URI)
                mlflow.set_tag("USI", self.USI)
                mlflow.set_tag("Run Time", runtime)
                mlflow.set_tag("Run ID", RunID)

                # Log the transformation pipeline
                experiment.logger.info(
                    "SubProcess save_model() called =================================="
                )
                experiment.save_model(
                    experiment.prep_pipe, "Transformation Pipeline", verbose=False
                )
                experiment.logger.info(
                    "SubProcess save_model() end =================================="
                )
                mlflow.log_artifact("Transformation Pipeline.pkl")
                os.remove("Transformation Pipeline.pkl")

                # Log pandas profile
                if log_profile:
                    # TODO decouple, make it a separate logger?
                    import pandas_profiling

                    pf = pandas_profiling.ProfileReport(
                        experiment.data_before_preprocess, **profile_kwargs
                    )
                    pf.to_file("Data Profile.html")
                    mlflow.log_artifact("Data Profile.html")
                    os.remove("Data Profile.html")
                    display.display(functions_styler, clear=True)

                # Log training and testing set
                if log_data:
                    experiment.X.to_csv("Dataset.csv")
                    mlflow.log_artifact("Dataset.csv")
                    os.remove("Dataset.csv")
        return

    def setup_logging(
        self,
        experiment: "_PyCaretExperiment",
        functions: Union[pd.DataFrame, Styler],
        runtime: float,
        log_profile: bool,
        profile_kwargs: dict,
        log_data: bool,
        display: Optional[Display],
    ) -> None:
        if experiment._is_unsupervised():
            return self._setup_unsupervised(
                experiment=experiment,
                functions=functions,
                runtime=runtime,
                log_profile=log_profile,
                profile_kwargs=profile_kwargs,
                log_data=log_data,
                display=display,
            )
        else:
            return self._setup_supervised(
                experiment=experiment,
                functions=functions,
                runtime=runtime,
                log_profile=log_profile,
                profile_kwargs=profile_kwargs,
                log_data=log_data,
                display=display,
            )

    def _log_model(
        self,
        experiment: "_PyCaretExperiment",
        model,
        model_results: Union[pd.DataFrame, Styler],
        score_dict: dict,
        source: str,
        runtime: float,
        model_fit_time: float,
        _prep_pipe,
        log_holdout: bool = True,
        log_plots: bool = False,
        tune_cv_results: Optional[dict] = None,
        URI: Optional[str] = None,
        display: Optional[Display] = None,
    ) -> None:
        experiment.logger.info("Creating MLFlow logs")

        # Creating Logs message monitor
        if display:
            display.update_monitor(1, "Creating Logs")
            display.display_monitor()

        mlflow.set_experiment(self.exp_name_log)

        full_name = experiment._get_model_name(model)
        experiment.logger.info(f"Model: {full_name}")

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
                except Exception:
                    params = params.get_params()
            except Exception:
                experiment.logger.warning("Couldn't get params for model. Exception:")
                experiment.logger.warning(traceback.format_exc())
                params = {}

            for i in list(params):
                v = params.get(i)
                if len(str(v)) > 250:
                    params.pop(i)

            experiment.logger.info(f"logged params: {params}")
            mlflow.log_params(params)

            # Log metrics
            mlflow.log_metrics(score_dict)

            # set tag of compare_models
            mlflow.set_tag("Source", source)

            if not URI:
                URI = secrets.token_hex(nbytes=4)
            mlflow.set_tag("URI", URI)
            mlflow.set_tag("USI", self.USI)
            mlflow.set_tag("Run Time", runtime)
            mlflow.set_tag("Run ID", RunID)

            # Log training time in seconds
            mlflow.log_metric("TT", model_fit_time)

            # Log the CV results as model_results.html artifact
            if not experiment._is_unsupervised():
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
                        holdout = experiment.predict_model(model, verbose=False)  # type: ignore
                        holdout_score = experiment.pull(pop=True)
                        del holdout
                        holdout_score.to_html(
                            "Holdout.html", col_space=65, justify="left"
                        )
                        mlflow.log_artifact("Holdout.html")
                        os.remove("Holdout.html")
                    except Exception:
                        experiment.logger.warning(
                            "Couldn't create holdout prediction for model, exception below:"
                        )
                        experiment.logger.warning(traceback.format_exc())

            # Log AUC and Confusion Matrix plot

            if log_plots:

                experiment.logger.info(
                    "SubProcess plot_model() called =================================="
                )

                def _log_plot(plot):
                    try:
                        plot_name = experiment.plot_model(
                            model, plot=plot, verbose=False, save=True, system=False
                        )
                        mlflow.log_artifact(plot_name)
                        os.remove(plot_name)
                    except Exception as e:
                        experiment.logger.warning(e)

                for plot in log_plots:
                    _log_plot(plot)

                experiment.logger.info(
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

            default_conda_env = mlflow.sklearn.get_default_conda_env()
            default_conda_env["name"] = f"{self.exp_name_log}-env"
            default_conda_env.get("dependencies").pop(-3)
            dependencies = default_conda_env.get("dependencies")[-1]
            from pycaret.utils import __version__

            dep = f"pycaret=={__version__}"
            dependencies["pip"] = [dep]

            try:
                signature = mlflow.models.signature.infer_signature(
                    experiment.data_before_preprocess.drop(
                        [experiment.target_param], axis=1
                    )
                )
            except Exception:
                experiment.logger.warning("Couldn't infer MLFlow signature.")
                signature = None
            if not experiment._is_unsupervised():
                input_example = (
                    experiment.data_before_preprocess.drop(
                        [experiment.target_param], axis=1
                    )
                    .iloc[0]
                    .to_dict()
                )
            else:
                input_example = experiment.data_before_preprocess.iloc[0].to_dict()

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

    def log_model(
        self,
        experiment: "_PyCaretExperiment",
        model,
        model_results: Union[pd.DataFrame, Styler],
        score_dict: dict,
        source: str,
        runtime: float,
        model_fit_time: float,
        _prep_pipe,
        log_holdout: bool = True,
        log_plots: bool = False,
        tune_cv_results: Optional[dict] = None,
        URI: Optional[str] = None,
        display: Optional[Display] = None,
    ) -> None:
        if self.errors == "raise":
            self._log_model(
                experiment=experiment,
                model=model,
                model_results=model_results,
                score_dict=score_dict,
                source=source,
                runtime=runtime,
                model_fit_time=model_fit_time,
                _prep_pipe=_prep_pipe,
                log_holdout=log_holdout,
                log_plots=log_plots,
                tune_cv_results=tune_cv_results,
                URI=URI,
                display=display,
            )
        else:
            try:
                self._log_model(
                    experiment=experiment,
                    model=model,
                    model_results=model_results,
                    score_dict=score_dict,
                    source=source,
                    runtime=runtime,
                    model_fit_time=model_fit_time,
                    _prep_pipe=_prep_pipe,
                    log_holdout=log_holdout,
                    log_plots=log_plots,
                    tune_cv_results=tune_cv_results,
                    URI=URI,
                    display=display,
                )
            except:
                experiment.logger.error(
                    f"{self.name} log_model() for {model} raised an exception:"
                )
                experiment.logger.error(traceback.format_exc())

    def get_logs(
        self, experiment: "_PyCaretExperiment"
    ) -> pd.DataFrame:
        client = MlflowClient()

        exp_id = self.exp_id
        experiment_by_id = client.get_experiment(exp_id)
        if experiment_by_id is None:
            raise ValueError(
                "No active run found. Check logging parameter in setup or to get logs for inactive run pass experiment_name."
            )

        runs = mlflow.search_runs(exp_id)

        return runs
