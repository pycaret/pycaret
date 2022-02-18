import os
import gc
from copy import deepcopy

from typing import Optional, Dict, Any
from pycaret.internal.Display import Display
from pycaret.internal.logging import get_logger
from pycaret.internal.pipeline import get_pipeline_estimator_label
from pycaret.internal.meta_estimators import get_estimator_from_meta_estimator
from pycaret.loggers import BaseLogger
import traceback
import functions

logger = get_logger()
try:
    import mlflow
    import mlflow.sklearn
except ImportError:
    logger.warning("mlflowLogger requires mlflow. Install using `pip install mlflow`")


class mlflowLogger(BaseLogger):
    def __init__(self) -> None:
        super().__init__()
        import pdb
        pdb.set_trace()

    @classmethod
    def log_model(
        model,
        model_results,
        ml_usecase,
        score_dict: dict,
        source: str,
        runtime: float,
        model_fit_time: float,
        _prep_pipe,
        log_holdout: bool = True,
        log_plots: bool = False,
        experiment_custom_tags: Optional[Dict[str, Any]] = None,
        tune_cv_results=None,
        URI=None,
        display: Optional[Display] = None
    ):
        from pycaret.internal.tabular import _get_model_name, exp_name_log, predict_model, MLUsecase, _is_unsupervised, data_before_preprocess, target_param
        logger = get_logger()

        logger.info("Creating MLFlow logs")

        # Creating Logs message monitor
        if display:
            display.update_monitor(1, "Creating Logs")
            display.display_monitor()

        mlflow.set_experiment(exp_name_log)

        full_name = _get_model_name(model)
        logger.info(f"Model: {full_name}")

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
                except:
                    params = params.get_params()
            except:
                logger.warning("Couldn't get params for model. Exception:")
                logger.warning(traceback.format_exc())
                params = {}

            for i in list(params):
                v = params.get(i)
                if len(str(v)) > 250:
                    params.pop(i)

            logger.info(f"logged params: {params}")
            mlflow.log_params(params)

            # Log metrics
            mlflow.log_metrics(score_dict)

            # set tag of compare_models
            mlflow.set_tag("Source", source)

            # set custom tags if applicable
            if isinstance(experiment_custom_tags, dict):
                mlflow.set_tags(experiment_custom_tags)

            if not URI:
                import secrets

                URI = secrets.token_hex(nbytes=4)
            mlflow.set_tag("URI", URI)
            mlflow.set_tag("USI", USI)
            mlflow.set_tag("Run Time", runtime)
            mlflow.set_tag("Run ID", RunID)

            # Log training time in seconds
            mlflow.log_metric("TT", model_fit_time)

            # Log the CV results as model_results.html artifact
            if not _is_unsupervised(ml_usecase):
                try:
                    model_results.data.to_html("Results.html", col_space=65, justify="left")
                except:
                    model_results.to_html("Results.html", col_space=65, justify="left")
                mlflow.log_artifact("Results.html")
                os.remove("Results.html")

                if log_holdout:
                    # Generate hold-out predictions and save as html
                    try:
                        holdout = predict_model(model, verbose=False)
                        holdout_score = pull(pop=True)
                        del holdout
                        holdout_score.to_html("Holdout.html", col_space=65, justify="left")
                        mlflow.log_artifact("Holdout.html")
                        os.remove("Holdout.html")
                    except:
                        logger.warning(
                            "Couldn't create holdout prediction for model, exception below:"
                        )
                        logger.warning(traceback.format_exc())

            # Log AUC and Confusion Matrix plot

            if log_plots:

                logger.info(
                    "SubProcess plot_model() called =================================="
                )

                def _log_plot(plot):
                    try:
                        plot_name = plot_model(
                            model, plot=plot, verbose=False, save=True, system=False
                        )
                        mlflow.log_artifact(plot_name)
                        os.remove(plot_name)
                    except Exception as e:
                        logger.warning(e)

                for plot in log_plots:
                    _log_plot(plot)

                logger.info(
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
            default_conda_env["name"] = f"{exp_name_log}-env"
            default_conda_env.get("dependencies").pop(-3)
            dependencies = default_conda_env.get("dependencies")[-1]
            from pycaret.utils import __version__

            dep = f"pycaret=={__version__}"
            dependencies["pip"] = [dep]

            # define model signature
            from mlflow.models.signature import infer_signature

            try:
                signature = infer_signature(
                    data_before_preprocess.drop([target_param], axis=1)
                )
            except:
                logger.warning("Couldn't infer MLFlow signature.")
                signature = None
            if not _is_unsupervised(ml_usecase):
                input_example = (
                    data_before_preprocess.drop([target_param], axis=1).iloc[0].to_dict()
                )
            else:
                input_example = data_before_preprocess.iloc[0].to_dict()

            # log model as sklearn flavor
            prep_pipe_temp = deepcopy(_prep_pipe)
            prep_pipe_temp.steps.append(["trained_model", model])
            mlflow.sklearn.log_model(
                prep_pipe_temp,
                "model",
                conda_env=default_conda_env,
                # signature=signature,
                # input_example=input_example,
            )
            del prep_pipe_temp
        gc.collect()

    def log_experiment(log_profile, log_data):
        from pycaret.internal.tabular import exp_name_log, _is_unsupervised, data_before_preprocess, USI
        logger.info("Logging experiment in MLFlow")

        import mlflow

        try:
            mlflow.create_experiment(exp_name_log)
        except:
            logger.warning("Couldn't create mlflow experiment. Exception:")
            logger.warning(traceback.format_exc())

        # mlflow logging
        mlflow.set_experiment(exp_name_log)

        run_name_ = f"Session Initialized {USI}"

        mlflow.end_run()
        mlflow.start_run(run_name=run_name_)

        # Get active run to log as tag
        RunID = mlflow.active_run().info.run_id

        k = functions.copy()
        k.set_index("Description", drop=True, inplace=True)
        kdict = k.to_dict()
        params = kdict.get("Value")
        mlflow.log_params(params)

        # set tag of compare_models
        mlflow.set_tag("Source", "setup")

        # set custom tags if applicable
        if isinstance(experiment_custom_tags, dict):
            mlflow.set_tags(experiment_custom_tags)

        import secrets

        URI = secrets.token_hex(nbytes=4)
        mlflow.set_tag("URI", URI)
        mlflow.set_tag("USI", USI)
        mlflow.set_tag("Run Time", runtime)
        mlflow.set_tag("Run ID", RunID)

        # Log the transformation pipeline
        logger.info("SubProcess save_model() called ==================================")
        save_model(prep_pipe, "Transformation Pipeline", verbose=False)
        logger.info("SubProcess save_model() end ==================================")
        mlflow.log_artifact("Transformation Pipeline.pkl")
        os.remove("Transformation Pipeline.pkl")

        # Log pandas profile
        if log_profile:
            import pandas_profiling

            pf = pandas_profiling.ProfileReport(
                data_before_preprocess, **profile_kwargs
            )
            pf.to_file("Data Profile.html")
            mlflow.log_artifact("Data Profile.html")
            os.remove("Data Profile.html")
            display.display(functions_, clear=True)

        # Log training and testing set
        if log_data:
            if not _is_unsupervised(_ml_usecase):
                X_train.join(y_train).to_csv("Train.csv")
                X_test.join(y_test).to_csv("Test.csv")
                mlflow.log_artifact("Train.csv")
                mlflow.log_artifact("Test.csv")
                os.remove("Train.csv")
                os.remove("Test.csv")
            else:
                X.to_csv("Dataset.csv")
                mlflow.log_artifact("Dataset.csv")
                os.remove("Dataset.csv")