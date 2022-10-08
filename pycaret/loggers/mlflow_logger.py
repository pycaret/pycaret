import secrets
from copy import deepcopy

import pycaret
from pycaret.loggers.base_logger import SETUP_TAG, BaseLogger
from pycaret.utils import __version__
from pycaret.utils.generic import mlflow_remove_bad_chars

try:
    import mlflow
    import mlflow.sklearn
except ImportError:
    mlflow = None


class MlflowLogger(BaseLogger):
    def __init__(self) -> None:
        if mlflow is None:
            raise ImportError(
                "MlflowLogger requires mlflow. Install using `pip install mlflow`"
            )
        super().__init__()
        self.run = None

    def init_experiment(self, exp_name_log, full_name=None):
        # get USI from nlp or tabular
        USI = None
        try:
            USI = pycaret.internal.tabular.USI
        except Exception:
            try:
                USI = pycaret.nlp.USI
            except Exception:
                pass
        full_name = full_name or f"{SETUP_TAG} {USI}"
        mlflow.set_experiment(exp_name_log)
        self.run = mlflow.start_run(run_name=full_name, nested=True)

        return self.run

    def finish_experiment(self):
        try:
            mlflow.end_run()
        except Exception:
            pass

    def log_params(self, params, model_name=None):
        params = {mlflow_remove_bad_chars(k): v for k, v in params.items()}
        mlflow.log_params(params)

    def log_metrics(self, metrics, source=None):
        mlflow.log_metrics(metrics)

    def set_tags(self, source, experiment_custom_tags, runtime, USI=None):
        # get USI from nlp or tabular
        if not USI:
            try:
                USI = pycaret.nlp.USI
            except Exception:
                pass

        # Get active run to log as tag
        RunID = mlflow.active_run().info.run_id

        # set tag of compare_models
        mlflow.set_tag("Source", source)

        # set custom tags if applicable
        if isinstance(experiment_custom_tags, dict):
            mlflow.set_tags(experiment_custom_tags)

        URI = secrets.token_hex(nbytes=4)
        mlflow.set_tag("URI", URI)
        mlflow.set_tag("USI", USI)
        mlflow.set_tag("Run Time", runtime)
        mlflow.set_tag("Run ID", RunID)

    def log_artifact(self, file, type="artifact"):
        mlflow.log_artifact(file)

    def log_plot(self, plot, title=None):
        self.log_artifact(plot)

    def log_hpram_grid(self, html_file, title="hpram_grid"):
        self.log_artifact(html_file)

    def log_sklearn_pipeline(self, experiment, prep_pipe, model, path=None):
        # get default conda env
        from mlflow.sklearn import get_default_conda_env

        default_conda_env = get_default_conda_env()
        default_conda_env["name"] = f"{experiment.exp_name_log}-env"
        default_conda_env.get("dependencies").pop(-3)
        dependencies = default_conda_env.get("dependencies")[-1]

        dep = f"pycaret=={__version__}"
        dependencies["pip"] = [dep]

        # # define model signature
        # from mlflow.models.signature import infer_signature

        # try:
        #     signature = infer_signature(
        #         data_before_preprocess.drop([target_param], axis=1)
        #     )
        # except Exception:
        #     logger.warning("Couldn't infer MLFlow signature.")
        #     signature = None
        # if not _is_unsupervised(_ml_usecase):
        #     input_example = (
        #         data_before_preprocess.drop([target_param], axis=1).iloc[0].to_dict()
        #     )
        # else:
        #     input_example = data_before_preprocess.iloc[0].to_dict()

        # log model as sklearn flavor
        prep_pipe_temp = deepcopy(prep_pipe)
        prep_pipe_temp.steps.append(["trained_model", model])
        mlflow.sklearn.log_model(
            prep_pipe_temp,
            "model",
            conda_env=default_conda_env,
            # signature=signature,
            # input_example=input_example,
        )
