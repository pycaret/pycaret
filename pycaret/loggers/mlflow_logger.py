import secrets
from contextlib import contextmanager

from pycaret.loggers.base_logger import BaseLogger
from pycaret.utils.generic import mlflow_remove_bad_chars

try:
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking.fluent import _active_run_stack
    from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
except ImportError:
    mlflow = None
    _active_run_stack = None


@contextmanager
def set_active_mlflow_run(run):
    """Set active MLFlow run to ``run`` and then back to what it was."""
    global _active_run_stack
    _active_run_stack.append(run)
    yield
    try:
        _active_run_stack.remove(run)
    except ValueError:
        pass


@contextmanager
def clean_active_mlflow_run():
    """Trick MLFLow into thinking there are no active runs."""
    global _active_run_stack
    old_run_stack = _active_run_stack.copy()
    _active_run_stack.clear()
    yield
    active_run = _active_run_stack[-1]
    _active_run_stack.clear()
    _active_run_stack.extend(old_run_stack)
    _active_run_stack.append(active_run)


class MlflowLogger(BaseLogger):
    def __init__(self) -> None:
        if mlflow is None:
            raise ImportError(
                "MlflowLogger requires mlflow. Install using `pip install mlflow`"
            )
        super().__init__()
        self.runs = []

    def init_experiment(self, exp_name_log, full_name=None, setup=True):
        full_name = full_name
        mlflow.set_experiment(exp_name_log)
        if setup:
            # If we are starting a new experiment with setup,
            # make sure we do not nest it in any other run
            with clean_active_mlflow_run():
                run = mlflow.start_run(run_name=full_name)
        else:
            run = mlflow.start_run(run_name=full_name, nested=True)
        self.runs.append(run)
        return self.runs

    @property
    def active_run(self):
        if not self.runs:
            return None
        return self.runs[-1]

    @property
    def parent_run(self):
        if len(self.runs) < 2:
            return None
        # Get the setup with the same USI as the current run.
        current_USI = mlflow.get_run(self.runs[-1].info.run_id).data.tags["USI"]
        for run in reversed(self.runs[:-1]):
            tags = mlflow.get_run(run.info.run_id).data.tags
            if tags["Source"] == "setup" and tags["USI"] == current_USI:
                return run
        return None

    @property
    def run_id(self):
        return self.active_run.info.run_id

    def finish_experiment(self):
        try:
            with set_active_mlflow_run(self.active_run):
                mlflow.end_run()
            self.runs.pop()
        except Exception:
            pass

    def log_params(self, params, model_name=None):
        params = {mlflow_remove_bad_chars(k): v for k, v in params.items()}
        with set_active_mlflow_run(self.active_run):
            mlflow.log_params(params)

    def log_metrics(self, metrics, source=None):
        with set_active_mlflow_run(self.active_run):
            mlflow.log_metrics(metrics)

    def set_tags(self, source, experiment_custom_tags, runtime, USI=None):
        # Get active run to log as tag
        with set_active_mlflow_run(self.active_run):
            RunID = self.active_run.info.run_id

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
            if self.parent_run and not source == "setup":
                mlflow.set_tag(MLFLOW_PARENT_RUN_ID, self.parent_run.info.run_id)

    def log_artifact(self, file, type="artifact"):
        with set_active_mlflow_run(self.active_run):
            mlflow.log_artifact(file)

    def log_plot(self, plot, title=None):
        self.log_artifact(plot)

    def log_hpram_grid(self, html_file, title="hpram_grid"):
        self.log_artifact(html_file)

    def log_sklearn_pipeline(self, experiment, prep_pipe, model, path=None):
        # get default conda env
        from mlflow.sklearn import get_default_conda_env

        from pycaret import __version__

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
        with set_active_mlflow_run(self.active_run):
            mlflow.sklearn.log_model(
                self._construct_pipeline_if_needed(model, prep_pipe),
                "model",
                conda_env=default_conda_env,
                # signature=signature,
                # input_example=input_example,
            )
