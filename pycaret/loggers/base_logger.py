from abc import ABC
from copy import deepcopy

from sklearn.pipeline import Pipeline

SETUP_TAG = "Session Initialized"


class BaseLogger(ABC):
    def init_logger(self):
        pass

    def __del__(self):
        try:
            self.finish_experiment()
        except Exception:
            pass

    def __repr__(self) -> str:
        return self.__class__.__name__

    def log_params(self, params, model_name=None):
        pass

    def init_experiment(self, exp_name_log, full_name=None, setup=True, **kwargs):
        pass

    def set_tags(self, source, experiment_custom_tags, runtime, USI=None):
        pass

    def _construct_pipeline_if_needed(self, model, prep_pipe: Pipeline) -> Pipeline:
        """If model is a pipeline, return it, else append model to copy of prep_pipe."""
        if not isinstance(model, Pipeline):
            prep_pipe_temp = deepcopy(prep_pipe)
            prep_pipe_temp.steps.append(["trained_model", model])
        else:
            prep_pipe_temp = model
        return prep_pipe_temp

    def log_sklearn_pipeline(self, experiment, prep_pipe, model, path=None):
        pass

    def log_model_comparison(self, model_result, source):
        pass

    def log_metrics(self, metrics, source=None):
        pass

    def log_plot(self, plot, title):
        pass

    def log_hpram_grid(self, html_file, title="hpram_grid"):
        pass

    def log_artifact(self, file, type="artifact"):
        pass

    def finish_experiment(self):
        pass
