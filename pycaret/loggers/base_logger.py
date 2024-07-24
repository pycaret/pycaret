from abc import ABC, abstractmethod
from copy import deepcopy

from sklearn.pipeline import Pipeline

SETUP_TAG = "Session Initialized"


class BaseLogger(ABC):
    @abstractmethod
    def init_logger(self):
        pass

    def __del__(self):
        try:
            self.finish_experiment()
        except Exception:
            pass

    def __repr__(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def log_params(self, params, model_name=None):
        pass

    @abstractmethod
    def init_experiment(self, exp_name_log, full_name=None, setup=True, **kwargs):
        pass

    @abstractmethod
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

    @abstractmethod
    def log_sklearn_pipeline(self, experiment, prep_pipe, model, path=None):
        pass

    @abstractmethod
    def log_model_comparison(self, model_result, source):
        pass

    @abstractmethod
    def log_metrics(self, metrics, source=None):
        pass

    @abstractmethod
    def log_plot(self, plot, title):
        pass

    @abstractmethod
    def log_hpram_grid(self, html_file, title="hpram_grid"):
        pass

    @abstractmethod
    def log_artifact(self, file, type="artifact"):
        pass

    @abstractmethod
    def finish_experiment(self):
        pass
