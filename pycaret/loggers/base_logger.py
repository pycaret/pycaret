from abc import ABC, abstractmethod


class BaseLogger(ABC):
    def init_logger():
        pass

    def __repr__(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def log_params(self, params, model_name=None):
        pass

    @abstractmethod
    def log_experiment(self, log_profile, log_data):
        pass

    @abstractmethod
    def set_tags(self, source, experiment_custom_tags, runtime):
        pass

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
