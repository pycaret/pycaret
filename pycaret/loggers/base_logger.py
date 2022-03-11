from abc import ABC, abstractclassmethod, abstractmethod


class BaseLogger:
    def init_logger():
        pass

    def log_params(params, model_name=None):
        pass

    def log_experiment(log_profile, log_data):
        pass

    def set_tags(self, source, experiment_custom_tags, runtime):
        pass

    def log_sklearn_pipeline(self, pipeline):
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
