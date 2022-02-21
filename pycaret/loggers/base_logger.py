from abc import ABC, abstractclassmethod, abstractmethod


class BaseLogger:
    def init_logger():
        pass
    
    @abstractmethod
    def log_model(
    model,
    model_results,
    score_dict: dict,
    source: str,
    runtime: float,
    model_fit_time: float,
    _prep_pipe,
    log_holdout: bool = True,
    log_plots: bool = False,
    tune_cv_results=None
    ):
        pass
    
    def log_params(params, model_name=None):
        pass
    
    def log_experiment(log_profile, log_data):
        pass
    
    def set_tags(self, source, experiment_custom_tags, runtime):
        pass

    def log_sklearn_pipeline(self, pipeline):
        pass
    
    def log_model_comparison(self, model_result):
        pass
    
    def log_metrics(self, metrics):
        pass
