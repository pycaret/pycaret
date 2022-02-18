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
    
    @abstractmethod
    def log_params(params):
        pass
    
    @abstractmethod
    def log_experiment(log_profile, log_data):
        pass
    

