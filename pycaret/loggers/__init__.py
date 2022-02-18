from typing import List
from .base_logger import BaseLogger

class DashboardLogger:
    def __init__(self, logger_list: List[BaseLogger]) -> None:
        self.loggers = logger_list
    
    def log_experment(self, log_profile, log_data, ml_usecase, functions, experiment_custom_tags, runtime, display):
        for logger in self.loggers:
            logger.log_experiment(log_profile, log_data, ml_usecase, functions, experiment_custom_tags, runtime, display)
    
    