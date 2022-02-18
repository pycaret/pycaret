from typing import List
from .base_logger import BaseLogger

class DashboardLogger:
    def __init__(self, logger_list: List[BaseLogger]) -> None:
        self.loggers = BaseLogger
    
    