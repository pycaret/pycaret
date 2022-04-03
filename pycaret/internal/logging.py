# Module: internal.logging
# Author: Antoni Baum (Yard1) <antoni.baum@protonmail.com>
# License: MIT

import logging
import traceback
import os
from typing import Union, Optional


class DummyLogger:
    """Logger class that does nothing.

    This class is a dummy logger class assigned to self.logger
    when system_log=None to overwrite commands to the logger.

    """

    def info(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        pass


def get_logger(name: str = "logs") -> logging.Logger:
    try:
        assert bool(LOGGER)
        return LOGGER
    except:
        return create_logger(name)


def create_logger(
    log: Union[bool, logging.Logger] = True
) -> Union[logging.Logger, DummyLogger]:
    """Create and return a logger object.

    Parameters
    ----------
    log: bool or logging.Logger, default = True
        - If False, don't create any logger (return dummy).
        - If True, create a default logger (logs.log).
        - If logging.Logger, returns the object unchanged.

    """
    if not log:
        return DummyLogger()
    elif isinstance(log, logging.Logger):
        return log

    logger = logging.getLogger("logs")
    level = os.getenv("PYCARET_CUSTOM_LOGGING_LEVEL", "DEBUG")
    logger.setLevel(level)

    # create console handler and set level to debug
    if logger.hasHandlers():
        logger.handlers.clear()

    try:
        ch = logging.FileHandler("logs.log")
    except:
        print("Could not attach a FileHandler to the logger! No logs will be saved.")
        traceback.print_exc()
        ch = logging.NullHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

    return logger


LOGGER = create_logger()
