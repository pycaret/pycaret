# Module: internal.logging
# Author: Antoni Baum (Yard1) <antoni.baum@protonmail.com>
# License: MIT

import logging
import traceback
from typing import Union, Optional


class DummyLogger:
    """Logger class that does nothing.

    This class is a dummy logger class assigned to self.logger when
    experiment_name is None to overwrite commands to the logger.
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


def create_logger(name: Optional[str] = None) -> Union[logging.Logger, DummyLogger]:
    if not name:
        return DummyLogger()

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    if logger.hasHandlers():
        logger.handlers.clear()

    try:
        ch = logging.FileHandler(f"{name}.log")
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
