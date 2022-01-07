# Module: internal.logging
# Author: Antoni Baum (Yard1) <antoni.baum@protonmail.com>
# License: MIT

import logging
import traceback
import os


def get_logger() -> logging.Logger:
    try:
        assert bool(LOGGER)
        return LOGGER
    except:
        return create_logger()


def create_logger() -> logging.Logger:
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
