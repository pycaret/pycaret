# Module: internal.logging
# Author: Antoni Baum (Yard1) <antoni.baum@protonmail.com>
# License: MIT

import logging


def get_logger() -> logging.Logger:
    try:
        assert bool(LOGGER)
        return LOGGER
    except:
        return create_logger()


def create_logger() -> logging.Logger:
    logger = logging.getLogger("logs")
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    if logger.hasHandlers():
        logger.handlers.clear()

    ch = logging.FileHandler("logs.log")
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

    return logger


LOGGER = create_logger()
