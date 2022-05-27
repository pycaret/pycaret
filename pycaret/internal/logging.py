# Module: internal.logging
# Author: Antoni Baum (Yard1) <antoni.baum@protonmail.com>
# License: MIT

import logging
import os
import traceback
import warnings
from contextlib import redirect_stderr, redirect_stdout
from typing import Callable, Optional, Union


# From https://stackoverflow.com/a/66209331
class LoggerWriter:
    """Writer allowing redirection of streams to logger methods."""

    def __init__(self, logfct: Callable):
        self.logfct = logfct
        self.buf = []

    def write(self, msg: str):
        if msg.endswith("\n"):
            self.buf.append(msg.rstrip("\n"))
            self.logfct("".join(self.buf))
            self.buf = []
        else:
            self.buf.append(msg)

    def flush(self):
        pass


class redirect_output:
    """Context manager to redirect stdout and stderr to logger."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or DummyLogger("dummy")
        self.redirect_stdout = redirect_stdout(LoggerWriter(logger.info))
        self.redirect_stderr = redirect_stderr(LoggerWriter(logger.warning))

    def __enter__(self):
        self.redirect_stdout.__enter__()
        self.redirect_stderr.__enter__()

    def __exit__(self, *args, **kwargs):
        self.redirect_stdout.__exit__(*args, **kwargs)
        self.redirect_stderr.__exit__(*args, **kwargs)


class DummyLogger(logging.Logger):
    """Logger class that does nothing.

    This class is a dummy logger class assigned to self.logger
    when system_log=None to overwrite commands to the logger.

    """

    def debug(*args, **kwargs):
        pass

    def info(*args, **kwargs):
        pass

    def warning(*args, **kwargs):
        pass

    def warn(*args, **kwargs):
        pass

    def error(*args, **kwargs):
        pass

    def exception(*args, **kwargs):
        pass

    def critical(*args, **kwargs):
        pass

    def log(*args, **kwargs):
        pass


def get_logger(name: str = "logs") -> logging.Logger:
    try:
        assert bool(LOGGER)
        return LOGGER
    except:
        return create_logger(name)


def create_logger(
    log: Union[bool, str, logging.Logger] = True
) -> Union[logging.Logger, DummyLogger]:
    """Create and return a logger object.

    Parameters
    ----------
    log: bool or str or logging.Logger, default = True
        - If False, don't create any logger (return dummy).
        - If True, create a default logger at logs.log.
        - If str, create a default logger at that path.
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

    path = "logs.log" if isinstance(log, bool) else log
    try:
        ch = logging.FileHandler(path)
    except:
        warnings.warn(
            f"Could not attach a FileHandler to the logger at path {path}! No logs will be saved."
        )
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
