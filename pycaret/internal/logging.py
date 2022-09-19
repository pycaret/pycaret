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


def get_logger() -> logging.Logger:
    try:
        assert bool(LOGGER)
        return LOGGER
    except Exception:
        return create_logger(True)


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
        return DummyLogger(name="DummyLogger")
    elif isinstance(log, logging.Logger):
        return log

    logger = logging.getLogger("logs")
    level = os.getenv("PYCARET_CUSTOM_LOGGING_LEVEL", "DEBUG")
    logger.setLevel(level)
    # Do not propagate to the root logger in Jupyter
    logger.propagate = False

    # create console handler and set level to debug
    if logger.hasHandlers():
        logger.handlers.clear()

    path = "logs.log" if isinstance(log, bool) else log
    try:
        ch = logging.FileHandler(path)
    except Exception:
        warnings.warn(
            f"Could not attach a FileHandler to the logger at path {path}! "
            "No logs will be saved."
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


# From https://stackoverflow.com/questions/28367810/how-to-change-the-logger-associated-to-logging-capturewarnings
# Redirect all warnings to our logger
_warnings_showwarning = None


def _showwarning(message, category, filename, lineno, file=None, line=None):
    """
    Implementation of showwarnings which redirects to logging, which will first
    check to see if the file parameter is None. If a file is specified, it will
    delegate to the original warnings implementation of showwarning. Otherwise,
    it will call warnings.formatwarning and will log the resulting string to a
    warnings logger named "py.warnings" with level logging.WARNING.
    """
    if file is not None:
        if _warnings_showwarning is not None:
            _warnings_showwarning(message, category, filename, lineno, file, line)
    else:
        s = warnings.formatwarning(message, category, filename, lineno, line)
        logger = LOGGER
        if not logger.handlers:
            logger.addHandler(logging.NullHandler())
        # bpo-46557: Log str(s) as msg instead of logger.warning("%s", s)
        # since some log aggregation tools group logs by the msg arg
        logger.warning(str(s))


def captureWarnings(capture):
    """
    If capture is true, redirect all warnings to the logging package.
    If capture is False, ensure that warnings are not redirected to logging
    but to their original destinations.
    """
    global _warnings_showwarning
    if capture:
        if _warnings_showwarning is None:
            _warnings_showwarning = warnings.showwarning
            warnings.showwarning = _showwarning
    else:
        if _warnings_showwarning is not None:
            warnings.showwarning = _warnings_showwarning
            _warnings_showwarning = None


captureWarnings(True)
