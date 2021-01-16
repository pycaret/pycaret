# Module: internal.plotting
# Author: Moez Ali <moez.ali@queensu.ca> and Antoni Baum (Yard1) <antoni.baum@protonmail.com>
# License: MIT

from typing import Any, Optional
from pycaret.internal.logging import get_logger
from pycaret.internal.Display import Display
import scikitplot as skplt
import matplotlib.pyplot as plt

def show_yellowbrick_in_streamlit(
    visualizer, outpath=None, clear_figure=False, **kwargs
):
    """
    Makes the magic happen and a visualizer appear! You can pass in a path to
    save the figure to disk with various backends, or you can call it with no
    arguments to show the figure either in a notebook or in a GUI window that
    pops up on screen.

    Parameters
    ----------
    outpath: string, default: None
        path or None. Save figure to disk or if None show in window

    clear_figure: boolean, default: False
        When True, this flag clears the figure after saving to file or
        showing on screen. This is useful when making consecutive plots.

    kwargs: dict
        generic keyword arguments.

    Notes
    -----
    Developers of visualizers don't usually override show, as it is
    primarily called by the user to render the visualization.
    """
    import streamlit as st

    # Finalize the figure
    visualizer.finalize()

    if outpath is not None:
        plt.savefig(outpath, **kwargs)
    else:
        st.write(visualizer.fig)

    if clear_figure:
        visualizer.fig.clear()

    # Return ax to ensure display in notebooks
    return visualizer.ax


def show_yellowbrick_plot(
    visualizer,
    X_train,
    y_train,
    X_test,
    y_test,
    name: str,
    handle_train: str = "fit",
    handle_test: str = "score",
    scale: float = 1,
    save: bool = False,
    fit_kwargs: Optional[dict] = None,
    groups: Optional[Any] = None,
    display: Optional[Display] = None,
    display_format: Optional[str] = None,
    **kwargs,
):
    """
    Generic method to handle yellowbrick plots.
    """
    logger = get_logger()
    visualizer.fig.set_dpi(visualizer.fig.dpi * scale)

    if not fit_kwargs:
        fit_kwargs = {}

    fit_kwargs_and_kwargs = {**fit_kwargs, **kwargs}

    if handle_train == "draw":
        logger.info("Drawing Model")
        visualizer.draw(X_train, y_train, **kwargs)
    elif handle_train == "fit":
        logger.info("Fitting Model")
        visualizer.fit(X_train, y_train, **fit_kwargs_and_kwargs)
    elif handle_train == "fit_transform":
        logger.info("Fitting & Transforming Model")
        visualizer.fit_transform(X_train, y_train, **fit_kwargs_and_kwargs)
    elif handle_train == "score":
        logger.info("Scoring train set")
        visualizer.score(X_train, y_train, **kwargs)

    display.move_progress()

    if handle_test == "draw":
        visualizer.draw(X_test, y_test)
    elif handle_test == "fit":
        visualizer.fit(X_test, y_test, **fit_kwargs)
    elif handle_test == "fit_transform":
        visualizer.fit_transform(X_test, y_test, **fit_kwargs)
    elif handle_test == "score":
        logger.info("Scoring test/hold-out set")
        visualizer.score(X_test, y_test)

    display.move_progress()
    display.clear_output()

    if save:
        logger.info(f"Saving '{name}.png' in current active directory")
        visualizer.show(outpath=f"{name}.png", clear_figure=True)
    else:
        if display_format == "streamlit":
            show_yellowbrick_in_streamlit(visualizer, clear_figure=True)
        else:
            visualizer.show(clear_figure=True)

    logger.info("Visual Rendered Successfully")


class MatplotlibDefaultDPI(object):
    def __init__(self, base_dpi: float = 100, scale_to_set: float = 1):
        try:
            self.default_skplt_dpit = skplt.metrics.plt.rcParams["figure.dpi"]
            skplt.metrics.plt.rcParams["figure.dpi"] = base_dpi * scale_to_set
        except:
            pass

    def __enter__(self) -> None:
        return None

    def __exit__(self, type, value, traceback):
        try:
            skplt.metrics.plt.rcParams["figure.dpi"] = self.default_skplt_dpit
        except:
            pass
