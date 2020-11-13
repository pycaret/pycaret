# Module: internal.plotting
# Author: Moez Ali <moez.ali@queensu.ca> and Antoni Baum (Yard1) <antoni.baum@protonmail.com>
# License: MIT

from typing import Any, Optional
from pycaret.internal.logging import get_logger
from pycaret.internal.Display import Display
import scikitplot as skplt


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
