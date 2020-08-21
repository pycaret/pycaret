from pycaret.internal.utils import get_logger


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
    system: bool = True,
    logger=None,
    display=None,
    **kwargs,
):
    if not logger:
        logger = get_logger()
    visualizer.fig.set_dpi(visualizer.fig.dpi * scale)

    if handle_train == "draw":
        logger.info("Drawing Model")
        visualizer.draw(X_train, y_train, **kwargs)
    elif handle_train == "fit":
        logger.info("Fitting Model")
        visualizer.fit(X_train, y_train, **kwargs)
    elif handle_train == "fit_transform":
        logger.info("Fitting & Transforming Model")
        visualizer.fit_transform(X_train, y_train, **kwargs)
    elif handle_train == "score":
        logger.info("Scoring train set")
        visualizer.score(X_train, y_train, **kwargs)

    display.move_progress()

    if handle_test == "draw":
        visualizer.draw(X_test, y_test)
    elif handle_test == "fit":
        visualizer.fit(X_test, y_test)
    elif handle_test == "fit_transform":
        visualizer.fit_transform(X_test, y_test)
    elif handle_test == "score":
        logger.info("Scoring test/hold-out set")
        visualizer.score(X_test, y_test)

    display.move_progress()
    display.clear_output()

    if save:
        logger.info(f"Saving '{name}.png' in current active directory")
        visualizer.show(outpath=f"{name}.png", clear_figure=(not system))
    else:
        visualizer.show()

    logger.info("Visual Rendered Successfully")
