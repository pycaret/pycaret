"""Stateless persistence helpers for PyCaret 4.0.

`save_model(pipeline, path)` and `load_model(path)` are the stateless,
top-level utilities for persisting fitted sklearn Pipelines. They're
equivalent to `joblib.dump` / `joblib.load` with a thin PyCaret metadata
header so loaded artifacts can be routed back to the right Experiment.

Unlike 3.x, these functions do not require an active experiment, do not
inspect global state, and do not mutate anything. A model saved from one
Experiment can be loaded into another (or into no Experiment at all).

Usage
-----

>>> from pycaret.tasks import ClassificationExperiment
>>> from pycaret import save_model, load_model
>>> exp = ClassificationExperiment(target="y").fit(df)
>>> model = exp.compare_models().best
>>> save_model(model, "artifacts/best.pkl")
>>> # Later, possibly in a different process:
>>> restored = load_model("artifacts/best.pkl")
>>> restored.predict(new_data)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

PathLike = str | Path


def save_model(model: Any, path: PathLike, *, verbose: bool = False) -> Path:
    """Persist a fitted model / pipeline to disk.

    Parameters
    ----------
    model : Any
        The sklearn Pipeline (or any picklable estimator) to persist.
    path : str | Path
        Destination. A ``.pkl`` suffix is added if missing.
    verbose : bool, default=False
        If True, print a confirmation line. Default silent for UI/agent use.

    Returns
    -------
    Path
        The absolute path of the written file.
    """
    import joblib

    dest = Path(path)
    if dest.suffix == "":
        dest = dest.with_suffix(".pkl")
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, dest)
    if verbose:
        # Intentional: top-level persistence helpers are allowed to print a
        # confirmation when the user explicitly opts in with verbose=True.
        print(f"Model saved: {dest.resolve()}")
    return dest.resolve()


def load_model(path: PathLike, *, verbose: bool = False) -> Any:
    """Load a model previously saved with `save_model`.

    Parameters
    ----------
    path : str | Path
        Path to the ``.pkl`` file. A ``.pkl`` suffix is added if missing.
    verbose : bool, default=False

    Returns
    -------
    Any
        The loaded object, typically a fitted sklearn Pipeline.
    """
    import joblib

    src = Path(path)
    if src.suffix == "" and not src.exists():
        src = src.with_suffix(".pkl")
    model = joblib.load(src)
    if verbose:
        print(f"Model loaded: {src.resolve()}")
    return model


__all__ = ["save_model", "load_model"]
