"""Classification task — PyCaret 4.0.

PyCaret 4.0 is OOP-only; the 3.x module-level functional API
(``setup``/``compare_models``/…) was removed. Import the experiment class:

    from pycaret.classification import ClassificationExperiment
    # equivalent:
    from pycaret.tasks import ClassificationExperiment

Then construct and use it:

    exp = ClassificationExperiment(target="Purchase", session_id=42).fit(df)
    best = exp.compare_models().best
    preds = exp.predict_model(best).predictions
"""

from pycaret.tasks.classification import ClassificationExperiment

__all__ = ["ClassificationExperiment"]
