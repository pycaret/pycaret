"""Regression task — PyCaret 4.0.

PyCaret 4.0 is OOP-only; the 3.x module-level functional API was removed.

    from pycaret.regression import RegressionExperiment
    exp = RegressionExperiment(target="medv", session_id=42).fit(df)
    best = exp.compare_models().best
    preds = exp.predict_model(best).predictions
"""

from pycaret.tasks.regression import RegressionExperiment

__all__ = ["RegressionExperiment"]
