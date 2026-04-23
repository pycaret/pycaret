"""Anomaly-detection task — PyCaret 4.0.

PyCaret 4.0 is OOP-only; the 3.x functional API was removed.

    from pycaret.anomaly import AnomalyExperiment
    exp = AnomalyExperiment(session_id=42).fit(df)
    iforest = exp.create_model("iforest").pipeline
    labelled = exp.assign_model(iforest)
"""

from pycaret.tasks.anomaly import AnomalyExperiment

__all__ = ["AnomalyExperiment"]
