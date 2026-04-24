"""Clustering task — PyCaret 4.0.

PyCaret 4.0 is OOP-only; the 3.x functional API was removed.

    from pycaret.clustering import ClusteringExperiment
    exp = ClusteringExperiment(session_id=42).fit(df)
    kmeans = exp.create_model("kmeans", num_clusters=4).pipeline
    labelled = exp.assign_model(kmeans)
"""

from pycaret.tasks.clustering import ClusteringExperiment

__all__ = ["ClusteringExperiment"]
