import os, sys
from pycaret.datasets import get_data
from pycaret.clustering import *

sys.path.insert(0, os.path.abspath(".."))


def _setup():
    dataset = get_data("mice")
    data = dataset.sample(frac=0.95, random_state=786)
    data_unseen = dataset.drop(data.index)
    data.reset_index(drop=True, inplace=True)
    data_unseen.reset_index(drop=True, inplace=True)
    exp_clu101 = setup(
        data, normalize=True, ignore_features=["MouseID"], session_id=123
    )


def test_report_kmeans():
    _setup()
    model = create_model("kmeans")
    create_report(model, "report_anomaly_kmeans", "html")


def test_report_ap():
    _setup()
    model = create_model("ap")
    create_report(model, "report_anomaly_ap", "html")


def test_report_meanshift():
    _setup()
    model = create_model("meanshift")
    create_report(model, "report_anomaly_meanshift", "html")


def test_report_sc():
    _setup()
    model = create_model("sc")
    create_report(model, "report_anomaly_sc", "html")


def test_report_hclust():
    _setup()
    model = create_model("hclust")
    create_report(model, "report_anomaly_hclust", "html")


def test_report_dbscan():
    _setup()
    model = create_model("dbscan")
    create_report(model, "report_anomaly_dbscan", "html")


def test_report_optics():
    _setup()
    model = create_model("optics")
    create_report(model, "report_anomaly_optics", "html")


def test_report_birch():
    _setup()
    model = create_model("birch")
    create_report(model, "report_anomaly_birch", "html")


def test_report_kmodes():
    _setup()
    model = create_model("kmodes")
    create_report(model, "report_anomaly_kmodes", "html")
