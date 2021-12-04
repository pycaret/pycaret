import os, sys
from pycaret.datasets import get_data
import pycaret.clustering

sys.path.insert(0, os.path.abspath(".."))


def _setup():
    dataset = get_data("mice")
    data = dataset.sample(frac=0.95, random_state=786)
    data_unseen = dataset.drop(data.index)
    data.reset_index(drop=True, inplace=True)
    data_unseen.reset_index(drop=True, inplace=True)
    exp_clu101 = pycaret.clustering.setup(
        data, normalize=True, ignore_features=["MouseID"], session_id=123
    )


def test_report_kmeans():
    _setup()
    model = pycaret.clustering.create_model("kmeans")
    pycaret.clustering.create_report(model, "report_anomaly_kmeans", "html")


def test_report_ap():
    _setup()
    model = pycaret.clustering.create_model("ap")
    pycaret.clustering.create_report(model, "report_anomaly_ap", "html")


def test_report_meanshift():
    _setup()
    model = pycaret.clustering.create_model("meanshift")
    pycaret.clustering.create_report(model, "report_anomaly_meanshift", "html")


def test_report_sc():
    _setup()
    model = pycaret.clustering.create_model("sc")
    pycaret.clustering.create_report(model, "report_anomaly_sc", "html")


def test_report_hclust():
    _setup()
    model = pycaret.clustering.create_model("hclust")
    pycaret.clustering.create_report(model, "report_anomaly_hclust", "html")


def test_report_dbscan():
    _setup()
    model = pycaret.clustering.create_model("dbscan")
    pycaret.clustering.create_report(model, "report_anomaly_dbscan", "html")


def test_report_optics():
    _setup()
    model = pycaret.clustering.create_model("optics")
    pycaret.clustering.create_report(model, "report_anomaly_optics", "html")


def test_report_birch():
    _setup()
    model = pycaret.clustering.create_model("birch")
    pycaret.clustering.create_report(model, "report_anomaly_birch", "html")


def test_report_kmodes():
    _setup()
    model = pycaret.clustering.create_model("kmodes")
    pycaret.clustering.create_report(model, "report_anomaly_kmodes", "html")
