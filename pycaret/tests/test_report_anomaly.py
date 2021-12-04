import os, sys
from pycaret.datasets import get_data
import pycaret.anomaly

sys.path.insert(0, os.path.abspath(".."))


def _setup():
    dataset = get_data("mice")
    data = dataset.sample(frac=0.95, random_state=786)
    data_unseen = dataset.drop(data.index)
    data.reset_index(drop=True, inplace=True)
    data_unseen.reset_index(drop=True, inplace=True)
    exp_ano101 = pycaret.anomaly.setup(
        data, normalize=True, ignore_features=["MouseID"], session_id=123
    )


def test_report_abod():
    _setup()
    model = pycaret.anomaly.create_model("abod")
    pycaret.anomaly.create_report(model, "report_anomaly_abod", "html")


def test_report_cluster():
    _setup()
    model = pycaret.anomaly.create_model("cluster")
    pycaret.anomaly.create_report(model, "report_anomaly_cluster", "html")


def test_report_cof():
    _setup()
    model = pycaret.anomaly.create_model("cof")
    pycaret.anomaly.create_report(model, "report_anomaly_cof", "html")


def test_report_histogram():
    _setup()
    model = pycaret.anomaly.create_model("histogram")
    pycaret.anomaly.create_report(model, "report_anomaly_histogram", "html")


def test_report_knn():
    _setup()
    model = pycaret.anomaly.create_model("knn")
    pycaret.anomaly.create_report(model, "report_anomaly_knn", "html")


def test_report_lof():
    _setup()
    model = pycaret.anomaly.create_model("lof")
    pycaret.anomaly.create_report(model, "report_anomaly_lof", "html")


def test_report_svm():
    _setup()
    model = pycaret.anomaly.create_model("svm")
    pycaret.anomaly.create_report(model, "report_anomaly_svm", "html")


def test_report_pca():
    _setup()
    model = pycaret.anomaly.create_model("pca")
    pycaret.anomaly.create_report(model, "report_anomaly_pca", "html")


def test_report_mcd():
    _setup()
    model = pycaret.anomaly.create_model("mcd")
    pycaret.anomaly.create_report(model, "report_anomaly_mcd", "html")


def test_report_sod():
    _setup()
    model = pycaret.anomaly.create_model("sod")
    pycaret.anomaly.create_report(model, "report_anomaly_sod", "html")


def test_report_sos():
    _setup()
    model = pycaret.anomaly.create_model("sos")
    pycaret.anomaly.create_report(model, "report_anomaly_sos", "html")
