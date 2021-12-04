import os, sys
from pycaret.datasets import get_data
from pycaret.anomaly import setup, create_model, create_report

sys.path.insert(0, os.path.abspath(".."))


def _setup():
    dataset = get_data("mice")
    data = dataset.sample(frac=0.95, random_state=786)
    data_unseen = dataset.drop(data.index)
    data.reset_index(drop=True, inplace=True)
    data_unseen.reset_index(drop=True, inplace=True)
    exp_ano101 = setup(
        data, normalize=True, ignore_features=["MouseID"], session_id=123
    )


def test_report_abod():
    _setup()
    model = create_model("abod")
    create_report(model, "report_anomaly_abod", "html")


def test_report_cluster():
    _setup()
    model = create_model("cluster")
    create_report(model, "report_anomaly_cluster", "html")


def test_report_cof():
    _setup()
    model = create_model("cof")
    create_report(model, "report_anomaly_cof", "html")


def test_report_histogram():
    _setup()
    model = create_model("histogram")
    create_report(model, "report_anomaly_histogram", "html")


def test_report_knn():
    _setup()
    model = create_model("knn")
    create_report(model, "report_anomaly_knn", "html")


def test_report_lof():
    _setup()
    model = create_model("lof")
    create_report(model, "report_anomaly_lof", "html")


def test_report_svm():
    _setup()
    model = create_model("svm")
    create_report(model, "report_anomaly_svm", "html")


def test_report_pca():
    _setup()
    model = create_model("pca")
    create_report(model, "report_anomaly_pca", "html")


def test_report_mcd():
    _setup()
    model = create_model("mcd")
    create_report(model, "report_anomaly_mcd", "html")


def test_report_sod():
    _setup()
    model = create_model("sod")
    create_report(model, "report_anomaly_sod", "html")


def test_report_sos():
    _setup()
    model = create_model("sos")
    create_report(model, "report_anomaly_sos", "html")
