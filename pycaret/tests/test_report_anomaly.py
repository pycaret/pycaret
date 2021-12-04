import os, sys
from pycaret.datasets import get_data
from pycaret.anomaly import *

sys.path.insert(0, os.path.abspath(".."))


def setup():
    dataset = get_data('mice')
    data = dataset.sample(frac=0.95, random_state=786)
    data_unseen = dataset.drop(data.index)
    data.reset_index(drop=True, inplace=True)
    data_unseen.reset_index(drop=True, inplace=True)
    exp_ano101 = setup(data, normalize=True,
                       ignore_features=['MouseID'],
                       session_id=123)


def test_report_abod():
    setup()
    model = create_model('abod')
    create_report(model, "report_anomaly_abod", "html")


def test_report_cluster():
    setup()
    model = create_model('cluster')
    create_report(model, "report_anomaly_cluster", "html")


def test_report_cof():
    setup()
    model = create_model('cof')
    create_report(model, "report_anomaly_cof", "html")


def test_report_histogram():
        setup()
        model = create_model('histogram')
        create_report(model, "report_anomaly_histogram", "html")


def test_report_knn():
    setup()
    model = create_model('knn')
    create_report(model, "report_anomaly_knn", "html")


def test_report_lof():
    setup()
    model = create_model('lof')
    create_report(model, "report_anomaly_lof", "html")


def test_report_svm():
    setup()
    model = create_model('svm')
    create_report(model, "report_anomaly_svm", "html")


def test_report_pca():
    setup()
    model = create_model('pca')
    create_report(model, "report_anomaly_pca", "html")


def test_report_mcd():
    setup()
    model = create_model('mcd')
    create_report(model, "report_anomaly_mcd", "html")


def test_report_sod():
    setup()
    model = create_model('sod')
    create_report(model, "report_anomaly_sod", "html")


def test_report_sos():
    setup()
    model = create_model('sos')
    create_report(model, "report_anomaly_sos", "html")


def test():
    test_report_abod()
    assert 1==1


if __name__ == "__main__":
    test()