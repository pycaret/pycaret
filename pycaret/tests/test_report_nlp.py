import os, sys
from pycaret.datasets import get_data
from pycaret.nlp import *


sys.path.insert(0, os.path.abspath(".."))


def _setup():
    data = get_data("kiva")
    # sampling the data to select only 1000 documents
    data = data.sample(1000, random_state=786).reset_index(drop=True)

    exp_nlp102 = setup(
        data=data,
        target="en",
        session_id=123,
        custom_stopwords=[
            "loan",
            "income",
            "usd",
            "many",
            "also",
            "make",
            "business",
            "buy",
            "sell",
            "purchase",
            "year",
            "people",
            "able",
            "enable",
            "old",
            "woman",
            "child",
            "school",
        ],
        log_experiment=True,
        experiment_name="kiva1",
    )


def test_report_lda():
    _setup()
    #  Latent Dirichlet Allocation
    model = create_model("lda")
    create_report(model, "report_nlp_lds", "html")


def test_report_lsi():
    _setup()
    # Latent Semantic Indexing
    model = create_model("lsi")
    create_report(model, "report_nlp_lsi", "html")


def test_report_hdp():
    _setup()
    # Hierarchical Dirichlet Process
    model = create_model("hdp")
    create_report(model, "report_nlp_hdp", "html")


def test_report_rp():
    _setup()
    # Random Projections
    model = create_model("rp")
    create_report(model, "report_nlp_rp", "html")


def test_report_nmf():
    _setup()
    # Non - Negative Matrix Factorization
    model = create_model("nmf")
    create_report(model, "report_nlp_nmf", "html")
