import os, sys

sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import pytest
import pycaret.nlp
import pycaret.datasets


def test():
    # loading dataset
    data = pycaret.datasets.get_data("kiva")
    data = data.head(1000)
    assert isinstance(data, pd.core.frame.DataFrame)

    # init setup
    nlp1 = pycaret.nlp.setup(
        data=data,
        target="en",
        log_experiment=True,
        html=False,
        session_id=123,
    )
    assert isinstance(nlp1, tuple)
    assert isinstance(nlp1[0], list)
    assert isinstance(nlp1[1], pd.core.frame.DataFrame)
    assert isinstance(nlp1[2], list)
    assert isinstance(nlp1[4], int)
    assert isinstance(nlp1[5], str)
    assert isinstance(nlp1[6], list)
    assert isinstance(nlp1[7], str)
    assert isinstance(nlp1[8], bool)
    assert isinstance(nlp1[9], bool)

    # create model
    lda = pycaret.nlp.create_model("lda")

    # assign model
    lda_results = pycaret.nlp.assign_model(lda)
    assert isinstance(lda_results, pd.core.frame.DataFrame)

    # evaluate model
    pycaret.nlp.evaluate_model(lda)

    # save model
    pycaret.nlp.save_model(lda, "lda_model_23122019")

    # load model
    saved_lda = pycaret.nlp.load_model("lda_model_23122019")

    # returns table of models
    all_models = pycaret.nlp.models()
    assert isinstance(all_models, pd.core.frame.DataFrame)

    # get config
    text = pycaret.nlp.get_config("text")
    assert isinstance(text, list)

    # set config
    pycaret.nlp.set_config("seed", 124)
    seed = pycaret.nlp.get_config("seed")
    assert seed == 124

    assert 1 == 1


if __name__ == "__main__":
    test()
