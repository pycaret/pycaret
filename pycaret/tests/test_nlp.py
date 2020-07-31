import os, sys
sys.path.insert(0, os.path.abspath(".."))

#compare_models_test
import pytest
import pycaret.nlp
import pycaret.datasets

def test():
    # loading dataset
    data = pycaret.datasets.get_data('kiva')
    data = data.head(1000)

    # init setup
    nlp1 = pycaret.nlp.setup(data, target = 'en', html=False, session_id=123)

    # create model
    lda = pycaret.nlp.create_model('lda')
    nmf = pycaret.nlp.create_model('nmf')

    # assign model
    lda_results = pycaret.nlp.assign_model(lda)
    nmf_results = pycaret.nlp.assign_model(nmf)

    # get config
    data_ = pycaret.nlp.get_config('data_')
    seed = pycaret.nlp.get_config('seed')

    assert 1 == 1
    
if __name__ == "__main__":
    test()
