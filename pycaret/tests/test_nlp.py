import os, sys
sys.path.insert(0, os.path.abspath(".."))

import pytest
import pycaret.nlp
import pycaret.datasets

def test():
    # loading dataset
    data = pycaret.datasets.get_data('kiva')
    data = data.head(1000)

    # init setup
    nlp1 = pycaret.nlp.setup(data = data, target = 'en', session_id = 123)
    
    # create model
    lda = pycaret.nlp.create_model('lda')

    # assign model
    lda_results = pycaret.nlp.assign_model(lda)

    # evaluate model
    pycaret.nlp.evaluate_model(lda)

    assert 1 == 1
    
if __name__ == "__main__":
    test()