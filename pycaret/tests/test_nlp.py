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
    
    # save model
    pycaret.nlp.save_model(lda, 'lda_model_23122019')
    
    # load model
    saved_lda = pycaret.nlp.load_model('lda_model_23122019')
    
    # returns table of models
    all_models = pycaret.nlp.models()
    
    # get config
    text = pycaret.nlp.get_config('text') 
    
    # set config
    pycaret.nlp.set_config('seed', 123) 

    assert 1 == 1
    
if __name__ == "__main__":
    test()
